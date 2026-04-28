import os
import pickle
import numpy as np
import pandas as pd

from clustering.train_kmeans import (
    MODEL_PATH, SCALER_PATH, LABELS_PATH, CLUSTER_FEATURES
)

# Description des zones environnementales (A, B, C, D) 
ZONE_INFO = {
    "A": {
        "label":       "Saine",
        "description": "Air propre, humidité normale",
        "recommendation": "Sport autorisé",
        "color":       "#2ecc71",
        "risk_score":  1,
    },
    "B": {
        "label":       "Modérée",
        "description": "Pollution légère, ventilation correcte",
        "recommendation": "Surveillance conseillée",
        "color":       "#f39c12",
        "risk_score":  2,
    },
    "C": {
        "label":       "Risquée",
        "description": "PM2.5 élevé, CO2 fort",
        "recommendation": "Éviter l'activité physique",
        "color":       "#e67e22",
        "risk_score":  3,
    },
    "D": {
        "label":       "Critique",
        "description": "Pollution forte + humidité élevée",
        "recommendation": "Alerte urgente — rester à l'intérieur",
        "color":       "#e74c3c",
        "risk_score":  4,
    },
}


# Chargement du modèle
_model_cache = {}


def _load_model() -> tuple:
    if "kmeans" in _model_cache:
        return (
            _model_cache["kmeans"],
            _model_cache["scaler"],
            _model_cache["labels"],
            _model_cache["features"],
        )

    for path, name in [(MODEL_PATH, "kmeans.pkl"),
                       (SCALER_PATH, "kmeans_scaler.pkl"),
                       (LABELS_PATH, "cluster_labels.pkl")]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Modèle '{name}' introuvable : {path}\n"
                "Lancez d'abord : python -m ai.clustering.train_kmeans"
            )

    with open(MODEL_PATH, "rb") as f:
        kmeans = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(LABELS_PATH, "rb") as f:
        meta = pickle.load(f)

    cluster_labels = meta["cluster_labels"]
    feature_names  = meta["feature_names"]

    _model_cache.update({
        "kmeans":   kmeans,
        "scaler":   scaler,
        "labels":   cluster_labels,
        "features": feature_names,
    })

    return kmeans, scaler, cluster_labels, feature_names


def reload_model():
    """Force le rechargement du modèle depuis le disque (après ré-entraînement)."""
    _model_cache.clear()
    _load_model()
    print("Modèle K-Means rechargé.")



# Prédiction


def predict_zone(
    pm25: float,
    co: float,
    humidity: float,
    temperature: float,
    ier_score: float,
) -> dict:
  
    kmeans, scaler, cluster_labels, feature_names = _load_model()
    input_dict = {
        "pm25": pm25,
        "co": co,
        "humidity": humidity,
        "temperature": temperature,
        "ier_score": ier_score,
    }
    X = np.array([[input_dict[f] for f in feature_names]])

    # Normalisation
    X_scaled = scaler.transform(X)

    # Prédiction
    cluster_id = int(kmeans.predict(X_scaled)[0])
    zone = cluster_labels.get(cluster_id, "A")
    zone_info = ZONE_INFO[zone].copy()

    return {
        "zone":           zone,
        "cluster_id":     cluster_id,
        "label":          zone_info["label"],
        "description":    zone_info["description"],
        "recommendation": zone_info["recommendation"],
        "color":          zone_info["color"],
        "risk_score":     zone_info["risk_score"],
        "input": {
            "pm25":        pm25,
            "co":          co,
            "humidity":    humidity,
            "temperature": temperature,
            "ier_score":   ier_score,
        },
    }


def predict_zone_batch(df: pd.DataFrame) -> pd.DataFrame:
    kmeans, scaler, cluster_labels, feature_names = _load_model()

    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes pour predict_zone_batch : {missing}")

    df = df.copy()
    X = df[feature_names].values
    X_scaled = scaler.transform(X)

    cluster_ids = kmeans.predict(X_scaled)
    zones = [cluster_labels.get(int(c), "A") for c in cluster_ids]

    df["cluster_id"]  = cluster_ids
    df["zone"]        = zones
    df["zone_label"]  = [ZONE_INFO[z]["label"] for z in zones]
    df["zone_color"]  = [ZONE_INFO[z]["color"] for z in zones]
    df["zone_recommendation"] = [ZONE_INFO[z]["recommendation"] for z in zones]

    print(f"Zones prédites pour {len(df)} observations :")
    print(df["zone"].value_counts().sort_index().to_string())
    return df



def get_zone_distances(
    pm25: float,
    co: float,
    humidity: float,
    temperature: float,
    ier_score: float,
) -> dict:
    kmeans, scaler, cluster_labels, feature_names = _load_model()

    input_dict = {
        "pm25": pm25,
        "co": co,
        "humidity": humidity,
        "temperature": temperature,
        "ier_score": ier_score,
    }
    X = np.array([[input_dict[f] for f in feature_names]])
    X_scaled = scaler.transform(X)

    distances = kmeans.transform(X_scaled)[0]
    result = {
        cluster_labels.get(i, str(i)): round(float(d), 4)
        for i, d in enumerate(distances)
    }
    return dict(sorted(result.items(), key=lambda x: x[1]))


# Statistiques par zone sur un DataFrame
def zone_statistics(df: pd.DataFrame) -> pd.DataFrame:
    
    if "zone" not in df.columns:
        df = predict_zone_batch(df)

    stat_cols = ["pm25", "co", "humidity", "temperature", "ier_score"]
    available = [c for c in stat_cols if c in df.columns]

    stats = df.groupby("zone")[available].agg(["mean", "std", "count"])
    stats.columns = ["_".join(col) for col in stats.columns]
    stats = stats.round(3)

    print("\nStatistiques par zone :")
    print(stats.to_string())
    return stats



if __name__ == "__main__":
    print("=" * 50)
    print("TEST — Prédiction de zone (temps réel)")
    print("=" * 50)

    # Scénarios test
    scenarios = [
        {"name": "Air sain",       "pm25": 5.0,  "co": 0.3, "humidity": 45.0, "temperature": 20.0, "ier_score": 10.0},
        {"name": "Modéré",         "pm25": 25.0, "co": 1.0, "humidity": 60.0, "temperature": 25.0, "ier_score": 38.0},
        {"name": "Risqué",         "pm25": 80.0, "co": 3.5, "humidity": 78.0, "temperature": 30.0, "ier_score": 65.0},
        {"name": "Critique",       "pm25": 200.0,"co": 8.0, "humidity": 90.0, "temperature": 35.0, "ier_score": 88.0},
    ]

    try:
        for s in scenarios:
            result = predict_zone(
                s["pm25"], s["co"], s["humidity"],
                s["temperature"], s["ier_score"]
            )
            print(f"\n  {s['name']:12s} → Zone {result['zone']} "
                  f"({result['label']}) | {result['recommendation']}")
    except FileNotFoundError as e:
        print(f"\n  Modèle non trouvé : {e}")
        print("  Lancez d'abord : python -m ai.clustering.train_kmeans")