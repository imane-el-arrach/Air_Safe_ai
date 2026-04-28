import os
import pickle
import numpy as np
import pandas as pd
from typing import Optional

from ier.weights import get_weights, get_risk_level, Pathologie, WEIGHTS

DEFAULT_BOUNDS: dict[str, dict[str, float]] = {
    "pm25":        {"min": 0.0,   "max": 500.0},
    "co":          {"min": 0.0,   "max": 50.0},
    "humidity":    {"min": 0.0,   "max": 100.0},
    "temperature": {"min": -30.0, "max": 50.0},
}

FEATURES = ["pm25", "co", "humidity", "temperature"]

# Normalisation


def normalize_value(
    value: float,
    feature: str,
    bounds: Optional[dict] = None,
) -> float:
    b = (bounds or DEFAULT_BOUNDS).get(feature, DEFAULT_BOUNDS.get(feature))
    if b is None:
        raise ValueError(f"Feature '{feature}' inconnue.")
    xmin, xmax = b["min"], b["max"]
    if xmax == xmin:
        return 0.0
    normalized = (value - xmin) / (xmax - xmin) * 100.0
    return float(np.clip(normalized, 0.0, 100.0))


def normalize_series(
    series: pd.Series,
    feature: str,
    bounds: Optional[dict] = None,
) -> pd.Series:
    """Normalise une Series pandas entière."""
    b = (bounds or DEFAULT_BOUNDS).get(feature, DEFAULT_BOUNDS.get(feature))
    xmin, xmax = b["min"], b["max"]
    if xmax == xmin:
        return pd.Series(np.zeros(len(series)), index=series.index)
    normalized = (series - xmin) / (xmax - xmin) * 100.0
    return normalized.clip(0.0, 100.0)


# Calcul IER — valeur unique (temps réel)


def compute_ier_single(
    pm25: float,
    co: float,
    humidity: float,
    temperature: float,
    pathologie: Pathologie = "general",
    bounds: Optional[dict] = None,
) -> dict:
    weights = get_weights(pathologie)
    values = {
        "pm25": pm25,
        "co": co,
        "humidity": humidity,
        "temperature": temperature,
    }

    details = {}
    score = 0.0

    for feature in FEATURES:
        raw_val = values[feature]
        norm_val = normalize_value(raw_val, feature, bounds)
        weight = weights[feature]
        contribution = weight * norm_val
        score += contribution
        details[feature] = {
            "raw":          round(raw_val, 3),
            "normalized":   round(norm_val, 2),
            "weight":       weight,
            "contribution": round(contribution, 2),
        }

    score = round(min(max(score, 0.0), 100.0), 2)
    risk = get_risk_level(score)

    return {
        "score":     score,
        "level":     risk["level"],
        "color":     risk["color"],
        "action":    risk["action"],
        "details":   details,
        "pathologie": pathologie,
    }


# Calcul IER — DataFrame complet (batch)

def compute_ier_dataframe(
    df: pd.DataFrame,
    pathologie: Pathologie = "general",
    bounds: Optional[dict] = None,
    add_details: bool = False,
) -> pd.DataFrame:
    df = df.copy()
    weights = get_weights(pathologie)

    # Vérification des colonnes obligatoires
    missing_cols = [f for f in FEATURES if f not in df.columns]
    if missing_cols:
        raise ValueError(f"Colonnes manquantes dans le DataFrame : {missing_cols}")

    # Normalisation de chaque feature
    norm_cols = {}
    for feature in FEATURES:
        norm_col = f"{feature}_norm_ier"
        df[norm_col] = normalize_series(df[feature], feature, bounds)
        norm_cols[feature] = norm_col

    # Score IER = somme pondérée
    df["ier_score"] = sum(
        weights[f] * df[norm_cols[f]] for f in FEATURES
    ).clip(0.0, 100.0).round(2)

    # Niveau de risque
    df["ier_level"] = df["ier_score"].apply(
        lambda s: get_risk_level(s)["level"]
    )
    df["ier_action"] = df["ier_score"].apply(
        lambda s: get_risk_level(s)["action"]
    )
    df["ier_color"] = df["ier_score"].apply(
        lambda s: get_risk_level(s)["color"]
    )

    # Contributions par feature 
    if add_details:
        for feature in FEATURES:
            df[f"{feature}_contrib"] = (
                weights[feature] * df[norm_cols[feature]]
            ).round(3)

    # Nettoyage des colonnes intermédiaires 
    if not add_details:
        df.drop(columns=list(norm_cols.values()), inplace=True)

    print(f"IER calculé pour {len(df)} lignes | pathologie: {pathologie}")
    print(df["ier_level"].value_counts().to_string())
    return df


# Calcul multi-pathologie 


def compute_ier_all_pathologies(
    pm25: float,
    co: float,
    humidity: float,
    temperature: float,
    bounds: Optional[dict] = None,
) -> dict[str, dict]:
    return {
        p: compute_ier_single(pm25, co, humidity, temperature, p, bounds)
        for p in WEIGHTS
    }


# Chargement des scalers du preprocessing

def load_bounds_from_scalers(scaler_path: str) -> dict:
    if not os.path.exists(scaler_path):
        print(f"  scalers.pkl introuvable ({scaler_path}). Utilisation des bornes par défaut.")
        return DEFAULT_BOUNDS
    with open(scaler_path, "rb") as f:
        scalers = pickle.load(f)
    print(f"  Bornes chargées depuis : {scaler_path}")
    return scalers

# Point d'entrée — test rapide


if __name__ == "__main__":
    print("=" * 50)
    print("TEST — Calcul IER (valeur unique)")
    print("=" * 50)

    # Exemple : environnement modérément pollué
    result = compute_ier_single(
        pm25=45.0,
        co=2.5,
        humidity=72.0,
        temperature=28.0,
        pathologie="asthme",
    )
    print(f"\nScore IER    : {result['score']}")
    print(f"Niveau       : {result['level']}")
    print(f"Action       : {result['action']}")
    print("\nDétail par feature :")
    for feat, detail in result["details"].items():
        print(f"  {feat:15s} brut={detail['raw']:6.1f} | "
              f"norm={detail['normalized']:5.1f} | "
              f"poids={detail['weight']} | "
              f"contrib={detail['contribution']}")

    print("\n" + "=" * 50)
    print("TEST — Toutes pathologies")
    print("=" * 50)
    all_results = compute_ier_all_pathologies(45.0, 2.5, 72.0, 28.0)
    for p, r in all_results.items():
        print(f"  {p:12s} → score={r['score']:6.2f} | {r['level']}")