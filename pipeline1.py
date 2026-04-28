

import os
import pickle
import pandas as pd

from data.loader import load_processed, PROCESSED_DIR
from data.preprocessing import run_preprocessing, AIRSAFE_FEATURES
from ier.calculator import compute_ier_single, compute_ier_dataframe, load_bounds_from_scalers
from ier.weights import get_risk_level
from clustering.predict_zone import predict_zone, predict_zone_batch, ZONE_INFO

# Chemin des scalers preprocessing 
SCALER_PATH = os.path.join(PROCESSED_DIR, "scalers.pkl")


# Pipeline temps réel — une mesure IoT

def run_realtime(
    measurement: dict,
    pathologie: str = "general",
    user_id: str = None,
) -> dict:
    
    # Validation des champs requis
    required = {"pm25", "co", "humidity", "temperature"}
    missing = required - set(measurement.keys())
    if missing:
        raise ValueError(f"Champs manquants dans la mesure : {missing}")

    pm25        = float(measurement["pm25"])
    co          = float(measurement["co"])
    humidity    = float(measurement["humidity"])
    temperature = float(measurement["temperature"])

    # Charger les bornes du preprocessing si disponibles
    bounds = load_bounds_from_scalers(SCALER_PATH)

    # ── Étape 1 : Calcul IER ──────────────────────────────────────────────
    ier_result = compute_ier_single(
        pm25=pm25,
        co=co,
        humidity=humidity,
        temperature=temperature,
        pathologie=pathologie,
        bounds=bounds,
    )

    # ── Étape 2 : Prédiction de zone K-Means ─────────────────────────────
    try:
        zone_result = predict_zone(
            pm25=pm25,
            co=co,
            humidity=humidity,
            temperature=temperature,
            ier_score=ier_result["score"],
        )
    except FileNotFoundError:
        # Modèle non entraîné → fallback basé sur IER seul
        zone_result = _zone_from_ier_fallback(ier_result["score"])

    # ── Étape 3 : Génération de l'alerte ─────────────────────────────────
    alert = _generate_alert(ier_result, zone_result)

    # ── Résultat final ────────────────────────────────────────────────────
    result = {
        "user_id":    user_id,
        "pathologie": pathologie,
        "input":      measurement,
        "ier":        ier_result,
        "zone":       zone_result,
        "alert":      alert,
    }

    return result


# Pipeline batch — DataFrame historique

def run_batch(
    df: pd.DataFrame = None,
    pathologie: str = "general",
    save_output: bool = True,
) -> pd.DataFrame:
    
    print("=" * 60)
    print("PIPELINE BATCH — AirSafe AI")
    print("=" * 60)

    # Chargement
    if df is None:
        print("\n[1/3] Chargement des données preprocessées...")
        try:
            df = load_processed("beijing_processed.csv")
        except FileNotFoundError:
            print("  Données non trouvées. Lancement du preprocessing...")
            df = run_preprocessing()

    # IER
    print(f"\n[2/3] Calcul IER (pathologie: {pathologie})...")
    bounds = load_bounds_from_scalers(SCALER_PATH)
    df = compute_ier_dataframe(df, pathologie=pathologie, bounds=bounds, add_details=True)

    # Zone K-Means
    print("\n[3/3] Prédiction des zones K-Means...")
    try:
        df = predict_zone_batch(df)
    except FileNotFoundError:
        print("  Modèle K-Means non trouvé. "
              "Lancez python -m ai.clustering.train_kmeans")
        df["zone"] = df["ier_score"].apply(_ier_to_zone_fallback)
        df["zone_label"] = df["zone"].map({z: ZONE_INFO[z]["label"] for z in ZONE_INFO})

    # Statistiques
    print("\n" + "=" * 60)
    print("RÉSUMÉ BATCH")
    print("=" * 60)
    print(f"  Lignes traitées : {len(df)}")
    print(f"\n  Distribution IER :")
    print(df["ier_level"].value_counts().to_string())
    if "zone" in df.columns:
        print(f"\n  Distribution zones :")
        print(df["zone"].value_counts().sort_index().to_string())

    if save_output:
        output_path = os.path.join(PROCESSED_DIR, f"beijing_ier_zones_{pathologie}.csv")
        df.to_csv(output_path, index=False)
        print(f"\n  Résultat sauvegardé : {output_path}")

    return df


# Helpers internes

def _zone_from_ier_fallback(ier_score: float) -> dict:
    """Fallback K-Means → zone basée uniquement sur l'IER."""
    zone = _ier_to_zone_fallback(ier_score)
    info = ZONE_INFO[zone]
    return {
        "zone":           zone,
        "cluster_id":     -1,
        "label":          info["label"],
        "description":    info["description"],
        "recommendation": info["recommendation"],
        "color":          info["color"],
        "risk_score":     info["risk_score"],
        "fallback":       True,
    }


def _ier_to_zone_fallback(ier_score: float) -> str:
    """Mapping IER → zone (fallback sans K-Means)."""
    if ier_score <= 25:   return "A"
    elif ier_score <= 50: return "B"
    elif ier_score <= 75: return "C"
    else:                 return "D"


def _generate_alert(ier_result: dict, zone_result: dict) -> dict:
    
    ier_score  = ier_result["score"]
    ier_level  = ier_result["level"]
    zone       = zone_result.get("zone", "A")

    # Seuils d'alerte
    HIGH_ZONES    = {"C", "D"}
    ALERT_IER_MIN = 51  # IER ≥ Élevé

    triggered = (ier_score >= ALERT_IER_MIN) and (zone in HIGH_ZONES)

    if zone == "D" or ier_level == "Critique":
        priority = "URGENT"
        message  = (
            f"⚠ Alerte critique — IER={ier_score:.1f} | Zone {zone}. "
            f"{ier_result['action']}. {zone_result.get('recommendation', '')}."
        )
    elif triggered:
        priority = "AVERTISSEMENT"
        message  = (
            f"Attention — IER={ier_score:.1f} | Zone {zone}. "
            f"{ier_result['action']}."
        )
    else:
        priority = "INFO"
        message  = (
            f"Air acceptable — IER={ier_score:.1f} | Zone {zone}. "
            f"{ier_result['action']}."
        )

    return {
        "triggered": triggered,
        "priority":  priority,
        "message":   message,
        "ier_score": ier_score,
        "zone":      zone,
    }


# Point d'entrée CLI

if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "realtime"

    if mode == "batch":
        df = run_batch(pathologie="general")

    else:
        print("=" * 50)
        print("TEST — Pipeline temps réel")
        print("=" * 50)

        test_cases = [
            {"name": "Sain",      "pm25": 5.0,   "co": 0.3, "humidity": 45.0, "temperature": 20.0},
            {"name": "Modéré",    "pm25": 30.0,  "co": 1.2, "humidity": 62.0, "temperature": 24.0},
            {"name": "Risqué",    "pm25": 90.0,  "co": 4.0, "humidity": 80.0, "temperature": 32.0},
            {"name": "Critique",  "pm25": 250.0, "co": 9.0, "humidity": 92.0, "temperature": 38.0},
        ]

        for tc in test_cases:
            name = tc.pop("name")
            result = run_realtime(tc, pathologie="asthme")

            print(f"\n── {name} ──")
            print(f"  IER   : {result['ier']['score']:.1f} ({result['ier']['level']})")
            print(f"  Zone  : {result['zone']['zone']} — {result['zone']['label']}")
            print(f"  Alerte: [{result['alert']['priority']}] {result['alert']['message']}")