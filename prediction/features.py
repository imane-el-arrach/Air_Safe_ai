

import numpy as np
import pandas as pd
from typing import Optional


# Constantes


PATHOLOGIE_MAP = {
    "asthme":    0,
    "rhinite":   1,
    "bronchite": 2,
    "copd":      3,
    "general":   4,
}

ZONE_RISK_MAP = {"A": 1, "B": 2, "C": 3, "D": 4}

FEATURE_NAMES = [
    "pm25", "co", "humidity", "temperature",
    "ier_score", "zone_risk_score",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "pm25_mean_24h", "co_mean_24h", "humidity_mean_24h",
    "pm25_max_24h", "co_max_24h", "ier_max_24h",
    "age", "pathologie_encoded", "is_smoker", "symptom_yesterday",
]

N_FEATURES = len(FEATURE_NAMES)  # 20



# Encodage cyclique du temps


def encode_time_cyclical(hour: int, month: int) -> dict:
    """
    Encode heure (0-23) et mois (1-12) en coordonnees sin/cos
    pour capturer la continuite cyclique.
    """
    return {
        "hour_sin":  round(float(np.sin(2 * np.pi * hour  / 24)), 6),
        "hour_cos":  round(float(np.cos(2 * np.pi * hour  / 24)), 6),
        "month_sin": round(float(np.sin(2 * np.pi * month / 12)), 6),
        "month_cos": round(float(np.cos(2 * np.pi * month / 12)), 6),
    }



# Historique glissant


def compute_rolling_features(
    history: list,
    window_hours: int = 24,
) -> dict:
    
    if not history:
        return {
            "pm25_mean_24h":     0.0,
            "co_mean_24h":       0.0,
            "humidity_mean_24h": 0.0,
            "pm25_max_24h":      0.0,
            "co_max_24h":        0.0,
            "ier_max_24h":       0.0,
        }

    window = history[-window_hours:] if len(history) > window_hours else history

    pm25_vals = [m.get("pm25",      0.0) for m in window]
    co_vals   = [m.get("co",        0.0) for m in window]
    hum_vals  = [m.get("humidity",  0.0) for m in window]
    ier_vals  = [m.get("ier_score", 0.0) for m in window]

    return {
        "pm25_mean_24h":     round(float(np.mean(pm25_vals)),  3),
        "co_mean_24h":       round(float(np.mean(co_vals)),    3),
        "humidity_mean_24h": round(float(np.mean(hum_vals)),   3),
        "pm25_max_24h":      round(float(np.max(pm25_vals)),   3),
        "co_max_24h":        round(float(np.max(co_vals)),     3),
        "ier_max_24h":       round(float(np.max(ier_vals)),    3),
    }


def compute_rolling_features_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.sort_values(["station", "datetime"] if "station" in df.columns else ["datetime"],
                       inplace=True)

    roll_specs = [
        ("pm25_mean_24h",     "pm25",      "mean", 24),
        ("co_mean_24h",       "co",        "mean", 24),
        ("humidity_mean_24h", "humidity",  "mean", 24),
        ("pm25_max_24h",      "pm25",      "max",  24),
        ("co_max_24h",        "co",        "max",  24),
        ("ier_max_24h",       "ier_score", "max",  24),
    ]

    group_col = "station" if "station" in df.columns else None

    for new_col, src_col, agg, window in roll_specs:
        if src_col not in df.columns:
            df[new_col] = 0.0
            continue
        if group_col:
            df[new_col] = (
                df.groupby(group_col)[src_col]
                .transform(lambda s: s.rolling(window, min_periods=1).agg(agg))
                .round(3)
            )
        else:
            df[new_col] = df[src_col].rolling(window, min_periods=1).agg(agg).round(3)

    df.reset_index(drop=True, inplace=True)
    return df



# Profil utilisateur


def encode_user_profile(
    age: int,
    pathologie: str = "general",
    is_smoker: bool = False,
    symptom_yesterday: bool = False,
) -> dict:
    """
    Encode le profil utilisateur en features numeriques.
    """
    return {
        "age":                int(age),
        "pathologie_encoded": PATHOLOGIE_MAP.get(pathologie.lower().strip(), 4),
        "is_smoker":          int(is_smoker),
        "symptom_yesterday":  int(symptom_yesterday),
    }



# Construction vecteur complet — temps reel


def build_feature_vector(
    pm25: float,
    co: float,
    humidity: float,
    temperature: float,
    ier_score: float,
    zone: str = "A",
    hour: int = 12,
    month: int = 6,
    history: Optional[list] = None,
    age: int = 35,
    pathologie: str = "general",
    is_smoker: bool = False,
    symptom_yesterday: bool = False,
) -> np.ndarray:
    
    time_feats    = encode_time_cyclical(hour, month)
    rolling_feats = compute_rolling_features(history or [])
    user_feats    = encode_user_profile(age, pathologie, is_smoker, symptom_yesterday)

    feature_dict = {
        "pm25":            pm25,
        "co":              co,
        "humidity":        humidity,
        "temperature":     temperature,
        "ier_score":       ier_score,
        "zone_risk_score": ZONE_RISK_MAP.get(zone, 1),
        **time_feats,
        **rolling_feats,
        **user_feats,
    }

    vector = np.array([[feature_dict[f] for f in FEATURE_NAMES]], dtype=np.float32)
    return vector


def build_feature_dict(
    pm25: float,
    co: float,
    humidity: float,
    temperature: float,
    ier_score: float,
    zone: str = "A",
    hour: int = 12,
    month: int = 6,
    history: Optional[list] = None,
    age: int = 35,
    pathologie: str = "general",
    is_smoker: bool = False,
    symptom_yesterday: bool = False,
) -> dict:
    """Meme chose que build_feature_vector mais retourne un dict (debug/API)."""
    time_feats    = encode_time_cyclical(hour, month)
    rolling_feats = compute_rolling_features(history or [])
    user_feats    = encode_user_profile(age, pathologie, is_smoker, symptom_yesterday)
    return {
        "pm25":            pm25,
        "co":              co,
        "humidity":        humidity,
        "temperature":     temperature,
        "ier_score":       ier_score,
        "zone_risk_score": ZONE_RISK_MAP.get(zone, 1),
        **time_feats,
        **rolling_feats,
        **user_feats,
    }



# Construction DataFrame batch — entrainement


def build_features_dataframe(
    df: pd.DataFrame,
    user_profiles: Optional[dict] = None,
) -> pd.DataFrame:
    
    df = df.copy()
    print(f"Construction des features -- {len(df)} lignes...")

    # Zone -> zone_risk_score
    if "zone" in df.columns:
        df["zone_risk_score"] = df["zone"].map(ZONE_RISK_MAP).fillna(1).astype(int)
    else:
        df["zone_risk_score"] = 1

    # Encodage temporel
    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"])
        df["hour_sin"]  = np.sin(2 * np.pi * dt.dt.hour  / 24).round(6)
        df["hour_cos"]  = np.cos(2 * np.pi * dt.dt.hour  / 24).round(6)
        df["month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12).round(6)
        df["month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12).round(6)
    else:
        for col in ["hour_sin", "hour_cos", "month_sin", "month_cos"]:
            df[col] = 0.0

    # Features glissantes 24h
    print("  Calcul des features glissantes 24h...")
    df = compute_rolling_features_dataframe(df)

    # Profil utilisateur
    if "pathologie" in df.columns:
        df["pathologie_encoded"] = df["pathologie"].map(PATHOLOGIE_MAP).fillna(4).astype(int)
    else:
        df["pathologie_encoded"] = 4

    for col, default in [("age", 35), ("is_smoker", 0), ("symptom_yesterday", 0)]:
        if col not in df.columns:
            df[col] = default

    # Verification
    missing = [f for f in FEATURE_NAMES if f not in df.columns]
    if missing:
        raise ValueError(f"Features manquantes apres construction : {missing}")

    print(f"  OK -- {N_FEATURES} features x {len(df)} lignes")
    return df[FEATURE_NAMES]



# Validation


def validate_feature_vector(vector: np.ndarray) -> bool:
    """Valide un vecteur de features avant inference."""
    if vector.shape != (1, N_FEATURES):
        raise ValueError(f"Shape invalide : attendu (1, {N_FEATURES}), obtenu {vector.shape}")
    if np.any(np.isnan(vector)):
        nan_idx = np.where(np.isnan(vector))[1]
        raise ValueError(f"NaN dans les features : {[FEATURE_NAMES[i] for i in nan_idx]}")
    if np.any(np.isinf(vector)):
        raise ValueError("Valeurs infinies dans le vecteur de features.")
    return True



# Point d'entree


if __name__ == "__main__":
    print("=" * 55)
    print("TEST -- Construction du vecteur de features")
    print("=" * 55)

    history = [
        {"pm25": 30.0, "co": 1.0, "humidity": 60.0, "ier_score": 30.0},
        {"pm25": 40.0, "co": 1.5, "humidity": 65.0, "ier_score": 38.0},
        {"pm25": 50.0, "co": 2.0, "humidity": 70.0, "ier_score": 45.0},
    ]

    vec = build_feature_vector(
        pm25=55.0, co=2.5, humidity=72.0, temperature=28.0,
        ier_score=52.0, zone="C",
        hour=14, month=7,
        history=history,
        age=42, pathologie="asthme",
        is_smoker=False, symptom_yesterday=True,
    )

    print(f"\nShape  : {vec.shape}")
    print(f"Dtype  : {vec.dtype}")
    print(f"\n{N_FEATURES} features :")
    feat_dict = build_feature_dict(
        pm25=55.0, co=2.5, humidity=72.0, temperature=28.0,
        ier_score=52.0, zone="C",
        hour=14, month=7, history=history,
        age=42, pathologie="asthme",
        is_smoker=False, symptom_yesterday=True,
    )
    for name, val in feat_dict.items():
        print(f"  {name:25s} = {val}")

    validate_feature_vector(vec)
    print("\nValidation : OK")