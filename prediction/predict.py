

import os
import pickle
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from prediction.features import (
    build_feature_vector, build_feature_dict,
    validate_feature_vector, FEATURE_NAMES, N_FEATURES
)

# Chemins

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# Seuils de probabilite -> niveau d'alerte

ALERT_THRESHOLDS = {
    "CRITIQUE":      0.80,   # P >= 80% -> alerte rouge + SMS
    "ELEVE":         0.60,   # P >= 60% -> alerte orange
    "MODERE":        0.40,   # P >= 40% -> notification jaune
    "FAIBLE":        0.0,    # P <  40% -> info verte
}

ALERT_COLORS = {
    "CRITIQUE": "#e74c3c",
    "ELEVE":    "#e67e22",
    "MODERE":   "#f39c12",
    "FAIBLE":   "#2ecc71",
}

# Resultat de prediction

@dataclass
class PredictionResult:
    """Resultat complet d'une prediction de crise."""
    user_id:          str
    proba_crisis:     float          # P(crise) [0, 1]
    alert_level:      str            # FAIBLE | MODERE | ELEVE | CRITIQUE
    alert_color:      str            # code hex
    should_notify:    bool           # True si >= MODERE
    message:          str            # message lisible (app mobile)
    feature_values:   dict = field(default_factory=dict)
    model_used:       str = ""       # 'rf_user_{id}' ou 'rf_global'
    top_contributors: list = field(default_factory=list)  # top features


# Cache modeles

_model_cache: dict = {}


def _load_model(user_id: Optional[str] = None):
    
    # Chercher modele personnalise
    if user_id:
        user_path = os.path.join(MODEL_DIR, f"rf_user_{user_id}.pkl")
        if os.path.exists(user_path):
            cache_key = f"user_{user_id}"
            if cache_key not in _model_cache:
                with open(user_path, "rb") as f:
                    _model_cache[cache_key] = pickle.load(f)
                print(f"  Modele charge : rf_user_{user_id}.pkl")
            return _model_cache[cache_key], f"rf_user_{user_id}"

    # Fallback modele global
    global_path = os.path.join(MODEL_DIR, "rf_global.pkl")
    if os.path.exists(global_path):
        if "global" not in _model_cache:
            with open(global_path, "rb") as f:
                _model_cache["global"] = pickle.load(f)
            print("  Modele charge : rf_global.pkl")
        return _model_cache["global"], "rf_global"

    raise FileNotFoundError(
        f"Aucun modele RF trouve dans {MODEL_DIR}.\n"
        "Lancez d'abord : python -m ai.prediction.train"
    )


def reload_models():
    """Force le rechargement de tous les modeles depuis le disque."""
    _model_cache.clear()
    print("Cache modeles vide.")


# Prediction principale

def predict_crisis(
    user_id: str,
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
) -> PredictionResult:
    
    # Construction du vecteur de features
    X = build_feature_vector(
        pm25=pm25, co=co, humidity=humidity, temperature=temperature,
        ier_score=ier_score, zone=zone,
        hour=hour, month=month, history=history,
        age=age, pathologie=pathologie,
        is_smoker=is_smoker, symptom_yesterday=symptom_yesterday,
    )
    validate_feature_vector(X)

    # Chargement du modele
    try:
        rf, model_name = _load_model(user_id)
    except FileNotFoundError:
        # Fallback heuristique si aucun modele entraine
        return _heuristic_prediction(
            user_id=user_id,
            ier_score=ier_score,
            zone=zone,
            symptom_yesterday=symptom_yesterday,
            pm25=pm25, co=co, humidity=humidity, temperature=temperature,
        )

    # Inference
    proba_crisis = float(rf.predict_proba(X)[0, 1])

    # Niveau d'alerte
    alert_level = _proba_to_alert(proba_crisis)
    alert_color = ALERT_COLORS[alert_level]
    should_notify = proba_crisis >= ALERT_THRESHOLDS["MODERE"]

    # Message lisible
    message = _build_message(proba_crisis, alert_level, pathologie)

    # Top features contributrices
    top_contributors = _get_top_contributors(rf, X)

    # Dict features pour debug
    feat_dict = build_feature_dict(
        pm25=pm25, co=co, humidity=humidity, temperature=temperature,
        ier_score=ier_score, zone=zone,
        hour=hour, month=month, history=history,
        age=age, pathologie=pathologie,
        is_smoker=is_smoker, symptom_yesterday=symptom_yesterday,
    )

    return PredictionResult(
        user_id=user_id,
        proba_crisis=round(proba_crisis, 4),
        alert_level=alert_level,
        alert_color=alert_color,
        should_notify=should_notify,
        message=message,
        feature_values=feat_dict,
        model_used=model_name,
        top_contributors=top_contributors,
    )


def predict_crisis_batch(
    df,
    user_id: str = None,
    pathologie: str = "general",
):
    """
    Prediction batch sur un DataFrame.
    Ajoute les colonnes 'proba_crisis', 'alert_level', 'should_notify'.
    """
    import pandas as pd
    from prediction.features import build_features_dataframe

    df = df.copy()
    try:
        rf, model_name = _load_model(user_id)
    except FileNotFoundError:
        print("  Modele absent - prediction heuristique")
        df["proba_crisis"]  = df["ier_score"].apply(_ier_to_proba)
        df["alert_level"]   = df["proba_crisis"].apply(_proba_to_alert)
        df["should_notify"] = df["proba_crisis"] >= ALERT_THRESHOLDS["MODERE"]
        return df

    if "pathologie" not in df.columns:
        df["pathologie"] = pathologie

    X_df = build_features_dataframe(df)
    X    = X_df.values.astype(np.float32)

    probas = rf.predict_proba(X)[:, 1]
    df["proba_crisis"]  = probas.round(4)
    df["alert_level"]   = [_proba_to_alert(p) for p in probas]
    df["should_notify"] = probas >= ALERT_THRESHOLDS["MODERE"]

    print(f"Predictions batch : {len(df)} lignes | modele={model_name}")
    print(df["alert_level"].value_counts().to_string())
    return df


# Helpers internes

def _proba_to_alert(proba: float) -> str:
    if proba >= ALERT_THRESHOLDS["CRITIQUE"]: return "CRITIQUE"
    if proba >= ALERT_THRESHOLDS["ELEVE"]:    return "ELEVE"
    if proba >= ALERT_THRESHOLDS["MODERE"]:   return "MODERE"
    return "FAIBLE"


def _ier_to_proba(ier: float) -> float:
    """Fallback heuristique IER -> probabilite."""
    if ier >= 76: return 0.88
    if ier >= 51: return 0.55
    if ier >= 26: return 0.22
    return 0.06


def _build_message(proba: float, level: str, pathologie: str) -> str:
    pct = int(proba * 100)
    msgs = {
        "CRITIQUE": f"Risque de crise {pathologie} tres eleve ({pct}%). Evitez toute exposition et prenez vos medicaments preventifs.",
        "ELEVE":    f"Risque de crise {pathologie} eleve ({pct}%). Reduisez votre activite physique et surveillez vos symptomes.",
        "MODERE":   f"Risque modere ({pct}%). Aerez les espaces interieurs et restez attentif.",
        "FAIBLE":   f"Risque faible ({pct}%). Conditions acceptables pour votre sante respiratoire.",
    }
    return msgs.get(level, f"Probabilite de crise : {pct}%")


def _get_top_contributors(rf, X: np.ndarray, top_n: int = 5) -> list:
    """
    Retourne les top N features les plus importantes pour cette prediction.
    Utilise l'importance globale du modele RF (feature_importances_).
    """
    importances = rf.feature_importances_
    top_idx     = np.argsort(importances)[::-1][:top_n]
    return [
        {
            "feature":    FEATURE_NAMES[i],
            "importance": round(float(importances[i]), 4),
            "value":      round(float(X[0, i]), 3),
        }
        for i in top_idx
    ]


def _heuristic_prediction(user_id, ier_score, zone, symptom_yesterday,
                           pm25, co, humidity, temperature) -> PredictionResult:
    """Prediction de secours basee sur IER + zone quand aucun modele n'est entraine."""
    proba = _ier_to_proba(ier_score)
    if zone in ("C", "D"):
        proba = min(proba + 0.15, 1.0)
    if symptom_yesterday:
        proba = min(proba + 0.10, 1.0)
    proba = round(proba, 4)
    level = _proba_to_alert(proba)
    return PredictionResult(
        user_id=user_id,
        proba_crisis=proba,
        alert_level=level,
        alert_color=ALERT_COLORS[level],
        should_notify=proba >= ALERT_THRESHOLDS["MODERE"],
        message=_build_message(proba, level, "general"),
        model_used="heuristic_fallback",
        feature_values={"pm25": pm25, "co": co, "humidity": humidity,
                        "temperature": temperature, "ier_score": ier_score},
    )


# Point d'entree

if __name__ == "__main__":
    print("=" * 55)
    print("TEST -- Prediction de crise")
    print("=" * 55)

    scenarios = [
        {"name": "Sain",     "pm25": 5.0,   "co": 0.3, "humidity": 45.0, "temperature": 20.0, "ier_score": 12.0, "zone": "A"},
        {"name": "Modere",   "pm25": 30.0,  "co": 1.2, "humidity": 62.0, "temperature": 24.0, "ier_score": 38.0, "zone": "B"},
        {"name": "Risque",   "pm25": 90.0,  "co": 4.0, "humidity": 78.0, "temperature": 32.0, "ier_score": 65.0, "zone": "C"},
        {"name": "Critique", "pm25": 250.0, "co": 9.0, "humidity": 90.0, "temperature": 38.0, "ier_score": 87.0, "zone": "D"},
    ]

    for s in scenarios:
        name = s.pop("name")
        r = predict_crisis(
            user_id="test_user",
            **s,
            hour=14, month=7,
            age=42, pathologie="asthme",
            is_smoker=False, symptom_yesterday=(name=="Risque"),
        )
        print(f"\n  {name:10s} | P(crise)={r.proba_crisis:.2f} "
              f"| {r.alert_level:8s} | {r.message[:60]}")
        print(f"             | modele={r.model_used}")