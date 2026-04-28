

import os
import csv
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, date
from typing import Optional

from prediction.train import (
    train_model, MIN_SAMPLES_USER, MODEL_DIR, LABELED_DIR
)
from prediction.features import build_features_dataframe, FEATURE_NAMES
from prediction.predict import reload_models

# Seuils de re-entrainement

RETRAIN_EVERY_N_SAMPLES = 10   # re-entrainer toutes les 10 nouvelles entrees
MIN_SAMPLES_RETRAIN     = MIN_SAMPLES_USER  # 30 entrees minimum


# Gestion du fichier labeled utilisateur

def get_labeled_path(user_id: str) -> str:
    """Retourne le chemin du fichier labeled de l'utilisateur."""
    os.makedirs(LABELED_DIR, exist_ok=True)
    return os.path.join(LABELED_DIR, f"labeled_{user_id}.csv")


def load_user_labeled(user_id: str) -> pd.DataFrame:
    """
    Charge le dataset labeled d'un utilisateur.
    Retourne un DataFrame vide si le fichier n'existe pas.
    """
    path = get_labeled_path(user_id)
    if not os.path.exists(path):
        return pd.DataFrame(columns=FEATURE_NAMES + ["symptom_label", "timestamp"])
    df = pd.read_csv(path, parse_dates=["timestamp"])
    print(f"  Labeled charge : {len(df)} entrees pour {user_id}")
    return df


def append_labeled_entry(
    user_id: str,
    features: dict,
    symptom_label: int,
    timestamp: Optional[datetime] = None,
) -> int:
    
    path = get_labeled_path(user_id)
    timestamp = timestamp or datetime.now()

    row = {f: features.get(f, 0.0) for f in FEATURE_NAMES}
    row["symptom_label"] = int(symptom_label)
    row["timestamp"]     = timestamp.isoformat()

    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FEATURE_NAMES + ["symptom_label", "timestamp"])
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    # Compter les entrees
    with open(path) as f:
        n = sum(1 for _ in f) - 1  # -1 pour le header

    print(f"  Entree ajoutee pour {user_id} | total={n} | label={symptom_label}")
    return n


def get_user_sample_count(user_id: str) -> int:
    """Retourne le nombre de samples labeled disponibles pour un utilisateur."""
    path = get_labeled_path(user_id)
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return max(0, sum(1 for _ in f) - 1)


# Logique de re-entrainement

def should_retrain(user_id: str) -> tuple:
    
    n_samples = get_user_sample_count(user_id)

    if n_samples < MIN_SAMPLES_RETRAIN:
        return False, (f"Donnees insuffisantes : {n_samples}/{MIN_SAMPLES_RETRAIN} "
                       f"({MIN_SAMPLES_RETRAIN - n_samples} sondages restants)")

    model_path = os.path.join(MODEL_DIR, f"rf_user_{user_id}.pkl")
    meta_path  = model_path.replace(".pkl", "_meta.pkl")

    if not os.path.exists(model_path):
        return True, f"Premier entrainement ({n_samples} samples disponibles)"

    # Verifier si assez de nouvelles donnees depuis le dernier entrainement
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        n_at_last_train = meta.get("n_samples", 0)
        new_samples     = n_samples - n_at_last_train
        if new_samples >= RETRAIN_EVERY_N_SAMPLES:
            return True, (f"Nouvelles donnees suffisantes : "
                          f"+{new_samples} depuis le dernier entrainement")
        return False, (f"Pas assez de nouvelles donnees : "
                       f"+{new_samples}/{RETRAIN_EVERY_N_SAMPLES}")

    return True, "Metadata absente - re-entrainement force"


def retrain_user_model(
    user_id: str,
    force: bool = False,
    include_global: bool = True,
) -> dict:
    
    print(f"\n{'='*55}")
    print(f"RE-ENTRAINEMENT -- {user_id}")
    print(f"{'='*55}")

    # Verification du seuil
    do_retrain, reason = should_retrain(user_id)
    if not do_retrain and not force:
        print(f"  Non necessaire : {reason}")
        return {"retrained": False, "reason": reason, "metrics": {}}

    print(f"  Raison : {reason}")

    # Charger les donnees labeled utilisateur
    df_user = load_user_labeled(user_id)
    if len(df_user) < MIN_SAMPLES_RETRAIN and not force:
        return {
            "retrained": False,
            "reason": f"Seuil non atteint : {len(df_user)}/{MIN_SAMPLES_RETRAIN}",
            "metrics": {}
        }

    # Data augmentation : combiner avec une fraction du dataset global
    if include_global:
        df_user = _augment_with_global(df_user, user_id)

    # Preparer X, y
    X_df = build_features_dataframe(df_user)
    X    = X_df.values.astype(np.float32)
    y    = df_user["symptom_label"].values.astype(int)

    # Nettoyage
    mask = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1))
    X, y = X[mask], y[mask]

    if len(X) < MIN_SAMPLES_USER:
        return {
            "retrained": False,
            "reason": f"Donnees insuffisantes apres nettoyage : {len(X)}",
            "metrics": {}
        }

    # Entrainement
    rf = train_model(X, y, user_id=user_id, cross_validate=(len(X) >= 200))

    # Invalider le cache
    reload_models()

    metrics = {
        "n_samples":    len(X),
        "positive_rate": round(float(y.mean()), 4),
        "timestamp":    datetime.now().isoformat(),
    }

    print(f"\n  Re-entrainement termine pour {user_id}")
    return {"retrained": True, "reason": reason, "metrics": metrics}


def _augment_with_global(df_user: pd.DataFrame, user_id: str,
                          fraction: float = 0.3) -> pd.DataFrame:
    
    global_ier_files = [
        f for f in os.listdir(os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "processed"
        ))
        if f.startswith("beijing_ier_zones") and f.endswith(".csv")
    ]

    if not global_ier_files:
        return df_user

    try:
        processed_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "processed"
        )
        df_global = pd.read_csv(os.path.join(processed_dir, global_ier_files[0]))

        # Labels synthetiques sur les donnees globales
        from prediction.train import generate_synthetic_labels
        df_global["symptom_label"] = generate_synthetic_labels(df_global)

        # Echantillonnage
        n_global = max(50, int(len(df_user) * fraction))
        df_sample = df_global.sample(
            n=min(n_global, len(df_global)),
            random_state=hash(user_id) % 2**31
        )

        # Aligner les colonnes
        common_cols = list(set(df_user.columns) & set(df_sample.columns))
        df_augmented = pd.concat(
            [df_user[common_cols], df_sample[common_cols]],
            ignore_index=True
        )
        print(f"  Augmentation : {len(df_user)} user + {len(df_sample)} global "
              f"= {len(df_augmented)} total")
        return df_augmented

    except Exception as e:
        print(f"  Augmentation echouee ({e}) -- donnees user seules")
        return df_user


# Batch : re-entrainer tous les utilisateurs eligibles

def retrain_all_eligible(force: bool = False) -> dict:
    """
    Re-entraine les modeles de tous les utilisateurs qui ont atteint le seuil.

    Returns
    -------
    dict : {user_id: resultat}
    """
    os.makedirs(LABELED_DIR, exist_ok=True)
    files = [f for f in os.listdir(LABELED_DIR)
             if f.startswith("labeled_") and f.endswith(".csv")]

    if not files:
        print("Aucun fichier labeled trouve.")
        return {}

    results = {}
    for fname in files:
        user_id = fname.replace("labeled_", "").replace(".csv", "")
        print(f"\n--- Utilisateur : {user_id} ---")
        results[user_id] = retrain_user_model(user_id, force=force)

    retrained = sum(1 for r in results.values() if r["retrained"])
    print(f"\nBilan : {retrained}/{len(files)} modeles re-entraines")
    return results


# Rapport d'etat utilisateur

def get_user_training_status(user_id: str) -> dict:
    
    n_samples = get_user_sample_count(user_id)
    model_path = os.path.join(MODEL_DIR, f"rf_user_{user_id}.pkl")
    has_model  = os.path.exists(model_path)

    last_trained = None
    n_at_last    = 0
    if has_model:
        meta_path = model_path.replace(".pkl", "_meta.pkl")
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            last_trained = meta.get("timestamp")
            n_at_last    = meta.get("n_samples", 0)

    new_since_train   = n_samples - n_at_last
    next_retrain_in   = max(0, RETRAIN_EVERY_N_SAMPLES - new_since_train)
    ready_to_train    = n_samples >= MIN_SAMPLES_RETRAIN
    progress_pct      = min(100.0, round(n_samples / MIN_SAMPLES_RETRAIN * 100, 1))

    return {
        "user_id":          user_id,
        "n_samples":        n_samples,
        "min_required":     MIN_SAMPLES_RETRAIN,
        "progress_pct":     progress_pct,
        "has_model":        has_model,
        "ready_to_train":   ready_to_train,
        "next_retrain_in":  next_retrain_in if has_model else max(0, MIN_SAMPLES_RETRAIN - n_samples),
        "last_trained":     last_trained,
    }


# Point d'entree

def main():
    parser = argparse.ArgumentParser(description="Re-entrainement RF AirSafe AI")
    parser.add_argument("--user",  type=str, default=None,
                        help="ID utilisateur (None = tous les eligibles)")
    parser.add_argument("--force", action="store_true",
                        help="Forcer le re-entrainement meme si seuil non atteint")
    parser.add_argument("--status", action="store_true",
                        help="Afficher l'etat sans re-entrainer")
    args = parser.parse_args()

    if args.status and args.user:
        status = get_user_training_status(args.user)
        print(f"\nEtat entrainement -- {args.user}")
        for k, v in status.items():
            print(f"  {k:20s} : {v}")
        return

    if args.user:
        retrain_user_model(args.user, force=args.force)
    else:
        retrain_all_eligible(force=args.force)


if __name__ == "__main__":
    main()