

import os
import pickle
import argparse
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix, f1_score
)

from prediction.features import build_features_dataframe, FEATURE_NAMES, N_FEATURES
from data.loader import PROCESSED_DIR


# Chemins


MODEL_DIR   = os.path.join(os.path.dirname(__file__), "models")
LABELED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "labeled")
RAW_DIR     = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")

KAGGLE_LABELED_FILE = "labeled_kaggle_global.csv"

# Hyperparametres

RF_PARAMS = {
    "n_estimators":      200,
    "max_depth":         12,
    "min_samples_split": 10,
    "min_samples_leaf":  5,
    "max_features":      "sqrt",
    "class_weight":      "balanced",
    "random_state":      42,
    "n_jobs":            -1,
}

MIN_SAMPLES_USER = 30


# Labels synthetiques (fallback si Kaggle absent)

def generate_synthetic_labels(df: pd.DataFrame, noise_rate: float = 0.08,
                               random_state: int = 42) -> pd.Series:
    """Labels heuristiques bases sur IER + zone (fallback sans Kaggle)."""
    rng   = np.random.default_rng(random_state)
    proba = np.where(df["ier_score"] >= 76, 0.90,
            np.where(df["ier_score"] >= 51, 0.55,
            np.where(df["ier_score"] >= 26, 0.20, 0.05)))
    if "zone" in df.columns:
        bonus = df["zone"].map({"A":0.0,"B":0.0,"C":0.15,"D":0.15}).fillna(0.0)
        proba = np.clip(proba + bonus.values, 0.0, 1.0)
    labels = rng.binomial(1, proba).astype(int)
    flip   = rng.random(len(labels)) < noise_rate
    labels[flip] = 1 - labels[flip]
    print(f"  Labels synthetiques : {len(labels)} | positifs={labels.mean()*100:.1f}%")
    return pd.Series(labels, index=df.index, name="symptom_label")


def _generate_dummy_dataset(n: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """Dataset entierement simule (pas de fichier externe disponible)."""
    rng = np.random.default_rng(random_state)
    df  = pd.DataFrame({
        "pm25":        np.clip(rng.lognormal(3.0, 0.8, n), 1, 400),
        "co":          np.clip(rng.lognormal(0.5, 0.6, n), 0, 30),
        "humidity":    np.clip(rng.normal(55, 18, n),  5, 99),
        "temperature": np.clip(rng.normal(14, 11, n), -15, 42),
        "station":     [f"station_{i%12}" for i in range(n)],
        "datetime":    pd.date_range("2013-03-01", periods=n, freq="1h"),
    })
    df["ier_score"] = (
        0.30*(df["pm25"]/500*100).clip(0,100) +
        0.25*(df["co"]/50*100).clip(0,100) +
        0.25*df["humidity"].clip(0,100) +
        0.20*((df["temperature"]+30)/80*100).clip(0,100)
    ).clip(0,100).round(2)
    df["zone"] = df["ier_score"].apply(
        lambda s: "A" if s<=25 else "B" if s<=50 else "C" if s<=75 else "D")
    return df


# Chargement dataset Kaggle preprocesse

def _load_kaggle_labeled() -> pd.DataFrame | None:
    """
    Charge le fichier Kaggle preprocesse depuis labeled/.
    Retourne None si absent (declenchera le pipeline Kaggle ou le fallback).
    """
    path = os.path.join(LABELED_DIR, KAGGLE_LABELED_FILE)
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"  Kaggle labeled charge : {len(df)} lignes ({path})")
        return df

    # Essayer de lancer le pipeline Kaggle automatiquement
    kaggle_csvs = [
        f for f in os.listdir(RAW_DIR)
        if f.endswith(".csv") and "air" in f.lower() and "quality" in f.lower()
    ] if os.path.exists(RAW_DIR) else []

    if kaggle_csvs:
        print(f"  Lancement automatique du pipeline Kaggle ({kaggle_csvs[0]})...")
        from data.kaggle_loader import run_kaggle_pipeline
        df = run_kaggle_pipeline(os.path.join(RAW_DIR, kaggle_csvs[0]), save=True)
        return df

    print("  Dataset Kaggle absent -> fallback synthetic")
    return None


# Preparation des donnees

def prepare_training_data(source: str = "kaggle", user_id: str = None,
                          pathologie: str = "general") -> tuple:
  
    print(f"\n[Preparation] source={source} | user_id={user_id or 'global'}")

    df = None

 
    if source == "kaggle":
        df = _load_kaggle_labeled()
        if df is not None:
            # Aligner les colonnes avec FEATURE_NAMES
            df = _align_kaggle_to_features(df, pathologie)


    elif source == "labeled":
        if user_id:
            path = os.path.join(LABELED_DIR, f"labeled_{user_id}.csv")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Donnees labeled introuvables : {path}\n"
                    f"Minimum requis : {MIN_SAMPLES_USER} sondages."
                )
            df = pd.read_csv(path)
        else:
            files = [f for f in os.listdir(LABELED_DIR)
                     if f.startswith("labeled_") and f.endswith(".csv")
                     and f != KAGGLE_LABELED_FILE]
            if not files:
                print("  Aucun fichier labeled utilisateur -> bascule Kaggle/synthetic")
                source = "kaggle"
                df = _load_kaggle_labeled()
                if df is not None:
                    df = _align_kaggle_to_features(df, pathologie)
            else:
                df = pd.concat(
                    [pd.read_csv(os.path.join(LABELED_DIR, f)) for f in files],
                    ignore_index=True
                )
                print(f"  {len(files)} fichiers labeled : {len(df)} lignes")


    if df is None or source == "synthetic":
        print("  Generation donnees synthetiques...")
        ier_file = os.path.join(PROCESSED_DIR, f"beijing_ier_zones_{pathologie}.csv")
        if os.path.exists(ier_file):
            df = pd.read_csv(ier_file)
        else:
            try:
                from data.loader import load_processed
                df = load_processed("beijing_processed.csv")
                from ier.calculator import compute_ier_dataframe
                df = compute_ier_dataframe(df, pathologie=pathologie)
                try:
                    from clustering.predict_zone import predict_zone_batch
                    df = predict_zone_batch(df)
                except FileNotFoundError:
                    df["zone"] = df["ier_score"].apply(
                        lambda s: "A" if s<=25 else "B" if s<=50 else "C" if s<=75 else "D")
            except FileNotFoundError:
                df = _generate_dummy_dataset(n=5000)

        df["symptom_label"]     = generate_synthetic_labels(df)
        rng = np.random.default_rng(42)
        df["age"]               = rng.integers(25, 70, len(df))
        df["pathologie"]        = pathologie
        df["is_smoker"]         = rng.binomial(1, 0.20, len(df))
        df["symptom_yesterday"] = df["symptom_label"].shift(1).fillna(0).astype(int)

    # ── Construction matrice features ────────────────────────────────────
    print("  Construction de la matrice features...")
    X_df = build_features_dataframe(df)
    X    = X_df.values.astype(np.float32)
    y_col = "symptom_label" if "symptom_label" in df.columns else "label"
    y    = df[y_col].values.astype(int)

    mask = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1))
    X, y = X[mask], y[mask]

    print(f"  Dataset final : {X.shape[0]} x {X.shape[1]} | positifs={y.mean()*100:.1f}%")
    return X, y, FEATURE_NAMES


def _align_kaggle_to_features(df: pd.DataFrame, pathologie: str) -> pd.DataFrame:
    
    df = df.copy()
    rng = np.random.default_rng(42)

    # Calculer IER depuis les 4 features si absent
    if "ier_score" not in df.columns:
        from ier.calculator import compute_ier_dataframe
        # Ajouter colonne station factice pour compute_ier_dataframe
        if "station" not in df.columns:
            df["station"] = "kaggle_global"
        df = compute_ier_dataframe(df, pathologie=pathologie)

    # Zone depuis IER
    if "zone" not in df.columns:
        df["zone"] = df["ier_score"].apply(
            lambda s: "A" if s<=25 else "B" if s<=50 else "C" if s<=75 else "D")

    # Datetime reconstruit depuis month (approximation)
    if "datetime" not in df.columns:
        month = df["month"].fillna(6).astype(int) if "month" in df.columns else 6
        df["datetime"] = pd.to_datetime(
            "2020-" + month.astype(str).str.zfill(2) + "-15"
        )

    # Profil utilisateur par defaut (non disponible dans Kaggle)
    if "age" not in df.columns:
        df["age"] = rng.integers(25, 70, len(df))
    if "pathologie" not in df.columns:
        df["pathologie"] = pathologie
    if "is_smoker" not in df.columns:
        df["is_smoker"] = rng.binomial(1, 0.20, len(df))
    if "symptom_yesterday" not in df.columns:
        df["symptom_yesterday"] = df["symptom_label"].shift(1).fillna(0).astype(int)
    if "station" not in df.columns:
        df["station"] = "kaggle_global"

    print(f"  Alignement Kaggle -> features AirSafe AI : {len(df)} lignes")
    return df


# Entrainement

def train_model(X: np.ndarray, y: np.ndarray, user_id: str = None,
                cross_validate: bool = True, save: bool = True) -> RandomForestClassifier:
    """Entraine le Random Forest et sauvegarde le modele."""
    if len(X) < MIN_SAMPLES_USER:
        raise ValueError(f"Donnees insuffisantes : {len(X)} < {MIN_SAMPLES_USER}")

    model_name = f"rf_user_{user_id}" if user_id else "rf_global"
    model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")

    print(f"\n{'='*55}")
    print(f"RANDOM FOREST -- {model_name.upper()}")
    print(f"{'='*55}")
    print(f"  N={len(X)} | features={X.shape[1]} | positifs={y.mean()*100:.1f}%")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Cross-validation
    if cross_validate and len(X_train) >= 200:
        print("\n  Cross-validation 5-fold...")
        cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        rf_cv = RandomForestClassifier(**RF_PARAMS)
        auc_cv = cross_val_score(rf_cv, X_train, y_train, cv=cv, scoring="roc_auc")
        f1_cv  = cross_val_score(rf_cv, X_train, y_train, cv=cv, scoring="f1")
        print(f"  AUC-ROC : {auc_cv.mean():.4f} +/- {auc_cv.std():.4f}")
        print(f"  F1      : {f1_cv.mean():.4f} +/- {f1_cv.std():.4f}")

    # Entrainement final
    print("\n  Entrainement final...")
    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_train, y_train)

    y_pred  = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    f1  = f1_score(y_test, y_pred, zero_division=0)

    print(f"\n  AUC-ROC  : {auc:.4f}")
    print(f"  F1-score : {f1:.4f}")
    print(f"  Matrice  :\n{confusion_matrix(y_test, y_pred)}")
    print(classification_report(y_test, y_pred,
          target_names=["Pas de crise","Crise"], zero_division=0))

    top = np.argsort(rf.feature_importances_)[::-1][:8]
    print("  Top 8 features :")
    for i in top:
        print(f"    {FEATURE_NAMES[i]:25s} {rf.feature_importances_[i]:.4f}")

    if save:
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(rf, f)
        meta = {
            "model_name":    model_name, "user_id": user_id,
            "n_samples":     len(X),     "n_features": N_FEATURES,
            "feature_names": FEATURE_NAMES,
            "auc_roc":       round(auc,4), "f1_score": round(f1,4),
            "rf_params":     RF_PARAMS,
            "positive_rate": round(float(y.mean()),4),
        }
        with open(model_path.replace(".pkl","_meta.pkl"), "wb") as f:
            pickle.dump(meta, f)
        print(f"\n  Modele sauvegarde : {model_path}")

    return rf


# Point d'entree

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user",       type=str, default=None)
    parser.add_argument("--source",     type=str, default="kaggle",
                        choices=["kaggle", "synthetic", "labeled"])
    parser.add_argument("--pathologie", type=str, default="general")
    parser.add_argument("--no-cv",      action="store_true")
    args = parser.parse_args()

    X, y, _ = prepare_training_data(args.source, args.user, args.pathologie)
    train_model(X, y, user_id=args.user, cross_validate=not args.no_cv)
    print("\nEntrainement termine.")


if __name__ == "__main__":
    main()