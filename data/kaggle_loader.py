
import os
import numpy as np
import pandas as pd

LABELED_DIR  = os.path.join(os.path.dirname(__file__), "labeled")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "processed")

# ---------------------------------------------------------------------------
# Colonnes a conserver depuis le CSV Kaggle
# ---------------------------------------------------------------------------

KAGGLE_KEEP = [
    "date",
    "PM2.5",
    "CO",
    "temperature",
    "humidity",
    "AQI",
    "population_density",
    "respiratory_admissions",
]

# Renommage Kaggle -> AirSafe AI
KAGGLE_RENAME = {
    "PM2.5":                 "pm25",
    "CO":                    "co",
    "AQI":                   "aqi",
    "respiratory_admissions": "respiratory_admissions",
}

# Features finales utilisees par le Random Forest (en plus des 20 de features.py)
# Ces colonnes enrichissent le dataset labeled
KAGGLE_FEATURES = ["pm25", "co", "temperature", "humidity",
                   "aqi", "population_density", "month", "day_of_week"]

LABEL_COL = "symptom_label"


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------

def load_kaggle_dataset(filepath: str) -> pd.DataFrame:
    """
    Charge le dataset Kaggle depuis un fichier CSV.

    Parameters
    ----------
    filepath : chemin vers le fichier CSV Kaggle telechargé

    Returns
    -------
    pd.DataFrame : donnees brutes avec toutes les colonnes

    Raises
    ------
    FileNotFoundError : si le fichier n'existe pas
    ValueError        : si les colonnes attendues sont absentes
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Fichier Kaggle introuvable : {filepath}\n"
            "Téléchargez le dataset depuis :\n"
            "https://www.kaggle.com/datasets/khushikyad001/"
            "air-quality-weather-and-respiratory-health\n"
            "et placez le CSV dans ai/data/raw/"
        )

    df = pd.read_csv(filepath)
    print(f"Dataset Kaggle charge : {len(df)} lignes | {len(df.columns)} colonnes")
    print(f"Colonnes : {list(df.columns)}")

    # Verification des colonnes obligatoires
    required = {"PM2.5", "CO", "temperature", "humidity", "respiratory_admissions"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans le dataset Kaggle : {missing}")

    return df


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_kaggle(df: pd.DataFrame,
                      label_percentile: float = 75.0) -> pd.DataFrame:
    """
    Nettoie et transforme le dataset Kaggle pour AirSafe AI.

    Etapes :
        1. Garder uniquement les colonnes utiles
        2. Renommer les colonnes
        3. Extraire features temporelles depuis 'date'
        4. Binariser respiratory_admissions -> symptom_label (seuil percentile)
        5. Traiter les valeurs manquantes
        6. Supprimer les outliers physiques

    Parameters
    ----------
    df               : DataFrame brut du dataset Kaggle
    label_percentile : percentile de respiratory_admissions pour le seuil Y=1
                       defaut 75 -> top 25% = crise (Y=1)

    Returns
    -------
    pd.DataFrame avec colonnes :
        pm25, co, temperature, humidity, aqi, population_density,
        month, day_of_week, symptom_label
    """
    df = df.copy()

    # 1. Garder les colonnes utiles
    keep = [c for c in KAGGLE_KEEP if c in df.columns]
    df = df[keep].copy()
    print(f"\n[1] Colonnes conservees ({len(keep)}) : {keep}")

    # 2. Renommer
    df.rename(columns=KAGGLE_RENAME, inplace=True)

    # 3. Features temporelles depuis 'date'
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["month"]       = df["date"].dt.month.fillna(6).astype(int)
        df["day_of_week"] = df["date"].dt.dayofweek.fillna(0).astype(int)  # 0=lundi
        df.drop(columns=["date"], inplace=True)
        print(f"[2] Features temporelles extraites : month, day_of_week")

    # 4. Binarisation du label
    if "respiratory_admissions" in df.columns:
        threshold = df["respiratory_admissions"].quantile(label_percentile / 100.0)
        df[LABEL_COL] = (df["respiratory_admissions"] >= threshold).astype(int)
        pos_rate = df[LABEL_COL].mean()
        print(f"[3] Label binarise (seuil percentile {label_percentile}%) : "
              f"threshold={threshold:.1f} | Y=1 : {pos_rate*100:.1f}%")
        df.drop(columns=["respiratory_admissions"], inplace=True)

    # 5. Valeurs manquantes
    feature_cols = [c for c in KAGGLE_FEATURES if c in df.columns]
    before = len(df)
    df[feature_cols] = df[feature_cols].interpolate(method="linear", limit=5)
    df.dropna(subset=feature_cols + [LABEL_COL], inplace=True)
    removed = before - len(df)
    if removed:
        print(f"[4] {removed} lignes supprimees (NaN irrecuperables)")

    # 6. Outliers physiques
    bounds = {
        "pm25":        (0.0,  999.0),
        "co":          (0.0, 50000.0),  # unites variables selon dataset
        "temperature": (-30.0, 55.0),
        "humidity":    (0.0, 100.0),
        "aqi":         (0.0, 500.0),
    }
    before = len(df)
    for col, (lo, hi) in bounds.items():
        if col in df.columns:
            df = df[(df[col] >= lo) & (df[col] <= hi)]
    removed_out = before - len(df)
    if removed_out:
        print(f"[5] {removed_out} outliers supprimes")

    df.reset_index(drop=True, inplace=True)

    print(f"\nDataset final : {len(df)} lignes")
    print(f"Colonnes      : {list(df.columns)}")
    print(f"\nStatistiques :")
    print(df[feature_cols].describe().round(2).to_string())
    print(f"\nLabel Y distribution :")
    print(df[LABEL_COL].value_counts().to_string())

    return df


# ---------------------------------------------------------------------------
# Sauvegarde dans labeled/ (format compatible avec retrain.py)
# ---------------------------------------------------------------------------

def save_as_labeled(df: pd.DataFrame,
                    filename: str = "labeled_kaggle_global.csv") -> str:
    """
    Sauvegarde le dataset Kaggle preprocesse dans ai/data/labeled/
    au format attendu par train.py et retrain.py.

    Le fichier sera utilise comme source d'entrainement global
    pour le Random Forest a la place des labels synthetiques.

    Returns
    -------
    str : chemin du fichier sauvegarde
    """
    os.makedirs(LABELED_DIR, exist_ok=True)
    path = os.path.join(LABELED_DIR, filename)
    df.to_csv(path, index=False)
    print(f"\nSauvegarde : {path} ({len(df)} lignes)")
    return path


# ---------------------------------------------------------------------------
# Pipeline complet
# ---------------------------------------------------------------------------

def run_kaggle_pipeline(filepath: str,
                        label_percentile: float = 75.0,
                        save: bool = True) -> pd.DataFrame:
    """
    Pipeline complet : chargement -> preprocessing -> sauvegarde.

    Parameters
    ----------
    filepath         : chemin vers le CSV Kaggle
    label_percentile : seuil de binarisation du label (defaut 75%)
    save             : si True, sauvegarde dans ai/data/labeled/

    Returns
    -------
    pd.DataFrame preprocesse
    """
    print("=" * 55)
    print("PIPELINE KAGGLE — Air Quality & Respiratory Health")
    print("=" * 55)

    df_raw  = load_kaggle_dataset(filepath)
    df_proc = preprocess_kaggle(df_raw, label_percentile=label_percentile)

    if save:
        save_as_labeled(df_proc)

    return df_proc


# ---------------------------------------------------------------------------
# Point d'entree
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Chercher automatiquement le CSV dans ai/data/raw/
    raw_dir  = os.path.join(os.path.dirname(__file__), "raw")
    csv_candidates = [
        f for f in os.listdir(raw_dir)
        if f.endswith(".csv") and "air" in f.lower() and "quality" in f.lower()
    ] if os.path.exists(raw_dir) else []

    if csv_candidates:
        filepath = os.path.join(raw_dir, csv_candidates[0])
        print(f"Fichier Kaggle detecte : {csv_candidates[0]}")
    elif len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        print("Usage : python -m ai.data.kaggle_loader <chemin_vers_csv_kaggle>")
        print("Ou placez le CSV dans ai/data/raw/ avec 'air' et 'quality' dans le nom.")
        sys.exit(1)

    df = run_kaggle_pipeline(filepath)