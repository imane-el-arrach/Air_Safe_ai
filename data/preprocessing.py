
import os
import pickle
import numpy as np
import pandas as pd

from data.loader import (
    load_raw_beijing, save_processed, load_processed,
    RAW_DIR, PROCESSED_DIR, AIRSAFE_FEATURES
)

# ---------------------------------------------------------------------------
# Seuils physiques acceptables (OMS + physique)
# ---------------------------------------------------------------------------

THRESHOLDS = {
    "pm25":        (0.0,   999.0),   # µg/m³  — OMS journalier 15 µg/m³ mais pics > 500 possibles
    "co":          (0.0,  10000.0),  # µg/m³  — Beijing UCI en µg/m³ (pas mg/m³)
    "temperature": (-30.0,  50.0),   # °C
    "humidity":    (  0.0, 100.0),   # %
}


# ---------------------------------------------------------------------------
# Etape 1 : Renommage + datetime
# ---------------------------------------------------------------------------

def rename_and_build_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renomme les colonnes UCI -> noms AirSafe AI
    et construit la colonne datetime depuis year/month/day/hour.
    """
    df = df.copy()

    # Renommage des 4 features
    rename_map = {
        "PM2.5": "pm25",
        "CO":    "co",
        "TEMP":  "temperature",
        "DEWP":  "dewpoint",
    }
    df.rename(columns=rename_map, inplace=True)

    # Construction datetime
    if all(c in df.columns for c in ["year", "month", "day", "hour"]):
        df["datetime"] = pd.to_datetime({
            "year":  df["year"],
            "month": df["month"],
            "day":   df["day"],
            "hour":  df["hour"],
        })
        print(f"  Datetime cree : {df['datetime'].min()} -> {df['datetime'].max()}")

    # Supprimer colonnes inutiles pour AirSafe AI
    drop_cols = ["No", "year", "month", "day", "hour",
                 "PM10", "SO2", "NO2", "O3", "PRES", "RAIN", "wd", "WSPM"]
    existing_drops = [c for c in drop_cols if c in df.columns]
    df.drop(columns=existing_drops, inplace=True)

    print(f"  Colonnes conservees : {list(df.columns)}")
    return df


# ---------------------------------------------------------------------------
# Etape 2 : Calcul humidite relative depuis DEWP + TEMP
# ---------------------------------------------------------------------------

def compute_humidity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule l'humidite relative (%) depuis temperature et point de rosee.

    Formule August-Roche-Magnus :
        RH = 100 x exp(17.625 x Td / (243.04 + Td))
                   / exp(17.625 x T  / (243.04 + T ))

    Le point de rosee (DEWP) est deja en °C dans le dataset Beijing.
    """
    df = df.copy()

    if "dewpoint" not in df.columns:
        raise KeyError("Colonne 'dewpoint' (DEWP) manquante.")

    T  = df["temperature"]
    Td = df["dewpoint"]

    a, b = 17.625, 243.04
    rh = 100.0 * (
        np.exp(a * Td / (b + Td)) /
        np.exp(a * T  / (b + T ))
    )

    df["humidity"] = rh.clip(0.0, 100.0).round(2)
    df.drop(columns=["dewpoint"], inplace=True)

    print(f"  Humidite calculee -> "
          f"min={df['humidity'].min():.1f}% | "
          f"moy={df['humidity'].mean():.1f}% | "
          f"max={df['humidity'].max():.1f}%")
    return df


# ---------------------------------------------------------------------------
# Etape 3 : Valeurs manquantes
# ---------------------------------------------------------------------------

def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Traite les valeurs manquantes sur les 4 features AirSafe AI.

    Strategie :
        - Interpolation lineaire par station (coherence temporelle)
        - Limite : 24h max de gap interpolable
        - Au-dela : lignes supprimees
    """
    df = df.copy()

    print(f"\n  Valeurs manquantes avant :")
    for col in AIRSAFE_FEATURES:
        if col in df.columns:
            n_nan = df[col].isna().sum()
            print(f"    {col:15s} : {n_nan:6,} NaN ({n_nan/len(df)*100:.2f}%)")

    # Trier par station + datetime pour interpolation coherente
    if "datetime" in df.columns:
        df.sort_values(["station", "datetime"], inplace=True)
        df.reset_index(drop=True, inplace=True)

    # Interpolation lineaire par station, limite 24 periodes
    for col in AIRSAFE_FEATURES:
        if col not in df.columns:
            continue
        df[col] = (
            df.groupby("station")[col]
            .transform(lambda s: s.interpolate(method="linear", limit=24))
        )

    # Supprimer les lignes encore manquantes (gaps > 24h)
    before = len(df)
    df.dropna(subset=AIRSAFE_FEATURES, inplace=True)
    removed = before - len(df)

    print(f"\n  Apres interpolation :")
    for col in AIRSAFE_FEATURES:
        if col in df.columns:
            n_nan = df[col].isna().sum()
            print(f"    {col:15s} : {n_nan} NaN")
    if removed > 0:
        print(f"  {removed:,} lignes supprimees (gaps > 24h)")
    print(f"  Lignes restantes : {len(df):,}")

    return df


# ---------------------------------------------------------------------------
# Etape 4 : Suppression des outliers
# ---------------------------------------------------------------------------

def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supprime les lignes dont les valeurs depassent les seuils physiques.
    """
    df   = df.copy()
    before = len(df)

    for col, (lo, hi) in THRESHOLDS.items():
        if col in df.columns:
            mask = (df[col] >= lo) & (df[col] <= hi)
            n_removed = (~mask).sum()
            if n_removed > 0:
                print(f"  {col:15s} : {n_removed:,} outliers supprimes "
                      f"(hors [{lo}, {hi}])")
            df = df[mask]

    total_removed = before - len(df)
    print(f"  Total supprime  : {total_removed:,} lignes ({total_removed/before*100:.2f}%)")
    print(f"  Lignes restantes: {len(df):,}")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Etape 5 : Sauvegarde des scalers min/max
# ---------------------------------------------------------------------------

def save_scalers(df: pd.DataFrame) -> dict:
    """
    Calcule et sauvegarde les min/max de chaque feature.
    Utilises pour la normalisation dans l'IER et le Random Forest.
    """
    scalers = {}
    for col in AIRSAFE_FEATURES:
        if col in df.columns:
            scalers[col] = {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
            }

    scaler_path = os.path.join(PROCESSED_DIR, "scalers.pkl")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    with open(scaler_path, "wb") as f:
        pickle.dump(scalers, f)

    print(f"\n  Scalers sauvegardes : {scaler_path}")
    for col, s in scalers.items():
        print(f"    {col:15s} min={s['min']:8.3f} | max={s['max']:8.3f}")

    return scalers


# ---------------------------------------------------------------------------
# Pipeline complet
# ---------------------------------------------------------------------------

def run_preprocessing(raw_dir: str = None, save: bool = True) -> pd.DataFrame:
    """
    Lance le pipeline de preprocessing complet sur les 12 CSV Beijing.

    Returns
    -------
    pd.DataFrame : dataset nettoye avec colonnes :
                   datetime, station, pm25, co, temperature, humidity
    """
    raw_dir = raw_dir or RAW_DIR

    print("=" * 55)
    print("PREPROCESSING — Beijing Multi-Site UCI 501")
    print("4 features : PM2.5 | CO | Temperature | Humidity")
    print("=" * 55)

    # 1. Chargement des 12 CSV
    print("\n[1/5] Chargement et combinaison des 12 CSV...")
    df = load_raw_beijing(raw_dir)

    # 2. Renommage + datetime
    print("\n[2/5] Renommage des colonnes + construction datetime...")
    df = rename_and_build_datetime(df)

    # 3. Calcul humidite
    print("\n[3/5] Calcul de l'humidite relative (DEWP -> RH)...")
    df = compute_humidity(df)

    # 4. Valeurs manquantes
    print("\n[4/5] Traitement des valeurs manquantes...")
    df = handle_missing(df)

    # 5. Outliers
    print("\n[5/5] Suppression des outliers...")
    df = remove_outliers(df)

    # Statistiques finales
    print("\n" + "=" * 55)
    print("RESULTAT FINAL")
    print("=" * 55)
    print(f"  Lignes       : {len(df):,}")
    print(f"  Stations     : {df['station'].nunique()}")
    print(f"  Periode      : {df['datetime'].min()} -> {df['datetime'].max()}")
    print(f"\n  Statistiques des 4 features :")

    stats = df[AIRSAFE_FEATURES].describe().round(3)
    print(stats.to_string())

    print(f"\n  Repartition par station :")
    print(df.groupby("station").size().rename("n_lignes").to_string())

    # Sauvegarde
    if save:
        save_scalers(df)
        save_processed(df[["datetime", "station"] + AIRSAFE_FEATURES],
                       "beijing_processed.csv")
        print("\nPreprocessing termine.")

    return df


# ---------------------------------------------------------------------------
# Point d'entree
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = run_preprocessing()