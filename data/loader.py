
import os
import glob
import pandas as pd

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------

BASE_DIR       = os.path.dirname(os.path.dirname(__file__))   # ai/
RAW_DIR        = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR  = os.path.join(BASE_DIR, "data", "processed")

# ---------------------------------------------------------------------------
# Colonnes a conserver depuis les CSV bruts
# ---------------------------------------------------------------------------

# Colonnes necessaires pour construire les 4 features AirSafe AI
KEEP_COLS = [
    "No",       # index original (supprime apres)
    "year",
    "month",
    "day",
    "hour",
    "PM2.5",    # -> pm25
    "CO",       # -> co
    "TEMP",     # -> temperature
    "DEWP",     # -> dewpoint -> humidity
    "station",  # nom de la station (region)
]

# Les 4 noms finaux apres renommage
AIRSAFE_FEATURES = ["pm25", "co", "temperature", "humidity"]

# Noms des 12 stations attendues
EXPECTED_STATIONS = [
    "Aotizhongxin", "Changping", "Dingling", "Dongsi",
    "Guanyuan", "Gucheng", "Huairou", "Nongzhanguan",
    "Shunyi", "Tiantan", "Wanliu", "Wanshouxigong",
]


# ---------------------------------------------------------------------------
# Chargement + combinaison des 12 CSV
# ---------------------------------------------------------------------------

def load_raw_beijing(raw_dir: str = RAW_DIR) -> pd.DataFrame:
    """
    Charge et concatene les 12 fichiers CSV des stations Beijing.

    Chaque fichier correspond a une station (region).
    La colonne 'station' est presente dans chaque fichier.

    Parameters
    ----------
    raw_dir : dossier contenant les fichiers PRSA_Data_*.csv

    Returns
    -------
    pd.DataFrame : toutes les stations concatenees

    Raises
    ------
    FileNotFoundError : si aucun fichier CSV trouve dans raw_dir
    """
    pattern = os.path.join(raw_dir, "PRSA_Data_*.csv")
    files   = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"\nAucun fichier CSV trouve dans : {raw_dir}\n"
            "Placez les 12 fichiers PRSA_Data_*.csv dans le dossier ai/data/raw/\n"
            "Exemple : ai/data/raw/PRSA_Data_Aotizhongxin_20130301-20170228.csv"
        )

    print(f"Fichiers trouves : {len(files)}")
    frames = []

    for fp in files:
        # Extraire le nom de station depuis le nom de fichier si besoin
        station_from_filename = (
            os.path.basename(fp)
            .replace("PRSA_Data_", "")
            .split("_")[0]
        )

        # Lire uniquement les colonnes utiles
        df = pd.read_csv(fp, usecols=lambda c: c in KEEP_COLS)

        # Si la colonne station est absente, l'ajouter depuis le nom de fichier
        if "station" not in df.columns:
            df["station"] = station_from_filename

        n = len(df)
        station = df["station"].iloc[0] if "station" in df.columns else station_from_filename
        print(f"  {station:20s} : {n:6d} lignes | "
              f"PM2.5 NaN={df['PM2.5'].isna().sum():4d} | "
              f"CO NaN={df['CO'].isna().sum():4d} | "
              f"TEMP NaN={df['TEMP'].isna().sum():4d} | "
              f"DEWP NaN={df['DEWP'].isna().sum():4d}")

        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    print(f"\n{'='*50}")
    print(f"COMBINE : {len(combined):,} lignes | {combined['station'].nunique()} stations")
    print(f"Periode : {combined['year'].min()}/{combined['month'].min():02d} "
          f"-> {combined['year'].max()}/{combined['month'].max():02d}")
    print(f"Stations : {sorted(combined['station'].unique().tolist())}")
    print(f"{'='*50}")

    return combined


# ---------------------------------------------------------------------------
# Sauvegarde / chargement du processed
# ---------------------------------------------------------------------------

def save_processed(df: pd.DataFrame, filename: str = "beijing_processed.csv") -> str:
    """Sauvegarde le DataFrame dans processed/ et retourne le chemin."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    path = os.path.join(PROCESSED_DIR, filename)
    df.to_csv(path, index=False)
    print(f"Sauvegarde : {path} ({len(df):,} lignes)")
    return path


def load_processed(filename: str = "beijing_processed.csv") -> pd.DataFrame:
    """Charge les donnees depuis processed/."""
    path = os.path.join(PROCESSED_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Fichier processed introuvable : {path}\n"
            "Lancez d'abord : python -m ai.data.preprocessing"
        )
    df = pd.read_csv(path)
    print(f"Charge (processed) : {len(df):,} lignes | colonnes : {list(df.columns)}")
    return df


# ---------------------------------------------------------------------------
# Point d'entree — verification rapide
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 55)
    print("VERIFICATION DES FICHIERS RAW — Beijing UCI 501")
    print("=" * 55)

    try:
        df = load_raw_beijing()
        print(f"\nColonnes disponibles : {list(df.columns)}")
        print(f"\nApercu (5 premieres lignes) :")
        print(df[["year","month","day","hour","PM2.5","CO","TEMP","DEWP","station"]].head())
        print(f"\nStatistiques brutes :")
        print(df[["PM2.5","CO","TEMP","DEWP"]].describe().round(3))
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)