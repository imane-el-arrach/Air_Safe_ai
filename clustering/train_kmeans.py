import os
import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

from data.loader import load_processed, PROCESSED_DIR
from ier.calculator import compute_ier_dataframe

# Chemin de sauvegarde du modèle
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "kmeans.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "kmeans_scaler.pkl")
LABELS_PATH = os.path.join(MODEL_DIR, "cluster_labels.pkl")

# Features utilisées pour le clustering
# IER est calculé et ajouté avant le clustering
CLUSTER_FEATURES = ["pm25", "co", "humidity", "temperature", "ier_score"]

# K fixé à 4 (zones A/B/C/D)
K = 4
RANDOM_STATE = 42


# Étape 1 : Préparation des features
def prepare_cluster_features(
    df: pd.DataFrame,
    pathologie: str = "general",
) -> pd.DataFrame:
    df = df.copy()

    # Calculer l'IER si pas encore présent
    if "ier_score" not in df.columns:
        print("  Calcul de l'IER (pathologie: general)...")
        df = compute_ier_dataframe(df, pathologie=pathologie)

    # Vérification des features
    missing = [f for f in CLUSTER_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Features manquantes pour le clustering : {missing}")

    # Agrégation journalière par station 
    if "datetime" in df.columns and "station" in df.columns:
        df["date"] = pd.to_datetime(df["datetime"]).dt.date
        df_agg = (
            df.groupby(["station", "date"])[CLUSTER_FEATURES]
            .mean()
            .reset_index()
        )
        print(f"  Agrégation journalière : {len(df)} → {len(df_agg)} observations")
        df = df_agg

    X = df[CLUSTER_FEATURES].dropna()
    print(f"  Features de clustering : {CLUSTER_FEATURES}")
    print(f"  Observations valides   : {len(X)}")
    return X

# Étape 2 : Méthode Elbow + Silhouette pour validation de K
def evaluate_k(
    X_scaled: np.ndarray,
    k_range: range = range(2, 9),
) -> dict:
    results = {"k": [], "inertia": [], "silhouette": [], "davies_bouldin": []}

    print(f"\n  Évaluation K-Means pour K={list(k_range)} :")
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)

        inertia = km.inertia_
        sil = silhouette_score(X_scaled, labels, sample_size=min(5000, len(X_scaled)))
        db = davies_bouldin_score(X_scaled, labels)

        results["k"].append(k)
        results["inertia"].append(round(inertia, 2))
        results["silhouette"].append(round(sil, 4))
        results["davies_bouldin"].append(round(db, 4))

        print(f"    K={k} | Inertie={inertia:10.1f} | "
              f"Silhouette={sil:.4f} | Davies-Bouldin={db:.4f}")

    return results


# Étape 3 : Attribution des labels A/B/C/D

def assign_zone_labels(
    kmeans: KMeans,
    scaler: StandardScaler,
    feature_names: list[str],
) -> dict[int, str]:
    # Centroïdes dans l'espace original (inverse_transform)
    centroids_scaled = kmeans.cluster_centers_
    centroids_orig = scaler.inverse_transform(centroids_scaled)
    centroids_df = pd.DataFrame(centroids_orig, columns=feature_names)

    # Classer par score IER du centroïde 
    if "ier_score" in centroids_df.columns:
        sort_col = "ier_score"
    else:
        # classer par PM2.5 + CO (corrélés au risque)
        sort_col = "pm25"

    centroids_df["cluster_id"] = range(K)
    centroids_df.sort_values(sort_col, inplace=True)
    centroids_df.reset_index(drop=True, inplace=True)

    zone_names = ["A", "B", "C", "D"]
    mapping = {
        int(row["cluster_id"]): zone_names[i]
        for i, row in centroids_df.iterrows()
    }

    print("\n  Attribution des zones :")
    for cluster_id, zone in mapping.items():
        ier_val = centroids_df.loc[
            centroids_df["cluster_id"] == cluster_id, sort_col
        ].values[0]
        print(f"    Cluster {cluster_id} → Zone {zone} "
              f"(centroïde {sort_col}={ier_val:.2f})")

    return mapping


# Entraînement principal


def train_kmeans(
    df: pd.DataFrame = None,
    pathologie: str = "general",
    evaluate_elbow: bool = True,
    save: bool = True,
) -> tuple[KMeans, StandardScaler, dict]:
    print("=" * 60)
    print("ENTRAÎNEMENT K-MEANS — AirSafe AI Clustering")
    print("=" * 60)

    # Chargement données
    if df is None:
        print("\n[1/5] Chargement des données preprocessées...")
        df = load_processed("beijing_processed.csv")
    else:
        print("\n[1/5] Utilisation du DataFrame fourni...")

    # Préparation features
    print("\n[2/5] Préparation des features de clustering...")
    X = prepare_cluster_features(df, pathologie=pathologie)

    # Normalisation StandardScaler (
    print("\n[3/5] Normalisation des features (StandardScaler)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[CLUSTER_FEATURES])
    print(f"  Forme des données : {X_scaled.shape}")

    # Évaluation Elbow
    if evaluate_elbow:
        print("\n[4/5] Évaluation des métriques Elbow + Silhouette...")
        eval_results = evaluate_k(X_scaled, k_range=range(2, 9))
        best_k_sil = eval_results["k"][
            np.argmax(eval_results["silhouette"])
        ]
        print(f"\n  K optimal (Silhouette) : {best_k_sil}")
        print(f"  K retenu (cahier de charges) : {K}")
    else:
        print("\n[4/5] Évaluation Elbow ignorée...")

    # Entraînement K=4
    print(f"\n[5/5] Entraînement K-Means (K={K}, n_init=10)...")
    kmeans = KMeans(
        n_clusters=K,
        random_state=RANDOM_STATE,
        n_init=10,
        max_iter=300,
    )
    kmeans.fit(X_scaled)

    # Métriques finales
    labels_pred = kmeans.labels_
    sil_final = silhouette_score(
        X_scaled, labels_pred,
        sample_size=min(5000, len(X_scaled))
    )
    db_final = davies_bouldin_score(X_scaled, labels_pred)
    print(f"\n  Métriques finales (K={K}) :")
    print(f"    Inertie          : {kmeans.inertia_:.2f}")
    print(f"    Silhouette       : {sil_final:.4f}")
    print(f"    Davies-Bouldin   : {db_final:.4f}")
    print(f"    Itérations       : {kmeans.n_iter_}")

    # Distribution des clusters
    unique, counts = np.unique(labels_pred, return_counts=True)
    print(f"\n  Distribution des clusters :")
    for c, n in zip(unique, counts):
        print(f"    Cluster {c} : {n} observations ({n/len(labels_pred)*100:.1f}%)")

    # Attribution des labels A/B/C/D
    cluster_labels = assign_zone_labels(kmeans, scaler, CLUSTER_FEATURES)

    # Sauvegarde
    if save:
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(kmeans, f)
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(scaler, f)
        with open(LABELS_PATH, "wb") as f:
            pickle.dump({
                "cluster_labels": cluster_labels,
                "feature_names":  CLUSTER_FEATURES,
                "k":              K,
                "silhouette":     sil_final,
                "davies_bouldin": db_final,
                "pathologie":     pathologie,
            }, f)
        print(f"\n  Modèle sauvegardé    : {MODEL_PATH}")
        print(f"  Scaler sauvegardé    : {SCALER_PATH}")
        print(f"  Labels sauvegardés   : {LABELS_PATH}")

    return kmeans, scaler, cluster_labels



# Point d'entrée
if __name__ == "__main__":
    kmeans, scaler, labels = train_kmeans(evaluate_elbow=True)
    print("\nEntraînement K-Means terminé.")
    print(f"Mapping clusters → zones : {labels}")