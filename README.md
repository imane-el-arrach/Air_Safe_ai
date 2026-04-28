# Air_Safe_ai

# ier/calculator.py

Calcul de l'Indice Environnemental Respiratoire (IER).

Formule (cahier de charges) :
    IER = Σᵢ wᵢ × x̃ᵢ
    où x̃ᵢ = (xᵢ - xmin) / (xmax - xmin) × 100

Le score IER est sur une échelle de 0 (air sain) à 100 (risque critique).

Deux modes d'utilisation :
    1. Batch  → calcul sur un DataFrame complet (entraînement / analyse)
    2. Temps réel → calcul sur une seule mesure dict (pipeline IoT)

Calcule l'IER pour une seule mesure IoT (usage temps réel).

    Parameters
    ----------
    pm25, co, humidity, temperature : valeurs brutes des capteurs
    pathologie : pathologie de l'utilisateur
    bounds     : bornes de normalisation (scalers du preprocessing)

    Returns
    -------
    dict :
        {
            'score':       float,   # score IER [0–100]
            'level':       str,     # 'Faible' | 'Modéré' | 'Élevé' | 'Critique'
            'color':       str,     # code couleur hex
            'action':      str,     # recommandation
            'details':     dict,    # contribution de chaque feature
        }

    Examples
    --------
    >>> result = compute_ier_single(45.0, 1.5, 70.0, 22.0, "asthme")
    >>> print(result['score'], result['level'])

# ier/weights.py

Pondérations de l'Indice Environnemental Respiratoire (IER) par pathologie.

Source : Tableau 1 du cahier de charges AirSafe AI.

    IER = Σ wᵢ × x̃ᵢ
    où x̃ᵢ = (xᵢ - xmin) / (xmax - xmin) × 100

Les poids sont normalisés à 1.0 pour chaque pathologie.

Retourne les poids IER pour une pathologie donnée.

    Parameters
    ----------
    pathologie : 'asthme' | 'rhinite' | 'bronchite' | 'general'

    Returns
    -------
    dict : {'pm25': w1, 'co': w2, 'humidity': w3, 'temperature': w4}

    Raises
    ------
    ValueError : si la pathologie n'est pas reconnue.

    Examples
    --------
    >>> get_weights("asthme")
    {'pm25': 0.40, 'co': 0.25, 'humidity': 0.20, 'temperature': 0.15}




# clustering/predict_zone.py
------------------------------
Prédiction de la zone environnementale (A/B/C/D) en temps réel.

Ce module charge le modèle K-Means entraîné et prédit la zone
d'une nouvelle mesure IoT ou d'un DataFrame de mesures.

Zones :
    A — Saine    : air propre, sport autorisé
    B — Modérée  : surveillance conseillée
    C — Risquée  : éviter l'activité physique
    D — Critique : alerte urgente

Usage :
    from clustering.predict_zone import predict_zone, predict_zone_batch
    zone = predict_zone(pm25=45.0, co=1.5, humidity=65.0, temperature=22.0, ier_score=38.0)


Prédit la zone environnementale pour une mesure IoT unique.

    Parameters
    ----------
    pm25        : concentration PM2.5 (µg/m³)
    co          : concentration CO (mg/m³)
    humidity    : humidité relative (%)
    temperature : température (°C)
    ier_score   : score IER calculé (0–100)

    Returns
    -------
    dict :
        {
            'zone':           str,   # 'A' | 'B' | 'C' | 'D'
            'label':          str,   # 'Saine' | 'Modérée' | 'Risquée' | 'Critique'
            'description':    str,
            'recommendation': str,
            'color':          str,   # hex color
            'cluster_id':     int,   # id brut du cluster K-Means
        }

    Examples
    --------
    >>> result = predict_zone(45.0, 1.5, 70.0, 22.0, ier_score=42.0)
    >>> print(result['zone'], result['label'])
    B Modérée


# clustering/train_kmeans.py
------------------------------
Entraînement du modèle K-Means pour le clustering des zones environnementales.

Zones cibles (cahier de charges) :
    A — Saine      : air propre, humidité normale  → sport autorisé
    B — Modérée    : pollution légère, ventilation ok → surveillance
    C — Risquée    : PM2.5 élevé, CO2 fort          → éviter activité
    D — Critique   : pollution + humidité, risque max → alerte urgente

Étapes :
    1. Chargement des données preprocessées
    2. Sélection et mise à l'échelle des features de clustering
    3. Détermination du K optimal (méthode Elbow + Silhouette)
    4. Entraînement K-Means avec K=4
    5. Attribution des labels A/B/C/D selon le centroïde
    6. Sauvegarde du modèle + scaler + mapping labels

Usage :
    python -m clustering.train_kmeans

# Pipeline complet d'entraînement du K-Means.

    Parameters
    ----------
    df              : DataFrame preprocessé (si None, charge depuis processed/)
    pathologie      : pathologie pour le calcul IER
    evaluate_elbow  : si True, affiche les métriques pour K=2..8
    save            : si True, sauvegarde le modèle

    Returns
    -------
    (KMeans, StandardScaler, dict_labels)


# prediction/features.py

Construction du vecteur de features pour le Random Forest prédictif.

Le modèle prédit P(crise respiratoire dans les 12-24h).

Vecteur de features final (20 features) :
    -- Mesures IoT brutes (4) --
    pm25, co, humidity, temperature

    -- Score IER + zone (2) --
    ier_score, zone_risk_score (A=1, B=2, C=3, D=4)

    -- Features temporelles (4) --
    hour_sin, hour_cos, month_sin, month_cos  (encodage cyclique)

    -- Historique glissant 24h (6) --
    pm25_mean_24h, co_mean_24h, humidity_mean_24h,
    pm25_max_24h,  co_max_24h,  ier_max_24h

    -- Profil utilisateur (4) --
    age, pathologie_encoded, is_smoker, symptom_yesterday

Usage :
    from prediction.features import build_feature_vector, build_features_dataframe

Charge le modele RF dans cet ordre de priorite :
    1. rf_user_{user_id}.pkl  (modele personnalise)
    2. rf_global.pkl          (modele global pre-entraine)

 Raises FileNotFoundError si aucun modele n'existe.


 Predit la probabilite de crise respiratoire dans les 12-24h.

    Parameters
    ----------
    user_id           : identifiant utilisateur (pour charger le bon modele)
    pm25 ... temperature : mesures IoT brutes
    ier_score         : score IER calcule (0-100)
    zone              : zone K-Means ('A'|'B'|'C'|'D')
    hour, month       : heure et mois courants
    history           : liste de mesures passees (24h)
    age, pathologie, is_smoker, symptom_yesterday : profil utilisateur

    Returns
    -------
    PredictionResult avec tous les details

#prediction/retrain.py
Re-entrainement automatique du modele personnalise d'un utilisateur.

Cycle d'apprentissage (cahier de charges) :
    Phase 1 (sem 1-2) : collecte des mesures IoT
    Phase 2 (sem 2-3) : sondage 1x/soir -> labels Y=0/1
    Phase 3 (sem 3-4) : dataset complet -> entrainement RF
    Phase 4 (continu) : nouvelles donnees -> re-entrainement periodique

Ce module est appele automatiquement par le backend FastAPI chaque soir
apres que l'utilisateur a soumis son sondage quotidien.

Usage :
    python -m prediction.retrain --user user_123
    python -m prediction.retrain --user user_123 --force

# Logique de re-entrainement
Ajoute une nouvelle entree labeled au fichier de l'utilisateur.
    Appelee chaque soir apres le sondage.

    Parameters
    ----------
    user_id        : identifiant utilisateur
    features       : dict des 20 features (depuis build_feature_dict)
    symptom_label  : 0 (pas de crise) ou 1 (crise/symptome)
    timestamp      : datetime de la mesure (defaut : maintenant)

    Returns
    -------
    int : nombre total d'entrees apres ajout

Determine si le modele de l'utilisateur doit etre re-entraine.

    Regles :
        1. Au moins MIN_SAMPLES_RETRAIN entrees labeled
        2. Pas de modele existant OU +RETRAIN_EVERY_N_SAMPLES depuis le dernier entrainement

    Returns
    -------
    (bool, str) : (doit_retrainer, raison)
    
 Re-entraine le modele personnalise d'un utilisateur.

    Parameters
    ----------
    user_id         : identifiant utilisateur
    force           : si True, ignore les seuils et force le re-entrainement
    include_global  : si True, augmente les donnees user avec une fraction du dataset global

# prediction/train.py

Entrainement du modele Random Forest — P(crise respiratoire dans les 12-24h).

Sources de donnees disponibles :
    'kaggle'    : dataset Kaggle "Air Quality, Weather & Respiratory Health"
                  -> vrais labels respiratory_admissions binarises
                  -> PRIORITAIRE par rapport a synthetic
    'synthetic' : Beijing UCI 501 + labels heuristiques IER
                  -> cold-start si Kaggle absent
    'labeled'   : sondages reels des utilisateurs
                  -> re-entrainement personnalise (retrain.py)

Usage :
    python -m prediction.train                      # Kaggle en priorite
    python -m prediction.train --source kaggle      # forcer Kaggle
    python -m prediction.train --source synthetic   # forcer synthetic
    python -m prediction.train --user user_123      # modele perso