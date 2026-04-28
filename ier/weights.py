from typing import Literal

# Type de pathologie supporté
Pathologie = Literal["asthme", "rhinite", "bronchite", "general"]

# Tableau des poids par pathologie 
WEIGHTS: dict[str, dict[str, float]] = {
    "asthme": {
        "pm25":        0.40,
        "co":          0.25,
        "humidity":    0.20,
        "temperature": 0.15,
    },
    "rhinite": {
        "pm25":        0.30,
        "co":          0.20,
        "humidity":    0.35,
        "temperature": 0.15,
    },
    "bronchite": {
        "pm25":        0.35,
        "co":          0.30,
        "humidity":    0.20,
        "temperature": 0.15,
    },
    "general": {
        "pm25":        0.30,
        "co":          0.25,
        "humidity":    0.25,
        "temperature": 0.20,
    },
}

# Niveaux de risque selon le score IER 
RISK_LEVELS = [
    {"min": 0,  "max": 25,  "level": "Faible",   "color": "#2ecc71",
     "action": "Aucune action requise"},
    {"min": 26, "max": 50,  "level": "Modéré",   "color": "#f39c12",
     "action": "Aérer si possible"},
    {"min": 51, "max": 75,  "level": "Élevé",    "color": "#e67e22",
     "action": "Réduire l'exposition"},
    {"min": 76, "max": 100, "level": "Critique", "color": "#e74c3c",
     "action": "Alerte + précautions urgentes"},
]


def get_weights(pathologie: Pathologie = "general") -> dict[str, float]:
    pathologie = pathologie.lower().strip()
    if pathologie not in WEIGHTS:
        raise ValueError(
            f"Pathologie '{pathologie}' inconnue. "
            f"Valeurs acceptées : {list(WEIGHTS.keys())}"
        )
    return WEIGHTS[pathologie]


def get_risk_level(score: float) -> dict:
    score = max(0.0, min(100.0, score))
    for risk in RISK_LEVELS:
        if risk["min"] <= score <= risk["max"]:
            return risk
    return RISK_LEVELS[-1]


def validate_weights(pathologie: Pathologie = "general") -> bool:
    """Vérifie que les poids d'une pathologie somment à 1.0."""
    w = get_weights(pathologie)
    total = sum(w.values())
    assert abs(total - 1.0) < 1e-9, f"Poids invalides pour {pathologie} : somme = {total}"
    return True


if __name__ == "__main__":
    print("Vérification des poids IER :")
    for p in WEIGHTS:
        w = get_weights(p)
        total = sum(w.values())
        print(f"  {p:12s} → somme = {total:.2f} | poids : {w}")
    print("\nNiveaux de risque :")
    for r in RISK_LEVELS:
        print(f"  [{r['min']:3d}–{r['max']:3d}] {r['level']:8s} → {r['action']}")