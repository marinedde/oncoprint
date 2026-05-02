---
title: OncoPrint API
emoji: 🧬
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# 🧬 OncoPrint — Classification Moléculaire & Pronostic de Survie

**API FastAPI — Certification CDSD Jedha 2026 — Marine Deldicque**

---

## Description

OncoPrint est une API de classification moléculaire du cancer du sein à partir de données génomiques multi-omiques (TCGA-BRCA).

Elle répond à deux questions cliniques :
1. **Quel est le sous-type moléculaire** de la tumeur ? (Luminal A, Luminal B/HER2+, HER2-enriched, Triple Négatif)
2. **Quel est le pronostic de survie** associé à ce sous-type ? (Kaplan-Meier + Cox PH)

---

## Endpoints

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/` | Informations générales |
| GET | `/health` | Statut de l'API + métriques du modèle |
| GET | `/features` | Liste des 1936 features attendues |
| POST | `/predict` | Prédiction sous-type + pronostic survie |
| GET | `/survival/{subtype}` | Infos survie pour un sous-type |
| GET | `/subtypes` | Description clinique des 4 sous-types |
| GET | `/docs` | Documentation Swagger interactive |

---

## Modèle

- **Algorithme** : XGBoost (optimisé GridSearchCV)
- **Dataset** : TCGA-BRCA Multi-Omics (536 patientes, 1936 features)
- **Accuracy test** : 84.3%
- **F1 Macro test** : 0.762
- **Survie** : Kaplan-Meier + Cox PH (lifelines)

---

## Exemple d'utilisation

```bash
curl -X POST "https://your-space.hf.space/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "rs_ESR1": 2.5,
      "pp_ER.alpha": 1.8,
      "rs_ERBB2": -0.3
    }
  }'
```

Réponse :
```json
{
  "subtype": "Luminal A",
  "confidence": 0.89,
  "probabilities": {
    "Luminal A": 0.89,
    "Triple Négatif": 0.06,
    "HER2-enriched": 0.03,
    "Luminal B / HER2+": 0.02
  },
  "survival": {
    "median_survival_months": null,
    "hazard_ratio": 1.0,
    "reference_subtype": "Luminal A"
  }
}
```

---

## Structure du projet

```
oncoprint-api/
├── main.py                      # API FastAPI
├── requirements.txt             # Dépendances
├── Dockerfile                   # Conteneurisation
├── README.md                    # Cette page
└── models/
    ├── oncoprint_xgb.joblib     # Modèle XGBoost
    ├── oncoprint_scaler.joblib  # StandardScaler
    ├── oncoprint_le.joblib      # LabelEncoder
    ├── oncoprint_features.joblib # Noms des 1936 features
    ├── oncoprint_info.joblib    # Métadonnées modèle
    ├── oncoprint_cox.joblib     # Modèle Cox PH (survie)
    └── oncoprint_kmf.joblib     # Estimateurs Kaplan-Meier
```

---

## Stack technique

`Python 3.10` · `FastAPI` · `XGBoost` · `scikit-learn` · `lifelines` · `Docker` · `HuggingFace Spaces`
