# OncoPrint 🧬

> **Chaque tumeur parle. OncoPrint traduit.**

Classification moléculaire du cancer du sein par apprentissage automatique — données TCGA-BRCA

[![HuggingFace Dashboard](https://img.shields.io/badge/🤗%20Dashboard-OncoPrint-blue)](https://huggingface.co/spaces/marinedde/oncoprint-dashboard)
[![HuggingFace API](https://img.shields.io/badge/🤗%20API-FastAPI-green)](https://huggingface.co/spaces/marinedde/oncoprint-api)
[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-orange)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## Présentation

Le cancer du sein regroupe **4 sous-types moléculaires biologiquement distincts**, dont le traitement dépend directement du profil génomique de la tumeur — pas seulement de sa localisation.

OncoPrint exploite les données multi-omiques du dataset public **TCGA-BRCA** (536 patientes, 1936 variables) pour prédire automatiquement ce sous-type à partir du profil génomique d'une patiente, avec interprétabilité des biomarqueurs et estimation du pronostic de survie.

| Sous-type | Biologie | Traitement | Pronostic |
|---|---|---|---|
| 🔵 Luminal A | ER+ PR+ HER2- | Hormonothérapie | Favorable ~90% à 5 ans |
| 🔴 Triple Négatif | ER- PR- HER2- | Chimiothérapie | Défavorable |
| 🟠 HER2-enriched | ER- PR- HER2+ | Trastuzumab (Herceptin) | Intermédiaire |
| 🟢 Luminal B / HER2+ | ER+ HER2+ | Hormo + anti-HER2 | Intermédiaire |

---

## Demo live

| Lien | Description |
|---|---|
| [Dashboard Streamlit](https://huggingface.co/spaces/marinedde/oncoprint-dashboard) | Interface complète — aide à la décision, exploration, survie |
| [API FastAPI](https://huggingface.co/spaces/marinedde/oncoprint-api) | API REST — `/predict`, `/survival`, `/health` |
| [Documentation API](https://marinedde-oncoprint-api.hf.space/docs) | Swagger auto-généré |

> **Note** : les Spaces HuggingFace gratuits se mettent en veille après inactivité. Si l'API ne répond pas, visiter l'URL de l'API directement pour la réveiller (30-60 secondes).

---

## Fonctionnalités

### Aide à la décision clinique
- Saisie des biomarqueurs via sliders (RNA-seq + phospho-protéines)
- Profils prédéfinis : Luminal A, Triple Négatif, HER2-enriched, Luminal B / HER2+, Profil ambigu
- Prédiction du sous-type avec score de confiance et probabilités par classe
- Validation de cohérence biomarqueurs
- Rapport clinique généré par Claude AI (Anthropic), adapté au profil prédit

### Pronostic de survie
- Médiane de survie par sous-type (Kaplan-Meier, cohorte TCGA-BRCA)
- Hazard Ratio vs Luminal A (Cox Proportional Hazards)
- Simulation pour un profil prédit

### Outil de recherche
- Analyse SHAP — top features déterminantes par prédiction
- Probabilités détaillées par sous-type
- Prédiction en batch via upload CSV

### Explorer & apprendre
- Description biologique des 4 sous-types
- Profils interactifs avec explication pédagogique générée par IA

### Performances
- Métriques globales et par sous-type
- Matrice de confusion, courbes ROC

---

## Résultats du modèle

| Métrique | Valeur |
|---|---|
| Accuracy (test set) | **84.3%** |
| F1 Macro | **76.3%** |
| Validation croisée 5-fold | **71.5% ± 2.5%** |
| Patientes test | 108 (20% holdout) |

**F1 par sous-type :**

| Sous-type | Precision | Recall | F1 |
|---|---|---|---|
| Luminal A | 0.90 | 0.89 | **0.89** |
| Triple Négatif | 0.76 | 0.89 | **0.82** |
| HER2-enriched | 0.71 | 1.00 | **0.83** |
| Luminal B / HER2+ | 0.62 | 0.42 | **0.50** |

> Luminal B F1=0.50 reflète l'hétérogénéité biologique connue de ce sous-type, pas une limite algorithmique.

**Validation SHAP — cohérence biologique :**

| Sous-type | Feature #1 | Cohérence clinique |
|---|---|---|
| Luminal A | pp_ER.alpha | Standard clinique depuis 40 ans — cible de l'hormonothérapie |
| Triple Négatif | pp_HER2 absent | Définition clinique du Triple Négatif |
| HER2-enriched | pp_HER2.pY1248 | Cible directe du Trastuzumab |
| Luminal B / HER2+ | cn_PPP1R1B | Marqueur d'amplification connu |

---

## Module de survie

Estimations issues de la cohorte TCGA-BRCA — à titre de recherche uniquement.

| Sous-type | Médiane OS | Hazard Ratio |
|---|---|---|
| Luminal A | > 180 mois | 1.00x (référence) |
| Luminal B / HER2+ | 120 mois | 1.45x |
| HER2-enriched | 96 mois | 1.82x |
| Triple Négatif | 72 mois | 2.31x |

---

## Architecture

```
TCGA-BRCA (705 patientes)
        ↓
Construction target ER/PR/HER2 → 4 sous-types (536 après nettoyage)
        ↓
Preprocessing : VarianceThreshold + StandardScaler (rs_ + pp_)
        ↓
SMOTE corrigé — appliqué uniquement dans le pipeline CV (pas de data leakage)
        ↓
XGBoost optimisé (GridSearchCV 5-fold) — Accuracy 84.3% / F1 Macro 76.3%
        ↓
SHAP — validation biologique des features
        ↓
lifelines — Kaplan-Meier + Cox Proportional Hazards
        ↓
FastAPI (Docker) + Streamlit + Claude AI — déployé sur HuggingFace Spaces
```

### Stack technique

| Couche | Technologies |
|---|---|
| Données & EDA | Python 3.12, Pandas, NumPy, Matplotlib, Seaborn, Plotly |
| ML | XGBoost 3.2.0, scikit-learn 1.6.1, imbalanced-learn (SMOTE), SHAP |
| Survie | lifelines 0.27.8, scipy 1.11.4 |
| API | FastAPI, Pydantic, uvicorn |
| Conteneurisation | Docker (python:3.12-slim) |
| Interface | Streamlit multipage |
| IA générative | Anthropic Claude API |
| Déploiement | HuggingFace Spaces (Docker + Streamlit) |

---

## Dataset

**The Cancer Genome Atlas — Breast Cancer (TCGA-BRCA)**
- Source : [NCI/NIH](https://www.cancer.gov/tcga) — données publiques de recherche
- 705 patientes initiales → 536 après nettoyage
- 1936 features multi-omiques : 249 mutations, 860 CNV, 604 RNA-seq, 223 phospho-protéines
- Variable cible construite depuis les valeurs ER/PR/HER2 par règles cliniques standard

---

## Installation locale

```bash
# Cloner le repo
git clone https://huggingface.co/spaces/marinedde/oncoprint-dashboard
cd oncoprint-dashboard

# Installer les dépendances
pip install -r requirements.txt

# Lancer le dashboard
streamlit run streamlit_app.py
```

**Pour l'API FastAPI :**
```bash
git clone https://huggingface.co/spaces/marinedde/oncoprint-api
cd oncoprint-api

# Via Docker
docker build -t oncoprint-api .
docker run -p 7860:7860 oncoprint-api

# Ou directement
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 7860
```

---

## Endpoints API

| Méthode | Endpoint | Description |
|---|---|---|
| GET | `/health` | Statut + métriques du modèle |
| GET | `/features` | Liste des 1936 features attendues |
| POST | `/predict` | Prédiction complète (sous-type + SHAP + survie) |
| GET | `/survival/{subtype}` | Données de survie pour un sous-type |
| GET | `/subtypes` | Description des 4 sous-types |

**Exemple `/predict` :**
```json
POST /predict
{
  "features": {
    "rs_ESR1": 10.0,
    "rs_PGR": 8.0,
    "rs_ERBB2": 2.0,
    "pp_ER.alpha": 0.8
  }
}
```

---

## Limites

- Données TCGA en conditions recherche — pas de validation clinique externe
- Luminal B recall 42% — hétérogénéité biologique inhérente à ce sous-type
- HER2-enriched n=5 dans le test — statistiquement fragile
- Interface démo : 9 features sur 1936 — simplification pédagogique
- HuggingFace gratuit : mise en veille après inactivité

**Pour aller plus loin :** plus de données (cohortes multi-centres), ajouter Ki67 et le stade, validation sur cohorte externe — objectif 90%+.

---

## Auteure

**Marine Deldicque**
Infirmière IDEL & Data Scientist — CDSD Jedha 2026

Projet de certification — Certification Data Scientist & Data Analyst, Jedha Bootcamp

> *Ce projet est un outil de recherche et d'aide à la décision. Il ne remplace pas le diagnostic clinique (anatomopathologie, IHC, FISH) ni l'avis d'un professionnel de santé.*
