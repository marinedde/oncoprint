# 🧬 OncoPrint
**Classification moléculaire automatique du cancer du sein**

Projet CDSD — Jedha Bootcamp 2026 | Marine Deldicque

---

## Contexte

OncoPrint est un outil d'aide à la décision basé sur le machine learning
pour la classification moléculaire du cancer du sein en 4 sous-types
(Luminal A, Triple Négatif, HER2-enriched, Luminal B / HER2+).

Il s'adresse à 3 profils d'utilisateurs :
- 🩺 **Médecin chercheur / CRO** — aide à la décision avec validation de cohérence
- 🔬 **Data scientist** — exploration SHAP, batch CSV, analyse de profils ambigus
- 📚 **Étudiant** — apprentissage interactif des sous-types moléculaires

---

## Dataset

- **Source :** The Cancer Genome Atlas (TCGA-BRCA) — NCI/NIH
- **Accès :** [Kaggle BRCA Multi-Omics](https://www.kaggle.com/datasets/samdemharter/brca-multiomics-tcga)
- **Population :** 536 patientes après nettoyage (705 initiales)
- **Features :** 1936 mesures multi-omiques
  - 249 mutations somatiques
  - 860 copy number variations
  - 604 gènes RNA-seq
  - 223 phospho-protéines

---

## Résultats

| Modèle | Accuracy | F1 Macro |
|---|---|---|
| Dummy baseline | 4.6% | 2.2% |
| Random Forest baseline | 80.6% | 69.2% |
| XGBoost baseline | 84.3% | 76.2% |
| Random Forest optimisé | 82.4% | 72.7% |
| **XGBoost optimisé** | **84.3%** | **76.3%** |

### Performances par sous-type (XGBoost optimisé)

| Sous-type | Precision | Recall | F1 |
|---|---|---|---|
| Luminal A | 0.90 | 0.89 | 0.89 |
| Triple Négatif | 0.76 | 0.89 | 0.82 |
| HER2-enriched | 0.71 | 1.00 | 0.83 |
| Luminal B / HER2+ | 0.62 | 0.42 | 0.50 |

---

## Pipeline

Données TCGA-BRCA (705 patientes)
↓
Construction target ER/PR/HER2 → 4 sous-types
↓
Preprocessing : SMOTE + StandardScaler + VarianceThreshold
↓
XGBoost optimisé (GridSearchCV)
↓
Validation SHAP — cohérence biologique confirmée
↓
API FastAPI + Dashboard Streamlit + Rapports Claude AI

---

## Stack technique

- **ML :** Python, XGBoost, scikit-learn, SHAP, imbalanced-learn
- **API :** FastAPI, Uvicorn
- **Dashboard :** Streamlit
- **LLM :** Anthropic Claude API
- **Environnement :** Google Colab, VS Code

---

## Installation

```bash
# Cloner le repo
git clone https://github.com/ton-username/oncoprint.git
cd oncoprint

# Créer l'environnement virtuel
python3 -m venv venv
source venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt

# Configurer la clé API
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Éditer secrets.toml et renseigner votre clé Anthropic

# Lancer l'API
uvicorn api.main:app --reload

# Dans un autre terminal, lancer le dashboard
streamlit run app/streamlit_app.py
```

---

## Notebooks

| Notebook | Contenu |
|---|---|
| 01_EDA | Audit dataset, outliers, distribution sous-types |
| 02_Preprocessing | SMOTE, scaling, correction data leakage |
| 03_Unsupervised_ML | PCA, t-SNE, UMAP, K-Means, clustering hiérarchique |
| 04_Supervised_ML | RF, XGBoost, GridSearch, validation croisée |
| 05_Interpretation | SHAP, validation biologique, rapport clinique |

---

## Limites

- Données TCGA en conditions recherche — validation clinique nécessaire
- Luminal B : recall 42% — hétérogénéité biologique connue
- HER2-enriched : n=5 dans le test — statistiquement fragile
- Interface démo : 9 biomarqueurs sur 1936 — usage réel via upload CSV

---

## Auteure

**Marine Deldicque**
Infirmière libérale — Data Scientist en transition
Jedha Bootcamp — CDSD 2026

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Marine_Deldicque-blue)](https://linkedin.com/in/ton-profil)
[![GitHub](https://img.shields.io/badge/GitHub-marinedde-black)](https://github.com/marinedde)