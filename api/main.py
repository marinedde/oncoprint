# ============================================
# OncoPrint — API FastAPI
# Marine Deldicque — CDSD Jedha 2026
# ============================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import joblib
import numpy as np
import pandas as pd
import os

# --- INITIALISATION ---
app = FastAPI(
    title="OncoPrint API",
    description="Classification moléculaire du cancer du sein — TCGA-BRCA",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# --- CHARGEMENT DES MODÈLES ---
# Dans Docker : __file__ = /app/main.py → BASE_DIR = /app → MODELS_DIR = /app/models
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

model = scaler = le = features = info = cph = None
kmf_dict = {}

def _load(name):
    path = os.path.join(MODELS_DIR, name)
    print(f"Chargement {path} ...")
    obj = joblib.load(path)
    print(f"  OK — {name}")
    return obj

try:
    model    = _load("oncoprint_xgb.joblib")
    scaler   = _load("oncoprint_scaler.joblib")
    le       = _load("oncoprint_le.joblib")
    features = _load("oncoprint_features.joblib")
    info     = _load("oncoprint_info.joblib")
    print("Modèles classification chargés")
except Exception as e:
    print(f"ERREUR chargement classification : {e}")

try:
    cph      = _load("oncoprint_cox.joblib")
    kmf_dict = _load("oncoprint_kmf.joblib")
    print("Modèles survie chargés")
except Exception as e:
    print(f"ERREUR chargement survie : {e} — survie désactivée")
    cph      = None
    kmf_dict = {}

# Colonnes à scaler (seulement si features chargé)
if features is not None:
    rs_cols    = [c for c in features if c.startswith('rs_')]
    pp_cols    = [c for c in features if c.startswith('pp_')]
    rs_pp_cols = rs_cols + pp_cols
else:
    rs_pp_cols = []

# --- SCHÉMAS ---
class PatientData(BaseModel):
    features: Dict[str, float]

class SurvivalInfo(BaseModel):
    median_survival_months: Optional[float]
    hazard_ratio:           Optional[float]
    reference_subtype:      str = "Luminal A"

class PredictionResponse(BaseModel):
    subtype:        str
    confidence:     float
    probabilities:  Dict[str, float]
    top_features:   List[Dict]
    survival:       Optional[SurvivalInfo]
    model_info:     Dict

# --- HELPERS ---
def get_survival_info(subtype: str) -> Optional[SurvivalInfo]:
    """Retourne médiane K-M et Hazard Ratio Cox pour un sous-type."""
    try:
        # Médiane Kaplan-Meier
        median = None
        if subtype in kmf_dict:
            m = kmf_dict[subtype].median_survival_time_
            median = float(m) if m != float('inf') else None

        # Hazard Ratio Cox
        hr = 1.0  # Luminal A = référence
        if cph is not None:
            col = subtype.replace('/', '_').replace(' ', '_')
            if col in cph.params_.index:
                hr = round(float(np.exp(cph.params_[col])), 3)

        return SurvivalInfo(
            median_survival_months=median,
            hazard_ratio=hr,
            reference_subtype="Luminal A"
        )
    except Exception:
        return None

# --- ENDPOINTS ---
@app.get("/")
def root():
    return {
        "projet"  : "OncoPrint",
        "version" : "1.1.0",
        "auteur"  : "Marine Deldicque",
        "modules" : ["classification", "survie (Kaplan-Meier + Cox PH)"],
        "status"  : "running"
    }

@app.get("/debug")
def debug():
    """Diagnostic des modèles — à supprimer en production."""
    import os
    results = {}
    files = [
        "oncoprint_xgb.joblib",
        "oncoprint_scaler.joblib",
        "oncoprint_le.joblib",
        "oncoprint_features.joblib",
        "oncoprint_info.joblib",
        "oncoprint_cox.joblib",
        "oncoprint_kmf.joblib",
    ]
    results["MODELS_DIR"] = MODELS_DIR
    results["dir_exists"] = os.path.isdir(MODELS_DIR)
    results["files"] = {}
    for f in files:
        path = os.path.join(MODELS_DIR, f)
        exists = os.path.isfile(path)
        size = os.path.getsize(path) if exists else 0
        results["files"][f] = {"exists": exists, "size_bytes": size}
        if exists and size > 0:
            try:
                joblib.load(path)
                results["files"][f]["load"] = "OK"
            except Exception as e:
                results["files"][f]["load"] = f"ERREUR: {str(e)}"
    results["model_loaded"] = model is not None
    results["python_version"] = __import__("sys").version
    try:
        import xgboost
        results["xgboost_version"] = xgboost.__version__
    except Exception:
        results["xgboost_version"] = "non disponible"
    try:
        import sklearn
        results["sklearn_version"] = sklearn.__version__
    except Exception:
        results["sklearn_version"] = "non disponible"
    return results

@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Modèles non chargés")
    return {
        "status"         : "healthy",
        "modele"         : info.get("modele", "XGBoost"),
        "accuracy"       : info.get("accuracy_test", 0.843),
        "f1_macro"       : info.get("f1_macro_test", 0.763),
        "classes"        : list(le.classes_),
        "survie_module"  : "ok" if cph is not None else "non disponible"
    }

@app.get("/features")
def get_features():
    if features is None:
        raise HTTPException(status_code=503, detail="Modèles non chargés")
    return {
        "n_features" : len(features),
        "types"      : {
            "mutations"      : len([f for f in features if f.startswith('mu_')]),
            "copy_number"    : len([f for f in features if f.startswith('cn_')]),
            "rna_seq"        : len([f for f in features if f.startswith('rs_')]),
            "phosphoproteins": len([f for f in features if f.startswith('pp_')])
        },
        "features": features[:20]
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(data: PatientData):
    if model is None:
        raise HTTPException(status_code=503, detail="Modèles non chargés")
    try:
        # Construire le vecteur de features
        X = pd.DataFrame([data.features])
        for feat in features:
            if feat not in X.columns:
                X[feat] = 0.0
        X = X[features]

        # Scaling rs_ et pp_
        X_scaled = X.copy()
        cols_to_scale = [c for c in rs_pp_cols if c in X.columns]
        X_scaled[cols_to_scale] = scaler.transform(X[cols_to_scale])

        # Prédiction classification
        prediction = model.predict(X_scaled)[0]
        probas     = model.predict_proba(X_scaled)[0]
        subtype    = le.classes_[prediction]
        confidence = float(probas[prediction])

        proba_dict = {
            le.classes_[i]: float(probas[i])
            for i in range(len(le.classes_))
        }

        # Top features importantes
        importances = model.feature_importances_
        top_idx     = np.argsort(importances)[::-1][:10]
        top_feats   = [
            {
                "feature"   : features[i],
                "importance": float(importances[i]),
                "value"     : float(X_scaled.iloc[0, i])
            }
            for i in top_idx
        ]

        # Module survie
        survival = get_survival_info(subtype)

        return PredictionResponse(
            subtype       = subtype,
            confidence    = confidence,
            probabilities = proba_dict,
            top_features  = top_feats,
            survival      = survival,
            model_info    = {
                "modele"  : "XGBoost",
                "accuracy": info.get("accuracy_test", 0.843),
                "f1_macro": info.get("f1_macro_test", 0.763)
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/survival/{subtype}")
def get_survival(subtype: str):
    """Retourne les infos de survie pour un sous-type donné."""
    survival = get_survival_info(subtype)
    if survival is None:
        raise HTTPException(status_code=404, detail=f"Sous-type '{subtype}' non trouvé")
    return {"subtype": subtype, "survival": survival}

@app.get("/model/info")
def model_info_endpoint():
    if info is None:
        raise HTTPException(status_code=503, detail="Modèles non chargés")
    return info

@app.get("/subtypes")
def get_subtypes():
    return {
        "subtypes": {
            "Luminal A": {
                "biologie"  : "ER+ PR+ HER2-",
                "traitement": "Hormonothérapie",
                "pronostic" : "Favorable — survie 5 ans ~90%",
                "f1_modele" : 0.89
            },
            "Triple Négatif": {
                "biologie"  : "ER- PR- HER2-",
                "traitement": "Chimiothérapie",
                "pronostic" : "Défavorable — récidive fréquente",
                "f1_modele" : 0.82
            },
            "HER2-enriched": {
                "biologie"  : "ER- PR- HER2+",
                "traitement": "Trastuzumab/Herceptin",
                "pronostic" : "Intermédiaire — amélioré avec Herceptin",
                "f1_modele" : 0.83
            },
            "Luminal B / HER2+": {
                "biologie"  : "ER+ HER2+ ou Ki67 élevé",
                "traitement": "Hormonothérapie + anti-HER2",
                "pronostic" : "Intermédiaire",
                "f1_modele" : 0.50
            }
        }
    }
