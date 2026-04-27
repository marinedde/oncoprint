# ============================================
# OncoPrint — API FastAPI
# Marine Deldicque — CDSD Jedha 2026
# ============================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import joblib
import numpy as np
import pandas as pd
import os

# --- INITIALISATION ---
app = FastAPI(
    title="OncoPrint API",
    description="Classification moléculaire du cancer du sein — TCGA-BRCA",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# --- CHARGEMENT DES MODÈLES ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

try:
    model    = joblib.load(os.path.join(MODELS_DIR, "oncoprint_xgb.joblib"))
    scaler   = joblib.load(os.path.join(MODELS_DIR, "oncoprint_scaler.joblib"))
    le       = joblib.load(os.path.join(MODELS_DIR, "oncoprint_le.joblib"))
    features = joblib.load(os.path.join(MODELS_DIR, "oncoprint_features.joblib"))
    info     = joblib.load(os.path.join(MODELS_DIR, "oncoprint_info.joblib"))
    print("Modèles chargés")
except Exception as e:
    print(f"Erreur chargement modèles : {e}")

# Colonnes RNA-seq et phospho-protéines pour le scaling
rs_cols = [c for c in features if c.startswith('rs_')]
pp_cols = [c for c in features if c.startswith('pp_')]
rs_pp_cols = rs_cols + pp_cols

# --- SCHÉMAS ---
class PatientData(BaseModel):
    features: Dict[str, float]

class PredictionResponse(BaseModel):
    subtype: str
    confidence: float
    probabilities: Dict[str, float]
    top_features: List[Dict]
    model_info: Dict

# --- ENDPOINTS ---
@app.get("/")
def root():
    return {
        "projet"  : "OncoPrint",
        "version" : "1.0.0",
        "auteur"  : "Marine Deldicque",
        "status"  : "running"
    }

@app.get("/health")
def health():
    return {
        "status"   : "healthy",
        "modele"   : info.get("modele", "XGBoost"),
        "accuracy" : info.get("accuracy_test", 0.843),
        "f1_macro" : info.get("f1_macro_test", 0.763),
        "classes"  : list(le.classes_)
    }

@app.get("/features")
def get_features():
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
    try:
        # Construire le vecteur de features
        X = pd.DataFrame([data.features])
        
        # Ajouter les features manquantes avec 0
        for feat in features:
            if feat not in X.columns:
                X[feat] = 0.0
        
        # Réordonner selon l'ordre d'entraînement
        X = X[features]
        
        # Appliquer le scaling sur rs_ et pp_
        X_scaled = X.copy()
        cols_to_scale = [c for c in rs_pp_cols if c in X.columns]
        X_scaled[cols_to_scale] = scaler.transform(X[cols_to_scale])
        
        # Prédiction
        prediction = model.predict(X_scaled)[0]
        probas     = model.predict_proba(X_scaled)[0]
        
        subtype    = le.classes_[prediction]
        confidence = float(probas[prediction])
        
        # Probabilités par classe
        proba_dict = {
            le.classes_[i]: float(probas[i])
            for i in range(len(le.classes_))
        }
        
        # Top features importantes pour ce modèle
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
        
        return PredictionResponse(
            subtype       = subtype,
            confidence    = confidence,
            probabilities = proba_dict,
            top_features  = top_feats,
            model_info    = {
                "modele"   : "XGBoost optimisé",
                "accuracy" : 0.843,
                "f1_macro" : 0.763
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info")
def model_info_endpoint():
    return info

@app.get("/subtypes")
def get_subtypes():
    return {
        "subtypes": {
            "Luminal A": {
                "biologie"   : "ER+ PR+ HER2-",
                "traitement" : "Hormonothérapie",
                "pronostic"  : "Favorable — survie 5 ans ~90%",
                "f1_modele"  : 0.89
            },
            "Triple Négatif": {
                "biologie"   : "ER- PR- HER2-",
                "traitement" : "Chimiothérapie",
                "pronostic"  : "Défavorable — récidive fréquente",
                "f1_modele"  : 0.82
            },
            "HER2-enriched": {
                "biologie"   : "ER- PR- HER2+",
                "traitement" : "Trastuzumab/Herceptin",
                "pronostic"  : "Intermédiaire — amélioré avec Herceptin",
                "f1_modele"  : 0.83
            },
            "Luminal B / HER2+": {
                "biologie"   : "ER+ HER2+ ou Ki67 élevé",
                "traitement" : "Hormonothérapie + anti-HER2",
                "pronostic"  : "Intermédiaire",
                "f1_modele"  : 0.50
            }
        }
    }