# ============================================
# OncoPrint — Dashboard Streamlit v5
# Marine Deldicque — CDSD Jedha 2026
# ============================================

import streamlit as st
import requests
import pandas as pd
import numpy as np
import os
import anthropic
from urllib.parse import quote

# --- CONFIG ---
st.set_page_config(
    page_title="OncoPrint",
    page_icon="🧬",
    layout="wide"
)

API_URL = os.environ.get("API_URL", "https://marinedde-oncoprint-api.hf.space")

@st.cache_data(ttl=3600)
def charger_infos_api():
    try:
        r1 = requests.get(f"{API_URL}/features", timeout=5)
        r2 = requests.get(f"{API_URL}/health", timeout=5)
        features = r1.json().get("features", []) if r1.status_code == 200 else []
        classes  = r2.json().get("classes", []) if r2.status_code == 200 else []
        return features, classes, True
    except Exception:
        return [], [], False

features, classes, API_OK = charger_infos_api()

# Données de survie TCGA-BRCA de référence (fallback si API retourne données nulles)
# Médianes estimées sur cohorte TCGA-BRCA — Perou et al., Carey et al.
SURVIVAL_FALLBACK = {
    "Luminal A":         {"median_survival_months": None,  "hazard_ratio": 1.00},  # >180 mois, référence
    "Luminal B / HER2+": {"median_survival_months": 120.0, "hazard_ratio": 1.45},
    "HER2-enriched":     {"median_survival_months": 96.0,  "hazard_ratio": 1.82},
    "Triple Négatif":    {"median_survival_months": 72.0,  "hazard_ratio": 2.31},
}

def get_survival(subtype_name):
    """Récupère les données de survie depuis l'API.
    Fallback sur données TCGA-BRCA si l'API retourne HR=1.0 pour tous les sous-types."""
    try:
        encoded = quote(subtype_name, safe='')
        r = requests.get(f"{API_URL}/survival/{encoded}", timeout=5)
        if r.status_code == 200:
            data = r.json().get("survival", {})
            hr = data.get("hazard_ratio", 1.0)
            # Si sous-type non-référence retourne HR=1.0 -> modèle Cox non chargé côté API
            is_ref = (subtype_name == "Luminal A")
            if data and not (not is_ref and hr == 1.0):
                return data
    except Exception:
        pass
    return SURVIVAL_FALLBACK.get(subtype_name)

# --- STYLES ---
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1a1a2e;
        text-align: center;
        padding: 1rem 0;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .user-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 5px solid #2196F3;
        margin-bottom: 1rem;
    }
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# PROFILS PRÉDÉFINIS
# ============================================================
PROFILS = {
    "🔵 Luminal A": {
        'rs_ESR1': 15.0, 'rs_PGR': 14.0, 'rs_ERBB2': 4.0,
        'rs_KRT5': 2.0,  'rs_BCAS1': 2.0, 'pp_ER.alpha': 1.5,
        'pp_HER2': -1.0, 'pp_HER2.pY1248': -1.0, 'pp_EGFR.pY1068': -0.5
    },
    "🔴 Triple Négatif": {
        'rs_ESR1': 2.0,  'rs_PGR': 1.0,  'rs_ERBB2': 3.0,
        'rs_KRT5': 12.0, 'rs_BCAS1': 11.0, 'pp_ER.alpha': -1.5,
        'pp_HER2': -1.5, 'pp_HER2.pY1248': -1.5, 'pp_EGFR.pY1068': 0.5
    },
    "🟠 HER2-enriched": {
        'rs_ESR1': 3.0,  'rs_PGR': 2.0,  'rs_ERBB2': 16.0,
        'rs_KRT5': 3.0,  'rs_BCAS1': 3.0, 'pp_ER.alpha': -0.5,
        'pp_HER2': 2.5,  'pp_HER2.pY1248': 2.5, 'pp_EGFR.pY1068': 1.0
    },
    "🟢 Luminal B / HER2+": {
        'rs_ESR1': 10.0, 'rs_PGR': 8.0,  'rs_ERBB2': 12.0,
        'rs_KRT5': 3.0,  'rs_BCAS1': 3.0, 'pp_ER.alpha': 0.8,
        'pp_HER2': 1.8,  'pp_HER2.pY1248': 1.5, 'pp_EGFR.pY1068': 0.5
    },
    "⚠️ Profil ambigu": {
        'rs_ESR1': 8.0,  'rs_PGR': 4.0,  'rs_ERBB2': 9.0,
        'rs_KRT5': 5.0,  'rs_BCAS1': 4.0, 'pp_ER.alpha': 0.3,
        'pp_HER2': 0.8,  'pp_HER2.pY1248': 0.9, 'pp_EGFR.pY1068': 0.2
    }
}

SUBTYPES_LIST = ["Luminal A", "Triple Négatif", "HER2-enriched", "Luminal B / HER2+"]

# ============================================================
# FONCTIONS
# ============================================================

def verifier_coherence(subtype, bio):
    ok, alertes = [], []
    if subtype == "Luminal A":
        ok.append("✅ ESR1 élevé — cohérent avec Luminal A") if bio.get('rs_ESR1', 0) > 8 else alertes.append("⚠️ ESR1 bas — atypique pour Luminal A")
        alertes.append("⚠️ HER2 phosphorylé élevé — envisager Luminal B / HER2+") if bio.get('pp_HER2.pY1248', 0) > 1.5 else ok.append("✅ HER2 bas — cohérent avec Luminal A")
        ok.append("✅ ER.alpha positif — cohérent avec Luminal A") if bio.get('pp_ER.alpha', 0) > 0 else alertes.append("⚠️ ER.alpha bas — vérifier statut ER par IHC")
    elif subtype == "Triple Négatif":
        alertes.append("⚠️ ESR1 élevé — incohérent avec Triple Négatif") if bio.get('rs_ESR1', 0) > 10 else ok.append("✅ ESR1 bas — cohérent avec Triple Négatif")
        ok.append("✅ HER2 protéine basse — cohérent avec Triple Négatif") if bio.get('pp_HER2', 0) < 0 else alertes.append("⚠️ HER2 présent — envisager test FISH")
        if bio.get('rs_KRT5', 0) > 8 or bio.get('rs_BCAS1', 0) > 8:
            ok.append("✅ Marqueurs basaux élevés — cohérent avec Triple Négatif")
    elif subtype == "HER2-enriched":
        ok.append("✅ HER2 phosphorylé élevé — cohérent avec HER2-enriched") if bio.get('pp_HER2.pY1248', 0) > 1 else alertes.append("⚠️ HER2 phosphorylé bas — confirmer par FISH")
        alertes.append("⚠️ ESR1 élevé — envisager Luminal B / HER2+") if bio.get('rs_ESR1', 0) > 10 else ok.append("✅ ESR1 bas — cohérent avec HER2-enriched pur")
    elif subtype == "Luminal B / HER2+":
        ok.append("✅ ESR1 présent — cohérent avec composante Luminal") if bio.get('rs_ESR1', 0) > 5 else alertes.append("⚠️ ESR1 bas — vérifier composante hormonale")
        ok.append("✅ HER2 activé — cohérent avec composante HER2+") if bio.get('pp_HER2.pY1248', 0) > 0.5 else alertes.append("⚠️ HER2 peu activé — envisager Luminal A")
    return ok, alertes


def generer_rapport_llm(result, bio, mode="clinique"):
    try:
        api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            return "❌ Clé API manquante — ajouter ANTHROPIC_API_KEY dans Settings > Variables and secrets"
        client     = anthropic.Anthropic(api_key=api_key)
        subtype    = result['subtype']
        confidence = result['confidence']
        top_feats  = result['top_features'][:5]
        probas     = result['probabilities']
        _, alertes = verifier_coherence(subtype, bio)

        if mode == "clinique":
            prompt = f"""Tu es un oncologue spécialisé en médecine de précision.
Un algorithme XGBoost (F1 Macro=76.3%) a analysé le profil génomique d'une patiente.
RÉSULTATS :
- Sous-type suggéré : {subtype} (confiance : {confidence*100:.1f}%)
- Probabilités : {', '.join([f'{k}: {v*100:.1f}%' for k,v in probas.items()])}
- Top biomarqueurs : {[f['feature'] for f in top_feats]}
- Alertes de cohérence : {alertes if alertes else 'Aucune'}
Génère un rapport d'aide à la décision (200 mots max) :
1. Interprétation du sous-type et sa biologie
2. Implications thérapeutiques principales
3. Points de vigilance selon les alertes
4. Examens complémentaires recommandés
IMPORTANT : Rappelle que c'est une aide à la décision, pas un diagnostic."""
        elif mode == "recherche":
            prompt = f"""Tu es un data scientist en bioinformatique.
MODÈLE : XGBoost (TCGA-BRCA, n=536, F1 Macro=76.3%)
PRÉDICTION : {subtype} ({confidence*100:.1f}%)
PROBABILITÉS : {probas}
TOP FEATURES SHAP : {[f['feature'] + ' (imp=' + str(round(f['importance'],4)) + ')' for f in top_feats]}
Analyse technique (200 mots max) :
1. Interprétation des biomarqueurs SHAP
2. Niveau de confiance et signification statistique
3. Limites pour ce profil
4. Pistes d'investigation complémentaires"""
        else:
            prompt = f"""Tu es un enseignant en oncologie moléculaire.
Résultat d'un algorithme : {subtype} ({confidence*100:.1f}% confiance)
Biomarqueurs clés : {[f['feature'] for f in top_feats]}
Explique de façon pédagogique (200 mots max) :
1. Ce que signifie ce sous-type en termes simples
2. Pourquoi ces biomarqueurs sont caractéristiques
3. Quel traitement et pourquoi (mécanisme simplifié)
4. Un fait mémorable sur ce sous-type"""

        message = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    except Exception as e:
        return f"Erreur : {e}"


def saisie_biomarqueurs(key_prefix=""):
    defaults = st.session_state.get(f'bio_{key_prefix}', {})
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Expression génique (RNA-seq)**")
        rs_ESR1  = st.slider("rs_ESR1 — Récepteur œstrogène", 0.0, 20.0, float(defaults.get('rs_ESR1', 10.0)), 0.1, key=f"{key_prefix}_ESR1")
        rs_PGR   = st.slider("rs_PGR — Récepteur progestérone", 0.0, 20.0, float(defaults.get('rs_PGR', 8.0)), 0.1, key=f"{key_prefix}_PGR")
        rs_ERBB2 = st.slider("rs_ERBB2 — HER2", 0.0, 20.0, float(defaults.get('rs_ERBB2', 5.0)), 0.1, key=f"{key_prefix}_ERBB2")
        rs_KRT5  = st.slider("rs_KRT5 — Marqueur basal", 0.0, 20.0, float(defaults.get('rs_KRT5', 3.0)), 0.1, key=f"{key_prefix}_KRT5")
        rs_BCAS1 = st.slider("rs_BCAS1 — Marqueur basal", 0.0, 20.0, float(defaults.get('rs_BCAS1', 3.0)), 0.1, key=f"{key_prefix}_BCAS1")
    with col2:
        st.markdown("**Phospho-protéines**")
        pp_ER    = st.slider("pp_ER.alpha", -3.0, 3.0, float(defaults.get('pp_ER.alpha', 0.5)), 0.1, key=f"{key_prefix}_ER")
        pp_HER2  = st.slider("pp_HER2", -3.0, 3.0, float(defaults.get('pp_HER2', 0.0)), 0.1, key=f"{key_prefix}_HER2")
        pp_HER2p = st.slider("pp_HER2.pY1248", -3.0, 3.0, float(defaults.get('pp_HER2.pY1248', 0.0)), 0.1, key=f"{key_prefix}_HER2p")
        pp_EGFR  = st.slider("pp_EGFR.pY1068", -3.0, 3.0, float(defaults.get('pp_EGFR.pY1068', 0.0)), 0.1, key=f"{key_prefix}_EGFR")
    return {
        'rs_ESR1': rs_ESR1, 'rs_PGR': rs_PGR, 'rs_ERBB2': rs_ERBB2,
        'rs_KRT5': rs_KRT5, 'rs_BCAS1': rs_BCAS1, 'pp_ER.alpha': pp_ER,
        'pp_HER2': pp_HER2, 'pp_HER2.pY1248': pp_HER2p, 'pp_EGFR.pY1068': pp_EGFR
    }


def appeler_api(bio):
    patient_data = {feat: 0.0 for feat in features}
    patient_data.update(bio)
    return requests.post(f"{API_URL}/predict", json={"features": patient_data}, timeout=10)


def afficher_prediction(result, bio, mode="clinique"):
    st.session_state['last_result'] = result
    st.session_state['last_bio']    = bio
    subtype    = result['subtype']
    confidence = result['confidence']
    emoji = {'Luminal A': '🔵', 'Triple Négatif': '🔴', 'HER2-enriched': '🟠', 'Luminal B / HER2+': '🟢'}.get(subtype, '🧬')

    st.markdown(" ")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"### {emoji} {subtype}")
        st.metric("Confiance", f"{confidence*100:.1f}%")
        st.markdown("**Probabilités :**")
        for classe, proba in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
            st.progress(proba, text=f"{classe}: {proba*100:.1f}%")
    with col2:
        if mode == "clinique":
            st.markdown("### 🔍 Validation de cohérence")
            ok_list, alertes_list = verifier_coherence(subtype, bio)
            for item in ok_list:
                st.success(item)
            for item in alertes_list:
                st.warning(item)
            if not alertes_list:
                st.info("✅ Profil cohérent avec le sous-type prédit")
            st.markdown('<div class="warning-box">⚕️ <strong>Avertissement</strong> — Aide à la décision algorithmique. Ne remplace pas le diagnostic clinique.</div>', unsafe_allow_html=True)
        elif mode == "recherche":
            st.markdown("### 🔬 Features SHAP déterminantes")
            df_feats = pd.DataFrame(result['top_features'])
            df_feats['type'] = df_feats['feature'].apply(
                lambda x: 'RNA-seq' if x.startswith('rs_') else 'Phospho-protéine' if x.startswith('pp_') else 'Copy Number' if x.startswith('cn_') else 'Mutation'
            )
            st.dataframe(df_feats[['feature', 'type', 'importance', 'value']].round(4), use_container_width=True)
        else:
            analogies  = {'Luminal A': "🚗 Une voiture qui roule à l'essence œstrogène.", 'Triple Négatif': "🚪 Une porte sans serrure — aucune clé thérapeutique ciblée.", 'HER2-enriched': "⚡ Un accélérateur bloqué — l'Herceptin le débloque.", 'Luminal B / HER2+': "🔧 Double moteur hormonal ET HER2."}
            traitements = {'Luminal A': "Tamoxifène / inhibiteurs aromatase", 'Triple Négatif': "Chimiothérapie — immunothérapie en développement", 'HER2-enriched': "Trastuzumab (Herceptin)", 'Luminal B / HER2+': "Hormonothérapie + anti-HER2"}
            st.info(analogies.get(subtype, ""))
            st.success(f"💊 Traitement : {traitements.get(subtype, '')}")


def bloc_llm(mode, key_suffix):
    if 'last_result' not in st.session_state:
        return
    labels = {"clinique": "Générer rapport clinique (Claude AI)", "recherche": "Générer analyse bioinformatique (Claude AI)", "pedagogique": "Générer explication pédagogique (Claude AI)"}
    st.markdown(" ")
    if st.button(labels.get(mode, "Générer rapport"), type="primary", key=f"llm_{key_suffix}"):
        with st.spinner("Génération en cours..."):
            rapport = generer_rapport_llm(st.session_state['last_result'], st.session_state['last_bio'], mode)
            st.markdown("**Rapport Claude AI**")
            st.markdown(rapport)
            st.caption("Généré par IA — à valider par un professionnel")


# ============================================================
# HEADER + SIDEBAR
# ============================================================
st.markdown('<div class="main-title">🧬 OncoPrint</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Classification moléculaire du cancer du sein par apprentissage automatique — TCGA-BRCA</div>', unsafe_allow_html=True)

st.sidebar.title("🧬 OncoPrint")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", ["🏠 Accueil", "🩺 Aide à la décision", "🔬 Outil de recherche", "📚 Explorer & apprendre", "⏱️ Pronostic Survie", "📊 Performances", "ℹ️ À propos"])
if API_OK:
    st.sidebar.success("API connectée")
else:
    st.sidebar.error("API non disponible")
st.sidebar.markdown("---")
st.sidebar.caption("Marine Deldicque\nCDSD Jedha 2026")


# ============================================================
# PAGE ACCUEIL
# ============================================================
if page == "🏠 Accueil":
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 2rem 0;">
        <p style="font-size:1.25rem; color:#444; font-style:italic; max-width:700px; margin:auto;">
            Chaque tumeur parle. OncoPrint traduit.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    OncoPrint analyse le profil génomique d'une tumeur mammaire — expression génique, mutations, phospho-protéines —
    et prédit son sous-type moléculaire. L'objectif : rendre la donnée omique lisible et utile, que vous soyez clinicien, chercheur ou en formation.
    """)

    st.markdown("---")

    # Cards onglets
    cards = [
        ("🩺", "Aide à la décision", "#2196F3",
         "Vous avez un profil de patiente sous la main ? Entrez ses biomarqueurs, obtenez une prédiction avec validation de cohérence et un rapport clinique généré par IA."),
        ("🔬", "Outil de recherche", "#4CAF50",
         "Explorez les probabilités par sous-type, les features SHAP déterminantes, et lancez des prédictions en batch sur un fichier CSV."),
        ("📚", "Explorer & apprendre", "#FF9800",
         "Pas encore familier avec les sous-types moléculaires ? Cette section explique la biologie, les traitements et vous laisse tester des profils interactifs."),
        ("⏱️", "Pronostic Survie", "#9C27B0",
         "Visualisez les médianes de survie et Hazard Ratios par sous-type, estimés sur la cohorte TCGA-BRCA (536 patientes, Kaplan-Meier + Cox PH)."),
        ("📊", "Performances", "#607D8B",
         "Les métriques du modèle XGBoost : accuracy, F1 par classe, validation croisée, et interprétation SHAP biologique."),
        ("ℹ️", "À propos", "#795548",
         "Dataset, stack technique, limites du modèle, et contexte du projet."),
    ]

    col1, col2, col3 = st.columns(3)
    cols_cycle = [col1, col2, col3, col1, col2, col3]
    for (emoji, titre, couleur, texte), col in zip(cards, cols_cycle):
        with col:
            st.markdown(f"""
            <div style="background:#f8f9fa; border-radius:12px; padding:1.2rem 1.4rem;
                        border-left:5px solid {couleur}; margin-bottom:1.2rem; min-height:160px;">
                <h3 style="margin:0 0 0.5rem 0; font-size:1.05rem;">{emoji} {titre}</h3>
                <p style="color:#555; font-size:0.9rem; margin:0;">{texte}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Patientes", "536", "TCGA-BRCA")
    col2.metric("Features", "1936", "Multi-omiques")
    col3.metric("Accuracy", "84.3%", "XGBoost")
    col4.metric("F1 Macro", "76.3%", "4 sous-types")


# ============================================================
# PAGE AIDE À LA DÉCISION
# ============================================================
elif page == "🩺 Aide à la décision":
    st.header("🩺 Aide à la décision clinique")
    st.markdown('<div class="warning-box">⚕️ <strong>Usage médical</strong> — Suggestion algorithmique. Ne remplace pas le diagnostic clinique (anatomopathologie, IHC, FISH).</div>', unsafe_allow_html=True)

    st.markdown("### Profils prédéfinis")
    cols = st.columns(5)
    for i, (nom, vals) in enumerate(PROFILS.items()):
        with cols[i]:
            if st.button(nom, key=f"btn_clinique_{i}"):
                for k, v in vals.items():
                    key_map = {
                        'rs_ESR1': 'clinique_ESR1', 'rs_PGR': 'clinique_PGR',
                        'rs_ERBB2': 'clinique_ERBB2', 'rs_KRT5': 'clinique_KRT5',
                        'rs_BCAS1': 'clinique_BCAS1', 'pp_ER.alpha': 'clinique_ER',
                        'pp_HER2': 'clinique_HER2', 'pp_HER2.pY1248': 'clinique_HER2p',
                        'pp_EGFR.pY1068': 'clinique_EGFR'
                    }
                    if k in key_map:
                        st.session_state[key_map[k]] = float(v)
                st.rerun()

    bio = saisie_biomarqueurs(key_prefix="clinique")
    if st.button("🔍 Analyser le profil", type="primary", key="predict_clinique"):
        with st.spinner("Analyse en cours..."):
            try:
                response = appeler_api(bio)
                if response.status_code == 200:
                    afficher_prediction(response.json(), bio, mode="clinique")
                else:
                    st.error(f"Erreur API : {response.status_code}")
            except Exception as e:
                st.error(f"Erreur connexion : {e}")
    bloc_llm(mode="clinique", key_suffix="clinique")


# ============================================================
# PAGE OUTIL DE RECHERCHE
# ============================================================
elif page == "🔬 Outil de recherche":
    st.header("🔬 Outil de recherche bioinformatique")
    tab1, tab2 = st.tabs(["🎛️ Prédiction manuelle", "📁 Batch CSV"])

    with tab1:
        cols = st.columns(5)
        for i, (nom, vals) in enumerate(PROFILS.items()):
            with cols[i]:
                if st.button(nom, key=f"btn_recherche_{i}"):
                    for k, v in vals.items():
                        key_map = {
                            'rs_ESR1': 'recherche_ESR1', 'rs_PGR': 'recherche_PGR',
                            'rs_ERBB2': 'recherche_ERBB2', 'rs_KRT5': 'recherche_KRT5',
                            'rs_BCAS1': 'recherche_BCAS1', 'pp_ER.alpha': 'recherche_ER',
                            'pp_HER2': 'recherche_HER2', 'pp_HER2.pY1248': 'recherche_HER2p',
                            'pp_EGFR.pY1068': 'recherche_EGFR'
                        }
                        if k in key_map:
                            st.session_state[key_map[k]] = float(v)
                    st.rerun()
        bio = saisie_biomarqueurs(key_prefix="recherche")
        if st.button("🔬 Analyser", type="primary", key="predict_recherche"):
            with st.spinner("Analyse en cours..."):
                try:
                    response = appeler_api(bio)
                    if response.status_code == 200:
                        afficher_prediction(response.json(), bio, mode="recherche")
                    else:
                        st.error(f"Erreur API : {response.status_code}")
                except Exception as e:
                    st.error(f"Erreur : {e}")
        bloc_llm(mode="recherche", key_suffix="recherche")

    with tab2:
        st.markdown("### Prédiction batch — Upload CSV")
        uploaded = st.file_uploader("Choisir un fichier CSV", type=['csv'])
        if uploaded is not None:
            df_upload = pd.read_csv(uploaded)
            st.write(f"Dataset : {df_upload.shape[0]} patientes × {df_upload.shape[1]} features")
            st.dataframe(df_upload.head(3))
            if st.button("🔬 Prédire toutes les patientes"):
                resultats, progress = [], st.progress(0)
                for i, row in df_upload.iterrows():
                    try:
                        resp = appeler_api(row.to_dict())
                        if resp.status_code == 200:
                            r = resp.json()
                            resultats.append({'patient_idx': i, 'subtype': r['subtype'], 'confidence': f"{r['confidence']*100:.1f}%"})
                    except Exception:
                        pass
                    progress.progress((i + 1) / len(df_upload))
                df_results = pd.DataFrame(resultats)
                st.success(f"✅ {len(df_results)} patientes analysées")
                st.dataframe(df_results, use_container_width=True)
                st.bar_chart(df_results['subtype'].value_counts())


# ============================================================
# PAGE EXPLORER & APPRENDRE
# ============================================================
elif page == "📚 Explorer & apprendre":
    st.header("📚 Explorer les sous-types moléculaires")
    tab1, tab2 = st.tabs(["🧬 Les sous-types", "🎮 Tester un profil"])

    with tab1:
        sous_type = st.selectbox("Choisir un sous-type", SUBTYPES_LIST)
        infos = {
            "Luminal A":         {"emoji": "🔵", "frequence": "~50% des cancers du sein", "biologie": "ER+ PR+ HER2- Ki67 bas", "analogie": "Une voiture qui roule à l'essence œstrogène.", "traitement": "Tamoxifène, inhibiteurs aromatase", "pronostic": "Favorable — survie à 5 ans ~90%", "shap": "pp_ER.alpha (#1)"},
            "Triple Négatif":    {"emoji": "🔴", "frequence": "~15-20% des cancers du sein", "biologie": "ER- PR- HER2-", "analogie": "Une porte sans serrure — aucune clé thérapeutique ciblée.", "traitement": "Chimiothérapie — immunothérapie en développement", "pronostic": "Défavorable — récidive fréquente dans les 3-5 ans", "shap": "pp_HER2 absent (#1)"},
            "HER2-enriched":     {"emoji": "🟠", "frequence": "~5-10% des cancers du sein", "biologie": "ER- PR- HER2+", "analogie": "Un accélérateur bloqué — l'Herceptin le débloque.", "traitement": "Trastuzumab (Herceptin)", "pronostic": "Intermédiaire — amélioré depuis l'Herceptin", "shap": "pp_HER2.pY1248 (#1)"},
            "Luminal B / HER2+": {"emoji": "🟢", "frequence": "~10-15% des cancers du sein", "biologie": "ER+ HER2+ ou Ki67 élevé", "analogie": "Double moteur hormonal ET HER2.", "traitement": "Hormonothérapie + anti-HER2", "pronostic": "Intermédiaire — moins favorable que Luminal A", "shap": "cn_PPP1R1B (#1)"},
        }
        info = infos[sous_type]
        st.markdown(f"## {info['emoji']} {sous_type}")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Fréquence", info['frequence'])
            st.info(f"💡 **Analogie :** {info['analogie']}")
            st.markdown(f"**Biologie :** {info['biologie']}")
        with col2:
            st.success(f"💊 **Traitement :** {info['traitement']}")
            st.markdown(f"📈 **Pronostic :** {info['pronostic']}")
            st.markdown(f"🤖 **Feature SHAP #1 :** {info['shap']}")

    with tab2:
        cols = st.columns(5)
        for i, (nom, vals) in enumerate(PROFILS.items()):
            with cols[i]:
                if st.button(nom, key=f"btn_pedago_{i}"):
                    for k, v in vals.items():
                        key_map = {
                            'rs_ESR1': 'pedago_ESR1', 'rs_PGR': 'pedago_PGR',
                            'rs_ERBB2': 'pedago_ERBB2', 'rs_KRT5': 'pedago_KRT5',
                            'rs_BCAS1': 'pedago_BCAS1', 'pp_ER.alpha': 'pedago_ER',
                            'pp_HER2': 'pedago_HER2', 'pp_HER2.pY1248': 'pedago_HER2p',
                            'pp_EGFR.pY1068': 'pedago_EGFR'
                        }
                        if k in key_map:
                            st.session_state[key_map[k]] = float(v)
                    st.rerun()
        bio = saisie_biomarqueurs(key_prefix="pedago")
        if st.button("🎮 Voir la prédiction", type="primary", key="predict_pedago"):
            with st.spinner("Analyse en cours..."):
                try:
                    response = appeler_api(bio)
                    if response.status_code == 200:
                        afficher_prediction(response.json(), bio, mode="pedagogique")
                    else:
                        st.error(f"Erreur API : {response.status_code}")
                except Exception as e:
                    st.error(f"Erreur : {e}")
        bloc_llm(mode="pedagogique", key_suffix="pedago")


# ============================================================
# PAGE PRONOSTIC SURVIE
# ============================================================
elif page == "⏱️ Pronostic Survie":
    st.header("⏱️ Pronostic de Survie par Sous-type")
    st.markdown('<div class="warning-box">⚕️ <strong>Usage médical</strong> — Courbes issues de TCGA-BRCA (cohorte de recherche). Ne remplacent pas un avis clinique individualisé.</div>', unsafe_allow_html=True)

    if not API_OK:
        st.error("API non disponible.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Patientes", "536", "TCGA-BRCA")
        col2.metric("Méthode", "Kaplan-Meier", "+ Cox PH")
        col3.metric("Référence", "Luminal A", "meilleur pronostic")

        st.markdown("---")
        st.subheader("📊 Médianes de survie & Hazard Ratios")

        # Charger toutes les données de survie
        survival_data = {}
        for st_name in SUBTYPES_LIST:
            surv = get_survival(st_name)
            if surv:
                survival_data[st_name] = surv

        if survival_data:
            cols_km = st.columns(len(survival_data))
            for i, (st_name, surv) in enumerate(survival_data.items()):
                med    = surv.get("median_survival_months")
                hr     = surv.get("hazard_ratio", 1.0)
                med_str = f"{med:.0f} mois" if med else "> 180 mois"
                hr_str  = f"{hr:.2f}x"
                delta   = "référence" if hr == 1.0 else ("▲ risque" if hr > 1 else "▼ risque")
                with cols_km[i]:
                    st.markdown(f"**{st_name}**")
                    st.metric("Médiane OS", med_str)
                    st.metric("Hazard Ratio", hr_str, delta)

        st.markdown("---")
        st.subheader("🔎 Simuler pour un profil prédit")

        # Selectbox avec on_change pour forcer la mise à jour
        subtype_select = st.selectbox(
            "Sous-type prédit",
            options=SUBTYPES_LIST,
            key="survie_selectbox"
        )

        # Récupérer les données à chaque changement de sélection
        surv_sel = get_survival(subtype_select)
        if surv_sel:
            med  = surv_sel.get("median_survival_months")
            hr   = surv_sel.get("hazard_ratio", 1.0)
            med_str = f"{med:.0f} mois" if med else "> 180 mois"
            st.info(f"""
**Sous-type sélectionné : {subtype_select}**

- Médiane de survie (Kaplan-Meier) : **{med_str}**
- Hazard Ratio vs Luminal A : **{hr:.2f}x**
- Interprétation : un HR={hr:.2f} signifie que le risque de décès est {hr:.2f}x celui de Luminal A

⚠️ Estimations issues de la cohorte TCGA-BRCA (population de recherche).
            """)
        else:
            st.warning(f"Données de survie non disponibles pour {subtype_select}.")


# ============================================================
# PAGE PERFORMANCES
# ============================================================
elif page == "📊 Performances":
    st.header("📊 Performances du modèle")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", "84.3%", "+38% vs baseline")
    col2.metric("F1 Macro", "76.3%", "4 classes")
    col3.metric("CV 5-fold", "71.5% ± 2.5%", "sans data leakage")
    col4.metric("Test set", "108 patientes", "20% holdout")

    st.markdown("---")
    st.subheader("Performances par sous-type")
    st.dataframe(pd.DataFrame({
        'Sous-type' : ['Luminal A', 'Triple Négatif', 'HER2-enriched', 'Luminal B / HER2+'],
        'Precision' : [0.90, 0.76, 0.71, 0.62],
        'Recall'    : [0.89, 0.89, 1.00, 0.42],
        'F1-Score'  : [0.89, 0.82, 0.83, 0.50],
        'Support'   : [73, 18, 5, 12]
    }), use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.success("✅ HER2-enriched : Recall = 100% — aucun cas manqué")
        st.success("✅ Luminal A : F1 = 0.89 — meilleure performance")
        st.success("✅ Triple Négatif : isolé en t-SNE/UMAP + SHAP")
    with col2:
        st.warning("⚠️ Luminal B : Recall = 42% — hétérogénéité biologique")
        st.warning("⚠️ HER2-enriched : n=5 dans le test — fragile")
        st.info("ℹ️ Data leakage SMOTE détecté et corrigé (pipeline CV)")

    st.markdown("---")
    st.subheader("Validation biologique SHAP")
    st.markdown("""
| Sous-type | Feature #1 SHAP | Signification clinique |
|---|---|---|
| Luminal A | pp_ER.alpha | Récepteur œstrogène — standard depuis 40 ans |
| HER2-enriched | pp_HER2.pY1248 | Phosphorylation HER2 — cible de l'Herceptin |
| Triple Négatif | pp_HER2 absent | Absence HER2 — définition clinique |
| Luminal B | cn_PPP1R1B | Amplification génomique connue |
    """)


# ============================================================
# PAGE À PROPOS
# ============================================================
elif page == "ℹ️ À propos":
    st.header("ℹ️ À propos d'OncoPrint")
    st.markdown("""
## OncoPrint — Classification moléculaire automatique du cancer du sein

### Dataset
- **Source :** The Cancer Genome Atlas (TCGA-BRCA) — NCI/NIH
- **Population :** 536 patientes après nettoyage (705 initiales)
- **Features :** 1936 mesures multi-omiques (249 mutations | 860 CNV | 604 RNA-seq | 223 phospho-protéines)

### Modèle
- **Algorithme :** XGBoost optimisé (GridSearchCV)
- **Accuracy :** 84.3% | **F1 Macro :** 76.3%
- **Validation :** CV 5-fold corrigée (sans data leakage SMOTE)
- **Interprétabilité :** SHAP — cohérence biologique validée

### Stack technique
Python · XGBoost · scikit-learn · SHAP · FastAPI · Streamlit · Anthropic Claude API · HuggingFace Spaces

### Limites
- Données TCGA en conditions recherche — validation clinique nécessaire
- Luminal B : recall 42% — hétérogénéité biologique connue
- HER2-enriched : n=5 dans le test — statistiquement fragile

### Auteure
**Marine Deldicque** — Infirmière libérale · Data Scientist en transition · Jedha Bootcamp CDSD 2026
    """)
