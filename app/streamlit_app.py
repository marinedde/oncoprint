# ============================================
# OncoPrint — Dashboard Streamlit v3
# Marine Deldicque — CDSD Jedha 2026
# ============================================

import streamlit as st
import requests
import pandas as pd
import numpy as np
import joblib
import os
import anthropic

# --- CONFIG ---
st.set_page_config(
    page_title="OncoPrint",
    page_icon="🧬",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
features = joblib.load(os.path.join(BASE_DIR, "models", "oncoprint_features.joblib"))
le       = joblib.load(os.path.join(BASE_DIR, "models", "oncoprint_le.joblib"))

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

# ============================================================
# FONCTIONS
# ============================================================

def verifier_coherence(subtype, bio):
    ok      = []
    alertes = []

    if subtype == "Luminal A":
        if bio.get('rs_ESR1', 0) > 8:
            ok.append("✅ ESR1 élevé — cohérent avec Luminal A")
        else:
            alertes.append("⚠️ ESR1 bas — atypique pour Luminal A, vérifier statut ER")
        if bio.get('pp_HER2.pY1248', 0) > 1.5:
            alertes.append("⚠️ HER2 phosphorylé élevé — envisager Luminal B / HER2+")
        else:
            ok.append("✅ HER2 bas — cohérent avec Luminal A")
        if bio.get('pp_ER.alpha', 0) > 0:
            ok.append("✅ ER.alpha positif — cohérent avec Luminal A")
        else:
            alertes.append("⚠️ ER.alpha bas — vérifier statut ER par IHC")

    elif subtype == "Triple Négatif":
        if bio.get('rs_ESR1', 0) > 10:
            alertes.append("⚠️ ESR1 élevé — incohérent avec Triple Négatif")
        else:
            ok.append("✅ ESR1 bas — cohérent avec Triple Négatif")
        if bio.get('pp_HER2', 0) < 0:
            ok.append("✅ HER2 protéine basse — cohérent avec Triple Négatif")
        else:
            alertes.append("⚠️ HER2 présent — envisager test FISH")
        if bio.get('rs_KRT5', 0) > 8 or bio.get('rs_BCAS1', 0) > 8:
            ok.append("✅ Marqueurs basaux élevés — cohérent avec Triple Négatif")

    elif subtype == "HER2-enriched":
        if bio.get('pp_HER2.pY1248', 0) > 1:
            ok.append("✅ HER2 phosphorylé élevé — cohérent avec HER2-enriched")
        else:
            alertes.append("⚠️ HER2 phosphorylé bas — confirmer par FISH")
        if bio.get('rs_ESR1', 0) > 10:
            alertes.append("⚠️ ESR1 élevé — envisager Luminal B / HER2+")
        else:
            ok.append("✅ ESR1 bas — cohérent avec HER2-enriched pur")

    elif subtype == "Luminal B / HER2+":
        if bio.get('rs_ESR1', 0) > 5:
            ok.append("✅ ESR1 présent — cohérent avec composante Luminal")
        else:
            alertes.append("⚠️ ESR1 bas — vérifier composante hormonale")
        if bio.get('pp_HER2.pY1248', 0) > 0.5:
            ok.append("✅ HER2 activé — cohérent avec composante HER2+")
        else:
            alertes.append("⚠️ HER2 peu activé — envisager Luminal A")

    return ok, alertes


def generer_rapport_llm(result, bio, mode="clinique"):
    try:
        api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            return "❌ Clé API manquante dans .streamlit/secrets.toml"

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
Analyse ces résultats de classification moléculaire :

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
    session_key = f'bio_{key_prefix}'
    defaults    = st.session_state.get(session_key, {})

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Expression génique (RNA-seq)**")
        rs_ESR1  = st.slider("rs_ESR1 — Récepteur œstrogène", 0.0, 20.0,
                             float(defaults.get('rs_ESR1', 10.0)), 0.1, key=f"{key_prefix}_ESR1")
        rs_PGR   = st.slider("rs_PGR — Récepteur progestérone", 0.0, 20.0,
                             float(defaults.get('rs_PGR', 8.0)), 0.1, key=f"{key_prefix}_PGR")
        rs_ERBB2 = st.slider("rs_ERBB2 — HER2", 0.0, 20.0,
                             float(defaults.get('rs_ERBB2', 5.0)), 0.1, key=f"{key_prefix}_ERBB2")
        rs_KRT5  = st.slider("rs_KRT5 — Marqueur basal", 0.0, 20.0,
                             float(defaults.get('rs_KRT5', 3.0)), 0.1, key=f"{key_prefix}_KRT5")
        rs_BCAS1 = st.slider("rs_BCAS1 — Marqueur basal", 0.0, 20.0,
                             float(defaults.get('rs_BCAS1', 3.0)), 0.1, key=f"{key_prefix}_BCAS1")
    with col2:
        st.markdown("**Phospho-protéines**")
        pp_ER    = st.slider("pp_ER.alpha", -3.0, 3.0,
                             float(defaults.get('pp_ER.alpha', 0.5)), 0.1, key=f"{key_prefix}_ER")
        pp_HER2  = st.slider("pp_HER2", -3.0, 3.0,
                             float(defaults.get('pp_HER2', 0.0)), 0.1, key=f"{key_prefix}_HER2")
        pp_HER2p = st.slider("pp_HER2.pY1248", -3.0, 3.0,
                             float(defaults.get('pp_HER2.pY1248', 0.0)), 0.1, key=f"{key_prefix}_HER2p")
        pp_EGFR  = st.slider("pp_EGFR.pY1068", -3.0, 3.0,
                             float(defaults.get('pp_EGFR.pY1068', 0.0)), 0.1, key=f"{key_prefix}_EGFR")

    return {
        'rs_ESR1'        : rs_ESR1,
        'rs_PGR'         : rs_PGR,
        'rs_ERBB2'       : rs_ERBB2,
        'rs_KRT5'        : rs_KRT5,
        'rs_BCAS1'       : rs_BCAS1,
        'pp_ER.alpha'    : pp_ER,
        'pp_HER2'        : pp_HER2,
        'pp_HER2.pY1248' : pp_HER2p,
        'pp_EGFR.pY1068' : pp_EGFR
    }


def appeler_api(bio):
    patient_data = {feat: 0.0 for feat in features}
    patient_data.update(bio)
    return requests.post(f"{API_URL}/predict", json={"features": patient_data})


def afficher_prediction(result, bio, mode="clinique"):
    st.session_state['last_result'] = result
    st.session_state['last_bio']    = bio
    st.session_state['last_mode']   = mode

    subtype    = result['subtype']
    confidence = result['confidence']

    colors = {
        'Luminal A'         : '🔵',
        'Triple Négatif'    : '🔴',
        'HER2-enriched'     : '🟠',
        'Luminal B / HER2+' : '🟢'
    }
    emoji = colors.get(subtype, '🧬')

    st.markdown("---")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(f"### {emoji} {subtype}")
        st.metric("Confiance", f"{confidence*100:.1f}%")
        st.markdown("**Probabilités :**")
        for classe, proba in sorted(
            result['probabilities'].items(),
            key=lambda x: x[1], reverse=True
        ):
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
            st.markdown("""
            <div class="warning-box">
            ⚕️ <strong>Avertissement</strong> — Aide à la décision algorithmique.
            Ne remplace pas le diagnostic clinique (IHC, FISH, anatomopathologie).
            </div>
            """, unsafe_allow_html=True)

        elif mode == "recherche":
            st.markdown("### 🔬 Features SHAP déterminantes")
            df_feats = pd.DataFrame(result['top_features'])
            df_feats['type'] = df_feats['feature'].apply(
                lambda x: 'RNA-seq' if x.startswith('rs_')
                else 'Phospho-protéine' if x.startswith('pp_')
                else 'Copy Number' if x.startswith('cn_')
                else 'Mutation'
            )
            st.dataframe(
                df_feats[['feature', 'type', 'importance', 'value']].round(4),
                use_container_width=True
            )

        else:
            st.markdown("### 📚 Comprendre ce sous-type")
            analogies = {
                'Luminal A'         : "🚗 Une voiture qui roule à l'essence œstrogène — couper l'approvisionnement suffit.",
                'Triple Négatif'    : "🚪 Une porte sans serrure — aucune clé thérapeutique ciblée disponible.",
                'HER2-enriched'     : "⚡ Un accélérateur bloqué — l'Herceptin est le mécanicien qui le débloque.",
                'Luminal B / HER2+' : "🔧 Double moteur hormonal ET HER2 — nécessite une double stratégie."
            }
            traitements = {
                'Luminal A'         : "Tamoxifène / inhibiteurs aromatase",
                'Triple Négatif'    : "Chimiothérapie — immunothérapie en développement",
                'HER2-enriched'     : "Trastuzumab (Herceptin) — révolution des années 2000",
                'Luminal B / HER2+' : "Hormonothérapie + anti-HER2 combinés"
            }
            st.info(analogies.get(subtype, ""))
            st.success(f"💊 Traitement : {traitements.get(subtype, '')}")


def bloc_llm(mode, key_suffix):
    """Bloc bouton LLM persistant."""
    if 'last_result' not in st.session_state:
        return

    labels = {
        "clinique"   : "📋 Générer rapport clinique (Claude AI)",
        "recherche"  : "🔬 Générer analyse bioinformatique (Claude AI)",
        "pedagogique": "📚 Générer explication pédagogique (Claude AI)"
    }

    st.markdown("---")
    if st.button(labels.get(mode, "🤖 Générer rapport"), type="primary", key=f"llm_{key_suffix}"):
        with st.spinner("Claude AI génère le rapport..."):
            rapport = generer_rapport_llm(
                st.session_state['last_result'],
                st.session_state['last_bio'],
                mode
            )
            st.markdown("### 📄 Rapport Claude AI")
            st.markdown(rapport)
            st.caption("⚠️ Généré par IA — à valider par un professionnel")


# ============================================================
# HEADER
# ============================================================
st.markdown('<div class="main-title">🧬 OncoPrint</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Classification moléculaire du cancer du sein '
    'par apprentissage automatique — TCGA-BRCA</div>',
    unsafe_allow_html=True
)

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("🧬 OncoPrint")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigation", [
    "🏠 Accueil",
    "🩺 Aide à la décision",
    "🔬 Outil de recherche",
    "📚 Explorer & apprendre",
    "📊 Performances",
    "ℹ️ À propos"
])

try:
    r = requests.get(f"{API_URL}/health", timeout=2)
    if r.status_code == 200:
        st.sidebar.success("✅ API connectée")
    else:
        st.sidebar.error("❌ Erreur API")
except Exception:
    st.sidebar.error("❌ API non disponible")

st.sidebar.markdown("---")
st.sidebar.caption("Marine Deldicque\nCDSD Jedha 2026")


# ============================================================
# PAGE ACCUEIL
# ============================================================
if page == "🏠 Accueil":

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Patientes", "536", "TCGA-BRCA")
    with col2:
        st.metric("Features", "1936", "Multi-omiques")
    with col3:
        st.metric("Accuracy", "84.3%", "XGBoost")
    with col4:
        st.metric("F1 Macro", "76.3%", "4 sous-types")

    st.markdown("---")
    st.subheader("À qui s'adresse OncoPrint ?")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="user-card">
        <h3>🩺 Médecin chercheur / CRO</h3>
        <p>Aide à la décision avec validation de cohérence
        biomarqueurs / prédiction et rapport clinique automatique.</p>
        <strong>→ Aide à la décision</strong>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="user-card" style="border-left-color: #4CAF50">
        <h3>🔬 Data scientist / Chercheur</h3>
        <p>Exploration SHAP, analyse des probabilités,
        investigation des profils ambigus, batch CSV.</p>
        <strong>→ Outil de recherche</strong>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="user-card" style="border-left-color: #FF9800">
        <h3>📚 Étudiant / Pédagogie</h3>
        <p>Comprendre les sous-types moléculaires,
        explorer les biomarqueurs, tester des profils interactifs.</p>
        <strong>→ Explorer & apprendre</strong>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Les 4 sous-types moléculaires")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("### 🔵 Luminal A")
        st.markdown("ER+ PR+ HER2- | Hormonothérapie | F1=0.89")
    with col2:
        st.markdown("### 🔴 Triple Négatif")
        st.markdown("ER- PR- HER2- | Chimiothérapie | F1=0.82")
    with col3:
        st.markdown("### 🟠 HER2-enriched")
        st.markdown("ER- PR- HER2+ | Herceptin | F1=0.83")
    with col4:
        st.markdown("### 🟢 Luminal B / HER2+")
        st.markdown("ER+ HER2+ | Hormo+anti-HER2 | F1=0.50")

    st.markdown("---")
    st.subheader("Pipeline OncoPrint")
    st.code("""
Données TCGA-BRCA (705 patientes)
     ↓
Construction target ER/PR/HER2 → 4 sous-types (536 après nettoyage)
     ↓
Preprocessing : SMOTE + StandardScaler + VarianceThreshold
     ↓
XGBoost optimisé (GridSearchCV) — Accuracy 84.3% / F1 Macro 76.3%
     ↓
Validation SHAP — cohérence biologique confirmée
     ↓
API FastAPI + Dashboard Streamlit + Rapports Claude AI
    """)


# ============================================================
# PAGE AIDE À LA DÉCISION
# ============================================================
elif page == "🩺 Aide à la décision":

    st.header("🩺 Aide à la décision clinique")
    st.markdown("""
    <div class="warning-box">
    ⚕️ <strong>Usage médical</strong> — Suggestion algorithmique basée sur le profil
    génomique multi-omique. Ne remplace pas le diagnostic clinique
    (anatomopathologie, IHC, FISH). Le diagnostic final appartient au médecin.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Profils prédéfinis")
    cols = st.columns(5)
    for i, (nom, vals) in enumerate(PROFILS.items()):
        with cols[i]:
            if st.button(nom, key=f"btn_clinique_{i}"):
                st.session_state['bio_clinique'] = vals
                st.rerun()

    st.markdown("### Saisie des biomarqueurs")
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
    st.info("Explorez les prédictions, les features SHAP et analysez des profils ambigus.")

    tab1, tab2 = st.tabs(["🎛️ Prédiction manuelle", "📁 Batch CSV"])

    with tab1:
        st.markdown("### Profils prédéfinis")
        cols = st.columns(5)
        for i, (nom, vals) in enumerate(PROFILS.items()):
            with cols[i]:
                if st.button(nom, key=f"btn_recherche_{i}"):
                    st.session_state['bio_recherche'] = vals
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
        st.info("Uploadez un CSV avec les colonnes de features pour prédire plusieurs patientes.")
        uploaded = st.file_uploader("Choisir un fichier CSV", type=['csv'])

        if uploaded is not None:
            df_upload = pd.read_csv(uploaded)
            st.write(f"Dataset : {df_upload.shape[0]} patientes × {df_upload.shape[1]} features")
            st.dataframe(df_upload.head(3))

            if st.button("🔬 Prédire toutes les patientes"):
                resultats = []
                progress  = st.progress(0)
                for i, row in df_upload.iterrows():
                    try:
                        resp = appeler_api(row.to_dict())
                        if resp.status_code == 200:
                            r = resp.json()
                            resultats.append({
                                'patient_idx': i,
                                'subtype'    : r['subtype'],
                                'confidence' : f"{r['confidence']*100:.1f}%"
                            })
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
        sous_type = st.selectbox("Choisir un sous-type", [
            "Luminal A", "Triple Négatif", "HER2-enriched", "Luminal B / HER2+"
        ])

        infos = {
            "Luminal A": {
                "emoji"      : "🔵",
                "frequence"  : "~50% des cancers du sein",
                "biologie"   : "ER+ PR+ HER2- Ki67 bas — tumeur hormono-dépendante",
                "analogie"   : "Une voiture qui roule uniquement à l'essence œstrogène — couper l'approvisionnement suffit.",
                "traitement" : "Tamoxifène, inhibiteurs aromatase",
                "pronostic"  : "Favorable — survie à 5 ans ~90%",
                "shap"       : "pp_ER.alpha (#1) — biomarqueur ER standard depuis 40 ans"
            },
            "Triple Négatif": {
                "emoji"      : "🔴",
                "frequence"  : "~15-20% des cancers du sein",
                "biologie"   : "ER- PR- HER2- — absence totale de récepteurs ciblables",
                "analogie"   : "Une porte sans serrure — aucune clé thérapeutique ciblée disponible.",
                "traitement" : "Chimiothérapie — immunothérapie en développement",
                "pronostic"  : "Défavorable — récidive fréquente dans les 3-5 ans",
                "shap"       : "pp_HER2 absent (#1) — l'absence de HER2 est diagnostique"
            },
            "HER2-enriched": {
                "emoji"      : "🟠",
                "frequence"  : "~5-10% des cancers du sein",
                "biologie"   : "ER- PR- HER2+ — surexpression / amplification HER2/ERBB2",
                "analogie"   : "Un accélérateur bloqué — l'Herceptin est le mécanicien qui le débloque.",
                "traitement" : "Trastuzumab (Herceptin) — révolution thérapeutique des années 2000",
                "pronostic"  : "Intermédiaire — dramatiquement amélioré depuis l'Herceptin",
                "shap"       : "pp_HER2.pY1248 (#1) — phosphorylation HER2, cible de l'Herceptin"
            },
            "Luminal B / HER2+": {
                "emoji"      : "🟢",
                "frequence"  : "~10-15% des cancers du sein",
                "biologie"   : "ER+ HER2+ ou Ki67 élevé — composante hormonale ET HER2",
                "analogie"   : "Double moteur hormonal ET HER2 — nécessite une double stratégie.",
                "traitement" : "Hormonothérapie + anti-HER2 combinés",
                "pronostic"  : "Intermédiaire — moins favorable que Luminal A",
                "shap"       : "cn_PPP1R1B (#1) — amplification génomique caractéristique"
            }
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
        st.markdown("### 🎮 Tester un profil interactif")

        cols = st.columns(5)
        for i, (nom, vals) in enumerate(PROFILS.items()):
            with cols[i]:
                if st.button(nom, key=f"btn_pedago_{i}"):
                    st.session_state['bio_pedago'] = vals
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
# PAGE PERFORMANCES
# ============================================================
elif page == "📊 Performances":

    st.header("📊 Performances du modèle")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", "84.3%", "+38% vs baseline")
    with col2:
        st.metric("F1 Macro", "76.3%", "4 classes")
    with col3:
        st.metric("CV 5-fold", "71.5% ± 2.5%", "sans data leakage")
    with col4:
        st.metric("Test set", "108 patientes", "20% holdout")

    st.markdown("---")
    st.subheader("Performances par sous-type")
    df_perf = pd.DataFrame({
        'Sous-type' : ['Luminal A', 'Triple Négatif', 'HER2-enriched', 'Luminal B / HER2+'],
        'Precision' : [0.90, 0.76, 0.71, 0.62],
        'Recall'    : [0.89, 0.89, 1.00, 0.42],
        'F1-Score'  : [0.89, 0.82, 0.83, 0.50],
        'Support'   : [73, 18, 5, 12]
    })
    st.dataframe(df_perf, use_container_width=True)

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
## OncoPrint
**Classification moléculaire automatique du cancer du sein**

### Contexte
Projet de data science appliqué à l'oncologie —
certification CDSD Jedha Bootcamp 2026.

### Dataset
- **Source :** The Cancer Genome Atlas (TCGA-BRCA) — NCI/NIH
- **Population :** 536 patientes après nettoyage (705 initiales)
- **Features :** 1936 mesures multi-omiques
  - 249 mutations somatiques | 860 copy number variations
  - 604 gènes RNA-seq | 223 phospho-protéines

### Modèle
- **Algorithme :** XGBoost optimisé (GridSearchCV)
- **Accuracy :** 84.3% | **F1 Macro :** 76.3%
- **Validation :** CV 5-fold corrigée (sans data leakage SMOTE)
- **Interprétabilité :** SHAP — cohérence biologique validée

### Stack technique
Python · XGBoost · scikit-learn · SHAP · imbalanced-learn
FastAPI · Streamlit · Anthropic Claude API
Google Colab · VS Code · GitHub

### Limites
- Données TCGA en conditions recherche — validation clinique nécessaire
- Luminal B : recall 42% — hétérogénéité biologique connue
- HER2-enriched : n=5 dans le test — statistiquement fragile

### Auteure
**Marine Deldicque**
Infirmière libérale — Data Scientist en transition
Jedha Bootcamp — CDSD 2026
    """)