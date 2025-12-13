"""
Student Dropout Prediction - Simple Predictor
==============================================

App semplificata per predire il dropout studentesco.
Usa i modelli RF e XGBoost trainati dalla pipeline.

MODALITÀ:
=========
1. STANDARD: Usa i modelli con tutte le 36 features (include performance universitaria)
2. PRE-IMMATRICOLAZIONE: Usa i modelli con solo 24 features (screening preventivo)

ESECUZIONE:
streamlit run app_simple.py
"""

import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path

# Importa configurazione colori centralizzata
from color_config import CLASS_COLORS, CLASS_COLORS_LIGHT

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

st.set_page_config(
    page_title="Student Dropout Predictor",
    page_icon="🎓",
    layout="wide"
)

# CSS dinamico con colori dalla configurazione
css = f"""
<style>
    .prediction-box {{
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 2rem 0;
    }}
    .dropout {{ background-color: {CLASS_COLORS_LIGHT['Dropout']}; color: {CLASS_COLORS['Dropout']}; }}
    .enrolled {{ background-color: {CLASS_COLORS_LIGHT['Enrolled']}; color: {CLASS_COLORS['Enrolled']}; }}
    .graduate {{ background-color: {CLASS_COLORS_LIGHT['Graduate']}; color: {CLASS_COLORS['Graduate']}; }}
    .metric-card {{
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }}
    .mode-badge {{
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem 0;
    }}
    .mode-standard {{ background-color: #e3f2fd; color: #1565c0; }}
    .mode-preadmission {{ background-color: #fff3e0; color: #e65100; }}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# ============================================================================
# SELEZIONE MODALITÀ (SIDEBAR)
# ============================================================================

st.sidebar.title("⚙️ Configurazione")
st.sidebar.markdown("---")

prediction_mode = st.sidebar.radio(
    "**Modalità Predizione**",
    ["Standard (Post-Iscrizione)", "Pre-Immatricolazione (Preventiva)"],
    index=0,
    help="Scegli se usare tutti i dati (incluse performance) o solo dati pre-universitari"
)

preadmission_mode = "Pre-Immatricolazione" in prediction_mode

# Info modalità
if preadmission_mode:
    st.sidebar.info("""
    **🔍 Modalità Pre-Immatricolazione**
    
    - **Features**: 24 (solo dati pre-universitari)
    - **Quando**: PRIMA dell'inizio corsi
    - **Accuracy**: ~58-60%
    - **Scopo**: Screening preventivo
    """)
else:
    st.sidebar.success("""
    **📊 Modalità Standard**
    
    - **Features**: 36 (include performance)
    - **Quando**: DOPO l'inizio corsi
    - **Accuracy**: ~70-72%
    - **Scopo**: Monitoraggio continuo
    """)

st.sidebar.markdown("---")
st.sidebar.caption("💡 **Suggerimento**: Usa Pre-Immatricolazione per screening all'iscrizione, Standard per monitoraggio durante l'anno.")

# ============================================================================
# CARICAMENTO MODELLI
# ============================================================================

@st.cache_resource
def load_models(mode="standard"):
    """Carica entrambi i modelli, metadata e risultati di training dalla pipeline"""
    try:
        # Percorsi dalla pipeline (cartella 03_training/ o 03_training_preadmission/)
        if mode == "preadmission":
            base_path = Path('03_training_preadmission')
            rf_file = 'rf_model_preadmission.pkl'
            xgb_file = 'xgb_model_preadmission.pkl'
            encoder_file = 'label_encoder_preadmission.pkl'
            features_file = 'feature_names_preadmission.pkl'
            results_file = 'training_results_preadmission.pkl'
        else:
            base_path = Path('03_training')
            rf_file = 'rf_model.pkl'
            xgb_file = 'xgb_model.pkl'
            encoder_file = 'label_encoder.pkl'
            features_file = 'feature_names.pkl'
            results_file = 'training_results.pkl'

        # Carica Random Forest
        rf_model = joblib.load(base_path / rf_file)

        # Carica XGBoost
        xgb_model = joblib.load(base_path / xgb_file)

        # Carica metadata
        label_encoder = joblib.load(base_path / encoder_file)
        feature_names = joblib.load(base_path / features_file)

        # Carica risultati di training (metriche reali)
        training_results = joblib.load(base_path / results_file)

        return {
            'rf': rf_model,
            'xgb': xgb_model,
            'encoder': label_encoder,
            'features': feature_names,
            'results': training_results  # ⭐ AGGIUNTO
        }, None

    except Exception as e:
        error_msg = f"Errore caricamento modelli: {e}\n\nAssicurati di aver eseguito la pipeline:\n"
        if mode == "preadmission":
            error_msg += "python train_models.py --preadmission"
        else:
            error_msg += "python train_models.py"
        return None, error_msg

# ============================================================================
# MAPPATURE CATEGORICHE
# ============================================================================

@st.cache_data
def load_feature_mappings(mode="standard"):
    """
    Carica i mappings delle feature dal file JSON generato da student_analysis.py

    Returns:
        mappings (dict): Dizionario con mapping label → code
        error (str): Messaggio di errore se caricamento fallisce
    """
    try:
        # Percorsi diversi per modalità
        if mode == "preadmission":
            mappings_path = Path('01_analysis_preadmission') / 'feature_mappings_reverse.json'
        else:
            mappings_path = Path('01_analysis') / 'feature_mappings_reverse.json'

        # Carica mapping inverso (label -> code)
        with open(mappings_path, 'r', encoding='utf-8') as f:
            mappings = json.load(f)

        return mappings, None

    except FileNotFoundError:
        error_msg = f"⚠️ File mappings non trovato: {mappings_path}\n\n"
        error_msg += "Assicurati di aver eseguito prima:\n"
        if mode == "preadmission":
            error_msg += "python student_analysis.py --preadmission"
        else:
            error_msg += "python student_analysis.py"
        return None, error_msg
    except Exception as e:
        return None, f"Errore caricamento mappings: {e}"

# ============================================================================
# HEADER
# ============================================================================

st.title("🎓 Student Dropout Predictor")

# Badge modalità
if preadmission_mode:
    st.markdown(
        '<div class="mode-badge mode-preadmission">🔍 MODALITÀ PRE-IMMATRICOLAZIONE</div>',
        unsafe_allow_html=True
    )
    st.markdown("Predizione **PREVENTIVA** basata solo su dati disponibili all'iscrizione")
else:
    st.markdown(
        '<div class="mode-badge mode-standard">📊 MODALITÀ STANDARD</div>',
        unsafe_allow_html=True
    )
    st.markdown("Predici se uno studente abbandonerà, rimarrà iscritto o si diplomerà")

st.markdown("---")

# ============================================================================
# CARICAMENTO E VERIFICA
# ============================================================================

current_mode = "preadmission" if preadmission_mode else "standard"
models_data, error = load_models(current_mode)

if error:
    st.error(f"❌ {error}")
    st.stop()

# Estrai metriche reali dai risultati di training
training_results = models_data['results']
rf_ba = training_results['random_forest']['test_ba']
rf_f1 = training_results['random_forest']['test_f1']
xgb_ba = training_results['xgboost']['test_ba']
xgb_f1 = training_results['xgboost']['test_f1']
best_model = training_results['best_model']

if preadmission_mode:
    st.success(f"✅ Modelli Pre-Immatricolazione caricati: Random Forest + XGBoost (24 features) | Miglior modello: {best_model} ({xgb_ba if best_model == 'XGBoost' else rf_ba:.2%})")
else:
    st.success(f"✅ Modelli caricati: Random Forest + XGBoost | Miglior modello: {best_model} ({xgb_ba if best_model == 'XGBoost' else rf_ba:.2%})")

# ============================================================================
# SELEZIONE MODELLO
# ============================================================================

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Seleziona Modello")

    # Crea le label con le metriche REALI
    xgb_label = f"XGBoost (BA: {xgb_ba:.2%}) - {'Consigliato' if best_model == 'XGBoost' else ''}"
    rf_label = f"Random Forest (BA: {rf_ba:.2%}) - {'Consigliato' if best_model == 'Random Forest' else ''}"

    selected_model = st.radio(
        "Quale modello vuoi usare?",
        [xgb_label, rf_label, "Confronta Entrambi"],
        index=0,
        horizontal=True
    )

with col2:
    st.markdown("### Performance")

    if "XGBoost" in selected_model:
        st.metric("Balanced Accuracy", f"{xgb_ba:.2%}")
        st.metric("F1-Score", f"{xgb_f1:.2%}")
    elif "Random Forest" in selected_model:
        st.metric("Balanced Accuracy", f"{rf_ba:.2%}")
        st.metric("F1-Score", f"{rf_f1:.2%}")
    else:
        st.info("Confronto tra modelli")

    if preadmission_mode:
        st.caption("⚠️ Accuracy ridotta ma ECCELLENTE per predizione senza dati universitari!")

st.markdown("---")

# ============================================================================
# FORM INPUT
# ============================================================================

st.markdown("### 📝 Dati Studente")

if preadmission_mode:
    st.info("""
    **ℹ️ Modalità Pre-Immatricolazione**  
    Inserisci solo i dati disponibili **PRIMA** dell'inizio dei corsi.
    Le sezioni sulla performance universitaria saranno automaticamente escluse.
    """)

# Carica i mappings completi dal file JSON
mappings, mapping_error = load_feature_mappings(current_mode)

if mapping_error:
    st.warning(f"{mapping_error}\n")
else:
    st.success(f"✅ Mappings completi caricati: {len(mappings)} feature con valori leggibili")

# Crea dizionario per raccogliere dati
student_data = {}

# ============================================================================
# 1. FATTORI DEMOGRAFICI
# ============================================================================
with st.expander("👥 Fattori Demografici", expanded=True):
    col1, col2, col3 = st.columns(3)

    with col1:
        student_data['Age at enrollment'] = st.number_input(
            "Età all'iscrizione",
            min_value=17, max_value=70, value=20
        )

        student_data['Marital status'] = mappings['Marital status'][
            st.selectbox("Stato civile", list(mappings['Marital status'].keys()), key="marital")
        ]

    with col2:
        student_data['Gender'] = 1 if st.selectbox("Genere", ["Femmina", "Maschio"]) == "Maschio" else 0

        # Nazionalità - salviamo il label testuale per il calcolo
        nationality_label = st.selectbox("Nazionalità", list(mappings['Nacionality'].keys()), key="nationality")
        student_data['Nacionality'] = mappings['Nacionality'][nationality_label]

    with col3:
        student_data['Displaced'] = 1 if st.selectbox(
            "Fuori sede", ["No", "Sì"], key="displaced"
        ) == "Sì" else 0

    # Calcola automaticamente International dalla nazionalità
    # Se nazionalità è "Portuguese" → International = 0, altrimenti = 1
    student_data['International'] = 0 if nationality_label == "Portuguese" else 1

    # Mostra lo status calcolato
    if student_data['International'] == 1:
        st.info(f"ℹ️ **Studente internazionale**: Sì (nazionalità: {nationality_label})")
    else:
        st.caption(f"💡 Studente locale (nazionalità: {nationality_label})")

# ============================================================================
# 2. PERCORSO ACCADEMICO PRECEDENTE
# ============================================================================
with st.expander("🎓 Percorso Accademico Precedente", expanded=True):
    col1, col2 = st.columns(2)

    with col1:
        student_data['Previous qualification'] = mappings['Previous qualification'][
            st.selectbox("Qualifica precedente", list(mappings['Previous qualification'].keys()), key="prev_qual")
        ]

    with col2:
        student_data['Previous qualification (grade)'] = st.number_input(
            "Voto qualifica precedente",
            min_value=0.0, max_value=200.0, value=120.0, step=0.1
        )

# ============================================================================
# 3. INFORMAZIONI SUL CORSO DI LAUREA
# ============================================================================
with st.expander("📚 Informazioni sul Corso di Laurea", expanded=True):
    col1, col2, col3 = st.columns(3)

    with col1:
        student_data['Application mode'] = mappings['Application mode'][
            st.selectbox("Modalità iscrizione", list(mappings['Application mode'].keys()), key="app_mode")
        ]

        student_data['Application order'] = st.number_input(
            "Ordine scelta corso (0-9)",
            min_value=0, max_value=9, value=0,
            help="0 = Prima scelta, 1 = Seconda scelta, ecc."
        )

    with col2:
        student_data['Course'] = mappings['Course'][
            st.selectbox("Corso", list(mappings['Course'].keys()), key="course")
        ]

        student_data['Daytime/evening attendance'] = 1 if st.selectbox(
            "Frequenza", ["Diurna", "Serale"], key="attendance"
        ) == "Diurna" else 0

    with col3:
        student_data['Admission grade'] = st.number_input(
            "Voto ammissione",
            min_value=0.0, max_value=200.0, value=120.0, step=0.1
        )

        student_data['Educational special needs'] = 1 if st.selectbox(
            "Bisogni educativi speciali", ["No", "Sì"], key="special_needs"
        ) == "Sì" else 0

# ============================================================================
# 4. BACKGROUND FAMILIARE
# ============================================================================
with st.expander("👨‍👩‍👧‍👦 Background Familiare", expanded=False):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 👩 Madre")
        # Mother's qualification
        if "Mother's qualification" in mappings:
            student_data["Mother's qualification"] = mappings["Mother's qualification"][
                st.selectbox("Qualifica madre", list(mappings["Mother's qualification"].keys()), key="mother_qual")
            ]
        else:
            student_data["Mother's qualification"] = st.number_input(
                "Qualifica madre (1-44)", min_value=1, max_value=44, value=1
            )

        # Mother's occupation
        if "Mother's occupation" in mappings:
            student_data["Mother's occupation"] = mappings["Mother's occupation"][
                st.selectbox("Occupazione madre", list(mappings["Mother's occupation"].keys()), key="mother_occ")
            ]
        else:
            student_data["Mother's occupation"] = st.number_input(
                "Occupazione madre (0-195)", min_value=0, max_value=195, value=0
            )

    with col2:
        st.markdown("#### 👨 Padre")
        # Father's qualification
        if "Father's qualification" in mappings:
            student_data["Father's qualification"] = mappings["Father's qualification"][
                st.selectbox("Qualifica padre", list(mappings["Father's qualification"].keys()), key="father_qual")
            ]
        else:
            student_data["Father's qualification"] = st.number_input(
                "Qualifica padre (1-44)", min_value=1, max_value=44, value=1
            )

        # Father's occupation
        if "Father's occupation" in mappings:
            student_data["Father's occupation"] = mappings["Father's occupation"][
                st.selectbox("Occupazione padre", list(mappings["Father's occupation"].keys()), key="father_occ")
            ]
        else:
            student_data["Father's occupation"] = st.number_input(
                "Occupazione padre (0-195)", min_value=0, max_value=195, value=0
            )

# ============================================================================
# 5. SITUAZIONE ECONOMICA
# ============================================================================
with st.expander("💰 Situazione Economica", expanded=False):
    col1, col2, col3 = st.columns(3)

    with col1:
        student_data['Tuition fees up to date'] = 1 if st.selectbox(
            "Tasse aggiornate", ["Sì", "No"], key="tuition"
        ) == "Sì" else 0

    with col2:
        student_data['Debtor'] = 1 if st.selectbox(
            "Debitore", ["No", "Sì"], key="debtor"
        ) == "Sì" else 0

    with col3:
        student_data['Scholarship holder'] = 1 if st.selectbox(
            "Borsa di studio", ["No", "Sì"], key="scholarship"
        ) == "Sì" else 0

# ============================================================================
# 6. PERFORMANCE UNIVERSITARIA - 1° SEMESTRE (SOLO MODALITÀ STANDARD)
# ============================================================================
if not preadmission_mode:
    with st.expander("📊 Performance Universitaria - 1° Semestre", expanded=False):
        st.warning("⚠️ Questi dati sono disponibili solo DOPO l'inizio del 1° semestre")

        col1, col2, col3 = st.columns(3)

        with col1:
            student_data['Curricular units 1st sem (credited)'] = st.number_input(
                "Crediti riconosciuti", min_value=0, max_value=30, value=0, key="1sem_credited"
            )

            student_data['Curricular units 1st sem (enrolled)'] = st.number_input(
                "Unità a cui iscritto", min_value=0, max_value=30, value=6, key="1sem_enrolled"
            )

        with col2:
            student_data['Curricular units 1st sem (evaluations)'] = st.number_input(
                "Numero di valutazioni", min_value=0, max_value=50, value=6, key="1sem_eval"
            )

            student_data['Curricular units 1st sem (approved)'] = st.number_input(
                "Unità approvate", min_value=0, max_value=30, value=5, key="1sem_approved"
            )

        with col3:
            student_data['Curricular units 1st sem (grade)'] = st.number_input(
                "Voto medio",
                min_value=0.0, max_value=20.0, value=12.0, step=0.1, key="1sem_grade"
            )

            student_data['Curricular units 1st sem (without evaluations)'] = st.number_input(
                "Senza valutazione", min_value=0, max_value=30, value=0, key="1sem_without"
            )

# ============================================================================
# 7. PERFORMANCE UNIVERSITARIA - 2° SEMESTRE (SOLO MODALITÀ STANDARD)
# ============================================================================
if not preadmission_mode:
    with st.expander("📊 Performance Universitaria - 2° Semestre", expanded=False):
        st.warning("⚠️ Questi dati sono disponibili solo DOPO l'inizio del 2° semestre")

        col1, col2, col3 = st.columns(3)

        with col1:
            student_data['Curricular units 2nd sem (credited)'] = st.number_input(
                "Crediti riconosciuti", min_value=0, max_value=30, value=0, key="2sem_credited"
            )

            student_data['Curricular units 2nd sem (enrolled)'] = st.number_input(
                "Unità a cui iscritto", min_value=0, max_value=30, value=6, key="2sem_enrolled"
            )

        with col2:
            student_data['Curricular units 2nd sem (evaluations)'] = st.number_input(
                "Numero di valutazioni", min_value=0, max_value=50, value=6, key="2sem_eval"
            )

            student_data['Curricular units 2nd sem (approved)'] = st.number_input(
                "Unità approvate", min_value=0, max_value=30, value=5, key="2sem_approved"
            )

        with col3:
            student_data['Curricular units 2nd sem (grade)'] = st.number_input(
                "Voto medio",
                min_value=0.0, max_value=20.0, value=12.0, step=0.1, key="2sem_grade"
            )

            student_data['Curricular units 2nd sem (without evaluations)'] = st.number_input(
                "Senza valutazione", min_value=0, max_value=30, value=0, key="2sem_without"
            )

# ============================================================================
# 8. INDICATORI ECONOMICI (AUTOMATICI - NON MODIFICABILI)
# ============================================================================
with st.expander("📈 Indicatori Economici (Automatici)", expanded=False):
    st.info("""
    **ℹ️ Valori Macroeconomici Automatici**  
    
    I seguenti indicatori vengono impostati automaticamente con i dati più recenti del Portogallo.
    Questi valori NON devono essere modificati dall'utente.
    
    - **Tasso di Disoccupazione**: 6.5% (Q4 2024)
    - **Tasso di Inflazione**: 2.3% (Novembre 2024)
    - **Crescita PIL**: 2.1% (Previsione 2024)
    
    *Fonte: PORDATA, Banco de Portugal, INE*
    """)

    # Mostra i valori (ma non permettere modifica)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Tasso Disoccupazione", "6.5%")
    with col2:
        st.metric("Tasso Inflazione", "2.3%")
    with col3:
        st.metric("Crescita PIL", "2.1%")

    # Imposta automaticamente i valori
    student_data['Unemployment rate'] = 6.5
    student_data['Inflation rate'] = 2.3
    student_data['GDP'] = 2.1

st.markdown("---")

# ============================================================================
# PREDIZIONE
# ============================================================================

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button(
        "🔮 PREDICI RISULTATO",
        type="primary",
        use_container_width=True
    )

if predict_button:
    try:
        # Prepara dati nell'ordine corretto
        X_new = pd.DataFrame([student_data])[models_data['features']]

        # ===== PREDIZIONI =====

        if "Confronta" in selected_model:
            # CONFRONTA ENTRAMBI I MODELLI
            st.markdown("### 🔄 Confronto Modelli")

            col1, col2 = st.columns(2)

            # Random Forest
            with col1:
                st.markdown("#### 🌲 Random Forest")
                rf_pred = models_data['rf'].predict(X_new)[0]
                rf_proba = models_data['rf'].predict_proba(X_new)[0]
                rf_conf = rf_proba.max()

                pred_class = {'Dropout': 'dropout', 'Enrolled': 'enrolled', 'Graduate': 'graduate'}
                st.markdown(
                    f'<div class="prediction-box {pred_class[rf_pred]}">{rf_pred}</div>',
                    unsafe_allow_html=True
                )
                st.metric("Confidenza", f"{rf_conf*100:.1f}%")

                # Probabilità
                classes = ['Dropout', 'Enrolled', 'Graduate']
                prob_df = pd.DataFrame({
                    'Classe': classes,
                    'Probabilità': [f"{p*100:.1f}%" for p in rf_proba]
                })
                st.dataframe(prob_df, use_container_width=True, hide_index=True)

            # XGBoost
            with col2:
                st.markdown("#### ⚡ XGBoost")
                xgb_pred_enc = models_data['xgb'].predict(X_new)[0]
                xgb_pred = models_data['encoder'].inverse_transform([xgb_pred_enc])[0]
                xgb_proba = models_data['xgb'].predict_proba(X_new)[0]
                xgb_conf = xgb_proba.max()

                st.markdown(
                    f'<div class="prediction-box {pred_class[xgb_pred]}">{xgb_pred}</div>',
                    unsafe_allow_html=True
                )
                st.metric("Confidenza", f"{xgb_conf*100:.1f}%")

                # Probabilità
                prob_df = pd.DataFrame({
                    'Classe': classes,
                    'Probabilità': [f"{p*100:.1f}%" for p in xgb_proba]
                })
                st.dataframe(prob_df, use_container_width=True, hide_index=True)

            # Accordo tra modelli
            st.markdown("---")
            if rf_pred == xgb_pred:
                st.success(f"✅ **ACCORDO**: Entrambi i modelli predicono **{rf_pred}**")
            else:
                st.warning(f"⚠️ **DISACCORDO**: RF predice **{rf_pred}**, XGB predice **{xgb_pred}**")

        else:
            # SINGOLO MODELLO
            st.markdown("### 🎯 Risultato Predizione")

            if "XGBoost" in selected_model:
                # XGBoost
                pred_encoded = models_data['xgb'].predict(X_new)[0]
                prediction = models_data['encoder'].inverse_transform([pred_encoded])[0]
                probabilities = models_data['xgb'].predict_proba(X_new)[0]
                model_name = "XGBoost"
            else:
                # Random Forest
                prediction = models_data['rf'].predict(X_new)[0]
                probabilities = models_data['rf'].predict_proba(X_new)[0]
                model_name = "Random Forest"

            confidence = probabilities.max()

            # Box predizione
            pred_classes = {'Dropout': 'dropout', 'Enrolled': 'enrolled', 'Graduate': 'graduate'}
            st.markdown(
                f'<div class="prediction-box {pred_classes[prediction]}">{prediction}</div>',
                unsafe_allow_html=True
            )

            # Metriche
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Modello", model_name)
            with col2:
                st.metric("Confidenza", f"{confidence*100:.1f}%")
            with col3:
                risk = "Alto" if prediction == "Dropout" else ("Medio" if prediction == "Enrolled" else "Basso")
                st.metric("Livello Rischio", risk)

            # Probabilità dettagliate
            st.markdown("#### 📊 Probabilità per Classe")

            classes = ['Dropout', 'Enrolled', 'Graduate']
            prob_data = pd.DataFrame({
                'Classe': classes,
                'Probabilità (%)': probabilities * 100
            }).sort_values('Probabilità (%)', ascending=False)

            col1, col2 = st.columns([1, 2])

            with col1:
                st.dataframe(prob_data, use_container_width=True, hide_index=True)

            with col2:
                # Bar chart semplice
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(8, 4))
                bar_colors = [CLASS_COLORS[c] for c in classes]

                ax.barh(classes, probabilities * 100, color=bar_colors)
                ax.set_xlabel('Probabilità (%)')
                ax.set_xlim(0, 100)
                for i, (cls, prob) in enumerate(zip(classes, probabilities * 100)):
                    ax.text(prob + 2, i, f'{prob:.1f}%', va='center')

                st.pyplot(fig)

            # Interpretazione
            st.markdown("---")
            st.markdown("#### 💡 Interpretazione")

            if preadmission_mode:
                # Interpretazione specifica per pre-immatricolazione
                if prediction == "Dropout":
                    st.error(f"""
                    **⚠️ STUDENTE AD ALTO RISCHIO (Predizione Preventiva)**
                    
                    Confidenza: {confidence*100:.1f}%
                    
                    **Raccomandazioni Pre-Iscrizione:**
                    - **Screening approfondito** prima dell'immatricolazione
                    - **Counseling orientativo** per valutare scelta corso
                    - **Colloquio con famiglia** per valutare supporto
                    - **Proposta tutoring proattivo** dal primo giorno
                    - **Valutare supporto finanziario** se necessario
                    
                    ⚠️ Nota: Predizione basata solo su dati pre-universitari (accuracy ~60%)
                    """)
                elif prediction == "Enrolled":
                    st.warning(f"""
                    **⚠️ STUDENTE A RISCHIO MEDIO (Predizione Preventiva)**
                    
                    Confidenza: {confidence*100:.1f}%
                    
                    **Raccomandazioni Pre-Iscrizione:**
                    - **Monitoraggio attento** dall'inizio
                    - **Assegnazione tutor** facoltativa
                    - **Presentazione servizi di supporto** disponibili
                    - **Check-in periodici** durante primo semestre
                    
                    ⚠️ Nota: Predizione basata solo su dati pre-universitari (accuracy ~60%)
                    """)
                else:
                    st.success(f"""
                    **✅ STUDENTE A BASSO RISCHIO (Predizione Preventiva)**
                    
                    Confidenza: {confidence*100:.1f}%
                    
                    **Raccomandazioni Pre-Iscrizione:**
                    - Supporto standard
                    - Possibile ruolo di **peer mentor** per altri studenti
                    - Opportunità di **programmi di eccellenza**
                    - Monitoraggio routinario
                    
                    ⚠️ Nota: Predizione basata solo su dati pre-universitari (accuracy ~60%)
                    """)
            else:
                # Interpretazione standard (post-iscrizione)
                if prediction == "Dropout":
                    st.error(f"""
                    **⚠️ STUDENTE AD ALTO RISCHIO**
                    
                    Confidenza: {confidence*100:.1f}%
                    
                    **Raccomandazioni:**
                    - Intervento immediato necessario
                    - Supporto accademico e tutoraggio
                    - Verifica situazione finanziaria
                    - Incontri regolari con tutor
                    """)
                elif prediction == "Enrolled":
                    st.warning(f"""
                    **⚠️ STUDENTE A RISCHIO MEDIO**
                    
                    Confidenza: {confidence*100:.1f}%
                    
                    **Raccomandazioni:**
                    - Monitoraggio costante
                    - Supporto su richiesta
                    - Verificare eventuali difficoltà
                    - Incoraggiare partecipazione
                    """)
                else:
                    st.success(f"""
                    **✅ STUDENTE A BASSO RISCHIO**
                    
                    Confidenza: {confidence*100:.1f}%
                    
                    **Raccomandazioni:**
                    - Mantenere supporto attuale
                    - Offrire opportunità di eccellenza
                    - Possibili ruoli di mentoring
                    - Continuare così!
                    """)

    except Exception as e:
        st.error(f"❌ Errore durante la predizione: {e}")
        st.exception(e)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")

if preadmission_mode:
    st.caption("Student Dropout Predictor | Modalità: PRE-IMMATRICOLAZIONE | Modelli: Random Forest + XGBoost | Dataset: UCI ML #697")
else:
    st.caption("Student Dropout Predictor | Modelli: Random Forest + XGBoost | Dataset: UCI ML #697")