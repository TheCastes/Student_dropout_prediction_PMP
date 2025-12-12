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

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

st.set_page_config(
    page_title="Student Dropout Predictor",
    layout="wide"
)

st.markdown("""
<style>
    .prediction-box {
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 2rem 0;
    }
    .dropout { background-color: #ffebee; color: #c62828; }
    .enrolled { background-color: #fff3e0; color: #ef6c00; }
    .graduate { background-color: #e8f5e9; color: #2e7d32; }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .mode-badge {
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem 0;
    }
    .mode-standard { background-color: #e3f2fd; color: #1565c0; }
    .mode-preadmission { background-color: #fff3e0; color: #e65100; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SELEZIONE MODALITÀ (SIDEBAR)
# ============================================================================

st.sidebar.title("Configurazione")
st.sidebar.markdown("---")

prediction_mode = st.sidebar.radio(
    "**Modalità Predizione**",
    ["Standard (Post-Iscrizione)", "Pre-Immatricolazione (Preventiva)"],
    index=0,
    help="Scegli se usare tutti i dati o solo dati pre-universitari"
)

preadmission_mode = "Pre-Immatricolazione" in prediction_mode

# Info modalità
if preadmission_mode:
    st.sidebar.info("""
    **Modalità Pre-Immatricolazione**
    
    **Features**: 24 (solo dati pre-universitari)
    """)
else:
    st.sidebar.success("""
    **Modalità Standard**
    
    **Features**: 36 (include performance)
    """)

# ============================================================================
# CARICAMENTO MODELLI
# ============================================================================

@st.cache_resource
def load_models(mode="standard"):
    """Carica entrambi i modelli e metadata dalla pipeline"""
    try:
        # Percorsi dalla pipeline (cartella 03_training/ o 03_training_preadmission/)
        if mode == "preadmission":
            base_path = Path('03_training_preadmission')
            rf_file = 'rf_model_preadmission.pkl'
            xgb_file = 'xgb_model_preadmission.pkl'
            encoder_file = 'label_encoder_preadmission.pkl'
            features_file = 'feature_names_preadmission.pkl'
        else:
            base_path = Path('03_training')
            rf_file = 'rf_model.pkl'
            xgb_file = 'xgb_model.pkl'
            encoder_file = 'label_encoder.pkl'
            features_file = 'feature_names.pkl'

        # Carica Random Forest
        rf_model = joblib.load(base_path / rf_file)

        # Carica XGBoost
        xgb_model = joblib.load(base_path / xgb_file)

        # Carica metadata
        label_encoder = joblib.load(base_path / encoder_file)
        feature_names = joblib.load(base_path / features_file)

        return {
            'rf': rf_model,
            'xgb': xgb_model,
            'encoder': label_encoder,
            'features': feature_names
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
        error_msg = f"File mappings non trovato: {mappings_path}\n\n"
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

st.title("Student Dropout Predictor")

# Badge modalità
if preadmission_mode:
    st.markdown(
        '<div class="mode-badge mode-preadmission">MODALITÀ PRE-IMMATRICOLAZIONE</div>',
        unsafe_allow_html=True
    )
    st.markdown("Predizione **PREVENTIVA** basata solo su dati disponibili all'iscrizione")
else:
    st.markdown(
        '<div class="mode-badge mode-standard">MODALITÀ STANDARD</div>',
        unsafe_allow_html=True
    )
    st.markdown("Predittore di abbandono universitario")

st.markdown("---")

# ============================================================================
# CARICAMENTO E VERIFICA
# ============================================================================

current_mode = "preadmission" if preadmission_mode else "standard"
models_data, error = load_models(current_mode)

if error:
    st.error(f"❌ {error}")
    st.stop()

if preadmission_mode:
    st.success("Modelli Pre-Immatricolazione caricati: Random Forest + XGBoost (24 features)")
else:
    st.success("Modelli caricati: Random Forest + XGBoost")

# ============================================================================
# SELEZIONE MODELLO
# ============================================================================

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Seleziona Modello")

    if preadmission_mode:
        # Performance pre-immatricolazione (esempio, da aggiornare con i tuoi valori reali)
        selected_model = st.radio(
            "Quale modello vuoi usare?",
            ["XGBoost (BA: ~59%) - Consigliato", "Random Forest (BA: ~58%)", "Confronta Entrambi"],
            index=0,
            horizontal=True
        )
    else:
        # Performance standard
        selected_model = st.radio(
            "Quale modello vuoi usare?",
            ["XGBoost (BA: 71.15%) - Consigliato", "Random Forest (BA: 70.82%)", "Confronta Entrambi"],
            index=0,
            horizontal=True
        )

with col2:
    st.markdown("### Performance")

    if preadmission_mode:
        if "XGBoost" in selected_model:
            st.metric("Balanced Accuracy", "~59%")
            st.metric("F1-Score", "~58%")
        elif "Random Forest" in selected_model:
            st.metric("Balanced Accuracy", "~58%")
            st.metric("F1-Score", "~57%")
        else:
            st.info("Confronto tra modelli")

        st.caption("Accuracy ridotta ma ECCELLENTE per predizione senza dati universitari!")
    else:
        if "XGBoost" in selected_model:
            st.metric("Balanced Accuracy", "71.15%")
            st.metric("F1-Score", "71.16%")
        elif "Random Forest" in selected_model:
            st.metric("Balanced Accuracy", "70.82%")
            st.metric("F1-Score", "70.80%")
        else:
            st.info("Confronto tra modelli")

st.markdown("---")

# ============================================================================
# FORM INPUT
# ============================================================================

st.markdown("### Dati Studente")

if preadmission_mode:
    st.info("""
    **Modalità Pre-Immatricolazione**  
    Inserisci solo i dati disponibili **PRIMA** dell'inizio dei corsi.
    Le sezioni sulla performance universitaria saranno automaticamente escluse.
    """)

# Carica i mappings completi dal file JSON
mappings, mapping_error = load_feature_mappings(current_mode)


st.success(f"Mappings completi caricati: {len(mappings)} feature con valori leggibili")

# Crea dizionario per raccogliere dati
student_data = {}

# ===== TAB 1: INFO PERSONALI =====
with st.expander("Informazioni Personali", expanded=True):
    col1, col2, col3 = st.columns(3)

    with col1:
        student_data['Age at enrollment'] = st.number_input(
            "Età all'iscrizione",
            min_value=17, max_value=70, value=20
        )

        student_data['Marital status'] = mappings['Marital status'][
            st.selectbox("Stato civile", list(mappings['Marital status'].keys()))
        ]

        student_data['Gender'] = 1 if st.selectbox("Genere", ["Femmina", "Maschio"]) == "Maschio" else 0

    with col2:
        student_data['Application mode'] = mappings['Application mode'][
            st.selectbox("Modalità iscrizione", list(mappings['Application mode'].keys()))
        ]

        student_data['Application order'] = st.number_input(
            "Ordine iscrizione (0-9)",
            min_value=0, max_value=9, value=0,
            help="0 = Prima scelta, 1 = Seconda scelta, ecc."
        )

        student_data['Course'] = mappings['Course'][
            st.selectbox("Corso", list(mappings['Course'].keys()))
        ]

        student_data['Daytime/evening attendance'] = 1 if st.selectbox(
            "Frequenza", ["Diurna", "Serale"]
        ) == "Diurna" else 0

    with col3:
        student_data['Previous qualification'] = mappings['Previous qualification'][
            st.selectbox("Qualifica precedente", list(mappings['Previous qualification'].keys()))
        ]

        student_data['Nacionality'] = mappings['Nacionality'][
            st.selectbox("Nazionalità", list(mappings['Nacionality'].keys()))
        ]

        student_data['International'] = 1 if st.selectbox(
            "Studente internazionale", ["No", "Sì"]
        ) == "Sì" else 0

# ===== TAB 2: BACKGROUND FAMILIARE =====
with st.expander("Background Familiare", expanded=False):
    col1, col2, col3 = st.columns(3)

    with col1:
        # MADRE - Qualifica e Occupazione
        # Mother's qualification - usa mapping se disponibile
        if "Mother's qualification" in mappings:
            student_data["Mother's qualification"] = mappings["Mother's qualification"][
                st.selectbox("Qualifica madre", list(mappings["Mother's qualification"].keys()), key="mother_qual")
            ]
        else:
            student_data["Mother's qualification"] = st.number_input(
                "Qualifica madre (1-44)", min_value=1, max_value=44, value=1
            )

        # Mother's occupation - usa mapping se disponibile
        if "Mother's occupation" in mappings:
            student_data["Mother's occupation"] = mappings["Mother's occupation"][
                st.selectbox("Occupazione madre", list(mappings["Mother's occupation"].keys()), key="mother_occ")
            ]
        else:
            student_data["Mother's occupation"] = st.number_input(
                "Occupazione madre (0-195)", min_value=0, max_value=195, value=0
            )

    with col2:
        # PADRE - Qualifica e Occupazione
        # Father's qualification - usa mapping se disponibile
        if "Father's qualification" in mappings:
            student_data["Father's qualification"] = mappings["Father's qualification"][
                st.selectbox("Qualifica padre", list(mappings["Father's qualification"].keys()), key="father_qual")
            ]
        else:
            student_data["Father's qualification"] = st.number_input(
                "Qualifica padre (1-44)", min_value=1, max_value=44, value=1
            )

        # Father's occupation - usa mapping se disponibile
        if "Father's occupation" in mappings:
            student_data["Father's occupation"] = mappings["Father's occupation"][
                st.selectbox("Occupazione padre", list(mappings["Father's occupation"].keys()), key="father_occ")
            ]
        else:
            student_data["Father's occupation"] = st.number_input(
                "Occupazione padre (0-195)", min_value=0, max_value=195, value=0
            )

    with col3:
        student_data['Admission grade'] = st.number_input(
            "Voto ammissione",
            min_value=0.0, max_value=200.0, value=120.0, step=0.1
        )

        student_data['Previous qualification (grade)'] = st.number_input(
            "Voto qualifica precedente",
            min_value=0.0, max_value=200.0, value=120.0, step=0.1
        )

# ===== TAB 3: PERFORMANCE UNIVERSITARIA (SOLO MODALITÀ STANDARD) =====
if not preadmission_mode:
    with st.expander("Performance Universitaria (1° Semestre)", expanded=False):
        st.warning("Questi dati sono disponibili solo DOPO l'inizio dei corsi")

        col1, col2, col3 = st.columns(3)

        with col1:
            student_data['Curricular units 1st sem (credited)'] = st.number_input(
                "Crediti riconosciuti 1° sem", min_value=0, max_value=30, value=0
            )

            student_data['Curricular units 1st sem (enrolled)'] = st.number_input(
                "Iscritto a (1° sem)", min_value=0, max_value=30, value=6
            )

        with col2:
            student_data['Curricular units 1st sem (evaluations)'] = st.number_input(
                "Valutazioni 1° sem", min_value=0, max_value=50, value=6
            )

            student_data['Curricular units 1st sem (approved)'] = st.number_input(
                "Approvati 1° sem", min_value=0, max_value=30, value=5
            )

        with col3:
            student_data['Curricular units 1st sem (grade)'] = st.number_input(
                "Voto medio 1° sem",
                min_value=0.0, max_value=20.0, value=12.0, step=0.1
            )

            student_data['Curricular units 1st sem (without evaluations)'] = st.number_input(
                "Senza valutazione 1° sem", min_value=0, max_value=30, value=0
            )

    with st.expander("Performance Universitaria (2° Semestre)", expanded=False):
        st.warning("Questi dati sono disponibili solo DOPO l'inizio dei corsi")

        col1, col2, col3 = st.columns(3)

        with col1:
            student_data['Curricular units 2nd sem (credited)'] = st.number_input(
                "Crediti riconosciuti 2° sem", min_value=0, max_value=30, value=0
            )

            student_data['Curricular units 2nd sem (enrolled)'] = st.number_input(
                "Iscritto a (2° sem)", min_value=0, max_value=30, value=6
            )

        with col2:
            student_data['Curricular units 2nd sem (evaluations)'] = st.number_input(
                "Valutazioni 2° sem", min_value=0, max_value=50, value=6
            )

            student_data['Curricular units 2nd sem (approved)'] = st.number_input(
                "Approvati 2° sem", min_value=0, max_value=30, value=5
            )

        with col3:
            student_data['Curricular units 2nd sem (grade)'] = st.number_input(
                "Voto medio 2° sem",
                min_value=0.0, max_value=20.0, value=12.0, step=0.1
            )

            student_data['Curricular units 2nd sem (without evaluations)'] = st.number_input(
                "Senza valutazione 2° sem", min_value=0, max_value=30, value=0
            )
else:
    # In modalità pre-immatricolazione, queste variabili NON esistono
    # Non vengono inserite nel dizionario student_data
    pass

# ===== TAB 4: SITUAZIONE FINANZIARIA =====
with st.expander("Situazione Finanziaria", expanded=False):
    col1, col2, col3 = st.columns(3)

    with col1:
        student_data['Tuition fees up to date'] = 1 if st.selectbox(
            "Tasse aggiornate", ["Sì", "No"]
        ) == "Sì" else 0

        student_data['Scholarship holder'] = 1 if st.selectbox(
            "Borsa di studio", ["No", "Sì"]
        ) == "Sì" else 0

    with col2:
        student_data['Debtor'] = 1 if st.selectbox(
            "Debitore", ["No", "Sì"]
        ) == "Sì" else 0

        student_data['Displaced'] = 1 if st.selectbox(
            "Fuori sede", ["No", "Sì"]
        ) == "Sì" else 0

    with col3:
        student_data['Educational special needs'] = 1 if st.selectbox(
            "Bisogni speciali", ["No", "Sì"]
        ) == "Sì" else 0

# ===== INDICATORI ECONOMICI (AUTOMATICI) =====
# Valori macroeconomici più recenti per Portogallo (2024-2025)
st.info("""
**Indicatori Economici (Automatici)**  
I seguenti valori macroeconomici vengono impostati automaticamente con i dati più recenti:
- **Tasso di Disoccupazione**: 6.5% (Portogallo, Q4 2024)
- **Tasso di Inflazione**: 2.3% (Portogallo, Nov 2024)
- **Crescita PIL**: 2.1% (Portogallo, previsione 2024)

*Fonte: PORDATA, Banco de Portugal, INE*
""")

# Imposta automaticamente i valori economici
student_data['Unemployment rate'] = 6.5   # Portogallo Q4 2024
student_data['Inflation rate'] = 2.3      # Portogallo Nov 2024
student_data['GDP'] = 2.1                 # Portogallo 2024

st.markdown("---")

# ============================================================================
# PREDIZIONE
# ============================================================================

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button(
        "PREDICI RISULTATO",
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
            st.markdown("### Confronto Modelli")

            col1, col2 = st.columns(2)

            # Random Forest
            with col1:
                st.markdown("#### Random Forest")
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
                st.markdown("#### XGBoost")
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
                st.success(f"**ACCORDO**: Entrambi i modelli predicono **{rf_pred}**")
            else:
                st.warning(f"**DISACCORDO**: RF predice **{rf_pred}**, XGB predice **{xgb_pred}**")

        else:
            # SINGOLO MODELLO
            st.markdown("### Risultato Predizione")

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
            st.markdown("#### Probabilità per Classe")

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
                colors = {'Dropout': '#c62828', 'Enrolled': '#ef6c00', 'Graduate': '#2e7d32'}
                bar_colors = [colors[c] for c in classes]

                ax.barh(classes, probabilities * 100, color=bar_colors)
                ax.set_xlabel('Probabilità (%)')
                ax.set_xlim(0, 100)
                for i, (cls, prob) in enumerate(zip(classes, probabilities * 100)):
                    ax.text(prob + 2, i, f'{prob:.1f}%', va='center')

                st.pyplot(fig)

            # Interpretazione
            st.markdown("---")
            st.markdown("#### Interpretazione")

            if preadmission_mode:
                # Interpretazione specifica per pre-immatricolazione
                if prediction == "Dropout":
                    st.error(f"""
                    ** STUDENTE AD ALTO RISCHIO (Predizione Preventiva)**
                    
                    Confidenza: {confidence*100:.1f}%
                    
                    **Raccomandazioni Pre-Iscrizione:**
                    - **Screening approfondito** prima dell'immatricolazione
                    - **Counseling orientativo** per valutare scelta corso
                    - **Colloquio con famiglia** per valutare supporto
                    - **Proposta tutoring proattivo** dal primo giorno
                    - **Valutare supporto finanziario** se necessario
                    
                    Nota: Predizione basata solo su dati pre-universitari (accuracy ~60%)
                    """)
                elif prediction == "Enrolled":
                    st.warning(f"""
                    **STUDENTE A RISCHIO MEDIO (Predizione Preventiva)**
                    
                    Confidenza: {confidence*100:.1f}%
                    
                    **Raccomandazioni Pre-Iscrizione:**
                    - **Monitoraggio attento** dall'inizio
                    - **Assegnazione tutor** facoltativa
                    - **Presentazione servizi di supporto** disponibili
                    - **Check-in periodici** durante primo semestre
                    
                    Nota: Predizione basata solo su dati pre-universitari (accuracy ~60%)
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
                    
                    Nota: Predizione basata solo su dati pre-universitari (accuracy ~60%)
                    """)
            else:
                # Interpretazione standard (post-iscrizione)
                if prediction == "Dropout":
                    st.error(f"""
                    **STUDENTE AD ALTO RISCHIO**
                    
                    Confidenza: {confidence*100:.1f}%
                    
                    **Raccomandazioni:**
                    - Intervento immediato necessario
                    - Supporto accademico e tutoraggio
                    - Verifica situazione finanziaria
                    - Incontri regolari con tutor
                    """)
                elif prediction == "Enrolled":
                    st.warning(f"""
                    **STUDENTE A RISCHIO MEDIO**
                    
                    Confidenza: {confidence*100:.1f}%
                    
                    **Raccomandazioni:**
                    - Monitoraggio costante
                    - Supporto su richiesta
                    - Verificare eventuali difficoltà
                    - Incoraggiare partecipazione
                    """)
                else:
                    st.success(f"""
                    **STUDENTE A BASSO RISCHIO**
                    
                    Confidenza: {confidence*100:.1f}%
                    
                    **Raccomandazioni:**
                    - Mantenere supporto attuale
                    - Offrire opportunità di eccellenza
                    - Possibili ruoli di mentoring
                    - Continuare così!
                    """)

    except Exception as e:
        st.error(f"Errore durante la predizione: {e}")
        st.exception(e)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")

if preadmission_mode:
    st.caption("Student Dropout Predictor | Modalità: PRE-IMMATRICOLAZIONE | Modelli: Random Forest + XGBoost | Dataset: UCI ML #697")
else:
    st.caption("Student Dropout Predictor | Modelli: Random Forest + XGBoost | Dataset: UCI ML #697")