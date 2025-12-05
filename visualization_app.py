"""
Streamlit Visualization & Simulation Application for Student Performance Prediction.
Loads trained models and provides visualization and prediction capabilities.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import warnings
import pickle
import json

warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data import FeaturePreprocessor

# Page configuration
st.set_page_config(
    page_title="Student Performance - Visualizzazione",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Models directory
MODELS_DIR = Path("trained_models")

# Class names
CLASS_NAMES_LIST = ["Dropout", "Enrolled", "Graduate"]


def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'model_loaded': False,
        'model': None,
        'metadata': None,
        'preprocessor_data': None,
        'available_experiments': []
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_available_experiments():
    """Get list of available trained model experiments."""
    if not MODELS_DIR.exists():
        return []

    experiments = []
    for exp_dir in MODELS_DIR.iterdir():
        if exp_dir.is_dir():
            metadata_path = exp_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                experiments.append({
                    'path': exp_dir,
                    'name': metadata.get('experiment_name', exp_dir.name),
                    'timestamp': metadata.get('timestamp', 'Unknown'),
                    'best_model': metadata.get('best_model_name', 'Unknown'),
                    'f1_macro': metadata.get('test_f1_macro', 0),
                    'accuracy': metadata.get('test_accuracy', 0)
                })

    return sorted(experiments, key=lambda x: x['timestamp'], reverse=True)


def load_experiment(exp_path):
    """Load a trained experiment."""
    exp_path = Path(exp_path)

    # Load metadata
    with open(exp_path / "metadata.json", 'r') as f:
        metadata = json.load(f)

    # Load model
    with open(exp_path / "best_model.pkl", 'rb') as f:
        model = pickle.load(f)

    # Load preprocessor data
    with open(exp_path / "preprocessor.pkl", 'rb') as f:
        preprocessor_data = pickle.load(f)

    st.session_state.model = model
    st.session_state.metadata = metadata
    st.session_state.preprocessor_data = preprocessor_data
    st.session_state.model_loaded = True

    return metadata


def plot_confusion_matrix(cm, class_names, title="Matrice di Confusione"):
    """Plot confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )

    ax.set_xlabel('Etichetta Predetta', fontsize=12)
    ax.set_ylabel('Etichetta Vera', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig


def plot_model_comparison(comparison_data, metric='F1 (Macro)'):
    """Plot model comparison from metadata."""
    df = pd.DataFrame(comparison_data)

    fig, ax = plt.subplots(figsize=(12, 6))

    df_sorted = df.sort_values(metric, ascending=True)
    colors = sns.color_palette("viridis", len(df_sorted))

    bars = ax.barh(df_sorted['Model'], df_sorted[metric], color=colors, edgecolor='black')

    for bar, val in zip(bars, df_sorted[metric]):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=10)

    ax.set_xlabel(metric, fontsize=12)
    ax.set_title(f'Confronto Modelli - {metric}', fontsize=14, fontweight='bold')
    ax.set_xlim(0, df_sorted[metric].max() * 1.15)

    plt.tight_layout()
    return fig


def create_feature_vector_from_input(user_inputs, feature_names, preprocessor_data):
    """
    Create a properly formatted feature vector from user inputs.
    Maps user inputs to the correct feature positions and applies preprocessing.
    """
    # Create a DataFrame with all features initialized to median/mode values
    X_template = preprocessor_data['X_train'].copy()

    # Get median values for numerical features and mode for categorical
    feature_values = {}
    for col in X_template.columns:
        if X_template[col].dtype in ['int64', 'float64']:
            feature_values[col] = X_template[col].median()
        else:
            feature_values[col] = X_template[col].mode()[0] if len(X_template[col].mode()) > 0 else 0

    # Update with user inputs (mapping common feature names)
    feature_mapping = {
        'age': 'Age at enrollment',
        'marital_status': 'Marital status',
        'admission_grade': 'Admission grade',
        'prev_qualification_grade': 'Previous qualification (grade)',
        'scholarship': 'Scholarship holder',
        'debtor': 'Debtor',
        'tuition_up_to_date': 'Tuition fees up to date',
        'gender': 'Gender',
        'units_1st_sem': 'Curricular units 1st sem (approved)',
        'units_2nd_sem': 'Curricular units 2nd sem (approved)'
    }

    for input_key, feature_name in feature_mapping.items():
        if input_key in user_inputs:
            # Find the actual column name (case-insensitive partial match)
            matching_cols = [col for col in X_template.columns
                           if feature_name.lower() in col.lower()]
            if matching_cols:
                feature_values[matching_cols[0]] = user_inputs[input_key]

    # Create DataFrame with single row
    X_input = pd.DataFrame([feature_values])

    # Apply the same preprocessing as during training
    preprocessor = FeaturePreprocessor(
        scaling=preprocessor_data.get('scaling', 'standard')
    )

    # Fit on training data and transform input
    preprocessor.fit(X_template)
    X_scaled = preprocessor.transform(X_input)

    return X_scaled.values


def main():
    init_session_state()

    # Header
    st.markdown('<p class="main-header">📊 Visualizzazione & Simulazione - Student Performance</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Analisi risultati e predizioni su nuovi studenti</p>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/graph.png", width=80)
        st.title("Carica Modello")

        # Get available experiments
        experiments = get_available_experiments()
        st.session_state.available_experiments = experiments

        if len(experiments) == 0:
            st.warning("⚠️ Nessun modello trovato. Addestra prima un modello usando l'app di training.")
        else:
            st.success(f"✅ {len(experiments)} modelli disponibili")

            # Select experiment
            exp_options = [
                f"{exp['name']} - {exp['timestamp']} (F1: {exp['f1_macro']:.4f})"
                for exp in experiments
            ]

            selected_idx = st.selectbox(
                "Seleziona esperimento:",
                range(len(exp_options)),
                format_func=lambda i: exp_options[i]
            )

            if st.button("Carica Modello", type="primary"):
                with st.spinner("Caricamento modello..."):
                    metadata = load_experiment(experiments[selected_idx]['path'])
                st.success(f"✅ Modello caricato: **{metadata['best_model_name']}**")

        st.markdown("---")

        if st.session_state.model_loaded:
            st.markdown("### Modello Caricato")
            st.info(f"""
            **Nome**: {st.session_state.metadata['best_model_name']}
            
            **Esperimento**: {st.session_state.metadata['experiment_name']}
            
            **Accuracy**: {st.session_state.metadata['test_accuracy']:.4f}
            
            **F1 Macro**: {st.session_state.metadata['test_f1_macro']:.4f}
            """)

    if not st.session_state.model_loaded:
        st.info("👈 Carica un modello dalla sidebar per iniziare")

        # Show available models table
        if len(st.session_state.available_experiments) > 0:
            st.markdown("---")
            st.subheader("Modelli Disponibili")

            experiments_df = pd.DataFrame([
                {
                    'Nome': exp['name'],
                    'Timestamp': exp['timestamp'],
                    'Modello': exp['best_model'],
                    'F1 Macro': f"{exp['f1_macro']:.4f}",
                    'Accuracy': f"{exp['accuracy']:.4f}"
                }
                for exp in st.session_state.available_experiments
            ])

            st.dataframe(experiments_df, use_container_width=True)

        st.stop()

    # Main tabs
    tab1, tab2, tab3 = st.tabs([
        "📈 Risultati Modello",
        "🔮 Simulatore Predizioni",
        "📋 Info Modello"
    ])

    # TAB 1: Model Results
    with tab1:
        st.header("📈 Risultati e Performance del Modello")

        metadata = st.session_state.metadata

        # Key metrics
        st.subheader("Metriche Principali")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Modello", metadata['best_model_name'])
        with col2:
            st.metric("Test Accuracy", f"{metadata['test_accuracy']:.4f}")
        with col3:
            st.metric("F1 Macro", f"{metadata['test_f1_macro']:.4f}")
        with col4:
            st.metric("Features", len(metadata['feature_names']))

        st.markdown("---")

        # Model comparison
        if 'comparison_results' in metadata:
            st.subheader("Confronto con Altri Modelli")

            metric_choice = st.selectbox(
                "Seleziona metrica:",
                ["F1 (Macro)", "Accuracy", "F1 (Weighted)"]
            )

            fig = plot_model_comparison(metadata['comparison_results'], metric_choice)
            st.pyplot(fig)
            plt.close()

            st.markdown("---")

            st.subheader("Tabella Risultati Completi")
            comparison_df = pd.DataFrame(metadata['comparison_results'])
            st.dataframe(comparison_df, use_container_width=True)

    # TAB 2: Prediction Simulator
    with tab2:
        st.header("🔮 Simulatore di Predizioni")

        st.info(f"Usando il modello **{metadata['best_model_name']}** per le predizioni")

        st.subheader("Inserisci Dati Studente")

        # Create input form with common features
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Dati Anagrafici**")
                age = st.number_input("Età all'Iscrizione", min_value=17, max_value=70, value=20)

                marital_status = st.selectbox(
                    "Stato Civile",
                    [1, 2, 3, 4, 5, 6],
                    format_func=lambda x: {
                        1: "Single",
                        2: "Sposato",
                        3: "Vedovo",
                        4: "Divorziato",
                        5: "Unione di fatto",
                        6: "Separato legalmente"
                    }[x]
                )

                gender = st.selectbox("Genere", [0, 1], format_func=lambda x: "Femmina" if x == 0 else "Maschio")

            with col2:
                st.markdown("**Dati Accademici**")
                admission_grade = st.slider("Voto Ammissione", 0.0, 200.0, 120.0)
                prev_qualification_grade = st.slider("Voto Qualifica Precedente", 0.0, 200.0, 120.0)

                scholarship = st.selectbox(
                    "Borsa di Studio",
                    [0, 1],
                    format_func=lambda x: "Sì" if x == 1 else "No"
                )

            with col3:
                st.markdown("**Dati Finanziari**")
                debtor = st.selectbox(
                    "Debitore",
                    [0, 1],
                    format_func=lambda x: "Sì" if x == 1 else "No"
                )

                tuition_up_to_date = st.selectbox(
                    "Tasse Pagate",
                    [0, 1],
                    format_func=lambda x: "Sì" if x == 1 else "No"
                )

                units_1st_sem = st.number_input("Unità 1° Sem Approvate", min_value=0, max_value=30, value=5)
                units_2nd_sem = st.number_input("Unità 2° Sem Approvate", min_value=0, max_value=30, value=5)

            submitted = st.form_submit_button("🔮 Predici Risultato", type="primary")

        if submitted:
            st.markdown("---")

            # Collect user inputs
            user_inputs = {
                'age': age,
                'marital_status': marital_status,
                'gender': gender,
                'admission_grade': admission_grade,
                'prev_qualification_grade': prev_qualification_grade,
                'scholarship': scholarship,
                'debtor': debtor,
                'tuition_up_to_date': tuition_up_to_date,
                'units_1st_sem': units_1st_sem,
                'units_2nd_sem': units_2nd_sem
            }

            # Make prediction
            with st.spinner("Esecuzione predizione..."):
                try:
                    # Create feature vector from user inputs
                    sample_features = create_feature_vector_from_input(
                        user_inputs,
                        metadata['feature_names'],
                        st.session_state.preprocessor_data
                    )

                    prediction = st.session_state.model.predict(sample_features)
                    probabilities = st.session_state.model.predict_proba(sample_features)

                    predicted_class = CLASS_NAMES_LIST[prediction[0]]

                    st.subheader("Risultato Predizione")

                    # Show input summary
                    with st.expander("📋 Riepilogo Input"):
                        input_df = pd.DataFrame([{
                            'Parametro': k.replace('_', ' ').title(),
                            'Valore': v
                        } for k, v in user_inputs.items()])
                        st.dataframe(input_df, use_container_width=True)

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Risultato Predetto", predicted_class)

                    with col2:
                        max_prob = max(probabilities[0]) * 100
                        st.metric("Confidenza", f"{max_prob:.1f}%")

                    with col3:
                        risk_level = "Alto" if predicted_class == "Dropout" else ("Medio" if predicted_class == "Enrolled" else "Basso")
                        risk_color = "🔴" if predicted_class == "Dropout" else ("🟡" if predicted_class == "Enrolled" else "🟢")
                        st.metric("Livello di Rischio", f"{risk_color} {risk_level}")

                    # Probability distribution
                    st.subheader("Distribuzione Probabilità")

                    prob_df = pd.DataFrame({
                        'Classe': CLASS_NAMES_LIST,
                        'Probabilità': probabilities[0]
                    })

                    fig, ax = plt.subplots(figsize=(10, 5))
                    colors = ['#ff6b6b', '#feca57', '#48dbfb']
                    bars = ax.bar(prob_df['Classe'], prob_df['Probabilità'], color=colors, edgecolor='black')

                    for bar, prob in zip(bars, prob_df['Probabilità']):
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.01,
                            f'{prob*100:.1f}%',
                            ha='center', va='bottom', fontsize=11, fontweight='bold'
                        )

                    ax.set_ylabel('Probabilità', fontsize=12)
                    ax.set_title('Probabilità di Predizione per Classe', fontsize=14, fontweight='bold')
                    ax.set_ylim(0, 1.1)
                    ax.grid(axis='y', alpha=0.3)

                    st.pyplot(fig)
                    plt.close()

                    # Show probability table
                    st.dataframe(
                        prob_df.style.format({'Probabilità': '{:.2%}'})
                        .background_gradient(cmap='RdYlGn', subset=['Probabilità']),
                        use_container_width=True
                    )

                    # Recommendations
                    st.subheader("Raccomandazioni")

                    if predicted_class == "Dropout":
                        st.error("""
                        ⚠️ **Alto Rischio di Abbandono**
                        
                        Interventi raccomandati:
                        - Fissare incontro urgente con tutor accademico
                        - Valutare opzioni di aiuto finanziario
                        - Fornire supporto di tutoraggio
                        - Monitorare attentamente la frequenza
                        - Considerare programma di mentoring
                        """)
                    elif predicted_class == "Enrolled":
                        st.warning("""
                        ⚡ **Rischio Moderato - Ancora Iscritto**
                        
                        Interventi raccomandati:
                        - Check-in regolari con lo studente
                        - Incoraggiare partecipazione a gruppi studio
                        - Monitorare i progressi accademici
                        - Offrire sessioni di orientamento
                        """)
                    else:
                        st.success("""
                        ✅ **Basso Rischio - Probabile Laurea**
                        
                        Lo studente mostra indicatori positivi per il completamento con successo.
                        Continuare a fornire supporto accademico standard e incoraggiamento.
                        """)

                except Exception as e:
                    st.error(f"Errore durante la predizione: {str(e)}")

                    # Debug information
                    with st.expander("🔍 Informazioni di Debug"):
                        st.write("**Errore completo:**")
                        st.code(str(e))
                        st.write("**Features disponibili:**")
                        st.write(metadata.get('feature_names', [])[:10])
                        st.write("**Shape atteso:**", len(metadata.get('feature_names', [])))

                    st.info("""
                    **Suggerimento**: Assicurati che i nomi delle features nel CSV di training
                    corrispondano a quelli utilizzati nel simulatore.
                    """)

    # TAB 3: Model Info
    with tab3:
        st.header("📋 Informazioni sul Modello")

        metadata = st.session_state.metadata

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Configurazione Training")

            if 'training_config' in metadata:
                config = metadata['training_config']
                st.json(config)
            else:
                st.info("Configurazione non disponibile")

        with col2:
            st.subheader("Classi Target")

            class_names_dict = metadata.get('class_names', {})
            for idx, name in class_names_dict.items():
                st.write(f"**{idx}**: {name}")

        st.markdown("---")

        st.subheader("Features Utilizzate")

        features = metadata.get('feature_names', [])

        # Display in columns
        n_cols = 3
        cols = st.columns(n_cols)

        for idx, feature in enumerate(features):
            col_idx = idx % n_cols
            with cols[col_idx]:
                st.write(f"- {feature}")

        st.markdown("---")

        st.subheader("Dettagli Esperimento")

        exp_details = {
            'Nome Esperimento': metadata.get('experiment_name', 'N/A'),
            'Timestamp': metadata.get('timestamp', 'N/A'),
            'Modello Migliore': metadata.get('best_model_name', 'N/A'),
            'Numero Features': len(metadata.get('feature_names', [])),
            'Numero Classi': len(metadata.get('class_names', {}))
        }

        st.json(exp_details)


if __name__ == "__main__":
    main()