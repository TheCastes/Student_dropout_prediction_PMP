"""
Streamlit Training Application for Student Performance Prediction System.
Handles data loading, preprocessing, model training, and saving trained models.
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
from datetime import datetime

warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data import (
    UCIDataLoader,
    DataValidator,
    FeaturePreprocessor,
    LabelEncoderWrapper,
    DataSplitter,
    DataPipeline,
    SamplingManager,
    SamplerFactory
)
from src.models import (
    ModelFactory,
    ModelTrainer,
    ModelEvaluator,
    ExperimentRunner,
    LogisticRegressionModel,
    SVMModel,
    DecisionTreeModel,
    RandomForestModel,
    GradientBoostingModel,
    XGBoostModel,
    CatBoostModel,
    LogitBoostModel
)

# Page configuration
st.set_page_config(
    page_title="Student Performance - Training",
    page_icon="🎓",
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
</style>
""", unsafe_allow_html=True)

# Class names mapping
CLASS_NAMES = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}
CLASS_NAMES_LIST = ["Dropout", "Enrolled", "Graduate"]

# Models directory
MODELS_DIR = Path("trained_models")
MODELS_DIR.mkdir(exist_ok=True)


def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'data_loaded': False,
        'X': None,
        'y': None,
        'processed_data': None,
        'X_train_sampled': None,
        'y_train_sampled': None,
        'models_trained': False,
        'comparison_df': None,
        'evaluation_results': None,
        'best_model': None,
        'best_model_name': None,
        'training_config': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_data(csv_file=None):
    """Load data from UCI or uploaded CSV."""
    with st.spinner("Caricamento dati..."):
        if csv_file is not None:
            df = pd.read_csv(csv_file, delimiter=';')
            if 'Target' in df.columns:
                y = df['Target']
                X = df.drop('Target', axis=1)
            else:
                y = df.iloc[:, -1]
                X = df.iloc[:, :-1]
        else:
            loader = UCIDataLoader(dataset_id=697)
            X, y = loader.load()
        
        st.session_state.X = X
        st.session_state.y = y
        st.session_state.data_loaded = True
        
    return X, y


def preprocess_data(test_size, random_state, scaling):
    """Preprocess the data."""
    with st.spinner("Preprocessing dei dati..."):
        preprocessor = FeaturePreprocessor(scaling=scaling)
        label_encoder = LabelEncoderWrapper()
        splitter = DataSplitter(test_size=test_size, random_state=random_state)
        
        pipeline = DataPipeline(preprocessor, label_encoder, splitter)
        processed = pipeline.run(st.session_state.X, st.session_state.y)
        
        st.session_state.processed_data = processed
        
    return processed


def apply_sampling(sampler_type, random_state):
    """Apply sampling to balance classes."""
    with st.spinner(f"Applicazione sampling {sampler_type.upper()}..."):
        sampling_manager = SamplingManager(random_state=random_state)
        
        X_train = st.session_state.processed_data['X_train_scaled'].values
        y_train = st.session_state.processed_data['y_train'].values
        
        X_resampled, y_resampled = sampling_manager.apply_sampling(
            X_train, y_train, sampler_type
        )
        
        st.session_state.X_train_sampled = X_resampled
        st.session_state.y_train_sampled = y_resampled
        
    return X_resampled, y_resampled


def train_models(selected_models, tune_hyperparameters, cv_folds, random_state):
    """Train selected models."""
    models = {}
    
    model_mapping = {
        "Logistic Regression": LogisticRegressionModel,
        "SVM": SVMModel,
        "Decision Tree": DecisionTreeModel,
        "Random Forest": RandomForestModel,
        "Gradient Boosting": GradientBoostingModel,
        "XGBoost": XGBoostModel,
        "CatBoost": CatBoostModel,
        "LogitBoost": LogitBoostModel
    }
    
    for model_name in selected_models:
        if model_name in model_mapping:
            models[model_name] = model_mapping[model_name](random_state=random_state)
    
    trainer = ModelTrainer(
        cv_folds=cv_folds,
        random_state=random_state,
        scoring='f1_macro'
    )
    evaluator = ModelEvaluator(CLASS_NAMES)
    runner = ExperimentRunner(trainer, evaluator, CLASS_NAMES)
    
    X_test = st.session_state.processed_data['X_test_scaled'].values
    y_test = st.session_state.processed_data['y_test'].values
    
    comparison_df, evaluation_results = runner.run_experiment(
        st.session_state.X_train_sampled,
        st.session_state.y_train_sampled,
        X_test,
        y_test,
        models,
        tune_hyperparameters
    )
    
    best_name, best_model = runner.get_best_model()
    
    st.session_state.comparison_df = comparison_df
    st.session_state.evaluation_results = evaluation_results
    st.session_state.best_model = best_model
    st.session_state.best_model_name = best_name
    st.session_state.models_trained = True
    
    return comparison_df, evaluation_results


def save_trained_models(experiment_name):
    """Save trained models and metadata to disk."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = MODELS_DIR / f"{experiment_name}_{timestamp}"
    experiment_dir.mkdir(exist_ok=True)
    
    # Save best model
    best_model_path = experiment_dir / "best_model.pkl"
    with open(best_model_path, 'wb') as f:
        pickle.dump(st.session_state.best_model, f)
    
    # Save preprocessor data (include X_train for feature reconstruction)
    preprocessor_data = {
        'X_train': st.session_state.processed_data['X_train'],
        'X_train_scaled': st.session_state.processed_data['X_train_scaled'],
        'feature_names': st.session_state.processed_data['feature_names'],
        'classes': st.session_state.processed_data['classes'],
        'class_mapping': st.session_state.processed_data['class_mapping'],
        'scaling': st.session_state.training_config.get('scaling', 'standard')
    }

    preprocessor_path = experiment_dir / "preprocessor.pkl"
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(preprocessor_data, f)

    # Save metadata
    metadata = {
        'experiment_name': experiment_name,
        'timestamp': timestamp,
        'best_model_name': st.session_state.best_model_name,
        'class_names': CLASS_NAMES,
        'feature_names': st.session_state.processed_data['feature_names'],
        'training_config': st.session_state.training_config,
        'comparison_results': st.session_state.comparison_df.to_dict('records'),
        'test_accuracy': float(st.session_state.evaluation_results[st.session_state.best_model_name]['accuracy']),
        'test_f1_macro': float(st.session_state.evaluation_results[st.session_state.best_model_name]['f1_macro']),
        'num_features': len(st.session_state.processed_data['feature_names']),
        'num_training_samples': len(st.session_state.y_train_sampled)
    }

    metadata_path = experiment_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return experiment_dir


def plot_class_distribution(y, title="Distribuzione Classi"):
    """Plot class distribution."""
    fig, ax = plt.subplots(figsize=(10, 6))

    counts = y.value_counts()
    colors = sns.color_palette("Set2", len(counts))

    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor='black')

    for bar, count in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            f'{count}\n({count/len(y)*100:.1f}%)',
            ha='center', va='bottom', fontsize=10
        )

    ax.set_xlabel('Classe', fontsize=12)
    ax.set_ylabel('Numero di Campioni', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig


def plot_model_comparison(df, metric='F1 (Macro)'):
    """Plot model comparison."""
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


def main():
    init_session_state()

    # Header
    st.markdown('<p class="main-header">🎓 Training Modelli - Student Performance</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Addestramento e salvataggio modelli predittivi</p>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=80)
        st.title("Configurazione Training")

        st.markdown("---")
        st.markdown("### Informazioni")
        st.info("""
        Questa applicazione permette di:
        - Caricare e preprocessare i dati
        - Addestrare multipli modelli
        - Salvare i modelli addestrati
        - Visualizzare i risultati
        """)

    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Caricamento Dati",
        "⚙️ Preprocessing",
        "🤖 Training Modelli",
        "📈 Risultati",
        "💾 Salvataggio"
    ])

    # TAB 1: Data Loading
    with tab1:
        st.header("📊 Caricamento Dati")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Sorgente Dati")

            data_source = st.radio(
                "Seleziona sorgente:",
                ["UCI Repository (o Sample Data)", "Carica CSV"]
            )

            if data_source == "Carica CSV":
                uploaded_file = st.file_uploader(
                    "Carica file CSV",
                    type=['csv'],
                    help="Carica un CSV con i dati degli studenti"
                )

                if uploaded_file is not None:
                    if st.button("Carica Dati CSV", type="primary"):
                        X, y = load_data(uploaded_file)
                        st.success(f"✅ Dati caricati! {len(X)} campioni, {len(X.columns)} features")
            else:
                if st.button("Carica da UCI Repository", type="primary"):
                    X, y = load_data()
                    st.success(f"✅ Dati caricati! {len(X)} campioni, {len(X.columns)} features")

        with col2:
            st.subheader("Statistiche Rapide")
            if st.session_state.data_loaded:
                st.metric("Campioni Totali", len(st.session_state.X))
                st.metric("Features", len(st.session_state.X.columns))
                st.metric("Classi", len(st.session_state.y.unique()))

        if st.session_state.data_loaded:
            st.markdown("---")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Anteprima Dati")
                st.dataframe(st.session_state.X.head(10), use_container_width=True)

            with col2:
                st.subheader("Distribuzione Classi")
                fig = plot_class_distribution(st.session_state.y)
                st.pyplot(fig)
                plt.close()

    # TAB 2: Preprocessing
    with tab2:
        st.header("⚙️ Preprocessing Dati")

        if not st.session_state.data_loaded:
            st.warning("⚠️ Carica prima i dati nella tab 'Caricamento Dati'")
            st.stop()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Impostazioni Preprocessing")

            test_size = st.slider(
                "Dimensione Test Set",
                min_value=0.1,
                max_value=0.4,
                value=0.2,
                step=0.05
            )

            random_state = st.number_input(
                "Random State",
                min_value=0,
                max_value=9999,
                value=42
            )

            scaling = st.selectbox(
                "Metodo Scaling",
                ["standard", "minmax", "none"]
            )

        with col2:
            st.subheader("Impostazioni Sampling")

            sampler_type = st.selectbox(
                "Tecnica Sampling",
                ["smote", "adasyn", "none"]
            )

            st.info("""
            **SMOTE**: Crea campioni sintetici per classi minoritarie
            
            **ADASYN**: Simile a SMOTE, focalizzato su esempi difficili
            """)

        if st.button("Esegui Preprocessing", type="primary"):
            processed = preprocess_data(test_size, random_state, scaling)

            if sampler_type != "none":
                X_resampled, y_resampled = apply_sampling(sampler_type, random_state)
            else:
                st.session_state.X_train_sampled = processed['X_train_scaled'].values
                st.session_state.y_train_sampled = processed['y_train'].values

            # Save config
            st.session_state.training_config = {
                'test_size': test_size,
                'random_state': random_state,
                'scaling': scaling,
                'sampler_type': sampler_type
            }

            st.success("✅ Preprocessing completato!")

        if st.session_state.processed_data is not None:
            st.markdown("---")
            st.subheader("Risultati Preprocessing")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Training (Originale)",
                         len(st.session_state.processed_data['y_train']))
            with col2:
                st.metric("Training (Dopo Sampling)",
                         len(st.session_state.y_train_sampled) if st.session_state.y_train_sampled is not None else "N/A")
            with col3:
                st.metric("Test Samples",
                         len(st.session_state.processed_data['y_test']))
            with col4:
                st.metric("Features",
                         len(st.session_state.processed_data['feature_names']))

    # TAB 3: Model Training
    with tab3:
        st.header("🤖 Training Modelli")

        if st.session_state.X_train_sampled is None:
            st.warning("⚠️ Completa prima il preprocessing")
            st.stop()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Modelli Standard")
            standard_models = st.multiselect(
                "Seleziona modelli standard:",
                ["Logistic Regression", "SVM", "Decision Tree", "Random Forest"],
                default=["Logistic Regression", "Decision Tree", "Random Forest"]
            )

        with col2:
            st.subheader("Modelli Boosting")
            boosting_models = st.multiselect(
                "Seleziona modelli boosting:",
                ["Gradient Boosting", "XGBoost", "CatBoost", "LogitBoost"],
                default=["Gradient Boosting", "XGBoost"]
            )

        selected_models = standard_models + boosting_models

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            cv_folds = st.slider(
                "Fold Cross-Validation",
                min_value=2,
                max_value=10,
                value=5
            )

        with col2:
            tune_hyperparameters = st.checkbox(
                "Tuning Hyperparametri",
                value=False,
                help="Abilita tuning (richiede più tempo)"
            )

        if st.button("Avvia Training", type="primary", disabled=len(selected_models) == 0):
            if len(selected_models) == 0:
                st.error("Seleziona almeno un modello.")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()

                with st.spinner("Training in corso... Questo può richiedere alcuni minuti."):
                    comparison_df, evaluation_results = train_models(
                        selected_models,
                        tune_hyperparameters,
                        cv_folds,
                        random_state=st.session_state.training_config['random_state']
                    )

                progress_bar.progress(100)
                status_text.text("Training completato!")

                st.success(f"✅ Training completato! Miglior modello: **{st.session_state.best_model_name}**")

        if st.session_state.models_trained:
            st.markdown("---")
            st.subheader("Risultati Rapidi")
            st.dataframe(st.session_state.comparison_df, use_container_width=True)

    # TAB 4: Results
    with tab4:
        st.header("📈 Risultati e Analisi")

        if not st.session_state.models_trained:
            st.warning("⚠️ Addestra prima i modelli")
            st.stop()

        st.success(f"🏆 **Miglior Modello**: {st.session_state.best_model_name} con F1 (Macro) = {st.session_state.comparison_df.iloc[0]['F1 (Macro)']:.4f}")

        st.subheader("Confronto Modelli")

        metric = st.selectbox(
            "Seleziona metrica:",
            ["F1 (Macro)", "Accuracy", "F1 (Weighted)"]
        )

        fig = plot_model_comparison(st.session_state.comparison_df, metric)
        st.pyplot(fig)
        plt.close()

        st.markdown("---")
        st.subheader("Tabella Completa Risultati")
        st.dataframe(st.session_state.comparison_df, use_container_width=True)

    # TAB 5: Save Models
    with tab5:
        st.header("💾 Salvataggio Modelli")

        if not st.session_state.models_trained:
            st.warning("⚠️ Addestra prima i modelli")
            st.stop()

        st.info("""
        Salva i modelli addestrati per utilizzarli nell'applicazione di visualizzazione e simulazione.
        Verranno salvati:
        - Il miglior modello addestrato
        - Il preprocessor e label encoder
        - Tutti i metadati e configurazioni
        """)

        experiment_name = st.text_input(
            "Nome Esperimento",
            value="student_performance_model",
            help="Nome identificativo per questo esperimento"
        )

        if st.button("💾 Salva Modelli", type="primary"):
            with st.spinner("Salvataggio in corso..."):
                experiment_dir = save_trained_models(experiment_name)

            st.success(f"✅ Modelli salvati con successo in: `{experiment_dir}`")

            # Show summary
            st.markdown("---")
            st.subheader("Riepilogo Salvataggio")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Modello Migliore", st.session_state.best_model_name)
                st.metric("Test Accuracy", f"{st.session_state.evaluation_results[st.session_state.best_model_name]['accuracy']:.4f}")

            with col2:
                st.metric("Test F1 (Macro)", f"{st.session_state.evaluation_results[st.session_state.best_model_name]['f1_macro']:.4f}")
                st.metric("Directory", str(experiment_dir.name))

            st.markdown("---")
            st.success("""
            ✅ **Modelli pronti per l'uso!**
            
            Ora puoi utilizzare l'applicazione di visualizzazione e simulazione per:
            - Caricare questo modello
            - Visualizzare le performance dettagliate
            - Fare predizioni su nuovi studenti
            """)


if __name__ == "__main__":
    main()