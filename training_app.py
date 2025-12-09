"""
Streamlit Training Application for Student Performance Prediction System.
Handles data loading, feature selection, preprocessing, model training, and saving trained models.
Enhanced with feature selection capabilities.
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

from src.data.data_mappings import (
    get_readable_value,
    get_feature_description,
    format_dataframe_for_display,
    MARITAL_STATUS,
    GENDER,
    YES_NO,
    ATTENDANCE
)
from src.utils.data_explorer import (
    show_data_dictionary,
    show_data_preview_enhanced,
    show_feature_explorer
)


# ============================================================================
# CONFIGURATION & STYLING
# ============================================================================

# Page configuration
st.set_page_config(
    page_title="Student Performance - Training",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Color palette - consistent across all visualizations
COLORS = {
    'primary': '#1E88E5',
    'secondary': '#43A047',
    'warning': '#FB8C00',
    'danger': '#E53935',
    'success': '#00897B',
    'info': '#5E35B1',
    'dropout': '#E53935',
    'enrolled': '#FB8C00',
    'graduate': '#43A047',
    'background': '#FAFAFA',
    'text': '#212121',
    'grid': '#E0E0E0'
}

# Seaborn style configuration
sns.set_style("whitegrid")
sns.set_palette([COLORS['primary'], COLORS['secondary'], COLORS['warning'],
                 COLORS['danger'], COLORS['success'], COLORS['info']])

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1E88E5;
        border-bottom: 2px solid #1E88E5;
        padding-bottom: 0.3rem;
        margin-bottom: 1rem;
    }
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1rem;
        color: white;
        margin-bottom: 0.5rem;
    }
    .metric-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        border-left: 4px solid #1E88E5;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E88E5;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Class names mapping
CLASS_NAMES = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}
CLASS_NAMES_LIST = ["Dropout", "Enrolled", "Graduate"]
CLASS_COLORS = [COLORS['dropout'], COLORS['enrolled'], COLORS['graduate']]

# Models directory
MODELS_DIR = Path("trained_models")
MODELS_DIR.mkdir(exist_ok=True)

# Feature categories for better organization
FEATURE_CATEGORIES = {
    'Demografiche': [
        'Marital status', 'Nacionality', 'Gender', 'Age at enrollment',
        'International', 'Displaced'
    ],
    'Background Familiare': [
        "Mother's qualification", "Father's qualification",
        "Mother's occupation", "Father's occupation"
    ],
    'Percorso Accademico Precedente': [
        'Previous qualification', 'Previous qualification (grade)',
        'Admission grade', 'Application mode', 'Application order', 'Course'
    ],
    'Situazione Economica': [
        'Scholarship holder', 'Tuition fees up to date', 'Debtor',
        'Educational special needs'
    ],
    'Frequenza e Orario': [
        'Daytime/evening attendance'
    ],
    'Performance 1° Semestre': [
        'Curricular units 1st sem (credited)',
        'Curricular units 1st sem (enrolled)',
        'Curricular units 1st sem (evaluations)',
        'Curricular units 1st sem (approved)',
        'Curricular units 1st sem (grade)',
        'Curricular units 1st sem (without evaluations)'
    ],
    'Performance 2° Semestre': [
        'Curricular units 2nd sem (credited)',
        'Curricular units 2nd sem (enrolled)',
        'Curricular units 2nd sem (evaluations)',
        'Curricular units 2nd sem (approved)',
        'Curricular units 2nd sem (grade)',
        'Curricular units 2nd sem (without evaluations)'
    ],
    'Indicatori Macroeconomici': [
        'Unemployment rate', 'Inflation rate', 'GDP'
    ]
}


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'data_loaded': False,
        'X': None,
        'y': None,
        'X_selected': None,
        'selected_features': None,
        'processed_data': None,
        'X_train_sampled': None,
        'y_train_sampled': None,
        'models_trained': False,
        'comparison_df': None,
        'evaluation_results': None,
        'best_model': None,
        'best_model_name': None,
        'training_config': None,
        'feature_importances': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================================
# DATA LOADING & PREPROCESSING FUNCTIONS
# ============================================================================

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
        st.session_state.selected_features = list(X.columns)
        st.session_state.X_selected = X.copy()

    return X, y


def get_feature_category(feature_name):
    """Get the category of a feature."""
    for category, features in FEATURE_CATEGORIES.items():
        for f in features:
            if f.lower() in feature_name.lower():
                return category
    return 'Altre Features'


def categorize_features(feature_list):
    """Organize features by category."""
    categorized = {}
    for feature in feature_list:
        category = get_feature_category(feature)
        if category not in categorized:
            categorized[category] = []
        categorized[category].append(feature)
    return categorized


def preprocess_data(test_size, random_state, scaling):
    """Preprocess the data with selected features."""
    with st.spinner("Preprocessing dei dati..."):
        # Use only selected features
        X_to_process = st.session_state.X_selected

        preprocessor = FeaturePreprocessor(scaling=scaling)
        label_encoder = LabelEncoderWrapper()
        splitter = DataSplitter(test_size=test_size, random_state=random_state)

        pipeline = DataPipeline(preprocessor, label_encoder, splitter)
        processed = pipeline.run(X_to_process, st.session_state.y)

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

    # Extract feature importances if available
    feature_importances = None
    if hasattr(best_model.model, 'feature_importances_'):
        feature_importances = best_model.model.feature_importances_

    st.session_state.comparison_df = comparison_df
    st.session_state.evaluation_results = evaluation_results
    st.session_state.best_model = best_model
    st.session_state.best_model_name = best_name
    st.session_state.models_trained = True
    st.session_state.feature_importances = feature_importances

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

    # Save preprocessor data
    preprocessor_data = {
        'X_train': st.session_state.processed_data['X_train'],
        'X_train_scaled': st.session_state.processed_data['X_train_scaled'],
        'feature_names': st.session_state.processed_data['feature_names'],
        'selected_features': st.session_state.selected_features,
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
        'selected_features': st.session_state.selected_features,
        'num_features_selected': len(st.session_state.selected_features),
        'num_features_total': len(st.session_state.X.columns),
        'training_config': st.session_state.training_config,
        'comparison_results': st.session_state.comparison_df.to_dict('records'),
        'test_accuracy': float(st.session_state.evaluation_results[st.session_state.best_model_name]['accuracy']),
        'test_f1_macro': float(st.session_state.evaluation_results[st.session_state.best_model_name]['f1_macro']),
    }

    # Add feature importances if available
    if st.session_state.feature_importances is not None:
        metadata['feature_importances'] = {
            name: float(imp) for name, imp in
            zip(st.session_state.selected_features, st.session_state.feature_importances)
        }

    metadata_path = experiment_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    return experiment_dir


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_figure_style():
    """Create consistent figure styling."""
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.color': COLORS['grid'],
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'font.family': 'sans-serif'
    })


def plot_class_distribution(y, title="Distribuzione Classi Target"):
    """Plot class distribution with consistent styling."""
    create_figure_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    counts = y.value_counts()

    bars = ax.bar(
        range(len(counts)),
        counts.values,
        color=CLASS_COLORS[:len(counts)],
        edgecolor='white',
        linewidth=2
    )

    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, fontsize=11)

    # Add value labels
    for i, (bar, count) in enumerate(zip(bars, counts.values)):
        pct = count / len(y) * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.02,
            f'{count:,}\n({pct:.1f}%)',
            ha='center', va='bottom', fontsize=11, fontweight='bold'
        )

    ax.set_xlabel('Classe', fontsize=12, fontweight='bold')
    ax.set_ylabel('Numero di Studenti', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, max(counts) * 1.2)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    return fig


def plot_model_comparison(df, metric='F1 (Macro)'):
    """Plot model comparison with consistent styling."""
    create_figure_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    df_sorted = df.sort_values(metric, ascending=True)

    # Create gradient colors
    n_models = len(df_sorted)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_models))

    bars = ax.barh(
        df_sorted['Model'],
        df_sorted[metric],
        color=colors,
        edgecolor='white',
        linewidth=2,
        height=0.7
    )

    # Add value labels
    for bar, val in zip(bars, df_sorted[metric]):
        ax.text(
            val + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f'{val:.4f}',
            va='center', fontsize=11, fontweight='bold'
        )

    ax.set_xlabel(metric, fontsize=12, fontweight='bold')
    ax.set_title(f'Confronto Modelli - {metric}', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(0, df_sorted[metric].max() * 1.15)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    return fig


def plot_feature_importance(importances, feature_names, top_n=15):
    """Plot feature importance with consistent styling."""
    create_figure_style()

    # Sort by importance
    indices = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(figsize=(12, 8))

    colors = plt.cm.Blues(np.linspace(0.4, 0.9, top_n))

    bars = ax.barh(
        range(top_n),
        importances[indices],
        color=colors,
        edgecolor='white',
        linewidth=1.5,
        height=0.7
    )

    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=10)

    for bar, val in zip(bars, importances[indices]):
        ax.text(
            val + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f'{val:.4f}',
            va='center', fontsize=9, fontweight='bold'
        )

    ax.set_xlabel('Importanza', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Features più Importanti', fontsize=14, fontweight='bold', pad=20)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    return fig


def plot_sampling_comparison(original_dist, resampled_dist):
    """Plot before/after sampling comparison."""
    create_figure_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (data, title) in zip(axes, [(original_dist, 'Prima del Sampling'),
                                         (resampled_dist, 'Dopo il Sampling')]):
        counts = pd.Series(data).value_counts().sort_index()

        bars = ax.bar(
            [CLASS_NAMES.get(i, str(i)) for i in counts.index],
            counts.values,
            color=CLASS_COLORS[:len(counts)],
            edgecolor='white',
            linewidth=2
        )

        for bar, count in zip(bars, counts.values):
            pct = count / len(data) * 100
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts) * 0.02,
                f'{count:,}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold'
            )

        ax.set_xlabel('Classe', fontsize=11, fontweight='bold')
        ax.set_ylabel('Conteggio', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_ylim(0, max(counts) * 1.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle('Effetto del Sampling sulla Distribuzione delle Classi',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    init_session_state()

    # Header
    st.markdown('<p class="main-header">🎓 Training - Student Performance Prediction</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Configura e addestra modelli di Machine Learning per la predizione della performance studentesca</p>',
                unsafe_allow_html=True)

    # Sidebar info
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/graduation-cap.png", width=80)
        st.title("Training Pipeline")

        st.markdown("---")

        # Status indicators
        st.markdown("### 📊 Stato Pipeline")

        status_items = [
            ("Dati Caricati", st.session_state.data_loaded),
            ("Features Selezionate", st.session_state.selected_features is not None),
            ("Preprocessing Completato", st.session_state.processed_data is not None),
            ("Modelli Addestrati", st.session_state.models_trained)
        ]

        for label, completed in status_items:
            icon = "✅" if completed else "⏳"
            st.write(f"{icon} {label}")

        st.markdown("---")
        st.markdown("### ℹ️ Info")
        st.markdown("""
        Questa applicazione permette di:
        1. Caricare dati
        2. **Selezionare features**
        3. Preprocessare i dati
        4. Addestrare modelli ML
        5. Salvare i modelli
        """)

    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📂 Caricamento Dati",
        "🎯 Selezione Features",
        "⚙️ Preprocessing",
        "🤖 Training",
        "📈 Risultati",
        "💾 Salvataggio"
    ])

    # =========================================================================
    # TAB 1: Data Loading
    # =========================================================================
    with tab1:
        st.header("📂 Caricamento Dati")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Sorgente Dati")

            data_source = st.radio(
                "Seleziona sorgente:",
                ["UCI Repository (o Dati Sample)", "Carica File CSV"],
                horizontal=True
            )

            if data_source == "Carica File CSV":
                uploaded_file = st.file_uploader(
                    "Carica il tuo file CSV",
                    type=['csv'],
                    help="Il file deve avere la colonna 'Target' come variabile target"
                )

                if uploaded_file is not None:
                    if st.button("📥 Carica Dati", type="primary"):
                        X, y = load_data(uploaded_file)
                        st.success(f"✅ Dati caricati: {len(X):,} campioni, {len(X.columns)} features")
            else:
                if st.button("📥 Carica da UCI", type="primary"):
                    X, y = load_data()
                    st.success(f"✅ Dati caricati: {len(X):,} campioni, {len(X.columns)} features")

        with col2:
            st.subheader("📊 Statistiche Rapide")
            if st.session_state.data_loaded:
                st.metric("Campioni Totali", f"{len(st.session_state.X):,}")
                st.metric("Features Totali", len(st.session_state.X.columns))
                st.metric("Classi Target", len(st.session_state.y.unique()))

        # Data preview

        if st.session_state.data_loaded:
            st.markdown("---")

            col1, col2 = st.columns([3, 2])

            with col1:
                st.subheader("Anteprima Dati")
                show_raw = st.checkbox("Mostra valori grezzi (codici numerici)", value=False)

                if show_raw:
                    st.dataframe(st.session_state.X.head(10), use_container_width=True)
                else:
                    df_display = format_dataframe_for_display(st.session_state.X.head(10))
                    st.dataframe(df_display, use_container_width=True)

                # Aggiungi legenda
                with st.expander("📖 Legenda Features"):
                    selected_feature = st.selectbox(
                        "Seleziona una feature:",
                        st.session_state.X.columns
                    )
                    st.info(get_feature_description(selected_feature))
                with st.expander("📖 Dizionario Dati Completo", expanded=False):
                    show_data_dictionary()
            with col2:
                st.subheader("Distribuzione Target")
                fig = plot_class_distribution(st.session_state.y)
                st.pyplot(fig)
                plt.close()

    # =========================================================================
    # TAB 2: Feature Selection
    # =========================================================================
    with tab2:
        st.header("🎯 Selezione Features")

        if not st.session_state.data_loaded:
            st.warning("⚠️ Carica prima i dati nella tab 'Caricamento Dati'")
            st.stop()

        st.markdown("""
        Seleziona le features da utilizzare per l'addestramento del modello.
        Le features sono organizzate per categoria per facilitare la selezione.
        """)

        all_features = list(st.session_state.X.columns)
        categorized = categorize_features(all_features)

        # Quick selection buttons
        st.subheader("⚡ Selezione Rapida")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("✅ Seleziona Tutte", use_container_width=True):
                st.session_state.selected_features = all_features.copy()
                st.rerun()

        with col2:
            if st.button("❌ Deseleziona Tutte", use_container_width=True):
                st.session_state.selected_features = []
                st.rerun()

        with col3:
            if st.button("📚 Solo Accademiche", use_container_width=True):
                academic = []
                for cat in ['Performance 1° Semestre', 'Performance 2° Semestre',
                           'Percorso Accademico Precedente']:
                    if cat in categorized:
                        academic.extend(categorized[cat])
                st.session_state.selected_features = [f for f in all_features if any(
                    ac.lower() in f.lower() for ac in academic)]
                st.rerun()

        with col4:
            if st.button("👤 Solo Demografiche", use_container_width=True):
                demo = categorized.get('Demografiche', []) + categorized.get('Background Familiare', [])
                st.session_state.selected_features = [f for f in all_features if any(
                    d.lower() in f.lower() for d in demo)]
                st.rerun()

        st.markdown("---")

        # Feature selection by category
        st.subheader("📋 Selezione per Categoria")

        # Initialize selected features if None
        if st.session_state.selected_features is None:
            st.session_state.selected_features = all_features.copy()

        selected = []

        # Create expandable sections for each category
        for category, features in categorized.items():
            # Filter to only include features that exist in the dataset
            category_features = [f for f in all_features if any(
                feat.lower() in f.lower() for feat in features)]

            if category_features:
                with st.expander(f"📁 {category} ({len(category_features)} features)", expanded=False):
                    col1, col2 = st.columns([3, 1])

                    with col2:
                        select_all = st.checkbox(f"Tutte", key=f"all_{category}",
                                                 value=all(f in st.session_state.selected_features
                                                          for f in category_features))

                    with col1:
                        for feature in category_features:
                            is_selected = feature in st.session_state.selected_features
                            if select_all:
                                is_selected = True

                            if st.checkbox(
                                    feature,
                                    value=is_selected,
                                    key=f"feat_{feature}",
                                    help=get_feature_description(feature)  # ⭐ AGGIUNGI QUESTO
                            ):
                                selected.append(feature)

        # Handle uncategorized features
        uncategorized = [f for f in all_features if f not in selected and
                        get_feature_category(f) == 'Altre Features']
        if uncategorized:
            with st.expander(f"📁 Altre Features ({len(uncategorized)} features)", expanded=False):
                for feature in uncategorized:
                    if st.checkbox(
                            feature,
                            value=is_selected,
                            key=f"feat_{feature}",
                            help=get_feature_description(feature)  # ⭐ AGGIUNGI QUESTO
                    ):
                        selected.append(feature)

        # Update button
        st.markdown("---")

        col1, col2, col3 = st.columns([2, 1, 2])

        with col2:
            if st.button("🔄 Aggiorna Selezione", type="primary", use_container_width=True):
                # Collect all selected features from checkboxes
                new_selected = []
                for feature in all_features:
                    if st.session_state.get(f"feat_{feature}", False):
                        new_selected.append(feature)

                if len(new_selected) == 0:
                    st.error("⚠️ Seleziona almeno una feature!")
                else:
                    st.session_state.selected_features = new_selected
                    st.session_state.X_selected = st.session_state.X[new_selected]
                    st.success(f"✅ Selezionate {len(new_selected)} features")

        # Summary
        st.markdown("---")
        st.subheader("📊 Riepilogo Selezione")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Features Totali", len(all_features))
        with col2:
            n_selected = len(st.session_state.selected_features) if st.session_state.selected_features else 0
            st.metric("Features Selezionate", n_selected)
        with col3:
            pct = (n_selected / len(all_features) * 100) if all_features else 0
            st.metric("Percentuale", f"{pct:.1f}%")

        if st.session_state.selected_features:
            with st.expander("📋 Lista Features Selezionate"):
                cols = st.columns(3)
                for i, feat in enumerate(st.session_state.selected_features):
                    with cols[i % 3]:
                        st.write(f"• {feat}")

    # =========================================================================
    # TAB 3: Preprocessing
    # =========================================================================

    with tab3:
        st.header("⚙️ Preprocessing Dati")

        if not st.session_state.data_loaded:
            st.warning("⚠️ Carica prima i dati nella tab 'Caricamento Dati'")
            st.stop()

        if not st.session_state.selected_features or len(st.session_state.selected_features) == 0:
            st.warning("⚠️ Seleziona almeno una feature nella tab 'Selezione Features'")
            st.stop()

        st.info("""
        **Configurazione Ottimale Applicata Automaticamente:**

        - ✅ **Test Set**: 20% (standard ottimale)
        - ✅ **Random State**: 42 (per riproducibilità)
        - ✅ **Scaling**: Standard Scaler (normalizzazione con media 0, deviazione std 1)

        Puoi personalizzare solo la **tecnica di sampling** per il bilanciamento delle classi.
        """)

        st.markdown("---")

        # Solo sampling configurabile
        st.subheader("⚖️ Bilanciamento Classi")

        col1, col2 = st.columns([2, 1])

        with col1:
            sampler_type = st.selectbox(
                "Tecnica di Sampling",
                ["smote", "adasyn", "none"],
                index=0,  # SMOTE come default
                help="Tecnica per bilanciare le classi sbilanciate nel dataset"
            )

        with col2:
            st.markdown("### ℹ️ Info Tecniche")
            if sampler_type == "smote":
                st.info("**SMOTE** crea campioni sintetici interpolando tra esempi della classe minoritaria.")
            elif sampler_type == "adasyn":
                st.info("**ADASYN** è simile a SMOTE ma si concentra sugli esempi più difficili da classificare.")
            else:
                st.info("**Nessun bilanciamento** - usa i dati così come sono (sconsigliato con classi sbilanciate).")

        # Configurazione fissa (non modificabile)
        test_size = 0.2
        random_state = 42
        scaling = "standard"

        st.markdown("---")

        # Mostra riepilogo configurazione
        st.subheader("📊 Configurazione Preprocessing")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Features Selezionate", len(st.session_state.selected_features))
        with col2:
            st.metric("Test Size", "20%")
        with col3:
            st.metric("Scaling", "Standard")
        with col4:
            st.metric("Sampling", sampler_type.upper() if sampler_type != "none" else "Nessuno")

        st.markdown("---")

        if st.button("▶️ Esegui Preprocessing", type="primary", use_container_width=True):
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
                'sampler_type': sampler_type,
                'num_features': len(st.session_state.selected_features),
                'selected_features': st.session_state.selected_features
            }

            st.success("✅ Preprocessing completato con configurazione ottimale!")

    # =========================================================================
    # TAB 4: Model Training
    # =========================================================================
    with tab4:
        st.header("🤖 Training Modelli")

        if st.session_state.X_train_sampled is None:
            st.warning("⚠️ Completa prima il preprocessing nella tab 'Preprocessing'")
            st.stop()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Modelli Standard")
            standard_models = st.multiselect(
                "Seleziona modelli standard:",
                ["Logistic Regression", "SVM", "Decision Tree", "Random Forest"],
                default=["Logistic Regression", "Decision Tree", "Random Forest"],
                help="Modelli classici di machine learning"
            )

        with col2:
            st.subheader("🚀 Modelli Boosting")
            boosting_models = st.multiselect(
                "Seleziona modelli boosting:",
                ["Gradient Boosting", "XGBoost", "CatBoost", "LogitBoost"],
                default=["Gradient Boosting", "XGBoost"],
                help="Modelli ensemble avanzati"
            )

        selected_models = standard_models + boosting_models

        st.markdown("---")

        col1, col2, col3 = st.columns(3)

        with col1:
            cv_folds = st.slider(
                "Fold Cross-Validation",
                min_value=2,
                max_value=10,
                value=5,
                help="Numero di fold per la validazione incrociata"
            )

        with col2:
            tune_hyperparameters = st.checkbox(
                "🔧 Tuning Iperparametri",
                value=False,
                help="Ottimizza gli iperparametri (richiede più tempo)"
            )

        with col3:
            st.metric("Modelli Selezionati", len(selected_models))

        st.markdown("---")

        if st.button("🚀 Avvia Training", type="primary", disabled=len(selected_models) == 0,
                    use_container_width=True):
            if len(selected_models) == 0:
                st.error("⚠️ Seleziona almeno un modello!")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()

                status_text.text("Inizializzazione training...")
                progress_bar.progress(10)

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
                st.balloons()

        # Quick results
        if st.session_state.models_trained:
            st.markdown("---")
            st.subheader("📊 Risultati Rapidi")
            st.dataframe(
                st.session_state.comparison_df.style.highlight_max(
                    subset=['F1 (Macro)', 'Accuracy'], color='lightgreen'
                ),
                use_container_width=True
            )

    # =========================================================================
    # TAB 5: Results
    # =========================================================================
    with tab5:
        st.header("📈 Risultati e Analisi")

        if not st.session_state.models_trained:
            st.warning("⚠️ Addestra prima i modelli nella tab 'Training'")
            st.stop()

        # Best model highlight
        best_f1 = st.session_state.comparison_df.iloc[0]['F1 (Macro)']
        st.success(f"🏆 **Miglior Modello**: {st.session_state.best_model_name} con F1 (Macro) = {best_f1:.4f}")

        # Model comparison
        st.subheader("📊 Confronto Modelli")

        metric = st.selectbox(
            "Seleziona metrica per il confronto:",
            ["F1 (Macro)", "Accuracy", "F1 (Weighted)"]
        )

        fig = plot_model_comparison(st.session_state.comparison_df, metric)
        st.pyplot(fig)
        plt.close()

        # Feature importance
        if st.session_state.feature_importances is not None:
            st.markdown("---")
            st.subheader("🎯 Importanza Features")

            fig = plot_feature_importance(
                st.session_state.feature_importances,
                st.session_state.selected_features,
                top_n=min(15, len(st.session_state.selected_features))
            )
            st.pyplot(fig)
            plt.close()

        st.markdown("---")
        st.subheader("📋 Tabella Completa Risultati")
        st.dataframe(
            st.session_state.comparison_df.style.format({
                'Accuracy': '{:.4f}',
                'F1 (Macro)': '{:.4f}',
                'F1 (Weighted)': '{:.4f}',
                'F1 Dropout': '{:.4f}',
                'F1 Enrolled': '{:.4f}',
                'F1 Graduate': '{:.4f}'
            }).highlight_max(subset=['F1 (Macro)'], color='lightgreen'),
            use_container_width=True
        )

    # =========================================================================
    # TAB 6: Save Models
    # =========================================================================
    with tab6:
        st.header("💾 Salvataggio Modelli")

        if not st.session_state.models_trained:
            st.warning("⚠️ Addestra prima i modelli nella tab 'Training'")
            st.stop()

        st.info("""
        Salva i modelli addestrati per utilizzarli nell'applicazione di visualizzazione e simulazione.
        
        Verranno salvati:
        - ✅ Il miglior modello addestrato
        - ✅ Il preprocessor e le configurazioni
        - ✅ Le features selezionate
        - ✅ Tutti i metadati dell'esperimento
        """)

        experiment_name = st.text_input(
            "Nome Esperimento",
            value="student_performance_model",
            help="Nome identificativo per questo esperimento"
        )

        st.markdown("---")

        if st.button("💾 Salva Modelli", type="primary", use_container_width=True):
            with st.spinner("Salvataggio in corso..."):
                experiment_dir = save_trained_models(experiment_name)

            st.success(f"✅ Modelli salvati con successo in: `{experiment_dir}`")
            st.balloons()

            # Summary
            st.markdown("---")
            st.subheader("📊 Riepilogo Salvataggio")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Modello Migliore", st.session_state.best_model_name)
            with col2:
                st.metric("Features Utilizzate", len(st.session_state.selected_features))
            with col3:
                st.metric("Test F1 (Macro)",
                         f"{st.session_state.evaluation_results[st.session_state.best_model_name]['f1_macro']:.4f}")

            st.markdown("---")
            st.success("""
            ✅ **Modelli pronti per l'uso!**
            
            Ora puoi utilizzare l'applicazione di **Visualizzazione e Simulazione** per:
            - 📊 Visualizzare le performance dettagliate
            - 🔮 Fare predizioni su nuovi studenti
            - 📈 Analizzare i risultati graficamente
            """)


if __name__ == "__main__":
    main()