"""
Streamlit Web Interface for Student Performance Prediction System.
Provides an interactive UI for data exploration, model training, and prediction.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import warnings
import io

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
from src.visualization import (
    VisualizationConfig,
    DataVisualizer,
    ModelVisualizer
)


# Page configuration
st.set_page_config(
    page_title="Student Performance Prediction",
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
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
def init_session_state():
    """Initialize session state variables."""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'X' not in st.session_state:
        st.session_state.X = None
    if 'y' not in st.session_state:
        st.session_state.y = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'X_train_sampled' not in st.session_state:
        st.session_state.X_train_sampled = None
    if 'y_train_sampled' not in st.session_state:
        st.session_state.y_train_sampled = None
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    if 'comparison_df' not in st.session_state:
        st.session_state.comparison_df = None
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = None
    if 'best_model' not in st.session_state:
        st.session_state.best_model = None
    if 'best_model_name' not in st.session_state:
        st.session_state.best_model_name = None


init_session_state()


# Class names mapping
CLASS_NAMES = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}
CLASS_NAMES_LIST = ["Dropout", "Enrolled", "Graduate"]


def load_data(csv_file=None):
    """Load data from UCI or uploaded CSV."""
    with st.spinner("Loading data..."):
        if csv_file is not None:
            # Load from uploaded file
            df = pd.read_csv(csv_file, delimiter=';')
            if 'Target' in df.columns:
                y = df['Target']
                X = df.drop('Target', axis=1)
            else:
                y = df.iloc[:, -1]
                X = df.iloc[:, :-1]
        else:
            # Load from UCI or generate sample
            loader = UCIDataLoader(dataset_id=697)
            X, y = loader.load()
        
        st.session_state.X = X
        st.session_state.y = y
        st.session_state.data_loaded = True
        
    return X, y


def preprocess_data(test_size, random_state, scaling):
    """Preprocess the data."""
    with st.spinner("Preprocessing data..."):
        preprocessor = FeaturePreprocessor(scaling=scaling)
        label_encoder = LabelEncoderWrapper()
        splitter = DataSplitter(
            test_size=test_size,
            random_state=random_state
        )
        
        pipeline = DataPipeline(preprocessor, label_encoder, splitter)
        processed = pipeline.run(st.session_state.X, st.session_state.y)
        
        st.session_state.processed_data = processed
        
    return processed


def apply_sampling(sampler_type, random_state):
    """Apply sampling to balance classes."""
    with st.spinner(f"Applying {sampler_type.upper()} sampling..."):
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


def plot_class_distribution(y, title="Class Distribution"):
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
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm, class_names, title="Confusion Matrix"):
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
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
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
    ax.set_title(f'Model Comparison - {metric}', fontsize=14, fontweight='bold')
    ax.set_xlim(0, df_sorted[metric].max() * 1.15)
    
    plt.tight_layout()
    return fig


def plot_per_class_metrics(results, metric='f1_score'):
    """Plot per-class metrics."""
    data = []
    for model_name, result in results.items():
        for class_name, class_metrics in result['per_class_metrics'].items():
            data.append({
                'Model': model_name,
                'Class': class_name,
                'Value': class_metrics[metric]
            })
    
    df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    classes = df['Class'].unique()
    models = df['Model'].unique()
    x = np.arange(len(models))
    width = 0.25
    
    colors = sns.color_palette("Set2", len(classes))
    
    for i, cls in enumerate(classes):
        cls_data = df[df['Class'] == cls]
        offset = (i - len(classes)/2 + 0.5) * width
        ax.bar(x + offset, cls_data['Value'], width, label=cls, color=colors[i], edgecolor='black')
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'Per-Class {metric.replace("_", " ").title()} by Model', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(title='Class', loc='upper right')
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    return fig


# Main App
def main():
    # Header
    st.markdown('<p class="main-header">🎓 Student Performance Prediction System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Machine Learning tool for predicting student academic outcomes</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/graduation-cap.png", width=80)
        st.title("Navigation")
        
        page = st.radio(
            "Select Page",
            ["📊 Data Loading", "⚙️ Preprocessing", "🤖 Model Training", "📈 Results", "🔮 Prediction"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This system predicts student outcomes:
        - **Dropout**: Student dropped out
        - **Enrolled**: Still enrolled
        - **Graduate**: Successfully graduated
        
        Based on the paper by Martins et al. (2021)
        """)
    
    # Data Loading Page
    if page == "📊 Data Loading":
        st.header("📊 Data Loading")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Load Dataset")
            
            data_source = st.radio(
                "Select data source:",
                ["UCI Repository (or Sample Data)", "Upload CSV File"]
            )
            
            if data_source == "Upload CSV File":
                uploaded_file = st.file_uploader(
                    "Upload your CSV file",
                    type=['csv'],
                    help="Upload a CSV file with student data. The last column should be the target variable."
                )
                
                if uploaded_file is not None:
                    if st.button("Load Uploaded Data", type="primary"):
                        X, y = load_data(uploaded_file)
                        st.success(f"✅ Data loaded successfully! {len(X)} samples, {len(X.columns)} features")
            else:
                if st.button("Load from UCI Repository", type="primary"):
                    X, y = load_data()
                    st.success(f"✅ Data loaded successfully! {len(X)} samples, {len(X.columns)} features")
        
        with col2:
            st.subheader("Quick Stats")
            if st.session_state.data_loaded:
                st.metric("Total Samples", len(st.session_state.X))
                st.metric("Features", len(st.session_state.X.columns))
                st.metric("Classes", len(st.session_state.y.unique()))
        
        # Show data if loaded
        if st.session_state.data_loaded:
            st.markdown("---")
            
            tab1, tab2, tab3 = st.tabs(["📋 Data Preview", "📊 Class Distribution", "📈 Feature Statistics"])
            
            with tab1:
                st.subheader("Data Preview")
                st.dataframe(st.session_state.X.head(20), use_container_width=True)
                
                st.subheader("Target Distribution")
                target_dist = st.session_state.y.value_counts()
                st.dataframe(target_dist.to_frame("Count"), use_container_width=True)
            
            with tab2:
                st.subheader("Class Distribution")
                fig = plot_class_distribution(st.session_state.y)
                st.pyplot(fig)
                plt.close()
            
            with tab3:
                st.subheader("Feature Statistics")
                st.dataframe(st.session_state.X.describe(), use_container_width=True)
    
    # Preprocessing Page
    elif page == "⚙️ Preprocessing":
        st.header("⚙️ Data Preprocessing")
        
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please load data first in the Data Loading page.")
            st.stop()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Preprocessing Settings")
            
            test_size = st.slider(
                "Test Set Size",
                min_value=0.1,
                max_value=0.4,
                value=0.2,
                step=0.05,
                help="Proportion of data to use for testing"
            )
            
            random_state = st.number_input(
                "Random State",
                min_value=0,
                max_value=9999,
                value=42,
                help="Random seed for reproducibility"
            )
            
            scaling = st.selectbox(
                "Scaling Method",
                ["standard", "minmax", "none"],
                help="Method to scale numerical features"
            )
        
        with col2:
            st.subheader("Sampling Settings")
            
            sampler_type = st.selectbox(
                "Sampling Technique",
                ["smote", "adasyn", "none"],
                help="Technique to handle class imbalance"
            )
            
            st.info("""
            **SMOTE**: Creates synthetic samples by interpolating between minority class examples.
            
            **ADASYN**: Similar to SMOTE but focuses on harder-to-learn examples.
            """)
        
        if st.button("Run Preprocessing", type="primary"):
            # Preprocess
            processed = preprocess_data(test_size, random_state, scaling)
            
            # Apply sampling
            if sampler_type != "none":
                X_resampled, y_resampled = apply_sampling(sampler_type, random_state)
            else:
                st.session_state.X_train_sampled = processed['X_train_scaled'].values
                st.session_state.y_train_sampled = processed['y_train'].values
            
            st.success("✅ Preprocessing completed!")
        
        # Show results if preprocessing done
        if st.session_state.processed_data is not None:
            st.markdown("---")
            st.subheader("Preprocessing Results")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Training Samples (Original)", 
                         len(st.session_state.processed_data['y_train']))
            with col2:
                st.metric("Training Samples (After Sampling)", 
                         len(st.session_state.y_train_sampled) if st.session_state.y_train_sampled is not None else "N/A")
            with col3:
                st.metric("Test Samples", 
                         len(st.session_state.processed_data['y_test']))
            with col4:
                st.metric("Features", 
                         len(st.session_state.processed_data['feature_names']))
            
            # Show class distribution comparison
            if st.session_state.y_train_sampled is not None:
                st.subheader("Class Distribution Comparison")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Before Sampling**")
                    original_dist = pd.Series(st.session_state.processed_data['y_train']).value_counts()
                    for cls, count in original_dist.items():
                        class_name = CLASS_NAMES.get(cls, str(cls))
                        st.write(f"- {class_name}: {count} ({count/len(st.session_state.processed_data['y_train'])*100:.1f}%)")
                
                with col2:
                    st.write("**After Sampling**")
                    resampled_dist = pd.Series(st.session_state.y_train_sampled).value_counts()
                    for cls, count in resampled_dist.items():
                        class_name = CLASS_NAMES.get(cls, str(cls))
                        st.write(f"- {class_name}: {count} ({count/len(st.session_state.y_train_sampled)*100:.1f}%)")
    
    # Model Training Page
    elif page == "🤖 Model Training":
        st.header("🤖 Model Training")
        
        if st.session_state.X_train_sampled is None:
            st.warning("⚠️ Please complete preprocessing first.")
            st.stop()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Standard Models")
            standard_models = st.multiselect(
                "Select standard models:",
                ["Logistic Regression", "SVM", "Decision Tree", "Random Forest"],
                default=["Logistic Regression", "Decision Tree", "Random Forest"]
            )
        
        with col2:
            st.subheader("Boosting Models")
            boosting_models = st.multiselect(
                "Select boosting models:",
                ["Gradient Boosting", "XGBoost", "CatBoost", "LogitBoost"],
                default=["Gradient Boosting", "XGBoost"]
            )
        
        selected_models = standard_models + boosting_models
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            cv_folds = st.slider(
                "Cross-Validation Folds",
                min_value=2,
                max_value=10,
                value=5,
                help="Number of folds for cross-validation"
            )
        
        with col2:
            tune_hyperparameters = st.checkbox(
                "Tune Hyperparameters",
                value=False,
                help="Enable hyperparameter tuning (takes longer)"
            )
        
        if st.button("Train Models", type="primary", disabled=len(selected_models) == 0):
            if len(selected_models) == 0:
                st.error("Please select at least one model.")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                with st.spinner("Training models... This may take a few minutes."):
                    comparison_df, evaluation_results = train_models(
                        selected_models,
                        tune_hyperparameters,
                        cv_folds,
                        random_state=42
                    )
                
                progress_bar.progress(100)
                status_text.text("Training completed!")
                
                st.success(f"✅ Training completed! Best model: **{st.session_state.best_model_name}**")
        
        # Show quick results if trained
        if st.session_state.models_trained:
            st.markdown("---")
            st.subheader("Quick Results")
            st.dataframe(st.session_state.comparison_df, use_container_width=True)
    
    # Results Page
    elif page == "📈 Results":
        st.header("📈 Results & Analysis")
        
        if not st.session_state.models_trained:
            st.warning("⚠️ Please train models first.")
            st.stop()
        
        # Best model highlight
        st.success(f"🏆 **Best Model**: {st.session_state.best_model_name} with F1 (Macro) = {st.session_state.comparison_df.iloc[0]['F1 (Macro)']:.4f}")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Comparison", "🎯 Per-Class Metrics", "📉 Confusion Matrices", "📋 Detailed Report"])
        
        with tab1:
            st.subheader("Model Comparison")
            
            metric = st.selectbox(
                "Select metric for comparison:",
                ["F1 (Macro)", "Accuracy", "F1 (Weighted)"]
            )
            
            fig = plot_model_comparison(st.session_state.comparison_df, metric)
            st.pyplot(fig)
            plt.close()
            
            st.subheader("Full Comparison Table")
            st.dataframe(st.session_state.comparison_df, use_container_width=True)
        
        with tab2:
            st.subheader("Per-Class Metrics")
            
            metric = st.selectbox(
                "Select metric:",
                ["f1_score", "precision", "recall"],
                format_func=lambda x: x.replace('_', ' ').title()
            )
            
            fig = plot_per_class_metrics(st.session_state.evaluation_results, metric)
            st.pyplot(fig)
            plt.close()
        
        with tab3:
            st.subheader("Confusion Matrices")
            
            model_name = st.selectbox(
                "Select model:",
                list(st.session_state.evaluation_results.keys())
            )
            
            cm = st.session_state.evaluation_results[model_name]['confusion_matrix']
            fig = plot_confusion_matrix(cm, CLASS_NAMES_LIST, f"Confusion Matrix - {model_name}")
            st.pyplot(fig)
            plt.close()
        
        with tab4:
            st.subheader("Detailed Classification Report")
            
            model_name = st.selectbox(
                "Select model for report:",
                list(st.session_state.evaluation_results.keys()),
                key="report_model"
            )
            
            report = st.session_state.evaluation_results[model_name]['classification_report']
            
            # Convert to DataFrame for display
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df, use_container_width=True)
        
        # Download results
        st.markdown("---")
        st.subheader("Download Results")
        
        csv = st.session_state.comparison_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="model_comparison.csv",
            mime="text/csv"
        )
    
    # Prediction Page
    elif page == "🔮 Prediction":
        st.header("🔮 Make Predictions")
        
        if not st.session_state.models_trained:
            st.warning("⚠️ Please train models first.")
            st.stop()
        
        st.info(f"Using **{st.session_state.best_model_name}** (best performing model) for predictions.")
        
        st.subheader("Enter Student Data")
        
        # Create input form
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                age = st.number_input("Age at Enrollment", min_value=17, max_value=70, value=20)
                marital_status = st.selectbox("Marital Status", [1, 2, 3, 4, 5, 6], 
                                             format_func=lambda x: {1: "Single", 2: "Married", 3: "Widower", 
                                                                   4: "Divorced", 5: "Facto Union", 6: "Legally Separated"}[x])
                admission_grade = st.slider("Admission Grade", 0.0, 200.0, 120.0)
            
            with col2:
                prev_qualification_grade = st.slider("Previous Qualification Grade", 0.0, 200.0, 120.0)
                scholarship = st.selectbox("Scholarship Holder", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
                debtor = st.selectbox("Debtor", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            
            with col3:
                tuition_up_to_date = st.selectbox("Tuition Fees Up to Date", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
                units_1st_sem_approved = st.number_input("Units 1st Sem Approved", min_value=0, max_value=30, value=5)
                units_2nd_sem_approved = st.number_input("Units 2nd Sem Approved", min_value=0, max_value=30, value=5)
            
            submitted = st.form_submit_button("Predict Outcome", type="primary")
        
        if submitted:
            st.markdown("---")
            
            # Create a sample with the same features as training data
            # This is a simplified version - in production, you'd need to match all features
            sample_features = np.zeros((1, st.session_state.processed_data['X_train_scaled'].shape[1]))
            
            # Set some key features (indices would need to match actual feature positions)
            # This is illustrative - actual implementation would need proper feature mapping
            
            with st.spinner("Making prediction..."):
                prediction = st.session_state.best_model.predict(sample_features)
                probabilities = st.session_state.best_model.predict_proba(sample_features)
            
            predicted_class = CLASS_NAMES.get(prediction[0], str(prediction[0]))
            
            st.subheader("Prediction Result")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Predicted Outcome", predicted_class)
            
            with col2:
                max_prob = max(probabilities[0]) * 100
                st.metric("Confidence", f"{max_prob:.1f}%")
            
            with col3:
                risk_level = "High" if predicted_class == "Dropout" else ("Medium" if predicted_class == "Enrolled" else "Low")
                st.metric("Risk Level", risk_level)
            
            # Show probability distribution
            st.subheader("Probability Distribution")
            
            prob_df = pd.DataFrame({
                'Class': CLASS_NAMES_LIST,
                'Probability': probabilities[0]
            })
            
            fig, ax = plt.subplots(figsize=(8, 4))
            colors = ['#ff6b6b', '#feca57', '#48dbfb']
            bars = ax.bar(prob_df['Class'], prob_df['Probability'], color=colors, edgecolor='black')
            
            for bar, prob in zip(bars, prob_df['Probability']):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                       f'{prob*100:.1f}%', ha='center', va='bottom', fontsize=11)
            
            ax.set_ylabel('Probability')
            ax.set_title('Prediction Probabilities')
            ax.set_ylim(0, 1.1)
            
            st.pyplot(fig)
            plt.close()
            
            # Recommendations based on prediction
            st.subheader("Recommendations")
            
            if predicted_class == "Dropout":
                st.error("""
                ⚠️ **High Risk of Dropout**
                
                Recommended interventions:
                - Schedule meeting with academic advisor
                - Consider financial aid options
                - Provide tutoring support
                - Monitor attendance closely
                """)
            elif predicted_class == "Enrolled":
                st.warning("""
                ⚡ **Moderate Risk - Still Enrolled**
                
                Recommended interventions:
                - Regular check-ins with student
                - Encourage participation in study groups
                - Monitor academic progress
                """)
            else:
                st.success("""
                ✅ **Low Risk - Likely to Graduate**
                
                The student shows positive indicators for successful completion.
                Continue to provide standard academic support.
                """)


if __name__ == "__main__":
    main()
