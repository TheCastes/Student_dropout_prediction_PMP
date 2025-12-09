"""
Streamlit Visualization & Simulation Application for Student Performance Prediction.
Enhanced with comprehensive visualizations and dynamic feature-based prediction form.
Only shows input fields for features that were used during model training.
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
sys.path.insert(0, str(Path(__file__).parent))

from src.data import FeaturePreprocessor

from src.data.data_mappings import (
    get_readable_value,
    get_feature_description,
    format_dataframe_for_display,
    MARITAL_STATUS,
    GENDER,
    YES_NO,
    ATTENDANCE,
    PARENT_QUALIFICATION,
    PARENT_OCCUPATION
)

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Student Performance - Visualizzazione",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Color palette
COLORS = {
    'primary': '#1E88E5', 'secondary': '#43A047', 'warning': '#FB8C00',
    'danger': '#E53935', 'success': '#00897B', 'info': '#5E35B1',
    'dropout': '#E53935', 'enrolled': '#FB8C00', 'graduate': '#43A047', 'grid': '#E0E0E0'
}

CLASS_COLORS = [COLORS['dropout'], COLORS['enrolled'], COLORS['graduate']]
CLASS_NAMES_LIST = ["Dropout", "Enrolled", "Graduate"]
MODELS_DIR = Path("trained_models")

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1E88E5; text-align: center; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.2rem; color: #666; text-align: center; margin-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# FEATURE CONFIGURATION
# ============================================================================

FEATURE_CONFIG = {
    'nacionality': {'display': 'Nazionalità', 'cat': 'Personali', 'type': 'number', 'min': 1, 'max': 110, 'default': 1},
    "mother's qualification": {'display': 'Titolo Madre', 'cat': 'Famiglia', 'type': 'number', 'min': 1, 'max': 44, 'default': 1},
    "father's qualification": {'display': 'Titolo Padre', 'cat': 'Famiglia', 'type': 'number', 'min': 1, 'max': 44, 'default': 1},
    "mother's occupation": {'display': 'Lavoro Madre', 'cat': 'Famiglia', 'type': 'number', 'min': 0, 'max': 200, 'default': 0},
    "father's occupation": {'display': 'Lavoro Padre', 'cat': 'Famiglia', 'type': 'number', 'min': 0, 'max': 200, 'default': 0},
    'admission grade': {'display': 'Voto Ammissione', 'cat': 'Accademici', 'type': 'slider', 'min': 0.0, 'max': 200.0, 'default': 120.0},
    'previous qualification': {'display': 'Qualifica Precedente', 'cat': 'Accademici', 'type': 'number', 'min': 1, 'max': 43, 'default': 1},
    'previous qualification (grade)': {'display': 'Voto Qualifica Prec.', 'cat': 'Accademici', 'type': 'slider', 'min': 0.0, 'max': 200.0, 'default': 120.0},
    'application mode': {'display': 'Modalità Iscrizione', 'cat': 'Accademici', 'type': 'number', 'min': 1, 'max': 57, 'default': 1},
    'application order': {'display': 'Ordine Preferenza', 'cat': 'Accademici', 'type': 'number', 'min': 0, 'max': 9, 'default': 1},
    'course': {'display': 'Corso', 'cat': 'Accademici', 'type': 'number', 'min': 1, 'max': 10000, 'default': 33},
    'curricular units 1st sem (credited)': {'display': '1°Sem Crediti', 'cat': 'Perf. 1° Sem', 'type': 'number', 'min': 0, 'max': 30, 'default': 0},
    'curricular units 1st sem (enrolled)': {'display': '1°Sem Iscritte', 'cat': 'Perf. 1° Sem', 'type': 'number', 'min': 0, 'max': 30, 'default': 6},
    'curricular units 1st sem (evaluations)': {'display': '1°Sem Valutazioni', 'cat': 'Perf. 1° Sem', 'type': 'number', 'min': 0, 'max': 50, 'default': 6},
    'curricular units 1st sem (approved)': {'display': '1°Sem Approvate', 'cat': 'Perf. 1° Sem', 'type': 'number', 'min': 0, 'max': 30, 'default': 5},
    'curricular units 1st sem (grade)': {'display': '1°Sem Media Voti', 'cat': 'Perf. 1° Sem', 'type': 'slider', 'min': 0.0, 'max': 20.0, 'default': 12.0},
    'curricular units 1st sem (without evaluations)': {'display': '1°Sem Senza Valut.', 'cat': 'Perf. 1° Sem', 'type': 'number', 'min': 0, 'max': 20, 'default': 0},
    'curricular units 2nd sem (credited)': {'display': '2°Sem Crediti', 'cat': 'Perf. 2° Sem', 'type': 'number', 'min': 0, 'max': 30, 'default': 0},
    'curricular units 2nd sem (enrolled)': {'display': '2°Sem Iscritte', 'cat': 'Perf. 2° Sem', 'type': 'number', 'min': 0, 'max': 30, 'default': 6},
    'curricular units 2nd sem (evaluations)': {'display': '2°Sem Valutazioni', 'cat': 'Perf. 2° Sem', 'type': 'number', 'min': 0, 'max': 50, 'default': 6},
    'curricular units 2nd sem (approved)': {'display': '2°Sem Approvate', 'cat': 'Perf. 2° Sem', 'type': 'number', 'min': 0, 'max': 30, 'default': 5},
    'curricular units 2nd sem (grade)': {'display': '2°Sem Media Voti', 'cat': 'Perf. 2° Sem', 'type': 'slider', 'min': 0.0, 'max': 20.0, 'default': 12.0},
    'curricular units 2nd sem (without evaluations)': {'display': '2°Sem Senza Valut.', 'cat': 'Perf. 2° Sem', 'type': 'number', 'min': 0, 'max': 20, 'default': 0},
    'unemployment rate': {'display': 'Tasso Disoccupazione', 'cat': 'Economici', 'type': 'slider', 'min': 0.0, 'max': 30.0, 'default': 10.0},
    'inflation rate': {'display': 'Tasso Inflazione', 'cat': 'Economici', 'type': 'slider', 'min': -5.0, 'max': 10.0, 'default': 1.0},
    'gdp': {'display': 'PIL', 'cat': 'Economici', 'type': 'slider', 'min': -10.0, 'max': 10.0, 'default': 1.0},
    'age at enrollment': {
        'display': 'Età Iscrizione',
        'cat': 'Personali',
        'type': 'number',
        'min': 17,
        'max': 70,
        'default': 20
    },
    'marital status': {
        'display': 'Stato Civile',
        'cat': 'Personali',
        'type': 'select',
        'opts': list(MARITAL_STATUS.keys()),
        'labels': MARITAL_STATUS,
        'default': 1
    },
    'gender': {
        'display': 'Genere',
        'cat': 'Personali',
        'type': 'select',
        'opts': list(GENDER.keys()),
        'labels': GENDER,
        'default': 0
    },
    'displaced': {
        'display': 'Fuori Sede',
        'cat': 'Personali',
        'type': 'select',
        'opts': list(YES_NO.keys()),
        'labels': YES_NO,
        'default': 0
    },
    'international': {
        'display': 'Internazionale',
        'cat': 'Personali',
        'type': 'select',
        'opts': list(YES_NO.keys()),
        'labels': YES_NO,
        'default': 0
    },
    'scholarship holder': {
        'display': 'Borsa di Studio',
        'cat': 'Finanziari',
        'type': 'select',
        'opts': list(YES_NO.keys()),
        'labels': YES_NO,
        'default': 0
    },
    'tuition fees up to date': {
        'display': 'Tasse Pagate',
        'cat': 'Finanziari',
        'type': 'select',
        'opts': list(YES_NO.keys()),
        'labels': YES_NO,
        'default': 1
    },
    'debtor': {
        'display': 'Debitore',
        'cat': 'Finanziari',
        'type': 'select',
        'opts': list(YES_NO.keys()),
        'labels': YES_NO,
        'default': 0
    },
    'educational special needs': {
        'display': 'Bisogni Speciali',
        'cat': 'Finanziari',
        'type': 'select',
        'opts': list(YES_NO.keys()),
        'labels': YES_NO,
        'default': 0
    },
    'daytime/evening attendance': {
        'display': 'Frequenza',
        'cat': 'Accademici',
        'type': 'select',
        'opts': list(ATTENDANCE.keys()),
        'labels': ATTENDANCE,
        'default': 1
    },
}

CATEGORY_ORDER = ['Personali', 'Famiglia', 'Accademici', 'Finanziari', 'Perf. 1° Sem', 'Perf. 2° Sem', 'Economici', 'Altro']
CATEGORY_ICONS = {'Personali': '👤', 'Famiglia': '👨‍👩‍👧', 'Accademici': '📚', 'Finanziari': '💰', 'Perf. 1° Sem': '📊', 'Perf. 2° Sem': '📈', 'Economici': '📉', 'Altro': '📌'}

# ============================================================================
# SESSION STATE & DATA LOADING
# ============================================================================

def init_session_state():
    for key in ['model_loaded', 'model', 'metadata', 'preprocessor_data']:
        if key not in st.session_state:
            st.session_state[key] = False if key == 'model_loaded' else None

def get_available_experiments():
    if not MODELS_DIR.exists(): return []
    experiments = []
    for exp_dir in MODELS_DIR.iterdir():
        if exp_dir.is_dir() and (exp_dir / "metadata.json").exists():
            with open(exp_dir / "metadata.json", 'r') as f:
                meta = json.load(f)
            experiments.append({'path': exp_dir, 'name': meta.get('experiment_name', exp_dir.name),
                'timestamp': meta.get('timestamp', ''), 'best_model': meta.get('best_model_name', ''),
                'f1_macro': meta.get('test_f1_macro', 0), 'accuracy': meta.get('test_accuracy', 0),
                'num_features': meta.get('num_features_selected', len(meta.get('feature_names', [])))})
    return sorted(experiments, key=lambda x: x['timestamp'], reverse=True)

def load_experiment(exp_path):
    exp_path = Path(exp_path)
    with open(exp_path / "metadata.json", 'r') as f: metadata = json.load(f)
    with open(exp_path / "best_model.pkl", 'rb') as f: model = pickle.load(f)
    with open(exp_path / "preprocessor.pkl", 'rb') as f: preprocessor_data = pickle.load(f)
    st.session_state.model, st.session_state.metadata = model, metadata
    st.session_state.preprocessor_data, st.session_state.model_loaded = preprocessor_data, True
    return metadata

# ============================================================================
# FEATURE HANDLING
# ============================================================================

def get_feature_config(col_name):
    col_lower = col_name.lower().strip()
    if col_lower in FEATURE_CONFIG: return FEATURE_CONFIG[col_lower]
    for key, cfg in FEATURE_CONFIG.items():
        if key in col_lower or col_lower in key: return cfg
    return {'display': col_name, 'cat': 'Altro', 'type': 'number', 'min': -1000, 'max': 1000, 'default': 0}

def get_active_features(metadata, preprocessor_data):
    return metadata.get('selected_features') or metadata.get('feature_names') or preprocessor_data.get('feature_names', [])

def organize_by_category(features):
    cats = {}
    for f in features:
        cfg = get_feature_config(f)
        cat = cfg.get('cat', 'Altro')
        if cat not in cats: cats[cat] = []
        cats[cat].append({'col': f, 'cfg': cfg})
    return {c: cats[c] for c in CATEGORY_ORDER if c in cats}

def render_input(feat_info, prefix=""):
    cfg, col = feat_info['cfg'], feat_info['col']
    key = f"{prefix}_{col}"
    t = cfg.get('type', 'number')
    help_text = get_feature_description(col)

    if t == 'select':
        opts = cfg.get('opts', [0, 1])
        labels = cfg.get('labels', {o: str(o) for o in opts})
        default_idx = opts.index(cfg.get('default', opts[0])) if cfg.get('default') in opts else 0
        return col, st.selectbox(cfg.get('display', col), opts, default_idx, format_func=lambda x: labels.get(x, str(x)), key=key, help=help_text)
    elif t == 'slider':
        return col, st.slider(cfg.get('display', col), float(cfg.get('min', 0)), float(cfg.get('max', 100)), float(cfg.get('default', 50)), key=key, help=help_text)
    else:
        return col, st.number_input(cfg.get('display', col), cfg.get('min', 0), cfg.get('max', 100), cfg.get('default', 0), key=key, help=help_text)

# ============================================================================
# VISUALIZATIONS
# ============================================================================

def setup_style():
    plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white', 'axes.grid': True,
        'grid.alpha': 0.3, 'grid.color': COLORS['grid'], 'axes.labelsize': 12, 'axes.titlesize': 14, 'axes.titleweight': 'bold'})

def plot_comparison(data, metric='F1 (Macro)'):
    setup_style()
    df = pd.DataFrame(data).sort_values(metric, ascending=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(df)))
    bars = ax.barh(df['Model'], df[metric], color=colors, edgecolor='white', linewidth=2, height=0.7)
    for bar, val in zip(bars, df[metric]):
        ax.text(val + 0.008, bar.get_y() + bar.get_height()/2, f'{val:.4f}', va='center', fontsize=11, fontweight='bold')
    ax.set_xlabel(metric, fontsize=12, fontweight='bold')
    ax.set_title(f'Confronto Modelli - {metric}', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(0, df[metric].max() * 1.15)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return fig

def plot_per_class(data):
    setup_style()
    df = pd.DataFrame(data)
    metrics = [m for m in ['F1 Dropout', 'F1 Enrolled', 'F1 Graduate'] if m in df.columns]
    if not metrics: return None
    fig, ax = plt.subplots(figsize=(14, 7))
    x, width = np.arange(len(df)), 0.25
    for i, m in enumerate(metrics):
        bars = ax.bar(x + (i-1)*width, df[m], width, label=m.replace('F1 ', ''), color=CLASS_COLORS[i], edgecolor='white')
        for bar in bars: ax.annotate(f'{bar.get_height():.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()), xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8, fontweight='bold')
    ax.set_xlabel('Modello', fontsize=12, fontweight='bold'); ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('Performance per Classe', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x); ax.set_xticklabels(df['Model'], rotation=45, ha='right')
    ax.legend(title='Classe'); ax.set_ylim(0, 1.1)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return fig

def plot_heatmap(data):
    setup_style()
    df = pd.DataFrame(data)
    cols = [c for c in ['Accuracy', 'F1 (Macro)', 'F1 (Weighted)', 'F1 Dropout', 'F1 Enrolled', 'F1 Graduate'] if c in df.columns]
    if not cols: return None
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df.set_index('Model')[cols], annot=True, fmt='.3f', cmap='RdYlGn', center=0.5, linewidths=2, linecolor='white', ax=ax)
    ax.set_title('Mappa Metriche', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right'); plt.tight_layout()
    return fig

def plot_radar(data):
    setup_style()
    df = pd.DataFrame(data)
    metrics = [m for m in ['Accuracy', 'F1 (Macro)', 'F1 Dropout', 'F1 Enrolled', 'F1 Graduate'] if m in df.columns]
    if len(metrics) < 3: return None
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist() + [0]
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    colors = plt.cm.Set2(np.linspace(0, 1, len(df)))
    for idx, (_, row) in enumerate(df.iterrows()):
        vals = [row[m] for m in metrics] + [row[metrics[0]]]
        ax.plot(angles, vals, 'o-', linewidth=2, label=row['Model'], color=colors[idx])
        ax.fill(angles, vals, alpha=0.15, color=colors[idx])
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1); ax.set_title('Radar Multi-Metrica', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0)); plt.tight_layout()
    return fig

def plot_importance(imp_dict, top_n=15):
    setup_style()
    if not imp_dict: return None
    sorted_f = sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features, values = [f[0] for f in sorted_f][::-1], [f[1] for f in sorted_f][::-1]
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(features)))
    bars = ax.barh(range(len(features)), values, color=colors, edgecolor='white')
    ax.set_yticks(range(len(features))); ax.set_yticklabels(features, fontsize=10)
    for bar, val in zip(bars, values): ax.text(val + 0.002, bar.get_y() + bar.get_height()/2, f'{val:.4f}', va='center', fontsize=9, fontweight='bold')
    ax.set_xlabel('Importanza', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {len(features)} Features', fontsize=14, fontweight='bold', pad=20)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); plt.tight_layout()
    return fig

def plot_probs(probs, names):
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, probs, color=CLASS_COLORS, edgecolor='white', linewidth=3, width=0.6)
    for bar, p in zip(bars, probs):
        bar.set_alpha(0.3 + 0.7*p)
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{p*100:.1f}%', ha='center', fontsize=14, fontweight='bold')
    ax.set_ylabel('Probabilità', fontsize=12, fontweight='bold')
    ax.set_title('Distribuzione Probabilità', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, 1.15); ax.axhline(y=0.333, color='gray', linestyle='--', alpha=0.5, label='Baseline 33%')
    ax.legend(); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); plt.tight_layout()
    return fig

def plot_dashboard(meta):
    setup_style()
    fig = plt.figure(figsize=(16, 10))
    data = meta.get('comparison_results', [])
    df = pd.DataFrame(data)
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.text(0.5, 0.7, '🏆', fontsize=50, ha='center', transform=ax1.transAxes)
    ax1.text(0.5, 0.4, meta.get('best_model_name', 'N/A'), fontsize=14, fontweight='bold', ha='center', transform=ax1.transAxes)
    ax1.text(0.5, 0.2, f"F1: {meta.get('test_f1_macro', 0):.4f}", fontsize=12, ha='center', transform=ax1.transAxes)
    ax1.set_title('Miglior Modello', fontsize=14, fontweight='bold'); ax1.axis('off')
    if 'Accuracy' in df.columns:
        ax2 = fig.add_subplot(2, 3, 2); dfs = df.sort_values('Accuracy', ascending=True)
        ax2.barh(dfs['Model'], dfs['Accuracy'], color=plt.cm.Greens(np.linspace(0.3, 0.9, len(dfs))), edgecolor='white')
        ax2.set_xlabel('Accuracy'); ax2.set_title('Accuracy', fontweight='bold'); ax2.set_xlim(0, 1)
    if 'F1 (Macro)' in df.columns:
        ax3 = fig.add_subplot(2, 3, 3); dfs = df.sort_values('F1 (Macro)', ascending=True)
        ax3.barh(dfs['Model'], dfs['F1 (Macro)'], color=plt.cm.Blues(np.linspace(0.3, 0.9, len(dfs))), edgecolor='white')
        ax3.set_xlabel('F1 (Macro)'); ax3.set_title('F1 Macro', fontweight='bold'); ax3.set_xlim(0, 1)
    ax4 = fig.add_subplot(2, 1, 2)
    metrics = [m for m in ['F1 Dropout', 'F1 Enrolled', 'F1 Graduate'] if m in df.columns]
    if metrics:
        x, width = np.arange(len(df)), 0.25
        for i, m in enumerate(metrics): ax4.bar(x + (i-1)*width, df[m], width, label=m.replace('F1 ', ''), color=CLASS_COLORS[i], edgecolor='white')
        ax4.set_xlabel('Modello'); ax4.set_ylabel('F1'); ax4.set_title('F1 per Classe', fontweight='bold')
        ax4.set_xticks(x); ax4.set_xticklabels(df['Model'], rotation=45, ha='right'); ax4.legend(); ax4.set_ylim(0, 1)
    plt.suptitle('Dashboard Performance', fontsize=16, fontweight='bold', y=1.02); plt.tight_layout()
    return fig

# ============================================================================
# PREDICTION
# ============================================================================

def create_features(inputs, preprocessor_data):
    X_tmpl = preprocessor_data['X_train'].copy()
    vals = {c: (X_tmpl[c].median() if X_tmpl[c].dtype in ['int64','float64'] else 0) for c in X_tmpl.columns}
    for k, v in inputs.items():
        if k in vals: vals[k] = v
        else:
            for c in vals:
                if c.lower() == k.lower(): vals[c] = v; break
    X = pd.DataFrame([vals])
    prep = FeaturePreprocessor(scaling=preprocessor_data.get('scaling', 'standard'))
    prep.fit(X_tmpl)
    return prep.transform(X).values

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    init_session_state()
    st.markdown('<p class="main-header">📊 Visualizzazione & Simulazione</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Analizza modelli e simula predizioni</p>', unsafe_allow_html=True)

    with st.sidebar:
        st.title("📂 Esperimenti")
        exps = get_available_experiments()
        if not exps: st.warning("Nessun esperimento."); st.stop()
        opts = {f"{e['name']} ({e['timestamp']})": e['path'] for e in exps}
        sel = st.selectbox("Seleziona:", list(opts.keys()))
        if st.button("📂 Carica", type="primary", use_container_width=True):
            load_experiment(opts[sel]); st.success("✅ Caricato!")
        if st.session_state.model_loaded:
            st.markdown("---"); m = st.session_state.metadata
            st.metric("Modello", m.get('best_model_name', 'N/A'))
            st.metric("F1 Macro", f"{m.get('test_f1_macro', 0):.4f}")
            st.metric("Features", m.get('num_features_selected', 'N/A'))

    if not st.session_state.model_loaded:
        st.info("👈 Carica un esperimento"); st.stop()

    meta = st.session_state.metadata
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance", "🔮 Predizione", "📋 Dettagli", "🎨 Dashboard"])

    with tab1:
        st.header("📈 Analisi Performance")
        data = meta.get('comparison_results', [])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🏆 Modello", meta.get('best_model_name', 'N/A'))
        c2.metric("📊 F1", f"{meta.get('test_f1_macro', 0):.4f}")
        c3.metric("🎯 Acc", f"{meta.get('test_accuracy', 0):.4f}")
        c4.metric("📋 Feat", meta.get('num_features_selected', 'N/A'))
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1: st.pyplot(plot_comparison(data, 'F1 (Macro)')); plt.close()
        with c2: st.pyplot(plot_comparison(data, 'Accuracy')); plt.close()
        st.markdown("---"); fig = plot_per_class(data)
        if fig: st.pyplot(fig); plt.close()
        st.markdown("---"); fig = plot_heatmap(data)
        if fig: st.pyplot(fig); plt.close()
        st.markdown("---"); fig = plot_radar(data)
        if fig: st.pyplot(fig); plt.close()
        if meta.get('feature_importances'):
            st.markdown("---"); fig = plot_importance(meta['feature_importances'])
            if fig: st.pyplot(fig); plt.close()
        st.markdown("---"); st.dataframe(pd.DataFrame(data), use_container_width=True)

    with tab2:
        st.header("🔮 Simulazione Predizione")
        active = get_active_features(meta, st.session_state.preprocessor_data)
        if not active: st.error("❌ Features non trovate"); st.stop()

        total_feat = meta.get('num_features_total', len(active))
        st.info(f"**Modello:** {meta.get('best_model_name', 'N/A')} | **Features attive:** {len(active)}/{total_feat}\n\n*Solo le features usate nel training sono visualizzate.*")

        cats = organize_by_category(active)
        with st.expander("📋 Features utilizzate", expanded=False):
            for cat, feats in cats.items():
                st.markdown(f"**{CATEGORY_ICONS.get(cat, '📌')} {cat}** ({len(feats)})")
                st.write(", ".join([f['cfg'].get('display', f['col']) for f in feats]))

        st.markdown("---")
        with st.form("pred"):
            st.subheader("📝 Dati Studente")
            inputs = {}
            cols = st.columns(3)
            for i, (cat, feats) in enumerate(cats.items()):
                with cols[i % 3]:
                    st.markdown(f"**{CATEGORY_ICONS.get(cat, '📌')} {cat}**")
                    for f in feats:
                        col, val = render_input(f, "p")
                        inputs[col] = val
                    st.markdown("---")
            submit = st.form_submit_button("🔮 Predici", type="primary", use_container_width=True)

        if submit:
            try:
                feat = create_features(inputs, st.session_state.preprocessor_data)
                pred = st.session_state.model.predict(feat)
                probs = st.session_state.model.predict_proba(feat)
                cls = CLASS_NAMES_LIST[pred[0]]
                st.markdown("---")
                c1, c2, c3 = st.columns(3)
                c1.metric("🎯 Risultato", cls)
                c2.metric("📈 Confidenza", f"{max(probs[0])*100:.1f}%")
                risk = "Alto 🔴" if cls == "Dropout" else ("Medio 🟡" if cls == "Enrolled" else "Basso 🟢")
                c3.metric("⚠️ Rischio", risk)
                st.markdown("---"); st.pyplot(plot_probs(probs[0], CLASS_NAMES_LIST)); plt.close()
                with st.expander("📋 Dati inseriti"):
                    input_data = []
                    for k, v in inputs.items():
                        readable_value = get_readable_value(k, v)
                        input_data.append({
                            'Feature': get_feature_config(k).get('display', k),
                            'Valore': readable_value,
                            'Descrizione': get_feature_description(k)
                        })

                    st.dataframe(
                        pd.DataFrame(input_data),
                        use_container_width=True
                    )
                st.markdown("---")
                if cls == "Dropout": st.error("⚠️ **Alto Rischio** - Tutoring urgente, supporto finanziario, mentoring")
                elif cls == "Enrolled": st.warning("⚡ **Rischio Moderato** - Check-in regolari, gruppi studio")
                else: st.success("✅ **Basso Rischio** - Buone prospettive")
            except Exception as e: st.error(f"Errore: {e}")

    with tab3:
        st.header("📋 Dettagli")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("⚙️ Config"); st.json(meta.get('training_config', {}))
        with c2:
            st.subheader("🎯 Classi")
            for i, n in enumerate(CLASS_NAMES_LIST): st.markdown(f"<span style='color:{CLASS_COLORS[i]}'>●</span> **{i}**: {n}", unsafe_allow_html=True)
        st.markdown("---"); st.subheader("📋 Features")
        for cat, feats in organize_by_category(get_active_features(meta, st.session_state.preprocessor_data)).items():
            with st.expander(f"{CATEGORY_ICONS.get(cat, '📌')} {cat} ({len(feats)})"):
                for f in feats: st.write(f"• **{f['cfg'].get('display', f['col'])}** (`{f['col']}`)")

    with tab4:
        st.header("🎨 Dashboard"); st.pyplot(plot_dashboard(meta)); plt.close()

if __name__ == "__main__":
    main()