# ============================================================================
# COLORI CLASSI TARGET (Dropout, Enrolled, Graduate)
# ============================================================================
CLASS_COLORS = {
    'Dropout': '#e74c3c',  # Rosso
    'Enrolled': '#f39c12',  # Arancione
    'Graduate': '#27ae60'  # Verde
}

CLASS_COLORS_LIST = ['#e74c3c', '#f39c12', '#27ae60']

CLASS_COLORS_LIGHT = {
    'Dropout': '#ffebee',
    'Enrolled': '#fff3e0',
    'Graduate': '#e8f5e9'
}

# ============================================================================
# COLORI MODELLI (Random Forest, XGBoost)
# ============================================================================

MODEL_COLORS = {
    'Random Forest': '#3498db',  # Blu
    'RandomForest': '#3498db',  # Alias
    'rf': '#3498db',  # Alias
    'XGBoost': '#9b59b6',  # Viola
    'xgb': '#9b59b6',  # Alias
    'XGB': '#9b59b6'  # Alias
}
MODEL_COLORS_LIST = ['#3498db', '#9b59b6']
# ============================================================================
# COLORI METRICHE
# ============================================================================
METRIC_COLORS = {
    'Balanced Accuracy': '#3498db',  # Blu
    'F1-Score': '#2ecc71',  # Verde
    'Precision': '#e67e22',  # Arancione
    'Recall': '#9b59b6',  # Viola
    'Accuracy': '#1abc9c'  # Turchese
}
# ============================================================================
# PALETTE PER HEATMAPS
# ============================================================================
HEATMAP_CMAPS = {
    'correlation': 'RdBu_r',  # Rosso-Blu
    'confusion_matrix': 'Blues',  # Blu graduato
    'importance': 'viridis',  # Viridis (percettualmente uniforme)
    'general': 'coolwarm'  # Alternativa per correlazioni
}
# ============================================================================
# PALETTE CATEGORICHE PER ALTRI GRAFICI
# ============================================================================
# Palette qualitativa per variabili categoriche (es. nazionalità, corsi)
CATEGORICAL_PALETTE = [
    '#3498db',  # Blu
    '#e74c3c',  # Rosso
    '#2ecc71',  # Verde
    '#f39c12',  # Arancione
    '#9b59b6',  # Viola
    '#1abc9c',  # Turchese
    '#34495e',  # Grigio scuro
    '#e67e22',  # Arancione scuro
    '#95a5a6',  # Grigio
    '#16a085'  # Verde acqua scuro
]
# ============================================================================
# COLORI UI
# ============================================================================
UI_COLORS = {
    'background': '#f0f2f6',
    'primary': '#3498db',
    'secondary': '#95a5a6',
    'success': '#27ae60',
    'warning': '#f39c12',
    'danger': '#e74c3c',
    'info': '#3498db'
}
# ============================================================================
# STILI MATPLOTLIB/SEABORN
# ============================================================================
# Stile matplotlib generale
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
# Parametri comuni per tutti i grafici
COMMON_PLOT_PARAMS = {
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#34495e',
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'legend.framealpha': 0.9,
    'grid.alpha': 0.3,
    'grid.color': '#95a5a6'
}


# ============================================================================
# FUNZIONI HELPER
# ============================================================================
def get_class_color(class_name):
    """Restituisce il colore per una classe target"""
    return CLASS_COLORS.get(class_name, '#95a5a6')


def get_model_color(model_name):
    """Restituisce il colore per un modello"""
    return MODEL_COLORS.get(model_name, '#95a5a6')


def get_class_colors_list():
    """Restituisce la lista ordinata dei colori delle classi"""
    return CLASS_COLORS_LIST


def get_model_colors_list():
    """Restituisce la lista ordinata dei colori dei modelli"""
    return MODEL_COLORS_LIST


def setup_plot_style():
    """Configura lo stile matplotlib per tutti i grafici"""
    import matplotlib.pyplot as plt
    plt.style.use(PLOT_STYLE)
    plt.rcParams.update(COMMON_PLOT_PARAMS)


def get_categorical_palette(n=None):
    """
    Restituisce una palette categorica.

    Args:
        n (int): Numero di colori necessari. Se None, restituisce tutta la palette.

    Returns:
        list: Lista di codici colore hex
    """
    if n is None:
        return CATEGORICAL_PALETTE
    return CATEGORICAL_PALETTE[:n]


# ============================================================================
# EXPORT PER SEABORN
# ============================================================================
def get_seaborn_palette(palette_name='classes'):
    """
    Restituisce una palette compatibile con seaborn

    Args:
        palette_name (str): 'classes', 'models', 'categorical'

    Returns:
        list: Lista di colori
    """
    if palette_name == 'classes':
        return CLASS_COLORS_LIST
    elif palette_name == 'models':
        return MODEL_COLORS_LIST
    elif palette_name == 'categorical':
        return CATEGORICAL_PALETTE
    else:
        return CATEGORICAL_PALETTE


def map_colors_to_labels(labels, color_dict=None):
    """
    Mappa i colori alle label in modo coerente.

    CRITICO: Questa funzione garantisce che ogni label abbia SEMPRE lo stesso colore,
    indipendentemente dall'ordine in cui appaiono nei dati.

    Args:
        labels (list, pd.Index, or iterable): Lista di label (es. ['Dropout', 'Graduate', 'Enrolled'])
        color_dict (dict): Dizionario label->colore. Default: CLASS_COLORS

    Returns:
        list: Lista di colori nell'ordine delle label fornite

    Examples:
        >>> map_colors_to_labels(['Dropout', 'Enrolled', 'Graduate'])
        ['#e74c3c', '#f39c12', '#27ae60']

        >>> map_colors_to_labels(['Graduate', 'Dropout'])  # Ordine diverso
        ['#27ae60', '#e74c3c']  # Ma i colori sono mappati correttamente!
    """
    if color_dict is None:
        color_dict = CLASS_COLORS

    return [color_dict.get(label, '#95a5a6') for label in labels]