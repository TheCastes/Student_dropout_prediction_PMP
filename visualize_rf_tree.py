"""
Visualizzazione della struttura ad albero della Random Forest
"""
import joblib
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import os
import sys

from color_config import setup_plot_style, CLASS_COLORS_LIST

# ============================================================================
# CONFIGURAZIONE
# ============================================================================
setup_plot_style()

PREADMISSION_MODE = '--preadmission' in sys.argv

print("=" * 80)
print("VISUALIZZAZIONE STRUTTURA RANDOM FOREST")
print("=" * 80)

base_dir = os.getcwd()

if PREADMISSION_MODE:
    input_dir = os.path.join(base_dir, '03_training_preadmission')
    output_dir = os.path.join(base_dir, '04_visualization_preadmission')
else:
    input_dir = os.path.join(base_dir, '03_training')
    output_dir = os.path.join(base_dir, '04_visualization')

os.makedirs(output_dir, exist_ok=True)

print(f"\nDirectory input: {input_dir}")
print(f"Output salvati in: {output_dir}\n")

# ============================================================================
# CARICAMENTO MODELLO
# ============================================================================
print("=" * 80)
print("1. CARICAMENTO MODELLO E METADATI")
print("=" * 80)

if PREADMISSION_MODE:
    rf_path = os.path.join(input_dir, 'rf_model_preadmission.pkl')
    features_path = os.path.join(input_dir, 'feature_names_preadmission.pkl')
else:
    rf_path = os.path.join(input_dir, 'rf_model.pkl')
    features_path = os.path.join(input_dir, 'feature_names.pkl')

# Carica Random Forest
print(f"\nCaricamento Random Forest: {rf_path}")
rf_model = joblib.load(rf_path)
print(f"✓ Modello caricato")
print(f"  - N. alberi: {rf_model.n_estimators}")
print(f"  - Max depth: {rf_model.max_depth if rf_model.max_depth else 'Unlimited'}")
print(f"  - N. features: {rf_model.n_features_in_}")

# Carica feature names
print(f"\nCaricamento feature names: {features_path}")
feature_names = joblib.load(features_path)
print(f"✓ Feature names caricate: {len(feature_names)} features")

# Classi
class_names = ['Dropout', 'Enrolled', 'Graduate']
print(f"\nClassi target: {class_names}")

# ============================================================================
# 2. VISUALIZZAZIONE SINGOLO ALBERO DETTAGLIATO
# ============================================================================
print("\n" + "=" * 80)
print("2. VISUALIZZAZIONE ALBERO SINGOLO (Dettagliato)")
print("=" * 80)

tree_index = 0
print(f"\nVisualizzazione albero #{tree_index} (primo della foresta)...")

# Crea figura grande per albero dettagliato
fig, ax = plt.subplots(figsize=(40, 20))

plot_tree(
    rf_model.estimators_[tree_index],
    feature_names=feature_names,
    class_names=class_names,
    filled=True,
    rounded=True,
    fontsize=8,
    ax=ax,
    proportion=True,
    precision=2
)

ax.set_title(f'Random Forest - Albero #{tree_index} (Dettagliato)\n'
             f'Modello: {rf_model.n_estimators} alberi totali',
             fontsize=24, fontweight='bold', pad=20)

plt.tight_layout()

if PREADMISSION_MODE:
    output_path = os.path.join(output_dir, 'rf_tree_detailed_preadmission.png')
else:
    output_path = os.path.join(output_dir, 'rf_tree_detailed.png')

print(f"Salvataggio: {output_path}")
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Albero dettagliato salvato")

# ============================================================================
# 3. VISUALIZZAZIONE GRIGLIA DI ALBERI
# ============================================================================
print("\n" + "=" * 80)
print("3. VISUALIZZAZIONE GRIGLIA DI ALBERI")
print("=" * 80)

# Seleziona 4 alberi rappresentativi
n_trees_to_show = 4
tree_indices = [0, 25, 50, 75]  # Primo, 25°, 50°, 75°

print(f"\nVisualizzazione di {n_trees_to_show} alberi: {tree_indices}")

fig, axes = plt.subplots(2, 2, figsize=(30, 24))
axes = axes.flatten()

for idx, tree_idx in enumerate(tree_indices):
    print(f"  - Albero #{tree_idx}")
    
    plot_tree(
        rf_model.estimators_[tree_idx],
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=6,
        ax=axes[idx],
        proportion=True,
        precision=2,
        max_depth=4  # Limita profondità per leggibilità
    )
    
    axes[idx].set_title(f'Albero #{tree_idx}', fontsize=16, fontweight='bold', pad=10)

fig.suptitle(f'Random Forest - Campione di {n_trees_to_show} Alberi su {rf_model.n_estimators} totali\n'
             f'(Profondità limitata a 4 livelli per leggibilità)',
             fontsize=20, fontweight='bold', y=0.995)

plt.tight_layout()

if PREADMISSION_MODE:
    output_path = os.path.join(output_dir, 'rf_trees_grid_preadmission.png')
else:
    output_path = os.path.join(output_dir, 'rf_trees_grid.png')

print(f"Salvataggio: {output_path}")
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Griglia alberi salvata")

# ============================================================================
# 4. VISUALIZZAZIONE ALBERO COMPATTO
# ============================================================================
print("\n" + "=" * 80)
print("4. VISUALIZZAZIONE ALBERO COMPATTO (Primi 3 livelli)")
print("=" * 80)

print(f"\nVisualizzazione albero #{tree_index} con max_depth=3...")

fig, ax = plt.subplots(figsize=(24, 14))

plot_tree(
    rf_model.estimators_[tree_index],
    feature_names=feature_names,
    class_names=class_names,
    filled=True,
    rounded=True,
    fontsize=10,
    ax=ax,
    proportion=True,
    precision=2,
    max_depth=3  # Solo primi 3 livelli
)

ax.set_title(f'Random Forest - Albero #{tree_index} (Primi 3 Livelli)\n'
             f'Vista compatta per maggiore leggibilità',
             fontsize=20, fontweight='bold', pad=20)

plt.tight_layout()

if PREADMISSION_MODE:
    output_path = os.path.join(output_dir, 'rf_tree_compact_preadmission.png')
else:
    output_path = os.path.join(output_dir, 'rf_tree_compact.png')

print(f"Salvataggio: {output_path}")
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Albero compatto salvato")

# ============================================================================
# RIEPILOGO FINALE
# ============================================================================
print("\n" + "=" * 80)
print("VISUALIZZAZIONE COMPLETATA")
print("=" * 80)

print(f"\nDirectory: {output_dir}\n")
print("File generati:")
if PREADMISSION_MODE:
    print(f"  1. rf_tree_detailed_preadmission.png  - Albero completo dettagliato")
    print(f"  2. rf_trees_grid_preadmission.png     - Griglia 2x2 di alberi")
    print(f"  3. rf_tree_compact_preadmission.png   - Albero compatto (3 livelli)")
else:
    print(f"  1. rf_tree_detailed.png  - Albero completo dettagliato")
    print(f"  2. rf_trees_grid.png     - Griglia 2x2 di alberi")
    print(f"  3. rf_tree_compact.png   - Albero compatto (3 livelli)")

print("\n" + "=" * 80)
print("LEGENDA:")
print("=" * 80)
print("""
Ogni nodo dell'albero mostra:
  • Feature e soglia di decisione (es. "Age at enrollment <= 25.5")
  • Gini impurity (misura di impurità del nodo)
  • Samples (numero di campioni che raggiungono il nodo)
  • Value (distribuzione campioni per classe: [Dropout, Enrolled, Graduate])
  • Class (classe maggioritaria in quel nodo)

Colori:
  • Arancione/Rosso = tendenza Dropout
  • Giallo/Arancione = tendenza Enrolled  
  • Verde = tendenza Graduate
  • Intensità colore = purezza della previsione
""")
print("=" * 80)
