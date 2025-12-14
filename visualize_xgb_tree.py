"""
Visualizzazione della struttura ad albero di XGBoost
"""
import joblib
import matplotlib.pyplot as plt
from xgboost import plot_tree
import os
import sys

from color_config import setup_plot_style

# ============================================================================
# CONFIGURAZIONE
# ============================================================================
setup_plot_style()

PREADMISSION_MODE = '--preadmission' in sys.argv

print("=" * 80)
print("VISUALIZZAZIONE STRUTTURA XGBOOST")
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
    xgb_path = os.path.join(input_dir, 'xgb_model_preadmission.pkl')
    features_path = os.path.join(input_dir, 'feature_names_preadmission.pkl')
else:
    xgb_path = os.path.join(input_dir, 'xgb_model.pkl')
    features_path = os.path.join(input_dir, 'feature_names.pkl')

# Carica XGBoost
print(f"\nCaricamento XGBoost: {xgb_path}")
xgb_model = joblib.load(xgb_path)
print(f"✓ Modello caricato")
print(f"  - N. boosting rounds: {xgb_model.n_estimators}")
print(f"  - Max depth: {xgb_model.max_depth}")
print(f"  - Learning rate: {xgb_model.learning_rate}")
print(f"  - N. features: {xgb_model.n_features_in_}")

# Carica feature names
print(f"\nCaricamento feature names: {features_path}")
feature_names = joblib.load(features_path)
print(f"✓ Feature names caricate: {len(feature_names)} features")

# Classi
class_names = ['Dropout', 'Enrolled', 'Graduate']
print(f"\nClassi target: {class_names}")

# ============================================================================
# 2. VISUALIZZAZIONE PRIMO ALBERO DETTAGLIATO
# ============================================================================
print("\n" + "=" * 80)
print("2. VISUALIZZAZIONE PRIMO ALBERO (Dettagliato)")
print("=" * 80)

tree_index = 0
print(f"\nVisualizzazione albero #{tree_index} (primo boosting round)...")

# Crea figura grande per albero dettagliato
fig, ax = plt.subplots(figsize=(40, 20))

plot_tree(
    xgb_model,
    num_trees=tree_index,
    ax=ax,
    rankdir='LR'  # Left to Right per migliore leggibilità
)

ax.set_title(f'XGBoost - Albero #{tree_index} (Dettagliato)\n'
             f'Modello: {xgb_model.n_estimators} alberi totali | Max Depth: {xgb_model.max_depth} | LR: {xgb_model.learning_rate}',
             fontsize=24, fontweight='bold', pad=20)

plt.tight_layout()

if PREADMISSION_MODE:
    output_path = os.path.join(output_dir, 'xgb_tree_detailed_preadmission.png')
else:
    output_path = os.path.join(output_dir, 'xgb_tree_detailed.png')

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

# Seleziona 6 alberi rappresentativi (primi rounds di boosting)
n_trees_to_show = 6
tree_indices = [0, 1, 2, 3, 4, 5]  # Primi 6 alberi

print(f"\nVisualizzazione di {n_trees_to_show} alberi: {tree_indices}")

fig, axes = plt.subplots(2, 3, figsize=(36, 20))
axes = axes.flatten()

for idx, tree_idx in enumerate(tree_indices):
    print(f"  - Albero #{tree_idx}")
    
    plot_tree(
        xgb_model,
        num_trees=tree_idx,
        ax=axes[idx],
        rankdir='TB'  # Top to Bottom per griglia
    )
    
    axes[idx].set_title(f'Boosting Round #{tree_idx}', fontsize=14, fontweight='bold', pad=10)

fig.suptitle(f'XGBoost - Primi {n_trees_to_show} Alberi su {xgb_model.n_estimators} totali\n'
             f'Sequential Boosting Process',
             fontsize=20, fontweight='bold', y=0.995)

plt.tight_layout()

if PREADMISSION_MODE:
    output_path = os.path.join(output_dir, 'xgb_trees_grid_preadmission.png')
else:
    output_path = os.path.join(output_dir, 'xgb_trees_grid.png')

print(f"Salvataggio: {output_path}")
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Griglia alberi salvata")

# ============================================================================
# 4. VISUALIZZAZIONE ALBERI TARDIVI (Late Boosting)
# ============================================================================
print("\n" + "=" * 80)
print("4. VISUALIZZAZIONE ALBERI TARDIVI (Late Boosting)")
print("=" * 80)

# Ultimi alberi del processo di boosting
late_tree_indices = [95, 96, 97, 98, 99]  # Ultimi 5 alberi
n_late_trees = len(late_tree_indices)

print(f"\nVisualizzazione ultimi {n_late_trees} alberi: {late_tree_indices}")

fig, axes = plt.subplots(1, 5, figsize=(40, 8))

for idx, tree_idx in enumerate(late_tree_indices):
    print(f"  - Albero #{tree_idx}")
    
    plot_tree(
        xgb_model,
        num_trees=tree_idx,
        ax=axes[idx],
        rankdir='TB'
    )
    
    axes[idx].set_title(f'Round #{tree_idx}', fontsize=12, fontweight='bold', pad=10)

fig.suptitle(f'XGBoost - Ultimi {n_late_trees} Alberi (Fine Boosting)\n'
             f'Alberi che raffinano le predizioni finali',
             fontsize=18, fontweight='bold', y=0.98)

plt.tight_layout()

if PREADMISSION_MODE:
    output_path = os.path.join(output_dir, 'xgb_trees_late_preadmission.png')
else:
    output_path = os.path.join(output_dir, 'xgb_trees_late.png')

print(f"Salvataggio: {output_path}")
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Alberi tardivi salvati")

# ============================================================================
# 5. VISUALIZZAZIONE SINGOLO ALBERO COMPATTO
# ============================================================================
print("\n" + "=" * 80)
print("5. VISUALIZZAZIONE ALBERO COMPATTO (Verticale)")
print("=" * 80)

tree_index = 0
print(f"\nVisualizzazione albero #{tree_index} in formato compatto...")

fig, ax = plt.subplots(figsize=(20, 28))

plot_tree(
    xgb_model,
    num_trees=tree_index,
    ax=ax,
    rankdir='TB'  # Top to Bottom
)

ax.set_title(f'XGBoost - Albero #{tree_index} (Vista Compatta)\n'
             f'Primo albero del processo di boosting',
             fontsize=20, fontweight='bold', pad=20)

plt.tight_layout()

if PREADMISSION_MODE:
    output_path = os.path.join(output_dir, 'xgb_tree_compact_preadmission.png')
else:
    output_path = os.path.join(output_dir, 'xgb_tree_compact.png')

print(f"Salvataggio: {output_path}")
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Albero compatto salvato")

# ============================================================================
# 6. VISUALIZZAZIONE CONFRONTO EARLY vs LATE BOOSTING
# ============================================================================
print("\n" + "=" * 80)
print("6. VISUALIZZAZIONE CONFRONTO EARLY vs LATE BOOSTING")
print("=" * 80)

print(f"\nConfronto albero iniziale vs albero finale...")

fig, axes = plt.subplots(1, 2, figsize=(32, 16))

# Primo albero
plot_tree(
    xgb_model,
    num_trees=0,
    ax=axes[0],
    rankdir='TB'
)
axes[0].set_title('Albero #0 - Early Boosting\n(Primo albero, pattern generali)',
                  fontsize=16, fontweight='bold', pad=15)

# Ultimo albero
plot_tree(
    xgb_model,
    num_trees=99,
    ax=axes[1],
    rankdir='TB'
)
axes[1].set_title('Albero #99 - Late Boosting\n(Ultimo albero, raffinamenti finali)',
                  fontsize=16, fontweight='bold', pad=15)

fig.suptitle(f'XGBoost - Confronto Early vs Late Boosting\n'
             f'Come cambia la struttura durante il processo di boosting',
             fontsize=20, fontweight='bold', y=0.995)

plt.tight_layout()

if PREADMISSION_MODE:
    output_path = os.path.join(output_dir, 'xgb_comparison_early_late_preadmission.png')
else:
    output_path = os.path.join(output_dir, 'xgb_comparison_early_late.png')

print(f"Salvataggio: {output_path}")
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Confronto salvato")

# ============================================================================
# RIEPILOGO FINALE
# ============================================================================
print("\n" + "=" * 80)
print("VISUALIZZAZIONE COMPLETATA")
print("=" * 80)

print(f"\nDirectory: {output_dir}\n")
print("File generati:")
if PREADMISSION_MODE:
    print(f"  1. xgb_tree_detailed_preadmission.png          - Albero completo dettagliato")
    print(f"  2. xgb_trees_grid_preadmission.png             - Griglia primi 6 alberi")
    print(f"  3. xgb_trees_late_preadmission.png             - Ultimi 5 alberi (late boosting)")
    print(f"  4. xgb_tree_compact_preadmission.png           - Albero compatto verticale")
    print(f"  5. xgb_comparison_early_late_preadmission.png  - Confronto primo vs ultimo albero")
else:
    print(f"  1. xgb_tree_detailed.png          - Albero completo dettagliato")
    print(f"  2. xgb_trees_grid.png             - Griglia primi 6 alberi")
    print(f"  3. xgb_trees_late.png             - Ultimi 5 alberi (late boosting)")
    print(f"  4. xgb_tree_compact.png           - Albero compatto verticale")
    print(f"  5. xgb_comparison_early_late.png  - Confronto primo vs ultimo albero")

print("\n" + "=" * 80)
print("LEGENDA XGBOOST:")
print("=" * 80)
print("""
Ogni nodo dell'albero mostra:
  • Feature e soglia di decisione (es. "f3 < 25.5")
    dove f0, f1, f2... sono gli indici delle features
  • Leaf values (valori alle foglie) = contributo alla predizione finale
  
DIFFERENZE CON RANDOM FOREST:
  • XGBoost usa BOOSTING sequenziale (alberi si correggono a vicenda)
  • Random Forest usa BAGGING parallelo (alberi indipendenti)
  • Gli alberi di XGBoost sono più piccoli e specializzati
  • Primi alberi catturano pattern generali
  • Alberi successivi correggono errori residui

NOTA:
  • f0, f1, f2... = indici numerici delle features
  • Per mappare a nomi reali: controlla feature_names.pkl
  • I valori alle foglie sono contributi additivi alla predizione finale
""")
print("=" * 80)
