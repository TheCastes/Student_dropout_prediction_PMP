"""
Se hai problemi con imbalanced-learn, esegui prima:
pip uninstall -y imbalanced-learn scikit-learn
pip install scikit-learn==1.5.2
pip install imbalanced-learn==0.12.4
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import os
import sys
# ============================================================================
# CONFIGURAZIONE MODALITÀ
# ============================================================================
PREADMISSION_MODE = '--preadmission' in sys.argv
import warnings
warnings.filterwarnings('ignore')

from color_config import (
    CLASS_COLORS_LIST,
    setup_plot_style,
    map_colors_to_labels
)
try:
    from imblearn.over_sampling import SMOTE
except ImportError as e:
    print(f"\n{'='*80}")
    print("ERRORE: imbalanced-learn non disponibile")
    print("="*80)
    exit(1)

setup_plot_style()
sns.set_palette(CLASS_COLORS_LIST)

print("\n" + "=" * 80)
if PREADMISSION_MODE:
    print("PREPROCESSING PRE-IMMATRICOLAZIONE CON SMOTE")
    print("=" * 80)
    print("\nMODALITÀ PRE-IMMATRICOLAZIONE ATTIVA")
else:
    print("PREPROCESSING CON SMOTE")
    print("=" * 80)

base_dir = os.getcwd()
if PREADMISSION_MODE:
    input_dir = os.path.join(base_dir, '01_analysis_preadmission')
    output_dir = os.path.join(base_dir, '02_preprocessing_preadmission')
else:
    input_dir = os.path.join(base_dir, '01_analysis')
    output_dir = os.path.join(base_dir, '02_preprocessing')
os.makedirs(output_dir, exist_ok=True)
print(f"\nDirectory base: {base_dir}")
print(f"Input da: {input_dir}")
print(f"Output salvati in: {output_dir}\n")
# ============================================================================
# 1. CARICAMENTO DATASET
# ============================================================================
print("=" * 80)
print("1. CARICAMENTO DATASET")
print("=" * 80)
if PREADMISSION_MODE:
    csv_files = [
        os.path.join(input_dir, 'student_data_preadmission.csv'),
        'student_data_preadmission.csv',
    ]
else:
    csv_files = [
        os.path.join(input_dir, 'student_data_original.csv'),
        'student_data_original.csv',
    ]
csv_path = None

for file in csv_files:
    if os.path.exists(file):
        csv_path = file
        break

if csv_path is None:
    print(f"\n Errore: File non trovato!")
    exit(1)
print(f"\nCaricamento: {csv_path}")
df = pd.read_csv(csv_path)
print(f" Dataset caricato: {len(df)} studenti, {len(df.columns)} variabili")

# ============================================================================
# 2. SPLIT TRAIN/TEST 80/20 STRATIFICATO
# ============================================================================
print("\n" + "=" * 80)
print("2. SPLIT TRAIN/TEST - 80/20 STRATIFICATO")
print("=" * 80)

target_col = 'Target'

X = df.drop(columns=[target_col])
y = df[target_col]

# Encoding target
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)
target_mapping = {i: label for i, label in enumerate(le_target.classes_)}

test_size = 0.20
random_state = 42

print(f"\n Split con test_size = {test_size}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=test_size,
    random_state=random_state,
    stratify=y_encoded
)
print(f"\n Split completato:")
print(f"   Training: {len(X_train)} campioni ({(1-test_size)*100:.0f}%)")
print(f"   Test:     {len(X_test)} campioni ({test_size*100:.0f}%)")

train_counts = Counter(y_train)
test_counts = Counter(y_test)

# ============================================================================
# 3. APPLICAZIONE SMOTE
# ============================================================================
print("\n" + "=" * 80)
print("3. APPLICAZIONE SMOTE AL TRAINING SET")
print("=" * 80)
print("\n Configurazione SMOTE:")
print("   Metodo: SMOTE standard (multiclasse)")
print("   Strategia: auto (bilancia tutte le classi)")
print("   K-neighbors: 5")
print(f"   Random state: {random_state}")
# Inizializza SMOTE
smote = SMOTE(
    sampling_strategy='auto',
    k_neighbors=5,
    random_state=random_state
)

print(f"\n Applicazione SMOTE...")
print("   (Generazione campioni sintetici in corso)")
try:
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print("   SMOTE applicato con successo!")
except Exception as e:
    print(f"   Errore durante SMOTE: {e}")
    exit(1)

# Statistiche post-SMOTE
smote_counts = Counter(y_train_smote)

print("\n" + "=" * 80)
print("RISULTATI SMOTE:")
print("=" * 80)
print(f"Training PRIMA di SMOTE:")
print(f"  Totale: {len(X_train)} campioni")
for label_encoded in sorted(train_counts.keys()):
    label = target_mapping[label_encoded]
    count = train_counts[label_encoded]
    pct = count / len(y_train) * 100
    print(f"  {label:15s}: {count:5d} ({pct:5.2f}%)")

print(f"\nTraining DOPO SMOTE:")
print(f"  Totale: {len(X_train_smote)} campioni")
for label_encoded in sorted(smote_counts.keys()):
    label = target_mapping[label_encoded]
    count = smote_counts[label_encoded]
    pct = count / len(y_train_smote) * 100
    print(f"  {label:15s}: {count:5d} ({pct:5.2f}%)")

synthetic_samples = len(X_train_smote) - len(X_train)
print(f"\n Campioni sintetici creati: {synthetic_samples}")
print(f"   Incremento: {synthetic_samples/len(X_train)*100:.1f}%")

# ============================================================================
# 5. SALVATAGGIO DATASET
# ============================================================================

print("\n" + "=" * 80)
print("5. SALVATAGGIO DATASET")
print("=" * 80)

# Decodifica target
y_train_decoded = le_target.inverse_transform(y_train)
y_train_smote_decoded = le_target.inverse_transform(y_train_smote)
y_test_decoded = le_target.inverse_transform(y_test)

if PREADMISSION_MODE:
    suffix = '_preadmission'
else:
    suffix = ''

# 1. Training originale
train_original = X_train.copy()
train_original[target_col] = y_train_decoded
train_original_path = os.path.join(output_dir, f'train_original{suffix}.csv')
train_original.to_csv(train_original_path, index=False)
print(f"\n Training ORIGINALE: {train_original_path}")
print(f"   {len(train_original)} righe × {len(train_original.columns)} colonne")
if PREADMISSION_MODE:
    print(f"   (features pre-immatricolazione - Sbilanciato)")
else:
    print(f"   (Sbilanciato - per confronto)")

# 2. Training con SMOTE
train_smote = X_train_smote.copy()
train_smote[target_col] = y_train_smote_decoded
train_smote_path = os.path.join(output_dir, f'train_smote{suffix}.csv')
train_smote.to_csv(train_smote_path, index=False)
print(f"\n Training con SMOTE: {train_smote_path}")
print(f"   {len(train_smote)} righe × {len(train_smote.columns)} colonne")

# 3. Test set
test_set = X_test.copy()
test_set[target_col] = y_test_decoded
test_set_path = os.path.join(output_dir, f'test_set{suffix}.csv')
test_set.to_csv(test_set_path, index=False)
print(f"\nTest set: {test_set_path}")
print(f"   {len(test_set)} righe × {len(test_set.columns)} colonne")



# ============================================================================
# 6. VISUALIZZAZIONI
# ============================================================================

print("\n" + "=" * 80)
print("6. GENERAZIONE VISUALIZZAZIONI")
print("=" * 80)

viz_dir = os.path.join(output_dir, 'visualizations')
os.makedirs(viz_dir, exist_ok=True)
print(f"Directory visualizzazioni: {viz_dir}\n")

n_viz = 0

# 1. Dataset originale
fig, ax = plt.subplots(figsize=(10, 6))
original_counts = df[target_col].value_counts().sort_index()
colors_mapped = map_colors_to_labels(original_counts.index)
bars = ax.bar(range(len(original_counts)), original_counts.values,
              color=colors_mapped, edgecolor='black', linewidth=1.5)
ax.set_title('Dataset Originale (4424 campioni)',fontsize=16, fontweight='bold')
ax.set_xlabel('Classe', fontsize=14)
ax.set_ylabel('Numero Campioni', fontsize=14)
ax.set_xticks(range(len(original_counts)))
ax.set_xticklabels(original_counts.index, rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3)
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}\n({height/len(df)*100:.1f}%)',
            ha='center', va='bottom', fontweight='bold', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(viz_dir, '01_dataset_original.png'), dpi=300, bbox_inches='tight')
plt.close()
n_viz += 1
print(f"  {n_viz}. 01_dataset_original.png")

# 2. Training prima di SMOTE
fig, ax = plt.subplots(figsize=(10, 6))
train_before = pd.Series(y_train).map(target_mapping).value_counts().sort_index()
train_before_colors = map_colors_to_labels(train_before.index)
bars = ax.bar(range(len(train_before)), train_before.values,
              color=train_before_colors, edgecolor='black', linewidth=1.5)
ax.set_title('Training Set PRIMA di SMOTE (3539 campioni - 80%)', fontsize=16, fontweight='bold')
ax.set_xlabel('Classe', fontsize=14)
ax.set_ylabel('Numero Campioni', fontsize=14)
ax.set_xticks(range(len(train_before)))
ax.set_xticklabels(train_before.index, rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3)
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}\n({height/len(y_train)*100:.1f}%)',
            ha='center', va='bottom', fontweight='bold', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(viz_dir, '02_train_before_smote.png'), dpi=300, bbox_inches='tight')
plt.close()
n_viz += 1
print(f"  {n_viz}. 02_train_before_smote.png")

# 3. Training dopo SMOTE
fig, ax = plt.subplots(figsize=(10, 6))
train_after = pd.Series(y_train_smote).map(target_mapping).value_counts().sort_index()
train_after_colors = map_colors_to_labels(train_after.index)
bars = ax.bar(range(len(train_after)), train_after.values,
              color=train_after_colors, edgecolor='black', linewidth=1.5)
ax.set_title(f'Training Set dopo SMOTE ({len(y_train_smote)} campioni)', fontsize=16, fontweight='bold')
ax.set_xlabel('Classe', fontsize=14)
ax.set_ylabel('Numero Campioni', fontsize=14)
ax.set_xticks(range(len(train_after)))
ax.set_xticklabels(train_after.index, rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3)
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}\n({height/len(y_train_smote)*100:.1f}%)',
            ha='center', va='bottom', fontweight='bold', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(viz_dir, '03_train_after_smote.png'), dpi=300, bbox_inches='tight')
plt.close()
n_viz += 1
print(f"  {n_viz}. 03_train_after_smote.png")

# 4. Test set
fig, ax = plt.subplots(figsize=(10, 6))
test_dist = pd.Series(y_test).map(target_mapping).value_counts().sort_index()
test_dist_colors = map_colors_to_labels(test_dist.index)
bars = ax.bar(range(len(test_dist)), test_dist.values,
              color=test_dist_colors, edgecolor='black', linewidth=1.5)
ax.set_title('Test Set (885 campioni - 20%)',fontsize=16, fontweight='bold')
ax.set_xlabel('Classe', fontsize=14)
ax.set_ylabel('Numero Campioni', fontsize=14)
ax.set_xticks(range(len(test_dist)))
ax.set_xticklabels(test_dist.index, rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3)
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}\n({height/len(y_test)*100:.1f}%)',
            ha='center', va='bottom', fontweight='bold', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(viz_dir, '04_test_set.png'), dpi=300, bbox_inches='tight')
plt.close()
n_viz += 1
print(f"  {n_viz}. 04_test_set.png")

print(f"\n {n_viz} visualizzazioni generate")
print(f" Directory: {viz_dir}")

# ============================================================================
# RIEPILOGO FINALE
# ============================================================================
print("\n" + "=" * 80)
if PREADMISSION_MODE:
    print("PREPROCESSING PRE-IMMATRICOLAZIONE COMPLETATO")
else:
    print("PREPROCESSING CON SMOTE COMPLETATO")
print("=" * 80)

print(f"\nDirectory: {output_dir}\n")
print("File generati:")

if PREADMISSION_MODE:
    print(f"  1. train_original_preadmission.csv     - Training originale")
    print(f"  2. train_smote_preadmission.csv        - Training con SMOTE")
    print(f"  3. test_set_preadmission.csv           - Test set")
    print(f"  4. visualizations/                     - Grafici e visualizzazioni")
else:
    print(f"  1. train_original.csv     - Training originale")
    print(f"  2. train_smote.csv        - Training con SMOTE")
    print(f"  3. test_set.csv           - Test set")
    print(f"  4. visualizations/        - Grafici e visualizzazioni")
plt.show()