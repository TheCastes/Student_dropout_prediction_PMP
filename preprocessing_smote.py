"""
Preprocessing con SMOTE per Random Forest e XGBoost
Dataset: Predict Students' Dropout and Academic Success

METODO: SMOTE (Synthetic Minority Over-sampling Technique)
SPLIT: 80% Train / 20% Test (stratificato)
OTTIMIZZAZIONI: No standardizzazione (inutile per alberi)

REQUISITI:
==========
pip install pandas numpy matplotlib seaborn scikit-learn
pip install imbalanced-learn==0.12.4

Se hai problemi con imbalanced-learn, esegui prima:
pip uninstall -y imbalanced-learn scikit-learn
pip install scikit-learn==1.5.2
pip install imbalanced-learn==0.12.4

OUTPUT:
=======
- train_original.csv (training originale 80% - sbilanciato)
- train_smote.csv (training con SMOTE - bilanciato)
- test_set.csv (test 20% - mai toccato da SMOTE!)
- class_distribution.png (visualizzazione)
- preprocessing_report.txt (report dettagliato)
- smote_info.pkl (informazioni SMOTE)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Import SMOTE con gestione errori
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
    print("✓ imbalanced-learn importato correttamente")
except ImportError as e:
    SMOTE_AVAILABLE = False
    print(f"\n{'='*80}")
    print("⚠️  ERRORE: imbalanced-learn non disponibile")
    print("="*80)
    print(f"\nDettagli: {e}")
    print("\n💡 SOLUZIONE:")
    print("   Nel tuo virtual environment (.venv), esegui:\n")
    print("   pip uninstall -y imbalanced-learn scikit-learn")
    print("   pip install scikit-learn==1.5.2")
    print("   pip install imbalanced-learn==0.12.4")
    print("\n" + "="*80)
    exit(1)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("\n" + "=" * 80)
print("PREPROCESSING CON SMOTE - Random Forest & XGBoost")
print("=" * 80)

output_dir = os.getcwd()
print(f"\n📁 Directory: {output_dir}\n")

# ============================================================================
# 1. CARICAMENTO DATASET
# ============================================================================

print("=" * 80)
print("1. CARICAMENTO DATASET")
print("=" * 80)

csv_files = ['student_data_original.csv', 'data.csv']
csv_path = None

for file in csv_files:
    if os.path.exists(file):
        csv_path = file
        break

if csv_path is None:
    print(f"\n✗ Errore: File non trovato!")
    print(f"   Cercati: {csv_files}")
    exit(1)

print(f"\n📂 Caricamento: {csv_path}")

if 'original' in csv_path:
    df = pd.read_csv(csv_path)
else:
    df = pd.read_csv(csv_path, sep=';', encoding='utf-8-sig')
    df.columns = df.columns.str.strip().str.replace('"', '').str.replace('\t', '')

print(f"✓ Dataset caricato: {len(df)} studenti, {len(df.columns)} variabili")

# ============================================================================
# 2. ANALISI BILANCIAMENTO
# ============================================================================

print("\n" + "=" * 80)
print("2. ANALISI BILANCIAMENTO CLASSI")
print("=" * 80)

target_col = 'Target'
if target_col not in df.columns:
    print(f"\n✗ Errore: Colonna '{target_col}' non trovata!")
    exit(1)

print(f"\nDistribuzione classi (SBILANCIATA):")
print("-" * 80)
target_counts = df[target_col].value_counts().sort_index()
target_pct = df[target_col].value_counts(normalize=True).sort_index() * 100

for status, count in target_counts.items():
    pct = target_pct[status]
    bar = '█' * int(pct / 2)
    print(f"  {status:15s}: {count:5d} ({pct:5.2f}%) {bar}")

imbalance_ratio = target_counts.max() / target_counts.min()
print(f"\n📊 Rapporto di sbilanciamento: {imbalance_ratio:.2f}:1")

if imbalance_ratio < 1.5:
    print("   ✓ Sbilanciamento lieve")
elif imbalance_ratio < 3:
    print("   ⚠️  Sbilanciamento moderato - SMOTE raccomandato")
else:
    print("   🔴 Sbilanciamento severo - SMOTE necessario!")

# ============================================================================
# 3. PREPARAZIONE FEATURES
# ============================================================================

print("\n" + "=" * 80)
print("3. PREPARAZIONE FEATURES")
print("=" * 80)

X = df.drop(columns=[target_col])
y = df[target_col]

print(f"\n✓ Features: {X.shape[1]} variabili")
print(f"✓ Target: {target_col}")

# Identifica tipi
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

print(f"\n📊 Tipi di variabili:")
print(f"   - Numeriche: {len(numeric_cols)}")
print(f"   - Categoriche: {len(categorical_cols)}")

# Gestione missing values
print(f"\n🔧 Controllo valori mancanti...")
missing_counts = X.isnull().sum()
total_missing = missing_counts.sum()

if total_missing > 0:
    print(f"   ⚠️  Trovati {total_missing} valori mancanti")
    for col in numeric_cols:
        if X[col].isnull().sum() > 0:
            X[col].fillna(X[col].median(), inplace=True)
    for col in categorical_cols:
        if X[col].isnull().sum() > 0:
            X[col].fillna(X[col].mode()[0], inplace=True)
    print(f"   ✓ Imputazione completata")
else:
    print(f"   ✓ Nessun valore mancante")

# Encoding categoriche
if len(categorical_cols) > 0:
    print(f"\n🔧 Encoding {len(categorical_cols)} variabili categoriche...")
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    print(f"   ✓ Label encoding completato")

# Encoding target
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)
target_mapping = {i: label for i, label in enumerate(le_target.classes_)}

print(f"\n✓ Target encoding: {target_mapping}")

# ============================================================================
# 4. SPLIT TRAIN/TEST 80/20 STRATIFICATO
# ============================================================================

print("\n" + "=" * 80)
print("4. SPLIT TRAIN/TEST - 80/20 STRATIFICATO")
print("=" * 80)

test_size = 0.20  # 20% test, 80% train
random_state = 42

print(f"\n🔧 Split con test_size={test_size} (stratificato)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=test_size,
    random_state=random_state,
    stratify=y_encoded  # Mantiene proporzioni
)

print(f"\n✓ Split completato:")
print(f"   Training: {len(X_train)} campioni ({(1-test_size)*100:.0f}%)")
print(f"   Test:     {len(X_test)} campioni ({test_size*100:.0f}%)")

train_counts = Counter(y_train)
test_counts = Counter(y_test)

print(f"\n📊 Distribuzione TRAINING SET (prima di SMOTE):")
for label_encoded in sorted(train_counts.keys()):
    label = target_mapping[label_encoded]
    count = train_counts[label_encoded]
    pct = count / len(y_train) * 100
    print(f"   {label:15s}: {count:5d} ({pct:5.2f}%)")

print(f"\n📊 Distribuzione TEST SET (mai toccato!):")
for label_encoded in sorted(test_counts.keys()):
    label = target_mapping[label_encoded]
    count = test_counts[label_encoded]
    pct = count / len(y_test) * 100
    print(f"   {label:15s}: {count:5d} ({pct:5.2f}%)")

# ============================================================================
# 5. APPLICAZIONE SMOTE (solo al training!)
# ============================================================================

print("\n" + "=" * 80)
print("5. APPLICAZIONE SMOTE AL TRAINING SET")
print("=" * 80)

print("\n🔧 Configurazione SMOTE:")
print("   Metodo: SMOTE standard (multiclasse)")
print("   Strategia: auto (bilancia tutte le classi)")
print("   K-neighbors: 5")
print(f"   Random state: {random_state}")

# Inizializza SMOTE
smote = SMOTE(
    sampling_strategy='auto',  # Bilancia tutte le classi
    k_neighbors=5,
    random_state=random_state
)

print(f"\n⏳ Applicazione SMOTE...")
print("   (Generazione campioni sintetici in corso...)")

try:
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print("   ✓ SMOTE applicato con successo!")
except Exception as e:
    print(f"   ✗ Errore durante SMOTE: {e}")
    print("\n💡 Possibili soluzioni:")
    print("   - Verifica che tutte le classi abbiano >= 6 campioni")
    print("   - Prova k_neighbors=3 se hai poche istanze")
    exit(1)

# Statistiche post-SMOTE
smote_counts = Counter(y_train_smote)

print(f"\n📊 RISULTATI SMOTE:")
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
print(f"\n✨ Campioni sintetici creati: {synthetic_samples}")
print(f"   Incremento: {synthetic_samples/len(X_train)*100:.1f}%")
print(f"\n💡 Campioni sintetici sono INTERPOLAZIONI tra campioni reali")
print(f"   (Non semplici duplicati, ma nuovi punti realistici)")

# ============================================================================
# 6. VERIFICA: NO STANDARDIZZAZIONE (per alberi)
# ============================================================================

print("\n" + "=" * 80)
print("6. OTTIMIZZAZIONE PER RANDOM FOREST & XGBOOST")
print("=" * 80)

print("\n✓ NO standardizzazione applicata")
print("   Motivo: Gli alberi decisionali sono invarianti alla scala")
print("   Beneficio: Training più veloce, stessa accuratezza")

print("\n✓ SMOTE applicato solo al training set")
print("   Motivo: Il test deve riflettere la distribuzione reale")
print("   Beneficio: Valutazione onesta del modello")

# ============================================================================
# 7. SALVATAGGIO DATASET
# ============================================================================

print("\n" + "=" * 80)
print("7. SALVATAGGIO DATASET")
print("=" * 80)

# Decodifica target
y_train_decoded = le_target.inverse_transform(y_train)
y_train_smote_decoded = le_target.inverse_transform(y_train_smote)
y_test_decoded = le_target.inverse_transform(y_test)

# 1. Training originale (per confronto)
train_original = X_train.copy()
train_original[target_col] = y_train_decoded
train_original_path = os.path.join(output_dir, 'train_original.csv')
train_original.to_csv(train_original_path, index=False)
print(f"\n✓ Training ORIGINALE: {train_original_path}")
print(f"   {len(train_original)} righe × {len(train_original.columns)} colonne")
print(f"   (Sbilanciato - per confronto)")

# 2. Training con SMOTE (da usare!)
train_smote = X_train_smote.copy()
train_smote[target_col] = y_train_smote_decoded
train_smote_path = os.path.join(output_dir, 'train_smote.csv')
train_smote.to_csv(train_smote_path, index=False)
print(f"\n✓ Training con SMOTE: {train_smote_path}")
print(f"   {len(train_smote)} righe × {len(train_smote.columns)} colonne")
print(f"   ⭐ USA QUESTO per training del modello!")

# 3. Test set (MAI toccato!)
test_set = X_test.copy()
test_set[target_col] = y_test_decoded
test_set_path = os.path.join(output_dir, 'test_set.csv')
test_set.to_csv(test_set_path, index=False)
print(f"\n✓ Test set: {test_set_path}")
print(f"   {len(test_set)} righe × {len(test_set.columns)} colonne")
print(f"   (NON modificato - distribuzione reale)")

# 4. Info SMOTE
smote_info = {
    'strategy': 'auto',
    'k_neighbors': 5,
    'random_state': random_state,
    'original_counts': dict(train_counts),
    'smote_counts': dict(smote_counts),
    'synthetic_samples': synthetic_samples,
    'target_mapping': target_mapping
}
smote_info_path = os.path.join(output_dir, 'smote_info.pkl')
joblib.dump(smote_info, smote_info_path)
print(f"\n✓ Info SMOTE: {smote_info_path}")

# 5. Target encoder
target_encoder_path = os.path.join(output_dir, 'target_encoder.pkl')
joblib.dump(le_target, target_encoder_path)
print(f"✓ Target encoder: {target_encoder_path}")

# ============================================================================
# 8. VISUALIZZAZIONE
# ============================================================================

print("\n" + "=" * 80)
print("8. VISUALIZZAZIONE")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(15, 11))
colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']

# 1. Dataset originale
ax1 = axes[0, 0]
original_counts = df[target_col].value_counts().sort_index()
bars1 = ax1.bar(range(len(original_counts)), original_counts.values, color=colors, edgecolor='black', linewidth=1.5)
ax1.set_title('Dataset Originale Completo\n(4424 campioni)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Classe', fontsize=12)
ax1.set_ylabel('Numero Campioni', fontsize=12)
ax1.set_xticks(range(len(original_counts)))
ax1.set_xticklabels(original_counts.index, rotation=45, ha='right')
ax1.grid(axis='y', alpha=0.3)
for i, bar in enumerate(bars1):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}\n({height/len(df)*100:.1f}%)',
            ha='center', va='bottom', fontweight='bold', fontsize=10)

# 2. Training prima di SMOTE
ax2 = axes[0, 1]
train_before = pd.Series(y_train).map(target_mapping).value_counts().sort_index()
bars2 = ax2.bar(range(len(train_before)), train_before.values, color=colors, edgecolor='black', linewidth=1.5)
ax2.set_title('Training Set PRIMA di SMOTE\n(3539 campioni - 80%)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Classe', fontsize=12)
ax2.set_ylabel('Numero Campioni', fontsize=12)
ax2.set_xticks(range(len(train_before)))
ax2.set_xticklabels(train_before.index, rotation=45, ha='right')
ax2.grid(axis='y', alpha=0.3)
for i, bar in enumerate(bars2):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}\n({height/len(y_train)*100:.1f}%)',
            ha='center', va='bottom', fontweight='bold', fontsize=10)

# 3. Training dopo SMOTE
ax3 = axes[1, 0]
train_after = pd.Series(y_train_smote).map(target_mapping).value_counts().sort_index()
bars3 = ax3.bar(range(len(train_after)), train_after.values, color=colors, edgecolor='black', linewidth=1.5)
ax3.set_title(f'Training Set DOPO SMOTE\n({len(y_train_smote)} campioni - BILANCIATO)', fontsize=14, fontweight='bold')
ax3.set_xlabel('Classe', fontsize=12)
ax3.set_ylabel('Numero Campioni', fontsize=12)
ax3.set_xticks(range(len(train_after)))
ax3.set_xticklabels(train_after.index, rotation=45, ha='right')
ax3.grid(axis='y', alpha=0.3)
for i, bar in enumerate(bars3):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}\n({height/len(y_train_smote)*100:.1f}%)',
            ha='center', va='bottom', fontweight='bold', fontsize=10)

# 4. Test set
ax4 = axes[1, 1]
test_dist = pd.Series(y_test).map(target_mapping).value_counts().sort_index()
bars4 = ax4.bar(range(len(test_dist)), test_dist.values, color=colors, edgecolor='black', linewidth=1.5)
ax4.set_title('Test Set - NON Modificato\n(885 campioni - 20%)', fontsize=14, fontweight='bold')
ax4.set_xlabel('Classe', fontsize=12)
ax4.set_ylabel('Numero Campioni', fontsize=12)
ax4.set_xticks(range(len(test_dist)))
ax4.set_xticklabels(test_dist.index, rotation=45, ha='right')
ax4.grid(axis='y', alpha=0.3)
for i, bar in enumerate(bars4):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}\n({height/len(y_test)*100:.1f}%)',
            ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()
viz_path = os.path.join(output_dir, 'class_distribution.png')
plt.savefig(viz_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Visualizzazione: {viz_path}")

# ============================================================================
# 9. REPORT
# ============================================================================

print("\n" + "=" * 80)
print("9. REPORT DETTAGLIATO")
print("=" * 80)

report_path = os.path.join(output_dir, 'preprocessing_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("PREPROCESSING CON SMOTE - Random Forest & XGBoost\n")
    f.write("=" * 80 + "\n\n")

    f.write("METODO: SMOTE (Synthetic Minority Over-sampling Technique)\n")
    f.write("SPLIT: 80% Training / 20% Test (stratificato)\n")
    f.write("OTTIMIZZAZIONI: No standardizzazione per alberi\n\n")

    f.write("DATASET ORIGINALE:\n")
    f.write("-" * 80 + "\n")
    f.write(f"Totale: {len(df)} campioni\n")
    f.write(f"Features: {len(X.columns)}\n\n")
    for status, count in target_counts.items():
        pct = target_pct[status]
        f.write(f"{status:15s}: {count:5d} ({pct:5.2f}%)\n")
    f.write(f"\nSbilanciamento: {imbalance_ratio:.2f}:1\n\n")

    f.write("SPLIT TRAIN/TEST:\n")
    f.write("-" * 80 + "\n")
    f.write(f"Training: {len(X_train)} campioni (80%)\n")
    f.write(f"Test: {len(X_test)} campioni (20%)\n")
    f.write("Stratificato: Sì (mantiene proporzioni)\n\n")

    f.write("SMOTE APPLICATO:\n")
    f.write("-" * 80 + "\n")
    f.write(f"Strategia: auto (bilancia tutte le classi)\n")
    f.write(f"K-neighbors: 5\n")
    f.write(f"Random state: {random_state}\n\n")

    f.write("Training DOPO SMOTE:\n")
    f.write(f"Totale: {len(X_train_smote)} campioni\n")
    for label_encoded in sorted(smote_counts.keys()):
        label = target_mapping[label_encoded]
        count = smote_counts[label_encoded]
        f.write(f"{label:15s}: {count:5d}\n")
    f.write(f"\nCampioni sintetici: {synthetic_samples}\n")
    f.write(f"Incremento: {synthetic_samples/len(X_train)*100:.1f}%\n\n")

    f.write("FILE GENERATI:\n")
    f.write("-" * 80 + "\n")
    f.write(f"1. {os.path.basename(train_original_path)}\n")
    f.write(f"2. {os.path.basename(train_smote_path)} ⭐ USA QUESTO!\n")
    f.write(f"3. {os.path.basename(test_set_path)}\n")
    f.write(f"4. {os.path.basename(smote_info_path)}\n")
    f.write(f"5. {os.path.basename(target_encoder_path)}\n")
    f.write(f"6. {os.path.basename(viz_path)}\n")
    f.write(f"7. {os.path.basename(report_path)}\n\n")

    f.write("USO CON RANDOM FOREST:\n")
    f.write("-" * 80 + "\n")
    f.write("from sklearn.ensemble import RandomForestClassifier\n")
    f.write("import pandas as pd\n\n")
    f.write("train = pd.read_csv('train_smote.csv')\n")
    f.write("X_train = train.drop('Target', axis=1)\n")
    f.write("y_train = train['Target']\n\n")
    f.write("model = RandomForestClassifier(random_state=42, n_jobs=-1)\n")
    f.write("model.fit(X_train, y_train)\n\n")

    f.write("USO CON XGBOOST:\n")
    f.write("-" * 80 + "\n")
    f.write("from xgboost import XGBClassifier\n\n")
    f.write("model = XGBClassifier(random_state=42, eval_metric='mlogloss')\n")
    f.write("model.fit(X_train, y_train)\n\n")

    f.write("VALUTAZIONE:\n")
    f.write("-" * 80 + "\n")
    f.write("from sklearn.metrics import classification_report, balanced_accuracy_score\n\n")
    f.write("test = pd.read_csv('test_set.csv')\n")
    f.write("X_test = test.drop('Target', axis=1)\n")
    f.write("y_test = test['Target']\n\n")
    f.write("y_pred = model.predict(X_test)\n")
    f.write("print(classification_report(y_test, y_pred))\n")
    f.write("print(f'Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.3f}')\n")

print(f"✓ Report: {report_path}")

# ============================================================================
# RIEPILOGO FINALE
# ============================================================================

print("\n" + "=" * 80)
print("✅ PREPROCESSING CON SMOTE COMPLETATO!")
print("=" * 80)

print(f"\n📁 Directory: {output_dir}\n")
print("📦 File generati:")
print(f"  1. train_original.csv     - Training originale (sbilanciato)")
print(f"  2. train_smote.csv        - Training con SMOTE ⭐ USA QUESTO!")
print(f"  3. test_set.csv           - Test set (mai modificato)")
print(f"  4. smote_info.pkl         - Informazioni SMOTE")
print(f"  5. target_encoder.pkl     - Encoder del target")
print(f"  6. class_distribution.png - Visualizzazione")
print(f"  7. preprocessing_report.txt - Report completo")

print("\n" + "=" * 80)
print("🚀 QUICK START - RANDOM FOREST")
print("=" * 80)
print("""
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, balanced_accuracy_score
import pandas as pd

# Carica training con SMOTE
train = pd.read_csv('train_smote.csv')
X_train = train.drop('Target', axis=1)
y_train = train['Target']

# Carica test
test = pd.read_csv('test_set.csv')
X_test = test.drop('Target', axis=1)
y_test = test['Target']

# Training
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Valutazione
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.3f}")
""")

print("=" * 80)
print("🚀 QUICK START - XGBOOST")
print("=" * 80)
print("""
from xgboost import XGBClassifier

# Stesso preprocessing di Random Forest
model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss')
model.fit(X_train, y_train)

# Valutazione
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
""")

print("=" * 80)
print("✨ CARATTERISTICHE:")
print("=" * 80)
print(f"✓ SMOTE applicato: {synthetic_samples} campioni sintetici creati")
print(f"✓ Split 80/20 stratificato")
print(f"✓ Training bilanciato: {len(X_train_smote)} campioni")
print(f"✓ Test intatto: {len(X_test)} campioni (distribuzione reale)")
print(f"✓ NO standardizzazione (inutile per alberi)")
print(f"✓ Pronto per Random Forest e XGBoost")
print("=" * 80)

plt.show()