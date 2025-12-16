import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
)
from sklearn.metrics import (
    confusion_matrix,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    make_scorer
)
from sklearn.preprocessing import LabelEncoder
import joblib
import time
import os
import sys
import warnings

warnings.filterwarnings('ignore')

# Importa SMOTE
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
except ImportError:
    print("ERRORE: imbalanced-learn non disponibile")
    exit(1)

from color_config import (
    CLASS_COLORS_LIST,
     MODEL_COLORS,
     MODEL_COLORS_LIST,
    HEATMAP_CMAPS,
    setup_plot_style,
)
setup_plot_style()
sns.set_palette(CLASS_COLORS_LIST)
HAS_COLOR_CONFIG = True

# ============================================================================
# CONFIGURAZIONE MODALITÀ
# ============================================================================
PREADMISSION_MODE = '--preadmission' in sys.argv

print("\n" + "=" * 80)
if PREADMISSION_MODE:
    print("TRAINING - MODALITÀ PRE-IMMATRICOLAZIONE")
else:
    print("TRAINING")
print("=" * 80)

base_dir = os.getcwd()

if PREADMISSION_MODE:
    output_dir = os.path.join(base_dir, '02_training_preadmission')
else:
    output_dir = os.path.join(base_dir, '02_training')

os.makedirs(output_dir, exist_ok=True)

print(f"\nDirectory base: {base_dir}")
print(f"Output salvati in: {output_dir}\n")
# ============================================================================
# 1. CARICAMENTO DATASET ORIGINALE
# ============================================================================
print("=" * 80)
print("1. CARICAMENTO DATASET")
print("=" * 80)

if PREADMISSION_MODE:
    possible_files = [
        'student_data_preadmission.csv',
        '01_analysis_preadmission/student_data_preadmission.csv',
        'data_preadmission.csv',
    ]
else:
    possible_files = [
        'student_data_original.csv',
        '01_analysis/student_data_original.csv',
        'data.csv',
        'student_data.csv',
    ]

csv_path = None
for file in possible_files:
    if os.path.exists(file):
        csv_path = file
        break

if csv_path is None:
    print(f"\nErrore: Dataset non trovato!")
    print(f"\nPercorsi cercati:")
    for file in possible_files:
        print(f"  - {file}")
    print("\nAssicurati che il file dataset sia nella directory corrente.")
    exit(1)

print(f"\nCaricamento: {csv_path}")
df = pd.read_csv(csv_path)
print(f"   Dataset caricato: {len(df)} studenti, {len(df.columns)} variabili")

if 'Target' not in df.columns:
    print(f"\nErrore: Colonna 'Target' non trovata!")
    print(f"Colonne disponibili: {df.columns.tolist()}")
    exit(1)

print(f"\nDistribuzione classi nel dataset completo:")
target_dist = df['Target'].value_counts().sort_index()
for cls, count in target_dist.items():
    pct = count / len(df) * 100
    print(f"  {cls:15s}: {count:5d} ({pct:5.2f}%)")

# ============================================================================
# 2. SPLIT TRAIN/TEST (80/20 STRATIFICATO)
# ============================================================================
print("\n" + "=" * 80)
print("2. SPLIT TRAIN/TEST - 80/20 STRATIFICATO")
print("=" * 80)

X = df.drop('Target', axis=1)
y = df['Target']

test_size = 0.20
random_state = 42

print(f"\nParametri split:")
print(f"  • Test size: {test_size*100:.0f}%")
print(f"  • Random state: {random_state}")
print(f"  • Stratificazione: Sì")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=test_size,
    random_state=random_state,
    stratify=y
)

print(f"\nSplit completato:")
print(f"  • Training: {len(X_train)} campioni ({(1-test_size)*100:.0f}%)")
print(f"  • Test:     {len(X_test)} campioni ({test_size*100:.0f}%)")

# Distribuzione training
print(f"\nDistribuzione training set:")
train_dist = y_train.value_counts().sort_index()
for cls, count in train_dist.items():
    pct = count / len(y_train) * 100
    print(f"  {cls:15s}: {count:5d} ({pct:5.2f}%)")

print(f"\nDistribuzione test set:")
test_dist = y_test.value_counts().sort_index()
for cls, count in test_dist.items():
    pct = count / len(y_test) * 100
    print(f"  {cls:15s}: {count:5d} ({pct:5.2f}%)")

feature_names = X_train.columns.tolist()
print(f"\nFeatures: {len(feature_names)} variabili")

print(f"\nEncoding classi per XGBoost...")
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)
classes = sorted(y_train.unique())
print(f"   Mapping: {dict(enumerate(le.classes_))}")

# ============================================================================
# 3. CONFIGURAZIONE CROSS-VALIDATION
# ============================================================================
print("\n" + "=" * 80)
print("3. CONFIGURAZIONE 5-FOLD CROSS-VALIDATION")
print("=" * 80)

n_folds = 5
cv = StratifiedKFold(
    n_splits=n_folds,
    shuffle=True,
    random_state=random_state
)

print(f"\nStrategia: StratifiedKFold")
print(f"N. folds: {n_folds}")
print(f"Shuffle: True")
print(f"Random state: {random_state}")

scorer = make_scorer(balanced_accuracy_score)

# ============================================================================
# 4. RANDOM FOREST - TRAINING & CV
# ============================================================================
print("\n" + "=" * 80)
print("4. RANDOM FOREST - TRAINING CON 5-FOLD CV")
print("=" * 80)

rf_pipeline = ImbPipeline([
    ('smote', SMOTE(k_neighbors=5, random_state=random_state)),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1
    ))
])

print(f"\nCross-validation in corso ({n_folds} folds)...")
start_time = time.time()

rf_cv_scores = cross_val_score(
    rf_pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring=scorer,
    n_jobs=-1
)

rf_cv_time = time.time() - start_time

print(f"\nTraining finale sul training set...")
start_time = time.time()

rf_pipeline.fit(X_train, y_train)

rf_train_time = time.time() - start_time
print(f"Training completato in {rf_train_time:.2f}s")

print(f"\nValutazione su test set:")
rf_test_pred = rf_pipeline.predict(X_test)
rf_test_ba = balanced_accuracy_score(y_test, rf_test_pred)
rf_test_f1 = f1_score(y_test, rf_test_pred, average='macro')

print(f"  • Test F1-Score (macro):  {rf_test_f1:.4f}")

# ============================================================================
# 5. XGBOOST - TRAINING & CV
# ============================================================================
print("\n" + "=" * 80)
print("5. XGBOOST - TRAINING CON 5-FOLD CV")
print("=" * 80)

xgb_pipeline = ImbPipeline([
    ('smote', SMOTE(k_neighbors=5, random_state=random_state)),
    ('classifier', XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=random_state,
        eval_metric='mlogloss'
    ))
])

print(f"\nCross-validation in corso ({n_folds} folds)...")

start_time = time.time()

xgb_cv_scores = cross_val_score(
    xgb_pipeline,
    X_train,
    y_train_encoded,
    cv=cv,
    scoring=scorer,
    n_jobs=-1
)

xgb_cv_time = time.time() - start_time

print(f"\nTraining finale sul training set...")
start_time = time.time()

xgb_pipeline.fit(X_train, y_train_encoded)

xgb_train_time = time.time() - start_time
print(f"Training completato in {xgb_train_time:.2f}s")

# Valutazione su test set
print(f"\nValutazione su test set:")
xgb_test_pred_encoded = xgb_pipeline.predict(X_test)
xgb_test_pred = le.inverse_transform(xgb_test_pred_encoded)
xgb_test_ba = balanced_accuracy_score(y_test, xgb_test_pred)
xgb_test_f1 = f1_score(y_test, xgb_test_pred, average='macro')

print(f"  • Test F1-Score (macro):  {xgb_test_f1:.4f}")

results = {
    'Random Forest': {
        'cv_mean': rf_cv_scores.mean(),
        'cv_std': rf_cv_scores.std(),
        'cv_scores': rf_cv_scores,
        'test_ba': rf_test_ba,
        'test_f1': rf_test_f1,
        'cv_time': rf_cv_time,
        'train_time': rf_train_time,
        'predictions': rf_test_pred,
        'pipeline': rf_pipeline
    },
    'XGBoost': {
        'cv_mean': xgb_cv_scores.mean(),
        'cv_std': xgb_cv_scores.std(),
        'cv_scores': xgb_cv_scores,
        'test_ba': xgb_test_ba,
        'test_f1': xgb_test_f1,
        'cv_time': xgb_cv_time,
        'train_time': xgb_train_time,
        'predictions': xgb_test_pred,
        'pipeline': xgb_pipeline
    }
}
# ============================================================================
# 7. VISUALIZZAZIONI
# ============================================================================
print("\n" + "=" * 80)
print("7. GENERAZIONE VISUALIZZAZIONI")
print("=" * 80)

viz_dir = os.path.join(output_dir, 'visualizations')
os.makedirs(viz_dir, exist_ok=True)
print(f"Directory visualizzazioni: {viz_dir}\n")

n_viz = 0

# 7.1 Confusion Matrices
print("Generazione confusion matrices...")
for model_name, model_data in results.items():
    safe_name = model_name.lower().replace(' ', '_')
    y_pred = model_data['predictions']
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = confusion_matrix(y_test, y_pred, normalize='true')

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap=HEATMAP_CMAPS['confusion_matrix'], ax=axes[0],
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'}, annot_kws={'fontsize': 11, 'fontweight': 'bold'})
    axes[0].set_xlabel('Predicted', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Actual', fontsize=13, fontweight='bold')
    axes[0].set_title(f'{model_name} - Absolute Counts', fontsize=14, fontweight='bold')

    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap=HEATMAP_CMAPS['confusion_matrix'], ax=axes[1],
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Proportion'}, annot_kws={'fontsize': 11, 'fontweight': 'bold'})
    axes[1].set_xlabel('Predicted', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Actual', fontsize=13, fontweight='bold')
    axes[1].set_title(f'{model_name} - Normalized (by true label)', fontsize=14, fontweight='bold')

    fig.suptitle(f'{model_name} - Confusion Matrices\n',
                 fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()
    cm_path = os.path.join(viz_dir, f'01_confusion_matrices_{safe_name}.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    n_viz += 1
    print(f"  {n_viz}. {os.path.basename(cm_path)}")

# 7.2 F1-Score per Classe
print("\nGenerazione F1-Score per classe...")
for model_name, model_data in results.items():
    safe_name = model_name.lower().replace(' ', '_')
    y_pred = model_data['predictions']

    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(12, 7))

    model_color = MODEL_COLORS.get(model_name, MODEL_COLORS_LIST[0])

    bars = ax.barh(classes, f1, color=model_color,
                   edgecolor='black', linewidth=1.5, alpha=0.85)
    ax.set_xlabel('F1-Score', fontsize=14, fontweight='bold')
    ax.set_title(f'{model_name} - F1-Score by Class', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlim(0, 1.05)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.invert_yaxis()

    ax.axvline(x=f1.mean(), color='red', linestyle='--', linewidth=2.5,
               label=f'Mean: {f1.mean():.3f}', alpha=0.7)
    ax.legend(fontsize=12)

    for i, (bar, val) in enumerate(zip(bars, f1)):
        ax.text(val + 0.015, bar.get_y() + bar.get_height() / 2, f'{val:.3f}',
                va='center', fontweight='bold', fontsize=11)

    plt.tight_layout()
    f1_path = os.path.join(viz_dir, f'02_f1score_by_class_{safe_name}.png')
    plt.savefig(f1_path, dpi=300, bbox_inches='tight')
    plt.close()
    n_viz += 1
    print(f"  {n_viz}. {os.path.basename(f1_path)}")

print(f"\n{n_viz} visualizzazioni generate")
print(f"Directory: {viz_dir}")

# ============================================================================
# 8. SALVATAGGIO MODELLI E RISULTATI
# ============================================================================
print("\n" + "=" * 80)
print("8. SALVATAGGIO MODELLI E RISULTATI")
print("=" * 80)

suffix = '_preadmission' if PREADMISSION_MODE else ''

rf_path = os.path.join(output_dir, f'rf_model{suffix}.pkl')
joblib.dump(rf_pipeline, rf_path)
print(f"\nRandom Forest: {rf_path}")

xgb_path = os.path.join(output_dir, f'xgb_model{suffix}.pkl')
joblib.dump(xgb_pipeline, xgb_path)
print(f"XGBoost: {xgb_path}")

features_path = os.path.join(output_dir, f'feature_names{suffix}.pkl')
joblib.dump(feature_names, features_path)
print(f"Feature names: {features_path}")

# Salva label encoder
le_path = os.path.join(output_dir, f'label_encoder{suffix}.pkl')
joblib.dump(le, le_path)
print(f"Label encoder: {le_path}")

results_data = {
    'random_forest': {
        'cv_scores': rf_cv_scores.tolist(),
        'cv_mean': float(rf_cv_scores.mean()),
        'cv_std': float(rf_cv_scores.std()),
        'test_ba': float(rf_test_ba),
        'test_f1': float(rf_test_f1),
        'gap_cv_test': float(abs(rf_cv_scores.mean() - rf_test_ba)),
        'cv_time': float(rf_cv_time),
        'train_time': float(rf_train_time)
    },
    'xgboost': {
        'cv_scores': xgb_cv_scores.tolist(),
        'cv_mean': float(xgb_cv_scores.mean()),
        'cv_std': float(xgb_cv_scores.std()),
        'test_ba': float(xgb_test_ba),
        'test_f1': float(xgb_test_f1),
        'gap_cv_test': float(abs(xgb_cv_scores.mean() - xgb_test_ba)),
        'cv_time': float(xgb_cv_time),
        'train_time': float(xgb_train_time)
    },
    'classes': classes,
    'feature_names': feature_names,
    'split_info': {
        'test_size': test_size,
        'random_state': random_state,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_features': len(feature_names)
    },
}

results_path = os.path.join(output_dir, f'training_results{suffix}.pkl')
joblib.dump(results_data, results_path)
test_results_df = pd.DataFrame({
    'True_Label': y_test,
    'RF_Prediction': rf_test_pred,
    'XGB_Prediction': xgb_test_pred,
    'RF_Correct': (y_test == rf_test_pred).astype(int),
    'XGB_Correct': (y_test == xgb_test_pred).astype(int)
})
test_csv_path = os.path.join(output_dir, f'test_predictions{suffix}.csv')
test_results_df.to_csv(test_csv_path, index=False)
print(f"Predizioni test set: {test_csv_path}")

# ============================================================================
# RIEPILOGO FINALE
# ============================================================================
print("\n" + "=" * 80)
if PREADMISSION_MODE:
    print("TRAINING PRE-IMMATRICOLAZIONE COMPLETATO")
else:
    print("TRAINING COMPLETATO CON SUCCESSO")
print("=" * 80)

print(f"\nDirectory: {output_dir}\n")
print("File generati:")
print(f"  1. rf_model{suffix}.pkl              - Modello RF")
print(f"  2. xgb_model{suffix}.pkl             - Modello XGB")
print(f"  3. feature_names{suffix}.pkl            - Nomi features")
print(f"  4. label_encoder{suffix}.pkl            - Encoder classi")
print(f"  5. training_results{suffix}.pkl         - Risultati dettagliati")
print(f"  6. test_predictions{suffix}.csv         - Predizioni su test set")
print(f"  7. visualizations/                      - {n_viz} grafici generati")
print("\n" + "=" * 80)