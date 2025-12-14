import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import (
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
import joblib
import time
import os
import sys
import warnings

warnings.filterwarnings('ignore')

from color_config import (
    CLASS_COLORS_LIST,
    MODEL_COLORS,
    MODEL_COLORS_LIST,
    HEATMAP_CMAPS,
    setup_plot_style,
)
# ============================================================================
# CONFIGURAZIONE MODALITÀ
# ============================================================================
PREADMISSION_MODE = '--preadmission' in sys.argv

setup_plot_style()
sns.set_palette(CLASS_COLORS_LIST)

print("=" * 80)
if PREADMISSION_MODE:
    print("TRAINING RANDOM FOREST & XGBOOST con 5-Fold Cross-Validation")
    print("=" * 80)
    print("\nMODALITÀ PRE-IMMATRICOLAZIONE ATTIVA")
else:
    print("TRAINING RANDOM FOREST & XGBOOST con 5-Fold Cross-Validation")
    print("=" * 80)

base_dir = os.getcwd()

if PREADMISSION_MODE:
    input_dir = os.path.join(base_dir, '02_preprocessing_preadmission')
    output_dir = os.path.join(base_dir, '03_training_preadmission')
else:
    input_dir = os.path.join(base_dir, '02_preprocessing')
    output_dir = os.path.join(base_dir, '03_training')

os.makedirs(output_dir, exist_ok=True)

print(f"\nDirectory base: {base_dir}")
print(f"Input da: {input_dir}")
print(f"Output salvati in: {output_dir}\n")

# ============================================================================
# 1. CARICAMENTO DATI
# ============================================================================
print("=" * 80)
print("1. CARICAMENTO DATI")
print("=" * 80)

if PREADMISSION_MODE:
    train_smote_path = os.path.join(input_dir, 'train_smote_preadmission.csv')
    test_set_path = os.path.join(input_dir, 'test_set_preadmission.csv')
else:
    train_smote_path = os.path.join(input_dir, 'train_smote.csv')
    test_set_path = os.path.join(input_dir, 'test_set.csv')

required_files = {
    'train_smote.csv': train_smote_path,
    'test_set.csv': test_set_path
}

for name, path in required_files.items():
    if not os.path.exists(path):
        print(f"\nErrore: File '{name}' non trovato!")
        print(f"   Percorso atteso: {path}")
        exit(1)
print("\nCaricamento training set")
train = pd.read_csv(train_smote_path)
X_train = train.drop('Target', axis=1)
y_train = train['Target']

print(f"Training set: {len(X_train)} campioni × {len(X_train.columns)} features")

print("\nCaricamento test set...")
test = pd.read_csv(test_set_path)
X_test = test.drop('Target', axis=1)
y_test = test['Target']

print(f"Test set: {len(X_test)} campioni × {len(X_test.columns)} features")

classes = sorted(y_train.unique())
print(f"\nClassi target: {classes}")

train_dist = y_train.value_counts().sort_index()
print(f"\nDistribuzione training:")
for cls in classes:
    count = train_dist[cls]
    pct = count / len(y_train) * 100
    print(f"  {cls:15s}: {count:5d} ({pct:5.2f}%)")

test_dist = y_test.value_counts().sort_index()
print(f"\nDistribuzione test:")
for cls in classes:
    count = test_dist[cls]
    pct = count / len(y_test) * 100
    print(f"  {cls:15s}: {count:5d} ({pct:5.2f}%)")

print(f"\nEncoding classi per XGBoost...")
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)
print(f"   Mapping: {dict(enumerate(le.classes_))}")

# ============================================================================
# 2. CONFIGURAZIONE CROSS-VALIDATION
# ============================================================================

print("\n" + "=" * 80)
print("2. CONFIGURAZIONE 5-FOLD CROSS-VALIDATION")
print("=" * 80)

n_folds = 5
random_state = 42

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
# 3. RANDOM FOREST - TRAINING & CV
# ============================================================================

print("\n" + "=" * 80)
print("3. RANDOM FOREST - TRAINING CON 5-FOLD CV")
print("=" * 80)

print("\n🌲 Configurazione Random Forest:")
print("   - n_estimators: 100")
print("   - random_state: 42")

rf_base = RandomForestClassifier(
    n_estimators=100,
    random_state=random_state,
    n_jobs=-1
)

print(f"\nCross-validation in corso ({n_folds} folds)...")
start_time = time.time()

rf_cv_scores = cross_val_score(
    rf_base,
    X_train,
    y_train,  # Random Forest accetta stringhe
    cv=cv,
    scoring=scorer,
    n_jobs=-1
)

rf_cv_time = time.time() - start_time

print(f"Cross-validation completata in {rf_cv_time:.2f}s")
print(f"\nRisultati CV:")
print(f"   Balanced Accuracy: {rf_cv_scores.mean():.4f} (+/- {rf_cv_scores.std():.4f})")
print(f"   Scores per fold: {[f'{s:.4f}' for s in rf_cv_scores]}")

print(f"\nTraining sul full training set...")
start_time = time.time()

rf_base.fit(X_train, y_train)

rf_train_time = time.time() - start_time
print(f"Training completato in {rf_train_time:.2f}s")

print(f"\nValutazione su test set...")
rf_test_pred = rf_base.predict(X_test)
rf_test_ba = balanced_accuracy_score(y_test, rf_test_pred)
rf_test_f1 = f1_score(y_test, rf_test_pred, average='macro')

print(f"Test Balanced Accuracy: {rf_test_ba:.4f}")
print(f"Test F1-Score (macro): {rf_test_f1:.4f}")

# ============================================================================
# 4. XGBOOST - TRAINING & CV
# ============================================================================

print("\n" + "=" * 80)
print("4. XGBOOST - TRAINING CON 5-FOLD CV")
print("=" * 80)

print("\nConfigurazione XGBoost:")
print("   - n_estimators: 100")
print("   - learning_rate: 0.1")
print("   - max_depth: 6")
print("   - eval_metric: mlogloss")
print("   - random_state: 42")

# Modello base
xgb_base = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=random_state,
    eval_metric='mlogloss'
)

print(f"\nCross-validation in corso ({n_folds} folds)...")
start_time = time.time()

xgb_cv_scores = cross_val_score(
    xgb_base,
    X_train,
    y_train_encoded,
    cv=cv,
    scoring=scorer,
    n_jobs=-1
)

xgb_cv_time = time.time() - start_time

print(f"Cross-validation completata in {xgb_cv_time:.2f}s")
print(f"\nRisultati CV:")
print(f"   Balanced Accuracy: {xgb_cv_scores.mean():.4f} (+/- {xgb_cv_scores.std():.4f})")
print(f"   Scores per fold: {[f'{s:.4f}' for s in xgb_cv_scores]}")

print(f"\nTraining sul full training set...")
start_time = time.time()

xgb_base.fit(X_train, y_train_encoded)

xgb_train_time = time.time() - start_time
print(f"Training completato in {xgb_train_time:.2f}s")

# Valutazione su test set
print(f"\nValutazione su test set...")
xgb_test_pred_encoded = xgb_base.predict(X_test)
xgb_test_pred = le.inverse_transform(xgb_test_pred_encoded)  # Decodifica per le metriche
xgb_test_ba = balanced_accuracy_score(y_test, xgb_test_pred)
xgb_test_f1 = f1_score(y_test, xgb_test_pred, average='macro')

print(f"Test Balanced Accuracy: {xgb_test_ba:.4f}")
print(f"Test F1-Score (macro): {xgb_test_f1:.4f}")

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
        'model': rf_base
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
        'model': xgb_base
    }
}

# ============================================================================
# 8. VISUALIZZAZIONI
# ============================================================================

print("\n" + "=" * 80)
print("8. GENERAZIONE VISUALIZZAZIONI")
print("=" * 80)

viz_dir = os.path.join(output_dir, 'visualizations')
os.makedirs(viz_dir, exist_ok=True)
print(f"Directory visualizzazioni: {viz_dir}\n")

n_viz = 0

# 8.1 Confusion Matrices
print("Generazione confusion matrices (assolute e normalizzate)...")
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

    fig.suptitle(f'{model_name} - Confusion Matrices\nBalanced Accuracy: {model_data["test_ba"]:.4f}',
                 fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()
    cm_path = os.path.join(viz_dir, f'01_confusion_matrices_{safe_name}.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    n_viz += 1
    print(f"  {n_viz}. {os.path.basename(cm_path)}")

# 8.2 F1-Score per Classe
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

print(f"\n {n_viz} visualizzazioni generate")
print(f" Directory: {viz_dir}")

# ============================================================================
# 9. SALVATAGGIO MODELLI
# ============================================================================

print("\n" + "=" * 80)
print("9. SALVATAGGIO MODELLI")
print("=" * 80)

if PREADMISSION_MODE:
    rf_path = os.path.join(output_dir, 'rf_model_preadmission.pkl')
else:
    rf_path = os.path.join(output_dir, 'rf_model.pkl')
joblib.dump(rf_base, rf_path)
print(f"\nRandom Forest salvato: {rf_path}")

if PREADMISSION_MODE:
    xgb_path = os.path.join(output_dir, 'xgb_model_preadmission.pkl')
else:
    xgb_path = os.path.join(output_dir, 'xgb_model.pkl')
joblib.dump(xgb_base, xgb_path)
print(f"XGBoost salvato: {xgb_path}")

if PREADMISSION_MODE:
    features_path = os.path.join(output_dir, 'feature_names_preadmission.pkl')
else:
    features_path = os.path.join(output_dir, 'feature_names.pkl')
joblib.dump(X_train.columns.tolist(), features_path)
print(f"Feature names salvati: {features_path}")

if PREADMISSION_MODE:
    le_path = os.path.join(output_dir, 'label_encoder_preadmission.pkl')
else:
    le_path = os.path.join(output_dir, 'label_encoder.pkl')
joblib.dump(le, le_path)
print(f"Label encoder salvato: {le_path}")

results_data = {
    'random_forest': {
        'cv_scores': rf_cv_scores.tolist(),
        'cv_mean': float(rf_cv_scores.mean()),
        'cv_std': float(rf_cv_scores.std()),
        'test_ba': float(rf_test_ba),
        'test_f1': float(rf_test_f1)
    },
    'xgboost': {
        'cv_scores': xgb_cv_scores.tolist(),
        'cv_mean': float(xgb_cv_scores.mean()),
        'cv_std': float(xgb_cv_scores.std()),
        'test_ba': float(xgb_test_ba),
        'test_f1': float(xgb_test_f1)
    },
    'classes': classes
}

if PREADMISSION_MODE:
    results_pkl_path = os.path.join(output_dir, 'training_results_preadmission.pkl')
else:
    results_pkl_path = os.path.join(output_dir, 'training_results.pkl')

joblib.dump(results_data, results_pkl_path)
print(f"Risultati salvati: {results_pkl_path}")
# ============================================================================
# RIEPILOGO FINALE
# ============================================================================
print("\n" + "=" * 80)
if PREADMISSION_MODE:
    print("TRAINING PRE-IMMATRICOLAZIONE COMPLETATO")
else:
    print("TRAINING COMPLETATO")
print("=" * 80)

print(f"\nDirectory: {output_dir}\n")
print("File generati:")
if PREADMISSION_MODE:
    print(f"  1. rf_model_preadmission.pkl            - Random Forest trainato")
    print(f"  2. xgb_model_preadmission.pkl           - XGBoost trainato")
    print(f"  3. feature_names_preadmission.pkl       - Nomi delle features")
    print(f"  4. label_encoder_preadmission.pkl       - Label encoder")
    print(f"  5. training_results_preadmission.pkl    - Risultati in formato pickle")
    print(f"  6. confusion_matrices_random_forest_preadmission.png  - Confusion matrices")
    print(f"  7. confusion_matrices_xgboost_preadmission.png  - Confusion matrices")
    print(f"  8. f1_score_by_class_random_forest_preadmission.png  - f1 score")
    print(f"  9. f1_score_by_class_xgboost_preadmission.png  - f1 score")

else:
    print(f"  1. rf_model.pkl            - Random Forest trainato")
    print(f"  2. xgb_model.pkl           - XGBoost trainato")
    print(f"  3. feature_names.pkl       - Nomi delle features")
    print(f"  4. label_encoder.pkl       - Label encoder")
    print(f"  5. training_results.pkl    - Risultati in formato pickle")
    print(f"  6. confusion_matrices_random_forest.png  - Confusion matrices")
    print(f"  7. confusion_matrices_xgboost.png  - Confusion matrices")
    print(f"  8. f1_score_by_class_random_forest.png  - f1 score")
    print(f"  9. f1_score_by_class_xgboost.png  - f1 score")
