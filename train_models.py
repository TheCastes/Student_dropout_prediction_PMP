import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import (
    cross_val_score,
    StratifiedKFold,
)
from sklearn.metrics import (
    classification_report,
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

# Importa configurazione colori centralizzata
from color_config import (
    CLASS_COLORS_LIST, MODEL_COLORS, MODEL_COLORS_LIST,
    HEATMAP_CMAPS, METRIC_COLORS, setup_plot_style, get_class_color
)

# ============================================================================
# CONFIGURAZIONE MODALITÀ
# ============================================================================
PREADMISSION_MODE = '--preadmission' in sys.argv

# Setup stile grafico coerente
setup_plot_style()
sns.set_palette(CLASS_COLORS_LIST)

print("=" * 80)
if PREADMISSION_MODE:
    print("TRAINING PRE-IMMATRICOLAZIONE - Random Forest & XGBoost (5-Fold CV)")
    print("=" * 80)
    print("\nMODALITÀ PRE-IMMATRICOLAZIONE ATTIVA")
else:
    print("TRAINING RANDOM FOREST & XGBOOST - 5-Fold Cross-Validation")
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

# Verifica file
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

# Scorer per CV
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
print("   - n_jobs: -1 (tutti i core)")

# Modello base
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

# Valutazione su test set
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

print(f"\n⏳ Cross-validation in corso ({n_folds} folds)...")
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

# Training sul full training set
print(f"\nTraining sul full training set...")
start_time = time.time()

xgb_base.fit(X_train, y_train_encoded)

xgb_train_time = time.time() - start_time
print(f"✓ Training completato in {xgb_train_time:.2f}s")

# Valutazione su test set
print(f"\nValutazione su test set...")
xgb_test_pred_encoded = xgb_base.predict(X_test)
xgb_test_pred = le.inverse_transform(xgb_test_pred_encoded)  # Decodifica per le metriche
xgb_test_ba = balanced_accuracy_score(y_test, xgb_test_pred)
xgb_test_f1 = f1_score(y_test, xgb_test_pred, average='macro')

print(f"Test Balanced Accuracy: {xgb_test_ba:.4f}")
print(f"Test F1-Score (macro): {xgb_test_f1:.4f}")

# ============================================================================
# 5. CONFRONTO MODELLI
# ============================================================================

print("\n" + "=" * 80)
print("5. CONFRONTO MODELLI")
print("=" * 80)

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

print(f"\n{'Modello':<20} {'CV Mean':<12} {'CV Std':<12} {'Test BA':<12} {'Test F1':<12}")
print("-" * 80)
for name, metrics in results.items():
    print(f"{name:<20} {metrics['cv_mean']:<12.4f} {metrics['cv_std']:<12.4f} "
          f"{metrics['test_ba']:<12.4f} {metrics['test_f1']:<12.4f}")

# Determina miglior modello
best_model_name = max(results, key=lambda x: results[x]['test_ba'])
best_model = results[best_model_name]['model']
best_ba = results[best_model_name]['test_ba']

print(f"\n MIGLIOR MODELLO: {best_model_name}")
print(f"   Test Balanced Accuracy: {best_ba:.4f}")

# ============================================================================
# 6. METRICHE DETTAGLIATE
# ============================================================================

print("\n" + "=" * 80)
print("6. METRICHE DETTAGLIATE PER CLASSE")
print("=" * 80)

for model_name, model_data in results.items():
    print(f"\n{model_name}:")
    print("-" * 80)

    y_pred = model_data['predictions']

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=classes))

    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)

    print("\nMetriche per classe:")
    print(f"{'Classe':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 80)
    for i, cls in enumerate(classes):
        print(f"{cls:<15} {precision[i]:<12.4f} {recall[i]:<12.4f} "
              f"{f1[i]:<12.4f} {support[i]:<10}")

# ============================================================================
# 7. FEATURE IMPORTANCE
# ============================================================================

print("\n" + "=" * 80)
print("7. FEATURE IMPORTANCE ANALYSIS")
print("=" * 80)

for model_name, model_data in results.items():
    print(f"\n{model_name} - Top 10 Features:")
    print("-" * 80)

    model = model_data['model']
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    top10 = feature_importance.head(10)
    for idx, row in top10.iterrows():
        print(f"  {row['feature']:<40} {row['importance']:.6f}")

# ============================================================================
# 8. VISUALIZZAZIONI
# ============================================================================
# 8. GENERAZIONE VISUALIZZAZIONI SEPARATE
# ============================================================================

print("\n" + "=" * 80)
print("8. GENERAZIONE VISUALIZZAZIONI SEPARATE")
print("=" * 80)

# Crea sottocartella per visualizzazioni
viz_dir = os.path.join(output_dir, 'visualizations')
os.makedirs(viz_dir, exist_ok=True)
print(f"Directory visualizzazioni: {viz_dir}\n")

n_viz = 0

# 8.1 Confusion Matrices - SEPARATE
print("Generazione confusion matrices...")
for model_name, model_data in results.items():
    fig, ax = plt.subplots(figsize=(8, 6))
    y_pred = model_data['predictions']
    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(cm, annot=True, fmt='d', cmap=HEATMAP_CMAPS['confusion_matrix'], ax=ax,
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'}, annot_kws={'fontsize': 12})

    ax.set_xlabel('Predicted', fontsize=14)
    ax.set_ylabel('Actual', fontsize=14)
    ax.set_title(f'{model_name} - Confusion Matrix\nBalanced Accuracy: {model_data["test_ba"]:.4f}',
                 fontsize=16, fontweight='bold')

    plt.tight_layout()
    safe_name = model_name.lower().replace(' ', '_')
    cm_path = os.path.join(viz_dir, f'01_confusion_matrix_{safe_name}.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    n_viz += 1
    print(f"  {n_viz}. {os.path.basename(cm_path)}")

# 8.2 Feature Importance - SEPARATE
print("\nGenerazione feature importance plots...")
for model_name, model_data in results.items():
    fig, ax = plt.subplots(figsize=(12, 8))
    model = model_data['model']

    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(20)  # Top 20

    color = MODEL_COLORS.get(model_name, MODEL_COLORS_LIST[0])

    ax.barh(range(len(feature_importance)), feature_importance['importance'],
            color=color, edgecolor='black', linewidth=1.5)
    ax.set_yticks(range(len(feature_importance)))
    ax.set_yticklabels(feature_importance['feature'], fontsize=10)
    ax.set_xlabel('Importance', fontsize=14)
    ax.set_title(f'{model_name} - Top 20 Feature Importance', fontsize=16, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    safe_name = model_name.lower().replace(' ', '_')
    fi_path = os.path.join(viz_dir, f'02_feature_importance_{safe_name}.png')
    plt.savefig(fi_path, dpi=300, bbox_inches='tight')
    plt.close()
    n_viz += 1
    print(f"  {n_viz}. {os.path.basename(fi_path)}")

# 8.3 Cross-Validation Scores
print("\nGenerazione CV scores plot...")
fig, ax = plt.subplots(figsize=(12, 7))

x_pos = np.arange(n_folds)
width = 0.35

rf_scores = results['Random Forest']['cv_scores']
xgb_scores = results['XGBoost']['cv_scores']

bars1 = ax.bar(x_pos - width / 2, rf_scores, width, label='Random Forest',
               color=MODEL_COLORS['Random Forest'], edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x_pos + width / 2, xgb_scores, width, label='XGBoost',
               color=MODEL_COLORS['XGBoost'], edgecolor='black', linewidth=1.5)

ax.axhline(y=rf_cv_scores.mean(), color=MODEL_COLORS['Random Forest'], linestyle='--',
           linewidth=2, label=f'RF Mean: {rf_cv_scores.mean():.4f}')
ax.axhline(y=xgb_cv_scores.mean(), color=MODEL_COLORS['XGBoost'], linestyle='--',
           linewidth=2, label=f'XGB Mean: {xgb_cv_scores.mean():.4f}')

ax.set_xlabel('Fold', fontsize=14)
ax.set_ylabel('Balanced Accuracy', fontsize=14)
ax.set_title('5-Fold Cross-Validation Scores', fontsize=16, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'Fold {i + 1}' for i in range(n_folds)])
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
cv_path = os.path.join(viz_dir, '03_cv_scores_comparison.png')
plt.savefig(cv_path, dpi=300, bbox_inches='tight')
plt.close()
n_viz += 1
print(f"  {n_viz}. {os.path.basename(cv_path)}")

# 8.4 Model Comparison - CV Mean
print("\nGenerazione model comparison plots...")
fig, ax = plt.subplots(figsize=(10, 6))
models = list(results.keys())
cv_means = [results[m]['cv_mean'] for m in models]
cv_stds = [results[m]['cv_std'] for m in models]

bars = ax.bar(models, cv_means, yerr=cv_stds, capsize=10,
              color=MODEL_COLORS_LIST, edgecolor='black', linewidth=2)
ax.set_ylabel('Balanced Accuracy', fontsize=14)
ax.set_title('Cross-Validation Mean Score (±std)', fontsize=16, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2., height,
            f'{height:.4f}',
            ha='center', va='bottom', fontweight='bold', fontsize=12)

plt.tight_layout()
comp1_path = os.path.join(viz_dir, '04_cv_mean_comparison.png')
plt.savefig(comp1_path, dpi=300, bbox_inches='tight')
plt.close()
n_viz += 1
print(f"  {n_viz}. {os.path.basename(comp1_path)}")

# 8.5 Test Set Performance
fig, ax = plt.subplots(figsize=(10, 6))
test_ba = [results[m]['test_ba'] for m in models]
test_f1 = [results[m]['test_f1'] for m in models]

x = np.arange(len(models))
width = 0.35
bars1 = ax.bar(x - width / 2, test_ba, width, label='Balanced Accuracy',
               color=METRIC_COLORS['Balanced Accuracy'], edgecolor='black', linewidth=2)
bars2 = ax.bar(x + width / 2, test_f1, width, label='F1-Score (macro)',
               color=METRIC_COLORS['F1-Score'], edgecolor='black', linewidth=2)

ax.set_ylabel('Score', fontsize=14)
ax.set_title('Test Set Performance', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend(fontsize=12)
ax.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
comp2_path = os.path.join(viz_dir, '05_test_performance.png')
plt.savefig(comp2_path, dpi=300, bbox_inches='tight')
plt.close()
n_viz += 1
print(f"  {n_viz}. {os.path.basename(comp2_path)}")

# 8.6 Training Time Comparison
fig, ax = plt.subplots(figsize=(10, 6))
train_times = [results[m]['train_time'] for m in models]
bars = ax.bar(models, train_times, color=MODEL_COLORS_LIST,
              edgecolor='black', linewidth=2)
ax.set_ylabel('Time (seconds)', fontsize=14)
ax.set_title('Training Time Comparison', fontsize=16, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2., height,
            f'{height:.2f}s',
            ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.tight_layout()
comp3_path = os.path.join(viz_dir, '06_training_time.png')
plt.savefig(comp3_path, dpi=300, bbox_inches='tight')
plt.close()
n_viz += 1
print(f"  {n_viz}. {os.path.basename(comp3_path)}")

print(f"\n✅ {n_viz} visualizzazioni generate")
print(f"📁 Directory: {viz_dir}")

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
    'best_model': best_model_name,
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
    print("TRAINING PRE-IMMATRICOLAZIONE COMPLETATO!")
else:
    print("TRAINING COMPLETATO!")
print("=" * 80)

print(f"\nDirectory: {output_dir}\n")
print("File generati:")
if PREADMISSION_MODE:
    print(f"  1. rf_model_preadmission.pkl            - Random Forest trainato")
    print(f"  2. xgb_model_preadmission.pkl           - XGBoost trainato")
    print(f"  3. feature_names_preadmission.pkl       - Nomi delle features")
    print(f"  4. label_encoder_preadmission.pkl       - Label encoder")
    print(f"  5. training_results_preadmission.pkl    - Risultati in formato pickle")
    print(f"  6. training_results_preadmission.txt    - Report testuale completo")
    print(f"  7. confusion_matrices_preadmission.png  - Confusion matrices")
    print(f"  8. feature_importance_preadmission.png  - Feature importance")
    print(f"  9. cv_scores_preadmission.png           - Cross-validation scores")
    print(f" 10. model_comparison_preadmission.png    - Confronto modelli")
else:
    print(f"  1. rf_model.pkl            - Random Forest trainato")
    print(f"  2. xgb_model.pkl           - XGBoost trainato")
    print(f"  3. feature_names.pkl       - Nomi delle features")
    print(f"  4. training_results.pkl    - Risultati in formato pickle")
    print(f"  5. training_results.txt    - Report testuale completo")
    print(f"  6. confusion_matrices.png  - Confusion matrices")
    print(f"  7. feature_importance.png  - Feature importance")
    print(f"  8. cv_scores.png           - Cross-validation scores")
    print(f"  9. model_comparison.png    - Confronto modelli")

print("\n" + "=" * 80)
print("RISULTATI FINALI")
print("=" * 80)

print(f"\n{'Modello':<20} {'CV Mean':<12} {'Test BA':<12} {'Test F1':<12}")
print("-" * 80)
for name, metrics in results.items():
    marker = " 🏆" if name == best_model_name else ""
    print(f"{name:<20} {metrics['cv_mean']:<12.4f} {metrics['test_ba']:<12.4f} "
          f"{metrics['test_f1']:<12.4f}{marker}")

print(f"\nMIGLIOR MODELLO: {best_model_name}")
print(f"   Cross-Validation BA: {results[best_model_name]['cv_mean']:.4f} "
      f"(+/- {results[best_model_name]['cv_std']:.4f})")
print(f"   Test Set BA: {results[best_model_name]['test_ba']:.4f}")
print(f"   Test Set F1: {results[best_model_name]['test_f1']:.4f}")

print("\n" + "=" * 80)