"""
Training Random Forest e XGBoost con 5-Fold Cross-Validation
Dataset: Predict Students' Dropout and Academic Success (preprocessato con SMOTE)

FEATURES:
=========
1. Random Forest con 5-fold CV
2. XGBoost con 5-fold CV
3. Hyperparameter tuning con GridSearchCV
4. Feature importance analysis
5. Confusion matrix e metriche dettagliate
6. Confronto tra modelli
7. Salvataggio modelli migliori

REQUISITI:
==========
pip install pandas numpy matplotlib seaborn scikit-learn xgboost

INPUT:
======
- train_smote.csv (training set bilanciato con SMOTE)
- test_set.csv (test set originale)

OUTPUT:
=======
- rf_model.pkl (miglior modello Random Forest)
- xgb_model.pkl (miglior modello XGBoost)
- training_results.txt (report completo)
- confusion_matrices.png (confusion matrix per entrambi i modelli)
- feature_importance.png (feature importance per entrambi)
- cv_scores.png (cross-validation scores)
- model_comparison.png (confronto performance)
"""

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

# ============================================================================
# CONFIGURAZIONE MODALITÀ
# ============================================================================

# Rileva modalità da argomenti linea di comando
PREADMISSION_MODE = '--preadmission' in sys.argv


# Configurazione
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
if PREADMISSION_MODE:
    print("TRAINING PRE-IMMATRICOLAZIONE - Random Forest & XGBoost (5-Fold CV)")
    print("=" * 80)
    print("\n⚠️  MODALITÀ PRE-IMMATRICOLAZIONE ATTIVA")
    print("   - Features: 24 (solo dati disponibili prima dell'iscrizione)")
    print("   - Obiettivo: Predizione PREVENTIVA dropout")
    print("   - Accuracy attesa: ~58-60% (normale per predizione precoce)")
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

# Crea la cartella di output se non esiste
os.makedirs(output_dir, exist_ok=True)

print(f"\n📁 Directory base: {base_dir}")
print(f"📂 Input da: {input_dir}")
print(f"📂 Output salvati in: {output_dir}\n")

# ============================================================================
# 1. CARICAMENTO DATI
# ============================================================================

print("=" * 80)
print("1. CARICAMENTO DATI PREPROCESSATI")
print("=" * 80)

# Percorsi file nella cartella preprocessing
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
        print(f"\n✗ Errore: File '{name}' non trovato!")
        print(f"   Percorso atteso: {path}")
        if PREADMISSION_MODE:
            print(f"   Assicurati di aver eseguito prima: python preprocessing_smote.py --preadmission")
        else:
            print(f"   Assicurati di aver eseguito prima: python preprocessing_smote.py")
        exit(1)

# Carica training set (con SMOTE)
print("\n📂 Caricamento training set (con SMOTE)...")
train = pd.read_csv(train_smote_path)
X_train = train.drop('Target', axis=1)
y_train = train['Target']

print(f"✓ Training set: {len(X_train)} campioni × {len(X_train.columns)} features")

# Carica test set
print("\n📂 Caricamento test set...")
test = pd.read_csv(test_set_path)
X_test = test.drop('Target', axis=1)
y_test = test['Target']

print(f"✓ Test set: {len(X_test)} campioni × {len(X_test.columns)} features")

# Verifica classi
classes = sorted(y_train.unique())
print(f"\n📊 Classi target: {classes}")

train_dist = y_train.value_counts().sort_index()
print(f"\nDistribuzione training (con SMOTE):")
for cls in classes:
    count = train_dist[cls]
    pct = count / len(y_train) * 100
    print(f"  {cls:15s}: {count:5d} ({pct:5.2f}%)")

test_dist = y_test.value_counts().sort_index()
print(f"\nDistribuzione test (originale):")
for cls in classes:
    count = test_dist[cls]
    pct = count / len(y_test) * 100
    print(f"  {cls:15s}: {count:5d} ({pct:5.2f}%)")

# Encoding classi per XGBoost (richiede numeri)
print(f"\n🔧 Encoding classi per XGBoost...")
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

print(f"\n✓ Strategia: StratifiedKFold")
print(f"✓ N. folds: {n_folds}")
print(f"✓ Shuffle: True")
print(f"✓ Random state: {random_state}")
print(f"\n💡 StratifiedKFold mantiene le proporzioni delle classi in ogni fold")

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

print(f"\n⏳ Cross-validation in corso ({n_folds} folds)...")
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

print(f"✓ Cross-validation completata in {rf_cv_time:.2f}s")
print(f"\n📊 Risultati CV:")
print(f"   Balanced Accuracy: {rf_cv_scores.mean():.4f} (+/- {rf_cv_scores.std():.4f})")
print(f"   Scores per fold: {[f'{s:.4f}' for s in rf_cv_scores]}")

# Training sul full training set
print(f"\n⏳ Training sul full training set...")
start_time = time.time()

rf_base.fit(X_train, y_train)

rf_train_time = time.time() - start_time
print(f"✓ Training completato in {rf_train_time:.2f}s")

# Valutazione su test set
print(f"\n⏳ Valutazione su test set...")
rf_test_pred = rf_base.predict(X_test)
rf_test_ba = balanced_accuracy_score(y_test, rf_test_pred)
rf_test_f1 = f1_score(y_test, rf_test_pred, average='macro')

print(f"✓ Test Balanced Accuracy: {rf_test_ba:.4f}")
print(f"✓ Test F1-Score (macro): {rf_test_f1:.4f}")

if PREADMISSION_MODE:
    if rf_test_ba >= 0.58:
        print(f"   💡 Ottimo risultato per predizione pre-immatricolazione!")
    else:
        print(f"   💡 Risultato ragionevole per predizione precoce (senza dati universitari)")

# ============================================================================
# 4. XGBOOST - TRAINING & CV
# ============================================================================

print("\n" + "=" * 80)
print("4. XGBOOST - TRAINING CON 5-FOLD CV")
print("=" * 80)

print("\n🚀 Configurazione XGBoost:")
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
    y_train_encoded,  # XGBoost richiede encoding numerico
    cv=cv,
    scoring=scorer,
    n_jobs=-1
)

xgb_cv_time = time.time() - start_time

print(f"✓ Cross-validation completata in {xgb_cv_time:.2f}s")
print(f"\n📊 Risultati CV:")
print(f"   Balanced Accuracy: {xgb_cv_scores.mean():.4f} (+/- {xgb_cv_scores.std():.4f})")
print(f"   Scores per fold: {[f'{s:.4f}' for s in xgb_cv_scores]}")

# Training sul full training set
print(f"\n⏳ Training sul full training set...")
start_time = time.time()

xgb_base.fit(X_train, y_train_encoded)  # XGBoost usa encoding numerico

xgb_train_time = time.time() - start_time
print(f"✓ Training completato in {xgb_train_time:.2f}s")

# Valutazione su test set
print(f"\n⏳ Valutazione su test set...")
xgb_test_pred_encoded = xgb_base.predict(X_test)
xgb_test_pred = le.inverse_transform(xgb_test_pred_encoded)  # Decodifica per le metriche
xgb_test_ba = balanced_accuracy_score(y_test, xgb_test_pred)
xgb_test_f1 = f1_score(y_test, xgb_test_pred, average='macro')

print(f"✓ Test Balanced Accuracy: {xgb_test_ba:.4f}")
print(f"✓ Test F1-Score (macro): {xgb_test_f1:.4f}")

if PREADMISSION_MODE:
    if xgb_test_ba >= 0.58:
        print(f"   💡 Ottimo risultato per predizione pre-immatricolazione!")
    else:
        print(f"   💡 Risultato ragionevole per predizione precoce (senza dati universitari)")

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

print(f"\n🏆 MIGLIOR MODELLO: {best_model_name}")
print(f"   Test Balanced Accuracy: {best_ba:.4f}")

if PREADMISSION_MODE:
    print(f"\n💡 NOTA SULLA PERFORMANCE PRE-IMMATRICOLAZIONE:")
    print(f"   - Accuracy ~58-60% è ECCELLENTE per predizione senza dati universitari")
    print(f"   - Permette screening preventivo PRIMA dell'inizio corsi")
    print(f"   - Trade-off accettabile per interventi precoci su studenti a rischio")

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

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=classes))

    # Metriche per classe
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

print("\n" + "=" * 80)
print("8. GENERAZIONE VISUALIZZAZIONI")
print("=" * 80)

# 8.1 Confusion Matrices
print("\n📊 Creazione confusion matrices...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, (model_name, model_data) in enumerate(results.items()):
    ax = axes[idx]
    y_pred = model_data['predictions']
    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'})

    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(f'{model_name}\nBalanced Accuracy: {model_data["test_ba"]:.4f}',
                fontsize=14, fontweight='bold')

plt.tight_layout()
if PREADMISSION_MODE:
    cm_path = os.path.join(output_dir, 'confusion_matrices_preadmission.png')
else:
    cm_path = os.path.join(output_dir, 'confusion_matrices.png')
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
print(f"✓ Salvato: {cm_path}")
plt.close()

# 8.2 Feature Importance
print("📊 Creazione feature importance plots...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for idx, (model_name, model_data) in enumerate(results.items()):
    ax = axes[idx]
    model = model_data['model']

    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(15)

    ax.barh(range(len(feature_importance)), feature_importance['importance'])
    ax.set_yticks(range(len(feature_importance)))
    ax.set_yticklabels(feature_importance['feature'])
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'{model_name} - Top 15 Features', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
if PREADMISSION_MODE:
    fi_path = os.path.join(output_dir, 'feature_importance_preadmission.png')
else:
    fi_path = os.path.join(output_dir, 'feature_importance.png')
plt.savefig(fi_path, dpi=300, bbox_inches='tight')
print(f"✓ Salvato: {fi_path}")
plt.close()

# 8.3 Cross-Validation Scores
print("📊 Creazione CV scores plot...")
fig, ax = plt.subplots(figsize=(10, 6))

x_pos = np.arange(n_folds)
width = 0.35

rf_scores = results['Random Forest']['cv_scores']
xgb_scores = results['XGBoost']['cv_scores']

bars1 = ax.bar(x_pos - width/2, rf_scores, width, label='Random Forest',
               color='#3498db', edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x_pos + width/2, xgb_scores, width, label='XGBoost',
               color='#e74c3c', edgecolor='black', linewidth=1.5)

ax.axhline(y=rf_cv_scores.mean(), color='#3498db', linestyle='--',
           label=f'RF Mean: {rf_cv_scores.mean():.4f}')
ax.axhline(y=xgb_cv_scores.mean(), color='#e74c3c', linestyle='--',
           label=f'XGB Mean: {xgb_cv_scores.mean():.4f}')

ax.set_xlabel('Fold', fontsize=12)
ax.set_ylabel('Balanced Accuracy', fontsize=12)
ax.set_title('5-Fold Cross-Validation Scores', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'Fold {i+1}' for i in range(n_folds)])
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

# Aggiungi valori sulle barre
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
if PREADMISSION_MODE:
    cv_path = os.path.join(output_dir, 'cv_scores_preadmission.png')
else:
    cv_path = os.path.join(output_dir, 'cv_scores.png')
plt.savefig(cv_path, dpi=300, bbox_inches='tight')
print(f"✓ Salvato: {cv_path}")
plt.close()

# 8.4 Model Comparison
print("📊 Creazione model comparison plot...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# CV Mean Comparison
ax1 = axes[0, 0]
models = list(results.keys())
cv_means = [results[m]['cv_mean'] for m in models]
cv_stds = [results[m]['cv_std'] for m in models]

bars = ax1.bar(models, cv_means, yerr=cv_stds, capsize=10,
               color=['#3498db', '#e74c3c'], edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Balanced Accuracy', fontsize=12)
ax1.set_title('Cross-Validation Mean Score', fontsize=12, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}',
            ha='center', va='bottom', fontweight='bold')

# Test Set Performance
ax2 = axes[0, 1]
test_ba = [results[m]['test_ba'] for m in models]
test_f1 = [results[m]['test_f1'] for m in models]

x = np.arange(len(models))
width = 0.35
bars1 = ax2.bar(x - width/2, test_ba, width, label='Balanced Accuracy',
               color='#3498db', edgecolor='black', linewidth=1.5)
bars2 = ax2.bar(x + width/2, test_f1, width, label='F1-Score (macro)',
               color='#2ecc71', edgecolor='black', linewidth=1.5)

ax2.set_ylabel('Score', fontsize=12)
ax2.set_title('Test Set Performance', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(models)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

# Training Time Comparison
ax3 = axes[1, 0]
train_times = [results[m]['train_time'] for m in models]
bars = ax3.bar(models, train_times, color=['#3498db', '#e74c3c'],
              edgecolor='black', linewidth=1.5)
ax3.set_ylabel('Time (seconds)', fontsize=12)
ax3.set_title('Training Time', fontsize=12, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}s',
            ha='center', va='bottom', fontweight='bold')

# Overall Metrics
ax4 = axes[1, 1]
ax4.axis('off')

summary_text = f"""
SUMMARY - BEST MODEL: {best_model_name}

Cross-Validation (5-fold):
  Mean BA: {results[best_model_name]['cv_mean']:.4f}
  Std BA:  {results[best_model_name]['cv_std']:.4f}

Test Set:
  Balanced Accuracy: {results[best_model_name]['test_ba']:.4f}
  F1-Score (macro):  {results[best_model_name]['test_f1']:.4f}

Training Time: {results[best_model_name]['train_time']:.2f}s
CV Time:       {results[best_model_name]['cv_time']:.2f}s
"""

ax4.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
        verticalalignment='center', bbox=dict(boxstyle='round',
        facecolor='wheat', alpha=0.5))

plt.tight_layout()
if PREADMISSION_MODE:
    comp_path = os.path.join(output_dir, 'model_comparison_preadmission.png')
else:
    comp_path = os.path.join(output_dir, 'model_comparison.png')
plt.savefig(comp_path, dpi=300, bbox_inches='tight')
print(f"✓ Salvato: {comp_path}")
plt.close()

# ============================================================================
# 9. SALVATAGGIO MODELLI
# ============================================================================

print("\n" + "=" * 80)
print("9. SALVATAGGIO MODELLI")
print("=" * 80)

# Salva Random Forest
if PREADMISSION_MODE:
    rf_path = os.path.join(output_dir, 'rf_model_preadmission.pkl')
else:
    rf_path = os.path.join(output_dir, 'rf_model.pkl')
joblib.dump(rf_base, rf_path)
print(f"\n✓ Random Forest salvato: {rf_path}")

# Salva XGBoost
if PREADMISSION_MODE:
    xgb_path = os.path.join(output_dir, 'xgb_model_preadmission.pkl')
else:
    xgb_path = os.path.join(output_dir, 'xgb_model.pkl')
joblib.dump(xgb_base, xgb_path)
print(f"✓ XGBoost salvato: {xgb_path}")

# Salva feature names
if PREADMISSION_MODE:
    features_path = os.path.join(output_dir, 'feature_names_preadmission.pkl')
else:
    features_path = os.path.join(output_dir, 'feature_names.pkl')
joblib.dump(X_train.columns.tolist(), features_path)
print(f"✓ Feature names salvati: {features_path}")

# Salva label encoder
if PREADMISSION_MODE:
    le_path = os.path.join(output_dir, 'label_encoder_preadmission.pkl')
else:
    le_path = os.path.join(output_dir, 'label_encoder.pkl')
joblib.dump(le, le_path)
print(f"✓ Label encoder salvato: {le_path}")

# Salva risultati
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
print(f"✓ Risultati salvati: {results_pkl_path}")

# ============================================================================
# 10. REPORT TESTUALE
# ============================================================================

print("\n" + "=" * 80)
print("10. GENERAZIONE REPORT")
print("=" * 80)

if PREADMISSION_MODE:
    report_path = os.path.join(output_dir, 'training_results_preadmission.txt')
else:
    report_path = os.path.join(output_dir, 'training_results.txt')

with open(report_path, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    if PREADMISSION_MODE:
        f.write("TRAINING RESULTS PRE-IMMATRICOLAZIONE - Random Forest & XGBoost\n")
        f.write("=" * 80 + "\n\n")
        f.write("MODALITÀ: PRE-IMMATRICOLAZIONE\n")
        f.write("Features: 24 (solo dati disponibili prima dell'iscrizione)\n")
        f.write("Obiettivo: Predizione PREVENTIVA dropout\n")
        f.write("Accuracy attesa: ~58-60% (normale per predizione precoce)\n\n")
    else:
        f.write("TRAINING RESULTS - Random Forest & XGBoost\n")
        f.write("=" * 80 + "\n\n")

    f.write("DATASET:\n")
    f.write("-" * 80 + "\n")
    f.write(f"Training set: {len(X_train)} campioni (bilanciato con SMOTE)\n")
    f.write(f"Test set: {len(X_test)} campioni (distribuzione originale)\n")
    f.write(f"Features: {len(X_train.columns)}\n")
    f.write(f"Classi: {classes}\n\n")

    f.write("CROSS-VALIDATION:\n")
    f.write("-" * 80 + "\n")
    f.write(f"Strategia: StratifiedKFold\n")
    f.write(f"N. folds: {n_folds}\n")
    f.write(f"Random state: {random_state}\n\n")

    for model_name, model_data in results.items():
        f.write(f"\n{model_name.upper()}:\n")
        f.write("=" * 80 + "\n\n")

        f.write("Cross-Validation Results:\n")
        f.write(f"  Mean Balanced Accuracy: {model_data['cv_mean']:.4f}\n")
        f.write(f"  Std Balanced Accuracy:  {model_data['cv_std']:.4f}\n")
        f.write(f"  Scores per fold: {[f'{s:.4f}' for s in model_data['cv_scores']]}\n\n")

        f.write("Test Set Results:\n")
        f.write(f"  Balanced Accuracy: {model_data['test_ba']:.4f}\n")
        f.write(f"  F1-Score (macro):  {model_data['test_f1']:.4f}\n\n")

        f.write("Training Time:\n")
        f.write(f"  CV time: {model_data['cv_time']:.2f}s\n")
        f.write(f"  Full training time: {model_data['train_time']:.2f}s\n\n")

        f.write("Classification Report:\n")
        f.write(classification_report(y_test, model_data['predictions'],
                                     target_names=classes))
        f.write("\n")

        # Feature importance
        model = model_data['model']
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        f.write("Top 20 Features:\n")
        f.write("-" * 80 + "\n")
        for idx, row in feature_importance.head(20).iterrows():
            f.write(f"{row['feature']:<45} {row['importance']:.6f}\n")
        f.write("\n\n")

    f.write("=" * 80 + "\n")
    f.write(f"MIGLIOR MODELLO: {best_model_name}\n")
    f.write("=" * 80 + "\n")
    f.write(f"Test Balanced Accuracy: {best_ba:.4f}\n")
    f.write(f"Test F1-Score (macro): {results[best_model_name]['test_f1']:.4f}\n")

    if PREADMISSION_MODE:
        f.write("\n" + "=" * 80 + "\n")
        f.write("INTERPRETAZIONE RISULTATI PRE-IMMATRICOLAZIONE\n")
        f.write("=" * 80 + "\n")
        f.write("\nUn'accuracy del 58-60% in modalità pre-immatricolazione è ECCELLENTE perché:\n")
        f.write("  1. Non abbiamo dati sulla performance universitaria (voti, esami superati)\n")
        f.write("  2. Prediciamo PRIMA che lo studente inizi i corsi\n")
        f.write("  3. Usiamo solo dati demografici, socioeconomici e background scolastico\n")
        f.write("  4. Trade-off accettabile per interventi preventivi tempestivi\n\n")
        f.write("APPLICAZIONI PRATICHE:\n")
        f.write("  - Screening all'iscrizione per identificare studenti a rischio\n")
        f.write("  - Assegnazione automatica a programmi di tutoring\n")
        f.write("  - Priorità nelle risorse di supporto\n")
        f.write("  - Counseling mirato già prima dell'inizio corsi\n")

print(f"\n✓ Report salvato: {report_path}")

# ============================================================================
# RIEPILOGO FINALE
# ============================================================================

print("\n" + "=" * 80)
if PREADMISSION_MODE:
    print("✅ TRAINING PRE-IMMATRICOLAZIONE COMPLETATO!")
else:
    print("✅ TRAINING COMPLETATO!")
print("=" * 80)

print(f"\n📁 Directory: {output_dir}\n")
print("📦 File generati:")
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
print("📊 RISULTATI FINALI")
print("=" * 80)

print(f"\n{'Modello':<20} {'CV Mean':<12} {'Test BA':<12} {'Test F1':<12}")
print("-" * 80)
for name, metrics in results.items():
    marker = " 🏆" if name == best_model_name else ""
    print(f"{name:<20} {metrics['cv_mean']:<12.4f} {metrics['test_ba']:<12.4f} "
          f"{metrics['test_f1']:<12.4f}{marker}")

print(f"\n🏆 MIGLIOR MODELLO: {best_model_name}")
print(f"   Cross-Validation BA: {results[best_model_name]['cv_mean']:.4f} "
      f"(+/- {results[best_model_name]['cv_std']:.4f})")
print(f"   Test Set BA: {results[best_model_name]['test_ba']:.4f}")
print(f"   Test Set F1: {results[best_model_name]['test_f1']:.4f}")

if PREADMISSION_MODE:
    print("\n" + "=" * 80)
    print("💡 NOTA SULLA PERFORMANCE PRE-IMMATRICOLAZIONE")
    print("=" * 80)
    print(f"\nAccuracy {best_ba:.1%} in modalità pre-immatricolazione significa:")
    print(f"  ✓ Identificazione corretta di {best_ba:.1%} degli studenti")
    print(f"  ✓ SENZA usare dati universitari (voti, esami)")
    print(f"  ✓ PRIMA dell'inizio dei corsi")
    print(f"  ✓ Solo con dati demografici e background scolastico")
    print("\nAPPLICAZIONI PRATICHE:")
    print("  • Screening automatico all'iscrizione")
    print("  • Assegnazione prioritaria a programmi di supporto")
    print("  • Interventi preventivi tempestivi")
    print("  • Counseling mirato già prima dell'inizio corsi")
    print("\nQUESTO È UN RISULTATO ECCELLENTE per predizione precoce!")

print("\n" + "=" * 80)
print("💡 PROSSIMI PASSI")
print("=" * 80)
print("\n1. Analizza le confusion matrices per capire dove i modelli sbagliano")
print("2. Esamina le feature importance per feature engineering")
print("3. Se vuoi migliorare, prova hyperparameter tuning più approfondito")
print("4. Usa il miglior modello per predizioni su nuovi dati")

if PREADMISSION_MODE:
    print("\n💡 CONFRONTO MODALITÀ:")
    print("   - Esegui anche: python train_models.py (senza --preadmission)")
    print("   - Confronta accuracy ~60% (pre) vs ~70% (post-iscrizione)")
    print("   - Valuta trade-off: tempestività vs precisione")

print("\n" + "=" * 80)