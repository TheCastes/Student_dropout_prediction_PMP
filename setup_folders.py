"""
Organizzatore Struttura Cartelle - Progetto Student Dropout Prediction

Questo script crea la struttura di cartelle per organizzare gli output dei 3 step:
1. Analysis (student_analysis.py)
2. Preprocessing (preprocessing_smote.py)  
3. Training (train_models.py)

STRUTTURA CREATA:
=================
project/
├── 01_analysis/          # Output di student_analysis.py
├── 02_preprocessing/     # Output di preprocessing_smote.py
├── 03_training/          # Output di train_models.py
└── data/                 # Dataset originale

ESECUZIONE:
===========
python setup_folders.py

Questo script:
- Crea le cartelle se non esistono
- Crea README.md in ogni cartella
- Stampa la struttura creata
"""

from pathlib import Path

def create_folder_structure():
    """Crea la struttura di cartelle per il progetto"""
    
    print("=" * 80)
    print("SETUP STRUTTURA CARTELLE - Student Dropout Prediction")
    print("=" * 80)
    
    # Directory base (dove si trova questo script)
    base_dir = Path.cwd()
    print(f"\n📁 Directory base: {base_dir}\n")
    
    # Definisci cartelle da creare
    folders = {
        # Versione completa (con performance universitaria)
        '01_analysis': 'Output analisi esplorativa dataset - COMPLETO',
        '02_preprocessing': 'Output preprocessing con SMOTE - COMPLETO',
        '03_training': 'Output training modelli ML - COMPLETO',

        # Versione pre-immatricolazione (solo dati pre-universitari)
        '01_analysis_preadmission': 'Output analisi - PRE-IMMATRICOLAZIONE',
        '02_preprocessing_preadmission': 'Output preprocessing - PRE-IMMATRICOLAZIONE',
        '03_training_preadmission': 'Output training - PRE-IMMATRICOLAZIONE',

        # Dataset originale
        'data': 'Dataset originale UCI'
    }

    # Crea cartelle
    print("📂 Creazione cartelle...")
    print("-" * 80)

    created = []
    existing = []

    for folder, description in folders.items():
        folder_path = base_dir / folder

        if not folder_path.exists():
            folder_path.mkdir(parents=True, exist_ok=True)
            print(f"✓ {folder:25s} - {description}")
            created.append(folder)
        else:
            print(f"→ {folder:25s} - Già esistente")
            existing.append(folder)

    print("\n" + "=" * 80)
    print("📝 CREAZIONE README")
    print("=" * 80)

    # README per ogni cartella
    readmes = {
        '01_analysis': """# Step 1: Analisi Esplorativa Dataset - VERSIONE COMPLETA

Questa cartella contiene tutti gli output dell'analisi esplorativa del dataset COMPLETO (con performance universitaria).

## Script:
`student_analysis.py`

## Output Generati:

### Dataset:
- **student_data_original.csv**: Dataset con valori numerici originali
- **student_data_mapped.csv**: Dataset con valori mappati (leggibile)

### Visualizzazioni:
- **student_analysis_visualizations.png**: 15 grafici principali
- **correlation_matrix_complete.png**: Matrice correlazione completa (tutte variabili)
- **correlation_matrix_academic.png**: Matrice correlazione accademica (variabili chiave)

## Come Eseguire:
```bash
python student_analysis.py
```

## Cosa Fa:
1. Carica il dataset da `data/data.csv`
2. Analizza TUTTE le 37 variabili (incluse performance universitaria)
3. Genera visualizzazioni e dataset mappati
4. Salva tutto in questa cartella

## Output Attesi:
- 2 file CSV (original, mapped)
- 3 file PNG (visualizations, 2 correlation matrices)
""",

        '02_preprocessing': """# Step 2: Preprocessing con SMOTE - VERSIONE COMPLETA

Questa cartella contiene tutti gli output del preprocessing del dataset COMPLETO.

## Script:
`preprocessing_smote.py`

## Output Generati:

### Dataset Preprocessati:
- **train_smote.csv**: Training set bilanciato con SMOTE (5,301 campioni)
- **test_set.csv**: Test set originale (885 campioni, 20%)
- **train_original.csv**: Training set originale sbilanciato (per confronto)

### Metadata:
- **smote_info.pkl**: Informazioni sulla procedura SMOTE
- **target_encoder.pkl**: Encoder per le classi target
- **preprocessing_report.txt**: Report testuale completo

### Visualizzazioni:
- **class_distribution.png**: Distribuzione classi prima/dopo SMOTE

## Come Eseguire:
```bash
python preprocessing_smote.py
```

## Cosa Fa:
1. Carica dataset da `01_analysis/student_data_original.csv`
2. Split 80/20 stratificato
3. Applica SMOTE al training set (bilancia le classi)
4. Usa TUTTE le 36 features (incluse performance universitaria)
5. Salva dataset preprocessati in questa cartella

## Requisiti:
```bash
pip install imbalanced-learn scikit-learn pandas numpy matplotlib seaborn
```

## Output Attesi:
- 3 file CSV (train_smote, test_set, train_original)
- 3 file PKL (smote_info, target_encoder)
- 1 file TXT (report)
- 1 file PNG (class_distribution)
""",

        '03_training': """# Step 3: Training Modelli ML - VERSIONE COMPLETA

Questa cartella contiene tutti gli output del training dei modelli COMPLETI.

## Script:
`train_models.py`

## Output Generati:

### Modelli Trainati:
- **xgb_model.pkl**: XGBoost (BA: 71.15%) ← MIGLIOR MODELLO
- **rf_model.pkl**: Random Forest (BA: 70.82%)

### Metadata:
- **label_encoder.pkl**: Encoder per classi (necessario per XGBoost)
- **feature_names.pkl**: Nomi delle 36 features
- **training_results.pkl**: Risultati serializzati
- **training_results.txt**: Report testuale completo

### Visualizzazioni:
- **confusion_matrices.png**: Confusion matrices dei 2 modelli
- **feature_importance.png**: Top 15 features più importanti
- **cv_scores.png**: Punteggi cross-validation (5-fold)
- **model_comparison.png**: Confronto performance completo

## Come Eseguire:
```bash
python train_models.py
```

## Cosa Fa:
1. Carica dataset da `02_preprocessing/train_smote.csv` e `test_set.csv`
2. Training Random Forest e XGBoost con TUTTE le 36 features
3. Valuta su test set
4. Genera visualizzazioni e report
5. Salva modelli trainati in questa cartella

## Requisiti:
```bash
pip install xgboost scikit-learn pandas numpy matplotlib seaborn
```

## Output Attesi:
- 2 modelli PKL (rf_model, xgb_model)
- 4 metadata PKL (label_encoder, feature_names, training_results)
- 1 report TXT
- 4 visualizzazioni PNG

## Uso Modelli:
```python
import joblib

# Carica miglior modello
model = joblib.load('03_training/xgb_model.pkl')
label_encoder = joblib.load('03_training/label_encoder.pkl')

# Predizioni
predictions = model.predict(X_new)
```
""",

        '01_analysis_preadmission': """# Step 1: Analisi Esplorativa - VERSIONE PRE-IMMATRICOLAZIONE

Questa cartella contiene gli output dell'analisi con SOLO dati pre-immatricolazione (NO performance universitaria).

## Script:
`student_analysis_preadmission.py`

## Output Generati:

### Dataset:
- **student_data_preadmission.csv**: Dataset filtrato (24 features + target)

### Visualizzazioni:
- **preadmission_analysis_visualizations.png**: 6 grafici chiave
- **preadmission_correlation_matrix.png**: Heatmap correlazioni

## Features ESCLUSE (12):
- Tutte le "Curricular units 1st sem" (6 variabili)
- Tutte le "Curricular units 2nd sem" (6 variabili)

**Motivo**: Non disponibili al momento dell'immatricolazione!

## Features INCLUSE (24):
- Demografiche (6): Età, genere, nazionalità, ecc.
- Pre-universitarie (7): Voti precedenti, corso, modalità
- Familiari (4): Qualifiche e occupazioni genitori
- Finanziarie (4): Tasse, borsa, debiti
- Economiche (3): Disoccupazione, inflazione, PIL

## Come Eseguire:
```bash
python student_analysis_preadmission.py
```

## Cosa Fa:
1. Carica dataset da `data/data.csv`
2. RIMUOVE le 12 variabili di performance universitaria
3. Analizza solo le 24 variabili pre-immatricolazione
4. Genera visualizzazioni mirate
5. Salva in questa cartella

## Obiettivo:
Predire dropout PRIMA dell'inizio corsi per interventi PREVENTIVI.

## Output Attesi:
- 1 file CSV (dataset filtrato)
- 2 file PNG (visualizations, correlation matrix)
""",

        '02_preprocessing_preadmission': """# Step 2: Preprocessing - VERSIONE PRE-IMMATRICOLAZIONE

Questa cartella contiene gli output del preprocessing con SOLO 24 features pre-immatricolazione.

## Script:
`preprocessing_smote_preadmission.py` (da creare)

## Output Generati:

### Dataset Preprocessati:
- **train_smote_preadmission.csv**: Training set bilanciato (24 features)
- **test_set_preadmission.csv**: Test set (24 features)
- **train_original_preadmission.csv**: Training originale (per confronto)

### Metadata:
- **smote_info_preadmission.pkl**: Info SMOTE
- **target_encoder_preadmission.pkl**: Encoder target
- **preprocessing_report_preadmission.txt**: Report

### Visualizzazioni:
- **class_distribution_preadmission.png**: Distribuzione classi

## Come Eseguire:
```bash
python preprocessing_smote_preadmission.py
```

## Differenza con Versione Completa:
- Features: 24 invece di 36 (-12 performance universitaria)
- Performance attesa: ~60% invece di 71% (-11%)
- Utilità: Predizione PRECOCE (prima inizio corsi)

## Output Attesi:
- 3 file CSV
- 3 file PKL
- 1 file TXT
- 1 file PNG
""",

        '03_training_preadmission': """# Step 3: Training Modelli - VERSIONE PRE-IMMATRICOLAZIONE

Questa cartella contiene i modelli trainati con SOLO 24 features pre-immatricolazione.

## Script:
`train_models_preadmission.py` (da creare)

## Output Generati:

### Modelli Trainati:
- **xgb_model_preadmission.pkl**: XGBoost (BA: ~60% stimata)
- **rf_model_preadmission.pkl**: Random Forest (BA: ~58% stimata)

### Metadata:
- **label_encoder_preadmission.pkl**: Encoder classi
- **feature_names_preadmission.pkl**: Nomi 24 features
- **training_results_preadmission.pkl**: Risultati
- **training_results_preadmission.txt**: Report

### Visualizzazioni:
- **confusion_matrices_preadmission.png**: Confusion matrices
- **feature_importance_preadmission.png**: Top features
- **cv_scores_preadmission.png**: CV scores
- **model_comparison_preadmission.png**: Confronto

## Come Eseguire:
```bash
python train_models_preadmission.py
```

## Performance Attese:
- Balanced Accuracy: 55-65% (vs 71% versione completa)
- Gap: -10-15% è NORMALE e ACCETTABILE
- Vantaggio: Predizione 6 mesi prima!

## Top Features Attese:
1. Tuition fees up to date
2. Scholarship holder
3. Admission grade
4. Previous qualification (grade)
5. Mother's/Father's qualification

## Uso Modelli:
```python
import joblib

model = joblib.load('03_training_preadmission/xgb_model_preadmission.pkl')
predictions = model.predict(X_new)  # X_new ha 24 features
```

## Confronto con Versione Completa:
Usa ENTRAMBI i modelli in sequenza:
1. Pre-immatricolazione: Screening iniziale (Agosto)
2. Completo: Monitoraggio continuo (Durante anno)
""",

        'data': """# Data - Dataset Originale

Questa cartella contiene il dataset originale.

## File Richiesto:
- **data.csv**: Dataset UCI Machine Learning Repository #697

## Download:
1. Vai su: https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
2. Scarica il file CSV
3. Metti `data.csv` in questa cartella

## Formato:
- Separatore: `;`
- Encoding: UTF-8 (con BOM)
- Righe: 4,424 studenti
- Colonne: 37 variabili

## Nota:
Il file `data.csv` NON è incluso nel repository.
Devi scaricarlo manualmente dal link sopra.
"""
    }

    # Crea README files
    for folder, content in readmes.items():
        readme_path = base_dir / folder / 'README.md'

        if not readme_path.exists():
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✓ {folder}/README.md")
        else:
            print(f"→ {folder}/README.md (già esistente)")

    # Crea .gitkeep per cartelle vuote
    print("\n" + "=" * 80)
    print("📌 CREAZIONE .gitkeep")
    print("=" * 80)

    for folder in folders.keys():
        gitkeep_path = base_dir / folder / '.gitkeep'
        if not gitkeep_path.exists():
            gitkeep_path.touch()
            print(f"✓ {folder}/.gitkeep")

    # Stampa struttura finale
    print("\n" + "=" * 80)
    print("📁 STRUTTURA FINALE")
    print("=" * 80)

    print("""
project/
│
├── 📊 VERSIONE COMPLETA (con performance universitaria)
│   ├── 01_analysis/
│   │   ├── README.md
│   │   ├── .gitkeep
│   │   └── [output di student_analysis.py - 36 features]
│   │
│   ├── 02_preprocessing/
│   │   ├── README.md
│   │   ├── .gitkeep
│   │   └── [output di preprocessing_smote.py - 36 features]
│   │
│   └── 03_training/
│       ├── README.md
│       ├── .gitkeep
│       └── [output di train_models.py - BA: 71%]
│
├── 🎯 VERSIONE PRE-IMMATRICOLAZIONE (solo dati pre-universitari)
│   ├── 01_analysis_preadmission/
│   │   ├── README.md
│   │   ├── .gitkeep
│   │   └── [output di student_analysis_preadmission.py - 24 features]
│   │
│   ├── 02_preprocessing_preadmission/
│   │   ├── README.md
│   │   ├── .gitkeep
│   │   └── [output di preprocessing_smote_preadmission.py - 24 features]
│   │
│   └── 03_training_preadmission/
│       ├── README.md
│       ├── .gitkeep
│       └── [output di train_models_preadmission.py - BA: ~60%]
│
├── data/
│   ├── README.md
│   ├── .gitkeep
│   └── data.csv (da scaricare)
│
├── student_analysis.py                    # Script versione completa
├── preprocessing_smote.py                 # Script versione completa
├── train_models.py                        # Script versione completa
│
├── student_analysis_preadmission.py       # Script versione pre-imm
├── preprocessing_smote_preadmission.py    # Script versione pre-imm (da creare)
├── train_models_preadmission.py           # Script versione pre-imm (da creare)
│
└── setup_folders.py (questo script)
""")

    # Riepilogo
    print("=" * 80)
    print("✅ SETUP COMPLETATO!")
    print("=" * 80)

    if created:
        print(f"\n📂 Cartelle create: {len(created)}")
        for folder in created:
            print(f"  ✓ {folder}/")

    if existing:
        print(f"\n📂 Cartelle già esistenti: {len(existing)}")
        for folder in existing:
            print(f"  → {folder}/")

    print(f"\n📝 README creati: {len(readmes)}")
    print(f"📌 .gitkeep creati: {len(folders)}")

    print("\n" + "=" * 80)
    print("PROSSIMI PASSI:")
    print("=" * 80)
    print("""
📊 PIPELINE COMPLETA (Accuracy 71%, con performance universitaria):
   1. Metti il file 'data.csv' nella cartella 'data/'
   2. Esegui: python student_analysis.py
   3. Esegui: python preprocessing_smote.py
   4. Esegui: python train_models.py
   → Output in: 01_analysis/, 02_preprocessing/, 03_training/

🎯 PIPELINE PRE-IMMATRICOLAZIONE (Accuracy ~60%, predizione precoce):
   1. Metti il file 'data.csv' nella cartella 'data/'
   2. Esegui: python student_analysis_preadmission.py
   3. Esegui: python preprocessing_smote_preadmission.py (da creare)
   4. Esegui: python train_models_preadmission.py (da creare)
   → Output in: 01_analysis_preadmission/, 02_preprocessing_preadmission/, 03_training_preadmission/

💡 RACCOMANDAZIONE:
   Esegui ENTRAMBE le pipeline e confronta i risultati!
   - Completa: Migliore accuracy, predice durante anno
   - Pre-imm: Predizione precoce, interventi preventivi
""")
    print("=" * 80)

if __name__ == "__main__":
    create_folder_structure()