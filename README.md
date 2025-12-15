# Student Dropout Prediction System

Sistema di Machine Learning per la predizione del dropout studentesco universitario, sviluppato con Random Forest e XGBoost. Il progetto offre due modalità di predizione: **standard** (con dati post-iscrizione) e **pre-immatricolazione** (solo dati disponibili prima dell'iscrizione).

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.2-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)

## Indice

- [Descrizione](#descrizione)
- [Dataset](#dataset)
- [Struttura del Progetto](#struttura-del-progetto)
- [Installazione](#installazione)
- [Utilizzo](#utilizzo)
- [Pipeline](#pipeline)

## Descrizione

Questo progetto implementa un sistema per la predizione del dropout studentesco. Classifichiamo gli studenti in tre categorie:

- **Dropout**: Studenti che abbandonano gli studi o che sono fuori corso da più di 3 anni
- **Enrolled**: Studenti fuori corso tra 1 e 3 anni
- **Graduate**: Studenti che si laureano con successo

## Dataset

Il progetto utilizza il dataset **"Predict Students' Dropout and Academic Success"** dall'UCI Machine Learning Repository.

### Caratteristiche del Dataset

- **Campioni**: 4,424 studenti
- **Features originali**: 36 variabili
- **Features pre-immatricolazione**: 24 variabili
- **Provenienza**: Dati provenienti dall'**Instituto Politécnico de Portalegre**, in Portogallo
- **Target**: 3 classi (Dropout, Enrolled, Graduate)


## Struttura del Progetto

```
student-dropout-prediction/
│
├── 📄 setup_folders.py                  # Setup struttura cartelle
├── 📄 color_config.py                   # Configurazione palette cromatiche per le visualizzazioni
├── 📄 student_analysis.py               # Analisi del dataset
├── 📄 preprocessing_smote.py            # Preprocessing mediante SMOTE
├── 📄 train_models.py                   # Training modelli ML
├── 📄 app.py                            # simulatore interattivo 
├── 📄 README.md                         
├── 📄 requirements.txt                  # Dipendenze
│
├── 📂 data/                             # Dataset
│   └── data.csv                         # File dati UCI
│
├── 📂 01_analysis/                      # Output analisi - STANDARD
│   ├── student_data_original.csv
│   ├── student_data_mapped.csv
│   ├── feature_mappings_reverse.json
│   └── visualizations/
│
├── 📂 02_preprocessing/                 # Output preprocessing - STANDARD
│   ├── train_original.csv
│   ├── train_smote.csv
│   ├── test_set.csv
│   └── visualizations/
│
├── 📂 03_training/                      # Output training - STANDARD
│   ├── rf_model.pkl
│   ├── xgb_model.pkl
│   ├── label_encoder.pkl
│   ├── feature_names.pkl
│   ├── training_results.pkl
│   └── visualizations/
│
├── 📂 01_analysis_preadmission/         # Output analisi - PRE-IMMATRICOLAZIONE
│   ├── student_data_preadmission.csv
│   ├── feature_mappings_reverse.json
│   └── visualizations/
│
├── 📂 02_preprocessing_preadmission/    # Output preprocessing - PRE-IMMATRICOLAZIONE
│   ├── train_original_preadmission.csv
│   ├── train_smote_preadmission.csv
│   ├── test_set_preadmission.csv
│   └── visualizations/
│
└── 📂 03_training_preadmission/         # Output training - PRE-IMMATRICOLAZIONE
    ├── rf_model_preadmission.pkl
    ├── xgb_model_preadmission.pkl
    ├── label_encoder_preadmission.pkl
    ├── feature_names_preadmission.pkl
    ├── training_results_preadmission.pkl
    └── visualizations/
```

## Installazione

### Requisiti di Sistema

- Python 3.8 o superiore
- pip package manager

### 1. Clona il Repository

```bash
git clone https://github.com/TheCastes/student_dropout_prediction_PMP.git
cd student_dropout_prediction_PMP
```

### 2. Crea un Ambiente Virtuale

```bash
# Linux/Mac Os
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Installa le Dipendenze

```bash
pip install -r requirements.txt
```

### Nota su imbalanced-learn

Se riscontri problemi con `imbalanced-learn`, esegui:

```bash
pip uninstall -y imbalanced-learn scikit-learn
pip install scikit-learn==1.5.2
pip install imbalanced-learn==0.12.4
```
### 4. Scarica il Dataset

1. Scarica il dataset dall'UCI ML Repository: [Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
2. Salva il file come `data.csv` nella cartella `data/`

## Utilizzo

```bash
# 1. Setup struttura cartelle
python setup_folders.py

# 2. Esegui la pipeline
python student_analysis.py
python preprocessing_smote.py
python train_models.py

# 3. Avvia l'interfaccia web
streamlit run app.py
```

### Modalità Pre-Immatricolazione
```bash
python student_analysis.py --preadmission
python preprocessing_smote.py --preadmission
python train_models.py --preadmission
```

L'app web rileverà automaticamente entrambe le modalità e permetterà di switchare tra loro.
## Pipeline

### Step 1: Setup Cartelle

```bash
python setup_folders.py
```
### Step 2: Analisi Dati

```bash
# Modalità STANDARD
python student_analysis.py
# Modalità PRE-IMMATRICOLAZIONE
python student_analysis.py --preadmission
```
**Cosa fa**:
- Carica e pulisce il dataset
- Genera statistiche descrittive
- Crea visualizzazioni (distribuzione target, correlazioni, etc.)
- Salva dataset processati e mappings

**Output**:
- `student_data_original.csv` / `student_data_preadmission.csv`
- `student_data_mapped.csv` (solo standard)
- `feature_mappings_reverse.json`
- Cartella `visualizations/` con grafici

### Step 3: Preprocessing con SMOTE

```bash
# Modalità STANDARD
python preprocessing_smote.py
# Modalità PRE-IMMATRICOLAZIONE
python preprocessing_smote.py --preadmission
```

**Cosa fa**:
- Split stratificato 80/20 (train/test)
- Applica SMOTE al training set per bilanciare le classi
- Genera visualizzazioni del bilanciamento

**Output**:
- `train_original.csv` - Training set sbilanciato (per confronto)
- `train_smote.csv` - Training set bilanciato con SMOTE
- `test_set.csv` - Test set (mantenuto sbilanciato)
- Visualizzazioni del bilanciamento classi

**Configurazione SMOTE**:
- Strategy: `auto` (bilancia tutte le classi)
- K-neighbors: 5

### Step 4: Training Modelli

```bash
# Modalità STANDARD
python train_models.py
# Modalità PRE-IMMATRICOLAZIONE
python train_models.py --preadmission
```

**Cosa fa**:
- Carica i dati preprocessati
- Esegue 5-fold stratified cross-validation
- Addestra Random Forest e XGBoost
- Valuta su test set
- Salva modelli e risultati

**Output**:
- `rf_model.pkl` / `rf_model_preadmission.pkl`
- `xgb_model.pkl` / `xgb_model_preadmission.pkl`
- `label_encoder.pkl` / `label_encoder_preadmission.pkl`
- `feature_names.pkl` / `feature_names_preadmission.pkl`
- `training_results.pkl` / `training_results_preadmission.pkl`
- Confusion matrices e grafici F1-Score

### Modelli Machine Learning

#### Random Forest

**Configurazione**:
```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
```
**Vantaggi**:
- Robusto a overfitting
- Gestisce bene feature categoriche
- Interpreta feature importance
- Non richiede scaling
#### XGBoost
**Configurazione**:
```python
XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    eval_metric='mlogloss'
)
```
**Vantaggi**:
- Performance superiore in molti casi
- Gestione nativa missing values
- Ottimizzazione gradient boosting

#### Metriche di Valutazione
- **F1-Score (macro)**: Media non pesata degli F1-Score per classe
- **Confusion Matrix**: Analisi dettagliata errori per classe
- **Cross-Validation**: 5-fold stratificato per robustezza

### Step 5. Interfaccia Web

```bash
streamlit run app.py
```
### Funzionalità

#### 1. Selezione Modalità (Sidebar)
- **Standard**: Usa tutte le features
- **Pre-Immatricolazione**: Solo features disponibili all'iscrizione

#### 2. Selezione Modello
- Random Forest
- XGBoost
- Confronta Entrambi (side-by-side)

#### 3. Form Input Dati Studente

**Sezioni del form**:

1. **Informazioni Base**
   - Stato civile
   - Modalità di candidatura
   - Corso di studio
   - Frequenza (diurna/serale)

2. **Informazioni Personali**
   - Età all'iscrizione
   - Genere
   - Nazionalità

3. **Background Educativo**
   - Qualifica precedente
   - Voto qualifica precedente
   - Voto d'ammissione

4. **Background Familiare**
   - Qualifica madre/padre
   - Occupazione madre/padre

5. **Situazione Economica**
   - Borsa di studio
   - Tasse universitarie
   - Debiti
   - Situazione lavorativa

6. **Performance Universitaria** (solo modalità Standard)
   - Crediti 1° semestre (iscritti, approvati, voti)
   - Crediti 2° semestre (iscritti, approvati, voti)

#### 4. Risultati Predizione

**Visualizzazioni**:
- Classe predetta con colore distintivo
- Livello di confidenza
- Livello di rischio dropout
- Tabella probabilità per classe
- Grafico a barre probabilità
- (Se confronto) Accordo/disaccordo tra modelli
