"""
Analisi Dataset: Predict Students' Dropout and Academic Success
Dataset UCI Machine Learning Repository - ID: 697
Versione corretta per il file CSV locale

ISTRUZIONI PER L'USO:
=====================
1. Scarica il file 'data.csv' dal dataset UCI o usa quello fornito
2. Metti data.csv nella STESSA CARTELLA di questo script
3. Apri il terminale nella cartella
4. Esegui: python student_analysis.py

REQUISITI:
==========
pip install pandas numpy matplotlib seaborn scikit-learn

OUTPUT:
=======
- student_data_original.csv (dati con valori numerici)
- student_data_mapped.csv (dati con valori leggibili)
- student_analysis_visualizations.png (15 grafici principali)
- correlation_matrix_complete.png (matrice correlazione COMPLETA - tutte le variabili)
- correlation_matrix_academic.png (matrice correlazione ACCADEMICA - variabili chiave)

Tutti i file vengono salvati nella directory corrente dove esegui lo script.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Configurazione stile grafici
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("CARICAMENTO DATASET: Predict Students' Dropout and Academic Success")
print("=" * 80)

# Directory di output (dove verranno salvati tutti i file)
output_dir = os.getcwd()
print(f"📁 Directory di lavoro: {output_dir}\n")

# Caricamento dataset dal file CSV
# Il file usa ; come separatore e ha un BOM all'inizio

# Cerca il file data.csv in diverse posizioni
possible_paths = [
    'data.csv',                          # Directory corrente
    os.path.join(os.path.dirname(__file__), 'data.csv'),  # Stessa dir dello script
    os.path.join(os.getcwd(), 'data.csv'),                # Working directory
]

csv_path = None
for path in possible_paths:
    if os.path.exists(path):
        csv_path = path
        break

if csv_path is None:
    print(f"\n✗ Errore: file 'data.csv' non trovato!")
    print("\nIl programma ha cercato in:")
    for path in possible_paths:
        print(f"  - {os.path.abspath(path)}")
    print("\n📁 ISTRUZIONI:")
    print("  1. Scarica il file 'data.csv' dal dataset UCI")
    print("  2. Metti data.csv nella stessa cartella di questo script")
    print("  3. Esegui di nuovo: python student_analysis.py")
    exit(1)

try:
    print(f"📂 Caricamento da: {os.path.abspath(csv_path)}")
    df = pd.read_csv(csv_path, sep=';', encoding='utf-8-sig')

    # Pulisci i nomi delle colonne (rimuovi spazi, tab e caratteri strani)
    df.columns = df.columns.str.strip().str.replace('"', '').str.replace('\t', '')

    print(f"\n✓ Dataset caricato con successo!")
    print(f"  - Numero di studenti: {len(df)}")
    print(f"  - Numero di variabili: {len(df.columns)}")

except Exception as e:
    print(f"\n✗ Errore nel caricamento del dataset: {e}")
    print("\n💡 Suggerimenti:")
    print("  - Verifica che il file non sia corrotto")
    print("  - Assicurati di avere i permessi di lettura")
    exit(1)

print("\n" + "=" * 80)
print("STRUTTURA DEL DATASET")
print("=" * 80)
print("\nNomi delle colonne nel dataset:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2d}. {col}")

target_col = 'Target'
print(f"\n✓ Variabile target: {target_col}")

print("\n" + "=" * 80)
print("MAPPATURA DELLE VARIABILI CATEGORICHE")
print("=" * 80)

# Dizionari di mappatura basati sulla Variables Table UCI
mappings = {
    'Marital status': {
        1: 'Single', 2: 'Married', 3: 'Widower',
        4: 'Divorced', 5: 'Facto union', 6: 'Legally separated'
    },
    'Application mode': {
        1: '1st phase - general', 2: 'Ordinance 612/93',
        5: '1st phase - Azores', 7: 'Other higher courses',
        10: 'Ordinance 854-B/99', 15: 'International (bachelor)',
        16: '1st phase - Madeira', 17: '2nd phase - general',
        18: '3rd phase - general', 26: 'Ordinance 533-A/99 b2',
        27: 'Ordinance 533-A/99 b3', 39: 'Over 23 years',
        42: 'Transfer', 43: 'Change of course',
        44: 'Tech specialization', 51: 'Change institution/course',
        53: 'Short cycle diploma', 57: 'Change institution (Intl)'
    },
    'Course': {
        33: 'Biofuel Production', 171: 'Animation & Multimedia',
        8014: 'Social Service (evening)', 9003: 'Agronomy',
        9070: 'Communication Design', 9085: 'Veterinary Nursing',
        9119: 'Informatics Engineering', 9130: 'Equinculture',
        9147: 'Management', 9238: 'Social Service',
        9254: 'Tourism', 9500: 'Nursing',
        9556: 'Oral Hygiene', 9670: 'Advertising & Marketing',
        9773: 'Journalism', 9853: 'Basic Education',
        9991: 'Management (evening)'
    },
    'Daytime/evening attendance': {1: 'Daytime', 0: 'Evening'},
    'Previous qualification': {
        1: 'Secondary education', 2: 'Higher - bachelor',
        3: 'Higher - degree', 4: 'Higher - master',
        5: 'Higher - doctorate', 6: 'Frequency higher ed',
        9: '12th year incomplete', 10: '11th year incomplete',
        12: 'Other - 11th year', 14: '10th year',
        15: '10th year incomplete', 19: 'Basic ed 3rd cycle',
        38: 'Basic ed 2nd cycle', 39: 'Tech specialization',
        40: 'Higher - degree (1st)', 42: 'Professional higher tech',
        43: 'Higher - master (2nd)'
    },
    'Nacionality': {
        1: 'Portuguese', 2: 'German', 6: 'Spanish', 11: 'Italian',
        13: 'Dutch', 14: 'English', 17: 'Lithuanian', 21: 'Angolan',
        22: 'Cape Verdean', 24: 'Guinean', 25: 'Mozambican',
        26: 'Santomean', 32: 'Turkish', 41: 'Brazilian',
        62: 'Romanian', 100: 'Moldova', 101: 'Mexican',
        103: 'Ukrainian', 105: 'Russian', 108: 'Cuban', 109: 'Colombian'
    },
    "Mother's qualification": {
        1: 'Secondary 12th', 2: 'Higher - Bachelor', 3: 'Higher - Degree',
        4: 'Higher - Master', 5: 'Higher - Doctorate', 6: 'Frequency Higher Ed',
        9: '12th incomplete', 10: '11th incomplete', 11: '7th Year (old)',
        12: 'Other - 11th', 14: '10th Year', 18: 'General commerce',
        19: 'Basic Ed 3rd Cycle', 22: 'Technical-professional', 26: '7th year',
        27: '2nd cycle high school', 29: '9th incomplete', 30: '8th year',
        34: 'Unknown', 35: "Can't read/write", 36: 'Can read no 4th year',
        37: 'Basic ed 1st cycle', 38: 'Basic Ed 2nd Cycle',
        39: 'Tech specialization', 40: 'Higher degree (1st)',
        41: 'Specialized higher', 42: 'Professional higher tech',
        43: 'Higher Master (2nd)', 44: 'Higher Doctorate (3rd)'
    },
    "Father's qualification": {
        1: 'Secondary 12th', 2: 'Higher - Bachelor', 3: 'Higher - Degree',
        4: 'Higher - Master', 5: 'Higher - Doctorate', 6: 'Frequency Higher Ed',
        9: '12th incomplete', 10: '11th incomplete', 11: '7th Year (old)',
        12: 'Other - 11th', 13: '2nd year compl high school', 14: '10th Year',
        18: 'General commerce', 19: 'Basic Ed 3rd Cycle',
        20: 'Complementary High School', 22: 'Technical-professional',
        25: 'Compl High School incomplete', 26: '7th year',
        27: '2nd cycle high school', 29: '9th incomplete', 30: '8th year',
        31: 'Admin & Commerce', 33: 'Accounting & Admin',
        34: 'Unknown', 35: "Can't read/write", 36: 'Can read no 4th year',
        37: 'Basic ed 1st cycle', 38: 'Basic Ed 2nd Cycle',
        39: 'Tech specialization', 40: 'Higher degree (1st)',
        41: 'Specialized higher', 42: 'Professional higher tech',
        43: 'Higher Master (2nd)', 44: 'Higher Doctorate (3rd)'
    },
    "Mother's occupation": {
        0: 'Student', 1: 'Representatives', 2: 'Intellectual & Scientific',
        3: 'Intermediate Level Technicians', 4: 'Administrative staff',
        5: 'Personal Services', 6: 'Protection & Security',
        7: 'Farmers', 8: 'Skilled Workers',
        9: 'Installation & Machine Operators', 10: 'Unskilled Workers',
        90: 'Armed Forces Professions', 99: 'Other', 101: 'Health professionals',
        122: 'Teachers', 123: 'Specialists', 125: 'ICT Technicians',
        131: 'Intermediate Level Technicians', 132: 'Technicians & Professionals',
        134: 'Administrative staff', 141: 'Office Workers',
        143: 'Data/Reception Operators', 144: 'Other Administrative',
        151: 'Personal Service Workers', 152: 'Sellers',
        153: 'Personal Care', 171: 'Skilled Construction',
        173: 'Skilled Workers', 175: 'Food Processing',
        191: 'Cleaning Workers', 192: 'Unskilled Workers (Agriculture)',
        193: 'Unskilled Workers (Industry)', 194: 'Meal Preparation'
    },
    "Father's occupation": {
        0: 'Student', 1: 'Representatives', 2: 'Intellectual & Scientific',
        3: 'Intermediate Level Technicians', 4: 'Administrative staff',
        5: 'Personal Services', 6: 'Protection & Security',
        7: 'Farmers', 8: 'Skilled Workers',
        9: 'Installation & Machine Operators', 10: 'Unskilled Workers',
        90: 'Armed Forces Professions', 99: 'Other', 101: 'Health professionals',
        102: 'Teachers', 103: 'Specialists', 112: 'Intermediate Level Health',
        114: 'Intermediate Level', 121: 'Specialists',
        122: 'Teachers', 123: 'Specialists', 124: 'Specialists',
        131: 'Intermediate Level Technicians', 132: 'Technicians & Professionals',
        134: 'Administrative staff', 135: 'Technicians',
        141: 'Office Workers', 143: 'Data/Reception Operators',
        144: 'Other Administrative', 151: 'Personal Service Workers',
        152: 'Sellers', 153: 'Personal Care', 154: 'Personal Services',
        161: 'Market-Oriented Farmers', 163: 'Farmers',
        171: 'Skilled Construction', 172: 'Skilled Workers',
        174: 'Skilled Workers', 175: 'Food Processing',
        181: 'Installation & Machine Operators', 182: 'Assemblers',
        183: 'Vehicle Drivers', 192: 'Unskilled Workers (Agriculture)',
        193: 'Unskilled Workers (Industry)', 194: 'Meal Preparation'
    }
}

# Crea dataset mappato
df_mapped = df.copy()
mapped_count = 0

for col, mapping_dict in mappings.items():
    if col in df_mapped.columns:
        df_mapped[col] = df_mapped[col].map(mapping_dict)
        print(f"✓ Mappata: {col}")
        mapped_count += 1
    else:
        print(f"⚠ Non trovata: {col}")

print(f"\n✓ {mapped_count} colonne mappate con successo!")

# Salva i dataset
original_csv = os.path.join(output_dir, 'student_data_original.csv')
mapped_csv = os.path.join(output_dir, 'student_data_mapped.csv')

df.to_csv(original_csv, index=False)
df_mapped.to_csv(mapped_csv, index=False)

print(f"\n✓ Dataset salvati:")
print(f"  - {original_csv}")
print(f"  - {mapped_csv}")

print("\n" + "=" * 80)
print("ANALISI ESPLORATIVA DATI")
print("=" * 80)

# Distribuzione target
print("\nDistribuzione variabile TARGET:")
print("-" * 80)
target_counts = df[target_col].value_counts()
target_pct = df[target_col].value_counts(normalize=True) * 100

for status, count in target_counts.items():
    pct = target_pct[status]
    bar = '█' * int(pct / 2)
    print(f"  {status:15s}: {count:5d} ({pct:5.2f}%) {bar}")

print("\n" + "=" * 80)
print("CREAZIONE VISUALIZZAZIONI")
print("=" * 80)

# Crea figura con visualizzazioni
fig = plt.figure(figsize=(20, 24))
colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']

# 1. Distribuzione Target
ax1 = plt.subplot(5, 3, 1)
target_counts.plot(kind='bar', ax=ax1, color=colors)
ax1.set_title('Distribuzione Status Studenti', fontsize=14, fontweight='bold')
ax1.set_xlabel('Status', fontsize=12)
ax1.set_ylabel('Numero Studenti', fontsize=12)
ax1.tick_params(axis='x', rotation=45)
for i, v in enumerate(target_counts.values):
    ax1.text(i, v + 50, str(v), ha='center', va='bottom', fontweight='bold')

# 2. Status per Genere
ax2 = plt.subplot(5, 3, 2)
gender_target = pd.crosstab(df['Gender'], df[target_col], normalize='index') * 100
gender_target.plot(kind='bar', ax=ax2, color=colors)
ax2.set_title('Status per Genere (%)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Genere (1=Male, 0=Female)', fontsize=12)
ax2.set_ylabel('Percentuale', fontsize=12)
ax2.legend(title='Status')
ax2.tick_params(axis='x', rotation=0)

# 3. Età all'iscrizione
ax3 = plt.subplot(5, 3, 3)
for status in df[target_col].unique():
    data = df[df[target_col] == status]['Age at enrollment']
    ax3.hist(data, alpha=0.6, label=status, bins=20)
ax3.set_title('Distribuzione Età all\'Iscrizione', fontsize=14, fontweight='bold')
ax3.set_xlabel('Età', fontsize=12)
ax3.set_ylabel('Frequenza', fontsize=12)
ax3.legend()

# 4. Voti 1° Semestre
ax4 = plt.subplot(5, 3, 4)
for status in df[target_col].unique():
    data = df[df[target_col] == status]['Curricular units 1st sem (grade)']
    ax4.hist(data, alpha=0.6, label=status, bins=20)
ax4.set_title('Voti 1° Semestre per Status', fontsize=14, fontweight='bold')
ax4.set_xlabel('Voto Medio', fontsize=12)
ax4.set_ylabel('Frequenza', fontsize=12)
ax4.legend()

# 5. Voti 2° Semestre
ax5 = plt.subplot(5, 3, 5)
for status in df[target_col].unique():
    data = df[df[target_col] == status]['Curricular units 2nd sem (grade)']
    ax5.hist(data, alpha=0.6, label=status, bins=20)
ax5.set_title('Voti 2° Semestre per Status', fontsize=14, fontweight='bold')
ax5.set_xlabel('Voto Medio', fontsize=12)
ax5.set_ylabel('Frequenza', fontsize=12)
ax5.legend()

# 6. Crediti Approvati 1° Semestre
ax6 = plt.subplot(5, 3, 6)
df.boxplot(column='Curricular units 1st sem (approved)', by=target_col, ax=ax6)
ax6.set_title('Crediti Approvati 1° Semestre', fontsize=14, fontweight='bold')
ax6.set_xlabel('Status', fontsize=12)
ax6.set_ylabel('Numero Crediti', fontsize=12)
plt.suptitle('')

# 7. Tasso di Approvazione
ax7 = plt.subplot(5, 3, 7)
df_temp = df.copy()
df_temp['approval_rate'] = (df_temp['Curricular units 1st sem (approved)'] /
                              df_temp['Curricular units 1st sem (enrolled)'] * 100)
for status in df[target_col].unique():
    data = df_temp[df_temp[target_col] == status]['approval_rate']
    ax7.hist(data.dropna(), alpha=0.6, label=status, bins=20)
ax7.set_title('Tasso Approvazione 1° Sem', fontsize=14, fontweight='bold')
ax7.set_xlabel('Tasso (%)', fontsize=12)
ax7.set_ylabel('Frequenza', fontsize=12)
ax7.legend()

# 8. Stato Civile
ax8 = plt.subplot(5, 3, 8)
marital_target = pd.crosstab(df_mapped['Marital status'], df_mapped[target_col])
marital_target.plot(kind='bar', ax=ax8, color=colors)
ax8.set_title('Status per Stato Civile', fontsize=14, fontweight='bold')
ax8.set_xlabel('Stato Civile', fontsize=12)
ax8.set_ylabel('Numero Studenti', fontsize=12)
ax8.legend(title='Status')
ax8.tick_params(axis='x', rotation=45)

# 9. Top 10 Corsi
ax9 = plt.subplot(5, 3, 9)
top_courses = df_mapped['Course'].value_counts().head(10)
top_courses.plot(kind='barh', ax=ax9, color='steelblue')
ax9.set_title('Top 10 Corsi più Frequentati', fontsize=14, fontweight='bold')
ax9.set_xlabel('Numero Studenti', fontsize=12)
ax9.set_ylabel('')

# 10. Debiti
ax10 = plt.subplot(5, 3, 10)
df.boxplot(column='Debtor', by=target_col, ax=ax10)
ax10.set_title('Debitori per Status', fontsize=14, fontweight='bold')
ax10.set_xlabel('Status', fontsize=12)
ax10.set_ylabel('Debiti (1=Sì, 0=No)', fontsize=12)
plt.suptitle('')

# 11. Qualificazione Precedente
ax11 = plt.subplot(5, 3, 11)
prev_qual_target = pd.crosstab(df_mapped['Previous qualification'],
                                 df_mapped[target_col], normalize='index') * 100
top_quals = df_mapped['Previous qualification'].value_counts().head(8)
prev_qual_target.loc[top_quals.index].plot(kind='bar', ax=ax11, color=colors)
ax11.set_title('Status per Qual. Precedente (Top 8)', fontsize=14, fontweight='bold')
ax11.set_xlabel('', fontsize=10)
ax11.set_ylabel('Percentuale', fontsize=12)
ax11.legend(title='Status')
ax11.tick_params(axis='x', rotation=45, labelsize=8)

# 12. Voto Qualificazione Precedente
ax12 = plt.subplot(5, 3, 12)
for status in df[target_col].unique():
    data = df[df[target_col] == status]['Previous qualification (grade)']
    ax12.hist(data, alpha=0.6, label=status, bins=20)
ax12.set_title('Voto Qualificazione Precedente', fontsize=14, fontweight='bold')
ax12.set_xlabel('Voto (0-200)', fontsize=12)
ax12.set_ylabel('Frequenza', fontsize=12)
ax12.legend()

# 13. Correlazione Voti
ax13 = plt.subplot(5, 3, 13)
for status, color in zip(df[target_col].unique(), colors):
    mask = df[target_col] == status
    ax13.scatter(df.loc[mask, 'Curricular units 1st sem (grade)'],
                df.loc[mask, 'Curricular units 2nd sem (grade)'],
                alpha=0.5, label=status, color=color, s=20)
ax13.set_title('Voti 1° vs 2° Semestre', fontsize=14, fontweight='bold')
ax13.set_xlabel('Voto 1° Sem', fontsize=12)
ax13.set_ylabel('Voto 2° Sem', fontsize=12)
ax13.legend()
ax13.plot([0, 20], [0, 20], 'k--', alpha=0.3)

# 14. Top Nazionalità
ax14 = plt.subplot(5, 3, 14)
top_nat = df_mapped['Nacionality'].value_counts().head(10)
top_nat.plot(kind='barh', ax=ax14, color='coral')
ax14.set_title('Top 10 Nazionalità', fontsize=14, fontweight='bold')
ax14.set_xlabel('Numero Studenti', fontsize=12)
ax14.set_ylabel('')

# 15. Borse di Studio
ax15 = plt.subplot(5, 3, 15)
schol_target = pd.crosstab(df['Scholarship holder'], df[target_col], normalize='index') * 100
schol_target.plot(kind='bar', ax=ax15, color=colors)
ax15.set_title('Status per Borsa Studio', fontsize=14, fontweight='bold')
ax15.set_xlabel('Borsa (1=Sì, 0=No)', fontsize=12)
ax15.set_ylabel('Percentuale', fontsize=12)
ax15.legend(title='Status')
ax15.tick_params(axis='x', rotation=0)

plt.tight_layout()
viz_path = os.path.join(output_dir, 'student_analysis_visualizations.png')
plt.savefig(viz_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Grafici salvati in: {viz_path}")

# Statistiche per status
print("\n" + "=" * 80)
print("STATISTICHE DESCRITTIVE PER STATUS")
print("=" * 80)

numeric_cols = ['Age at enrollment', 'Previous qualification (grade)',
                'Curricular units 1st sem (approved)', 'Curricular units 1st sem (grade)',
                'Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (grade)',
                'Admission grade']

for status in df[target_col].unique():
    print(f"\n{str(status).upper()}")
    print("-" * 80)
    status_data = df[df[target_col] == status][numeric_cols]
    print(status_data.describe().round(2))

# Matrice di correlazione
print("\n" + "=" * 80)
print("MATRICE DI CORRELAZIONE")
print("=" * 80)

# Seleziona TUTTE le colonne numeriche per una correlazione completa
all_numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Rimuovi il target dalla lista (verrà analizzato separatamente)
if target_col in all_numeric_cols:
    all_numeric_cols.remove(target_col)

print(f"\nCalcolo correlazioni per {len(all_numeric_cols)} variabili numeriche...")

# Matrice di correlazione COMPLETA
fig1, ax1 = plt.subplots(figsize=(20, 18))
corr_matrix_full = df[all_numeric_cols].corr()
sns.heatmap(corr_matrix_full, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, ax=ax1, cbar_kws={"shrink": 0.8},
            annot_kws={"size": 7})
ax1.set_title('Matrice di Correlazione COMPLETA - Tutte le Variabili',
              fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
corr_path_full = os.path.join(output_dir, 'correlation_matrix_complete.png')
plt.savefig(corr_path_full, dpi=300, bbox_inches='tight')
print(f"\n✓ Matrice COMPLETA salvata in: {corr_path_full}")

# Matrice di correlazione RIDOTTA (solo variabili più rilevanti per leggibilità)
correlation_cols_key = [
    'Age at enrollment',
    'Previous qualification (grade)',
    'Admission grade',
    'Curricular units 1st sem (credited)',
    'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (evaluations)',
    'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)',
    'Curricular units 2nd sem (credited)',
    'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)',
    'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)',
    'Unemployment rate',
    'Inflation rate',
    'GDP'
]

fig2, ax2 = plt.subplots(figsize=(14, 12))
corr_matrix_key = df[correlation_cols_key].corr()
sns.heatmap(corr_matrix_key, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, ax=ax2, cbar_kws={"shrink": 0.8},
            annot_kws={"size": 9})
ax2.set_title('Matrice di Correlazione - Variabili Performance Accademica',
              fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
corr_path_key = os.path.join(output_dir, 'correlation_matrix_academic.png')
plt.savefig(corr_path_key, dpi=300, bbox_inches='tight')
print(f"✓ Matrice ACCADEMICA salvata in: {corr_path_key}")

# Stampa le correlazioni più forti (>0.5)
print("\n" + "=" * 80)
print("CORRELAZIONI PIÙ FORTI (|r| > 0.5)")
print("=" * 80)
print("\nCoppie di variabili con correlazione forte:")
print("-" * 80)

# Estrai coppie con correlazione > 0.5 (escludendo diagonale)
strong_corr = []
for i in range(len(corr_matrix_full.columns)):
    for j in range(i+1, len(corr_matrix_full.columns)):
        corr_val = corr_matrix_full.iloc[i, j]
        if abs(corr_val) > 0.5:
            var1 = corr_matrix_full.columns[i]
            var2 = corr_matrix_full.columns[j]
            strong_corr.append((var1, var2, corr_val))

# Ordina per valore assoluto di correlazione
strong_corr.sort(key=lambda x: abs(x[2]), reverse=True)

for var1, var2, corr_val in strong_corr[:20]:  # Top 20
    direction = "+" if corr_val > 0 else "-"
    bar = "█" * int(abs(corr_val) * 20)
    print(f"  {var1:45s} <-> {var2:45s} : {direction}{abs(corr_val):.3f} {bar}")

# Feature Importance
print("\n" + "=" * 80)
print("CORRELAZIONI CON IL TARGET")
print("=" * 80)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_temp = df.copy()
df_temp['target_encoded'] = le.fit_transform(df_temp[target_col])

correlations = df_temp.select_dtypes(include=[np.number]).corr()['target_encoded'].abs().sort_values(ascending=False)
print("\nTop 20 variabili correlate con il target:")
print("-" * 80)
for i, (col, corr) in enumerate(correlations.head(20).items(), 1):
    bar = '█' * int(corr * 50)
    print(f"{i:2d}. {col:50s} {corr:6.4f} {bar}")

# Insights chiave
print("\n" + "=" * 80)
print("INSIGHTS CHIAVE PER MODELLO PREDITTIVO")
print("=" * 80)

print("\n1. VARIABILI PIÙ IMPORTANTI (correlazione > 0.3):")
important_vars = correlations[correlations > 0.3].drop('target_encoded')
for var in important_vars.index:
    print(f"   • {var}")

print("\n2. DIFFERENZE TRA STATUS:")
print(f"   • Dropout: {target_counts['Dropout']} studenti ({target_pct['Dropout']:.1f}%)")
print(f"   • Graduate: {target_counts['Graduate']} studenti ({target_pct['Graduate']:.1f}%)")
if 'Enrolled' in target_counts:
    print(f"   • Enrolled: {target_counts['Enrolled']} studenti ({target_pct['Enrolled']:.1f}%)")

print("\n3. MEDIE VOTI PER STATUS:")
for status in df[target_col].unique():
    mean_1sem = df[df[target_col] == status]['Curricular units 1st sem (grade)'].mean()
    mean_2sem = df[df[target_col] == status]['Curricular units 2nd sem (grade)'].mean()
    print(f"   • {status:10s}: 1° sem = {mean_1sem:5.2f}, 2° sem = {mean_2sem:5.2f}")

print("\n4. RACCOMANDAZIONI PER IL MODELLO:")
print("   • Usare algoritmi che gestiscono classi sbilanciate (es. Random Forest, XGBoost)")
print("   • Considerare feature engineering: tasso approvazione, trend voti, ecc.")
print("   • Variabili più predittive: performance 1° e 2° semestre")
print("   • Valutare tecniche di oversampling/undersampling per bilanciare le classi")

print("\n" + "=" * 80)
print("ANALISI COMPLETATA")
print("=" * 80)
print(f"\n📁 Tutti i file sono stati salvati in: {output_dir}\n")
print("File generati:")
print(f"  1. {os.path.basename(original_csv)} - Dataset con valori numerici")
print(f"  2. {os.path.basename(mapped_csv)} - Dataset con valori leggibili")
print(f"  3. {os.path.basename(viz_path)} - 15 visualizzazioni principali")
print(f"  4. {os.path.basename(corr_path_full)} - Matrice correlazione COMPLETA")
print(f"  5. {os.path.basename(corr_path_key)} - Matrice correlazione ACCADEMICA")
print("\nIl dataset è pronto per la costruzione di modelli predittivi!")
print("=" * 80)

plt.show()