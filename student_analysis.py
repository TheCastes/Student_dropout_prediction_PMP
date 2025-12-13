import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import json
import warnings

warnings.filterwarnings('ignore')

# Importa configurazione colori centralizzata
from color_config import (
    CLASS_COLORS, CLASS_COLORS_LIST, HEATMAP_CMAPS,
    CATEGORICAL_PALETTE, setup_plot_style, get_class_color,
    map_colors_to_labels
)

# ============================================================================
# CONFIGURAZIONE MODALITÀ
# ============================================================================
PREADMISSION_MODE = '--preadmission' in sys.argv
EXCLUDE_FEATURES_PREADMISSION = [
    'Curricular units 1st sem (credited)',
    'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (evaluations)',
    'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)',
    'Curricular units 1st sem (without evaluations)',
    'Curricular units 2nd sem (credited)',
    'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)',
    'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)',
    'Curricular units 2nd sem (without evaluations)'
]

# Setup stile grafico coerente
setup_plot_style()
sns.set_palette(CLASS_COLORS_LIST)

print("=" * 80)
if PREADMISSION_MODE:
    print("ANALISI PRE-IMMATRICOLAZIONE: Predict Students' Dropout")
    print("=" * 80)
    print("\nMODALITÀ PRE-IMMATRICOLAZIONE ATTIVA")
    print("CARICAMENTO DATASET: Predict Students' Dropout and Academic Success")
else:
    print("CARICAMENTO DATASET: Predict Students' Dropout and Academic Success")
    print("=" * 80)
base_dir = os.getcwd()
if PREADMISSION_MODE:
    output_dir = os.path.join(base_dir, '01_analysis_preadmission')
else:
    output_dir = os.path.join(base_dir, '01_analysis')
os.makedirs(output_dir, exist_ok=True)
print(f"Directory di lavoro: {base_dir}")
print(f"Output salvati in: {output_dir}\n")

possible_paths = [
    'data.csv',  # Directory corrente
    os.path.join(os.path.dirname(__file__), 'data.csv'),  # Stessa dir dello script
    os.path.join(os.getcwd(), 'data.csv'),  # Working directory
    'data/data.csv',
]

csv_path = None
for path in possible_paths:
    if os.path.exists(path):
        csv_path = path
        break

if csv_path is None:
    print(f"\nErrore: file 'data.csv' non trovato!")
    print("\nIl programma ha cercato in:")
    for path in possible_paths:
        print(f"  - {os.path.abspath(path)}")
    exit(1)
try:
    print(f"Caricamento da: {os.path.abspath(csv_path)}")
    df = pd.read_csv(csv_path, sep=';', encoding='utf-8-sig')
    df.columns = df.columns.str.strip().str.replace('"', '').str.replace('\t', '')
    print(f"\nDataset caricato con successo!")
    print(f"  - Numero di studenti: {len(df)}")
    print(f"  - Numero di variabili: {len(df.columns)}")
except Exception as e:
    print(f"\nErrore nel caricamento del dataset: {e}")
    exit(1)
# ============================================================================
# FILTRAGGIO FEATURES (se modalità pre-immatricolazione)
# ============================================================================
if PREADMISSION_MODE:
    print("\n" + "=" * 80)
    print("FILTRAGGIO FEATURES PRE-IMMATRICOLAZIONE")
    print("=" * 80)
    for i, feature in enumerate(EXCLUDE_FEATURES_PREADMISSION, 1):
        if feature in df.columns:
            print(f"   {i:2d}. {feature}")
    df = df.drop(columns=EXCLUDE_FEATURES_PREADMISSION, errors='ignore')

    print(f"\nDataset filtrato:")
    print(f"  - Variabili rimosse: {len(EXCLUDE_FEATURES_PREADMISSION)}")
    print(f"  - Variabili rimanenti: {len(df.columns)}")
    print(f"  - Features (escluso target): {len(df.columns) - 1}")
print("\n" + "=" * 80)
print("STRUTTURA DEL DATASET")
print("=" * 80)
print("\nNomi delle colonne nel dataset:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2d}. {col}")
target_col = 'Target'
print(f"\nVariabile target: {target_col}")
print("\n" + "=" * 80)
print("MAPPATURA DELLE VARIABILI CATEGORICHE")
print("=" * 80)
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
# ============================================================================
# SALVA MAPPINGS PER L'APP STREAMLIT
# ============================================================================
print("\nSalvataggio mappings")
mappings_json_path = os.path.join(output_dir, 'feature_mappings.json')
mappings_json = {}
for feature_name, mapping_dict in mappings.items():
    mappings_json[feature_name] = {str(k): v for k, v in mapping_dict.items()}
with open(mappings_json_path, 'w', encoding='utf-8') as f:
    json.dump(mappings_json, f, ensure_ascii=False, indent=2)
print(f"Mappings salvati: {mappings_json_path}")
print(f"  - {len(mappings)} features mappate")
reverse_mappings = {}
for feature_name, mapping_dict in mappings.items():
    reverse_mappings[feature_name] = {v: k for k, v in mapping_dict.items()}

reverse_mappings_path = os.path.join(output_dir, 'feature_mappings_reverse.json')
with open(reverse_mappings_path, 'w', encoding='utf-8') as f:
    json.dump(reverse_mappings, f, ensure_ascii=False, indent=2)
print(f"Mappings inversi salvati: {reverse_mappings_path}")
# ============================================================================
# APPLICAZIONE MAPPINGS AI DATI
# ============================================================================
df_mapped = df.copy()
mapped_count = 0
for col, mapping_dict in mappings.items():
    if col in df_mapped.columns:
        df_mapped[col] = df_mapped[col].map(mapping_dict)
        print(f"Mappata: {col}")
        mapped_count += 1
    else:
        print(f"Non trovata: {col}")
print(f"\n{mapped_count} colonne mappate con successo!")
if PREADMISSION_MODE:
    original_csv = os.path.join(output_dir, 'student_data_preadmission.csv')
    df.to_csv(original_csv, index=False)
    print(f"\nDataset pre-immatricolazione salvato:")
    print(f"  - {original_csv}")
    print(f"  - {len(df.columns)} variabili (24 features + target)")
else:
    original_csv = os.path.join(output_dir, 'student_data_original.csv')
    mapped_csv = os.path.join(output_dir, 'student_data_mapped.csv')
    df.to_csv(original_csv, index=False)
    df_mapped.to_csv(mapped_csv, index=False)
    print(f"\nDataset salvati:")
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
print("CREAZIONE VISUALIZZAZIONI SEPARATE")
print("=" * 80)

# Crea sottocartella per visualizzazioni
viz_dir = os.path.join(output_dir, 'visualizations')
os.makedirs(viz_dir, exist_ok=True)
print(f"Directory visualizzazioni: {viz_dir}\n")

n_viz = 0

if PREADMISSION_MODE:
    # ========================================================================
    # VISUALIZZAZIONI PRE-IMMATRICOLAZIONE - FILE SEPARATI
    # ========================================================================
    # Mappa i colori alle label effettive (NON usa lista fissa)
    colors = map_colors_to_labels(target_counts.index)

    # 1. Distribuzione Target
    fig, ax = plt.subplots(figsize=(10, 6))
    target_counts.plot(kind='bar', ax=ax, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_title('Distribuzione Status Studenti\n(Pre-Immatricolazione)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Status', fontsize=14)
    ax.set_ylabel('Numero Studenti', fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(target_counts.values):
        pct = target_pct.values[i]
        ax.text(i, v + 50, f'{v}\n({pct:.1f}%)', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, '01_target_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    n_viz += 1
    print(f"  {n_viz}. 01_target_distribution.png")

    # 2. Età all'iscrizione
    fig, ax = plt.subplots(figsize=(10, 6))
    for status in df[target_col].unique():
        data = df[df[target_col] == status]['Age at enrollment']
        ax.hist(data, alpha=0.6, label=status, bins=20, color=CLASS_COLORS[status], edgecolor='black')
    ax.set_title('Distribuzione Età all\'Iscrizione', fontsize=16, fontweight='bold')
    ax.set_xlabel('Età', fontsize=14)
    ax.set_ylabel('Frequenza', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, '02_age_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    n_viz += 1
    print(f"  {n_viz}. 02_age_distribution.png")

    # 3. Voto Qualificazione Precedente
    fig, ax = plt.subplots(figsize=(10, 6))
    for status in df[target_col].unique():
        data = df[df[target_col] == status]['Previous qualification (grade)']
        ax.hist(data, alpha=0.6, label=status, bins=20, color=CLASS_COLORS[status], edgecolor='black')
    ax.set_title('Voto Qualificazione Precedente', fontsize=16, fontweight='bold')
    ax.set_xlabel('Voto (0-200)', fontsize=14)
    ax.set_ylabel('Frequenza', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, '03_previous_qualification_grade.png'), dpi=300, bbox_inches='tight')
    plt.close()
    n_viz += 1
    print(f"  {n_viz}. 03_previous_qualification_grade.png")

    # 4. Voto Ammissione
    fig, ax = plt.subplots(figsize=(10, 6))
    for status in df[target_col].unique():
        data = df[df[target_col] == status]['Admission grade']
        ax.hist(data, alpha=0.6, label=status, bins=20, color=CLASS_COLORS[status], edgecolor='black')
    ax.set_title('Voto di Ammissione', fontsize=16, fontweight='bold')
    ax.set_xlabel('Voto (0-200)', fontsize=14)
    ax.set_ylabel('Frequenza', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, '04_admission_grade.png'), dpi=300, bbox_inches='tight')
    plt.close()
    n_viz += 1
    print(f"  {n_viz}. 04_admission_grade.png")

    # 5. Status per Genere
    fig, ax = plt.subplots(figsize=(10, 6))
    gender_target = pd.crosstab(df['Gender'], df[target_col], normalize='index') * 100
    gender_colors_pre = map_colors_to_labels(gender_target.columns)
    gender_target.plot(kind='bar', ax=ax, color=gender_colors_pre, edgecolor='black', linewidth=1.5)
    ax.set_title('Status per Genere (%)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Genere (1=Male, 0=Female)', fontsize=14)
    ax.set_ylabel('Percentuale', fontsize=14)
    ax.legend(title='Status', fontsize=11)
    ax.tick_params(axis='x', rotation=0)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, '05_gender_status.png'), dpi=300, bbox_inches='tight')
    plt.close()
    n_viz += 1
    print(f"  {n_viz}. 05_gender_status.png")

    # 6. Borse di Studio
    fig, ax = plt.subplots(figsize=(10, 6))
    schol_target = pd.crosstab(df['Scholarship holder'], df[target_col],
                               normalize='index') * 100
    schol_colors_pre = map_colors_to_labels(schol_target.columns)
    schol_target.plot(kind='bar', ax=ax, color=schol_colors_pre, edgecolor='black', linewidth=1.5)
    ax.set_title('Status per Borsa Studio', fontsize=16, fontweight='bold')
    ax.set_xlabel('Borsa (1=Sì, 0=No)', fontsize=14)
    ax.set_ylabel('Percentuale', fontsize=14)
    ax.legend(title='Status', fontsize=11)
    ax.tick_params(axis='x', rotation=0)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, '06_scholarship_status.png'), dpi=300, bbox_inches='tight')
    plt.close()
    n_viz += 1
    print(f"  {n_viz}. 06_scholarship_status.png")
else:
    # ========================================================================
    # VISUALIZZAZIONI COMPLETE - FILE SEPARATI
    # ========================================================================
    # Mappa i colori alle label effettive
    colors = map_colors_to_labels(target_counts.index)

    # 1. Distribuzione Target
    fig, ax = plt.subplots(figsize=(10, 6))
    target_counts.plot(kind='bar', ax=ax, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_title('Distribuzione Status Studenti', fontsize=16, fontweight='bold')
    ax.set_xlabel('Status', fontsize=14)
    ax.set_ylabel('Numero Studenti', fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(target_counts.values):
        pct = target_pct.values[i]
        ax.text(i, v + 50, f'{v}\n({pct:.1f}%)', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, '01_target_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    n_viz += 1
    print(f"  {n_viz}. 01_target_distribution.png")

    # 2. Status per Genere
    fig, ax = plt.subplots(figsize=(10, 6))
    gender_target = pd.crosstab(df['Gender'], df[target_col], normalize='index') * 100
    gender_colors = map_colors_to_labels(gender_target.columns)
    gender_target.plot(kind='bar', ax=ax, color=gender_colors, edgecolor='black', linewidth=1.5)
    ax.set_title('Status per Genere (%)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Genere (1=Male, 0=Female)', fontsize=14)
    ax.set_ylabel('Percentuale', fontsize=14)
    ax.legend(title='Status', fontsize=11)
    ax.tick_params(axis='x', rotation=0)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, '02_gender_status.png'), dpi=300, bbox_inches='tight')
    plt.close()
    n_viz += 1
    print(f"  {n_viz}. 02_gender_status.png")

    # 3. Età all'iscrizione
    fig, ax = plt.subplots(figsize=(10, 6))
    for status in df[target_col].unique():
        data = df[df[target_col] == status]['Age at enrollment']
        ax.hist(data, alpha=0.6, label=status, bins=20, color=CLASS_COLORS[status], edgecolor='black')
    ax.set_title('Distribuzione Età all\'Iscrizione', fontsize=16, fontweight='bold')
    ax.set_xlabel('Età', fontsize=14)
    ax.set_ylabel('Frequenza', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, '03_age_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    n_viz += 1
    print(f"  {n_viz}. 03_age_distribution.png")

    # 4. Voti 1° Semestre
    fig, ax = plt.subplots(figsize=(10, 6))
    for status in df[target_col].unique():
        data = df[df[target_col] == status]['Curricular units 1st sem (grade)']
        ax.hist(data, alpha=0.6, label=status, bins=20, color=CLASS_COLORS[status], edgecolor='black')
    ax.set_title('Voti 1° Semestre per Status', fontsize=16, fontweight='bold')
    ax.set_xlabel('Voto Medio', fontsize=14)
    ax.set_ylabel('Frequenza', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, '04_grades_1st_semester.png'), dpi=300, bbox_inches='tight')
    plt.close()
    n_viz += 1
    print(f"  {n_viz}. 04_grades_1st_semester.png")

    # 5. Voti 2° Semestre
    fig, ax = plt.subplots(figsize=(10, 6))
    for status in df[target_col].unique():
        data = df[df[target_col] == status]['Curricular units 2nd sem (grade)']
        ax.hist(data, alpha=0.6, label=status, bins=20, color=CLASS_COLORS[status], edgecolor='black')
    ax.set_title('Voti 2° Semestre per Status', fontsize=16, fontweight='bold')
    ax.set_xlabel('Voto Medio', fontsize=14)
    ax.set_ylabel('Frequenza', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, '05_grades_2nd_semester.png'), dpi=300, bbox_inches='tight')
    plt.close()
    n_viz += 1
    print(f"  {n_viz}. 05_grades_2nd_semester.png")

    # 6. Crediti Approvati 1° Semestre
    fig, ax = plt.subplots(figsize=(10, 6))
    df.boxplot(column='Curricular units 1st sem (approved)', by=target_col, ax=ax)
    ax.set_title('Crediti Approvati 1° Semestre', fontsize=16, fontweight='bold')
    ax.set_xlabel('Status', fontsize=14)
    ax.set_ylabel('Numero Crediti', fontsize=14)
    plt.suptitle('')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, '06_approved_1st_semester.png'), dpi=300, bbox_inches='tight')
    plt.close()
    n_viz += 1
    print(f"  {n_viz}. 06_approved_1st_semester.png")

    # 7. Tasso di Approvazione
    fig, ax = plt.subplots(figsize=(10, 6))
    df_temp = df.copy()
    df_temp['approval_rate'] = (df_temp['Curricular units 1st sem (approved)'] /
                                df_temp['Curricular units 1st sem (enrolled)'] * 100)
    for status in df[target_col].unique():
        data = df_temp[df_temp[target_col] == status]['approval_rate']
        ax.hist(data.dropna(), alpha=0.6, label=status, bins=20, color=CLASS_COLORS[status], edgecolor='black')
    ax.set_title('Tasso Approvazione 1° Sem', fontsize=16, fontweight='bold')
    ax.set_xlabel('Tasso (%)', fontsize=14)
    ax.set_ylabel('Frequenza', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, '07_approval_rate.png'), dpi=300, bbox_inches='tight')
    plt.close()
    n_viz += 1
    print(f"  {n_viz}. 07_approval_rate.png")

    # 8. Stato Civile
    fig, ax = plt.subplots(figsize=(12, 6))
    marital_target = pd.crosstab(df_mapped['Marital status'], df_mapped[target_col])
    marital_colors = map_colors_to_labels(marital_target.columns)
    marital_target.plot(kind='bar', ax=ax, color=marital_colors, edgecolor='black', linewidth=1.5)
    ax.set_title('Status per Stato Civile', fontsize=16, fontweight='bold')
    ax.set_xlabel('Stato Civile', fontsize=14)
    ax.set_ylabel('Numero Studenti', fontsize=14)
    ax.legend(title='Status', fontsize=11)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, '08_marital_status.png'), dpi=300, bbox_inches='tight')
    plt.close()
    n_viz += 1
    print(f"  {n_viz}. 08_marital_status.png")

    # 9. Top 10 Corsi
    fig, ax = plt.subplots(figsize=(10, 8))
    top_courses = df_mapped['Course'].value_counts().head(10)
    top_courses.plot(kind='barh', ax=ax, color=CATEGORICAL_PALETTE[0], edgecolor='black', linewidth=1.5)
    ax.set_title('Top 10 Corsi più Frequentati', fontsize=16, fontweight='bold')
    ax.set_xlabel('Numero Studenti', fontsize=14)
    ax.set_ylabel('')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, '09_top_courses.png'), dpi=300, bbox_inches='tight')
    plt.close()
    n_viz += 1
    print(f"  {n_viz}. 09_top_courses.png")

    # 10. Debiti
    fig, ax = plt.subplots(figsize=(10, 6))
    df.boxplot(column='Debtor', by=target_col, ax=ax)
    ax.set_title('Debitori per Status', fontsize=16, fontweight='bold')
    ax.set_xlabel('Status', fontsize=14)
    ax.set_ylabel('Debiti (1=Sì, 0=No)', fontsize=14)
    plt.suptitle('')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, '10_debtor_status.png'), dpi=300, bbox_inches='tight')
    plt.close()
    n_viz += 1
    print(f"  {n_viz}. 10_debtor_status.png")

    # 11. Qualificazione Precedente
    fig, ax = plt.subplots(figsize=(12, 6))
    prev_qual_target = pd.crosstab(df_mapped['Previous qualification'],
                                   df_mapped[target_col], normalize='index') * 100
    top_quals = df_mapped['Previous qualification'].value_counts().head(8)
    prev_qual_colors = map_colors_to_labels(prev_qual_target.columns)
    prev_qual_target.loc[top_quals.index].plot(kind='bar', ax=ax, color=prev_qual_colors, edgecolor='black',
                                               linewidth=1.5)
    ax.set_title('Status per Qualificazione Precedente (Top 8)', fontsize=16, fontweight='bold')
    ax.set_xlabel('', fontsize=10)
    ax.set_ylabel('Percentuale', fontsize=14)
    ax.legend(title='Status', fontsize=11)
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, '11_previous_qualification.png'), dpi=300, bbox_inches='tight')
    plt.close()
    n_viz += 1
    print(f"  {n_viz}. 11_previous_qualification.png")

    # 12. Voto Qualificazione Precedente
    fig, ax = plt.subplots(figsize=(10, 6))
    for status in df[target_col].unique():
        data = df[df[target_col] == status]['Previous qualification (grade)']
        ax.hist(data, alpha=0.6, label=status, bins=20, color=CLASS_COLORS[status], edgecolor='black')
    ax.set_title('Voto Qualificazione Precedente', fontsize=16, fontweight='bold')
    ax.set_xlabel('Voto (0-200)', fontsize=14)
    ax.set_ylabel('Frequenza', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, '12_previous_qualification_grade.png'), dpi=300, bbox_inches='tight')
    plt.close()
    n_viz += 1
    print(f"  {n_viz}. 12_previous_qualification_grade.png")

    # 13. Correlazione Voti
    fig, ax = plt.subplots(figsize=(10, 8))
    for status in df[target_col].unique():
        mask = df[target_col] == status
        ax.scatter(df.loc[mask, 'Curricular units 1st sem (grade)'],
                   df.loc[mask, 'Curricular units 2nd sem (grade)'],
                   alpha=0.5, label=status, color=CLASS_COLORS[status], s=30)
    ax.plot([0, 20], [0, 20], 'k--', alpha=0.3, linewidth=2, label='y=x')
    ax.set_title('Correlazione: Voti 1° vs 2° Semestre', fontsize=16, fontweight='bold')
    ax.set_xlabel('Voto 1° Sem', fontsize=14)
    ax.set_ylabel('Voto 2° Sem', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, '13_grades_correlation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    n_viz += 1
    print(f"  {n_viz}. 13_grades_correlation.png")

    # 14. Top Nazionalità
    fig, ax = plt.subplots(figsize=(10, 8))
    top_nat = df_mapped['Nacionality'].value_counts().head(10)
    top_nat.plot(kind='barh', ax=ax, color=CATEGORICAL_PALETTE[0], edgecolor='black', linewidth=1.5)
    ax.set_title('Top 10 Nazionalità', fontsize=16, fontweight='bold')
    ax.set_xlabel('Numero Studenti', fontsize=14)
    ax.set_ylabel('')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, '14_top_nationalities.png'), dpi=300, bbox_inches='tight')
    plt.close()
    n_viz += 1
    print(f"  {n_viz}. 14_top_nationalities.png")

    # 15. Borse di Studio
    fig, ax = plt.subplots(figsize=(10, 6))
    schol_target = pd.crosstab(df['Scholarship holder'], df[target_col], normalize='index') * 100
    schol_colors = map_colors_to_labels(schol_target.columns)
    schol_target.plot(kind='bar', ax=ax, color=schol_colors, edgecolor='black', linewidth=1.5)
    ax.set_title('Status per Borsa Studio', fontsize=16, fontweight='bold')
    ax.set_xlabel('Borsa (1=Sì, 0=No)', fontsize=14)
    ax.set_ylabel('Percentuale', fontsize=14)
    ax.legend(title='Status', fontsize=11)
    ax.tick_params(axis='x', rotation=0)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, '15_scholarship_status.png'), dpi=300, bbox_inches='tight')
    plt.close()
    n_viz += 1
    print(f"  {n_viz}. 15_scholarship_status.png")

print(f"\n✅ {n_viz} visualizzazioni generate")
print(f"📁 Directory: {viz_dir}")

print("\n" + "=" * 80)
print("STATISTICHE DESCRITTIVE PER STATUS")
print("=" * 80)

if PREADMISSION_MODE:
    numeric_cols = ['Age at enrollment', 'Previous qualification (grade)',
                    'Admission grade', 'Unemployment rate', 'Inflation rate', 'GDP']
else:
    numeric_cols = ['Age at enrollment', 'Previous qualification (grade)',
                    'Curricular units 1st sem (approved)', 'Curricular units 1st sem (grade)',
                    'Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (grade)',
                    'Admission grade']

for status in df[target_col].unique():
    print(f"\n{str(status).upper()}")
    print("-" * 80)
    status_data = df[df[target_col] == status][numeric_cols]
    print(status_data.describe().round(2))
print("\n" + "=" * 80)
print("MATRICE DI CORRELAZIONE")
print("=" * 80)
all_numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if target_col in all_numeric_cols:
    all_numeric_cols.remove(target_col)
print(f"\nCalcolo correlazioni per {len(all_numeric_cols)} variabili numeriche...")
fig1, ax1 = plt.subplots(figsize=(20, 18))
corr_matrix_full = df[all_numeric_cols].corr()
sns.heatmap(corr_matrix_full, annot=True, fmt='.2f', cmap=HEATMAP_CMAPS['correlation'],
            center=0, square=True, ax=ax1, cbar_kws={"shrink": 0.8},
            annot_kws={"size": 7})
ax1.set_title('Matrice di Correlazione COMPLETA - Tutte le Variabili',
              fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
corr_path_full = os.path.join(viz_dir, '16_correlation_matrix_complete.png')
plt.savefig(corr_path_full, dpi=300, bbox_inches='tight')
n_viz += 1
print(f"  {n_viz}. 16_correlation_matrix_complete.png")
if not PREADMISSION_MODE:
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
    sns.heatmap(corr_matrix_key, annot=True, fmt='.2f', cmap=HEATMAP_CMAPS['correlation'],
                center=0, square=True, ax=ax2, cbar_kws={"shrink": 0.8},
                annot_kws={"size": 9})
    ax2.set_title('Matrice di Correlazione - Variabili Performance Accademica',
                  fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    corr_path_key = os.path.join(viz_dir, '17_correlation_matrix_academic.png')
    plt.savefig(corr_path_key, dpi=300, bbox_inches='tight')
    n_viz += 1
    print(f"  {n_viz}. 17_correlation_matrix_academic.png")
print("\n" + "=" * 80)
print("CORRELAZIONI PIÙ FORTI (|r| > 0.5)")
print("=" * 80)
print("\nCoppie di variabili con correlazione forte:")
print("-" * 80)
strong_corr = []
for i in range(len(corr_matrix_full.columns)):
    for j in range(i + 1, len(corr_matrix_full.columns)):
        corr_val = corr_matrix_full.iloc[i, j]
        if abs(corr_val) > 0.5:
            var1 = corr_matrix_full.columns[i]
            var2 = corr_matrix_full.columns[j]
            strong_corr.append((var1, var2, corr_val))
strong_corr.sort(key=lambda x: abs(x[2]), reverse=True)

for var1, var2, corr_val in strong_corr[:20]:  # Top 20
    direction = "+" if corr_val > 0 else "-"
    bar = "█" * int(abs(corr_val) * 20)
    print(f"  {var1:45s} <-> {var2:45s} : {direction}{abs(corr_val):.3f} {bar}")
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

print("\n3. MEDIE CHIAVE PER STATUS:")
if PREADMISSION_MODE:
    for status in df[target_col].unique():
        mean_age = df[df[target_col] == status]['Age at enrollment'].mean()
        mean_prev = df[df[target_col] == status]['Previous qualification (grade)'].mean()
        mean_adm = df[df[target_col] == status]['Admission grade'].mean()
        print(f"   • {status:10s}: Età={mean_age:5.2f}, Voto prev={mean_prev:6.2f}, Voto amm={mean_adm:6.2f}")
else:
    for status in df[target_col].unique():
        mean_1sem = df[df[target_col] == status]['Curricular units 1st sem (grade)'].mean()
        mean_2sem = df[df[target_col] == status]['Curricular units 2nd sem (grade)'].mean()
        print(f"   • {status:10s}: 1° sem = {mean_1sem:5.2f}, 2° sem = {mean_2sem:5.2f}")
print("\n" + "=" * 80)
print("ANALISI COMPLETATA")
print("=" * 80)
print(f"\nTutti i file sono stati salvati in: {output_dir}\n")
print("File generati:")
if PREADMISSION_MODE:
    print(f"  1. {os.path.basename(original_csv)} - Dataset pre-immatricolazione (24 features)")
    print(f"  2. visualizations/ - Cartella con 6 visualizzazioni separate")
    print(f"  3. {os.path.basename(corr_path_full)} - Matrice correlazione")
else:
    # Modalità completa
    print(f"  1. {os.path.basename(original_csv)} - Dataset con valori numerici")
    print(f"  2. {os.path.basename(mapped_csv)} - Dataset con valori leggibili")
    print(f"  3. visualizations/ - Cartella con 17 visualizzazioni separate")
    print(f"  4. {os.path.basename(corr_path_full)} - Matrice correlazione COMPLETA")
    print(f"  5. {os.path.basename(corr_path_key)} - Matrice correlazione ACCADEMICA")
print(f"\n📁 Visualizzazioni: {viz_dir}")
print(f"📊 Totale grafici: {n_viz}")
print("=" * 80)
plt.show()