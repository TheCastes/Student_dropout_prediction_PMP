import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import json
import warnings

warnings.filterwarnings('ignore')

from color_config import (
    CLASS_COLORS_LIST,
    HEATMAP_CMAPS,
    setup_plot_style,
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

setup_plot_style()
sns.set_palette(CLASS_COLORS_LIST)

print("=" * 80)
if PREADMISSION_MODE:
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
    'data.csv',
    os.path.join(os.path.dirname(__file__), 'data.csv'),
    os.path.join(os.getcwd(), 'data.csv'),
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
# SALVA MAPPINGS
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
print("CREAZIONE VISUALIZZAZIONI")
print("=" * 80)

viz_dir = os.path.join(output_dir, 'visualizations')
os.makedirs(viz_dir, exist_ok=True)
print(f"Directory visualizzazioni: {viz_dir}\n")
n_viz = 0

if PREADMISSION_MODE:
    # 01. Status per Genere
    fig, ax = plt.subplots(figsize=(12, 7))
    gender_target = pd.crosstab(df['Gender'], df[target_col], normalize='index') * 100
    gender_colors_pre = map_colors_to_labels(gender_target.columns)
    x = np.arange(len(gender_target.index))
    width = 0.25
    for i, col in enumerate(gender_target.columns):
        offset = (i - len(gender_target.columns) / 2 + 0.5) * width
        bars = ax.bar(x + offset, gender_target[col], width,
                      label=col, color=gender_colors_pre[i],
                      edgecolor='black', linewidth=1.5, alpha=0.85)
        for j, (bar, val) in enumerate(zip(bars, gender_target[col])):
            if val > 3:  # Mostra solo se > 3%
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                        f'{val:.1f}%',
                        ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax.set_title('Status per Genere\n(percentuale all\'interno di ogni gruppo)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Genere', fontsize=14, fontweight='bold')
    ax.set_ylabel('Percentuale', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Femmine (0)', 'Maschi (1)'], fontsize=12)
    ax.legend(title='Status', fontsize=11, title_fontsize=12, framealpha=0.95)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(gender_target.max()) * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, '01_gender_status.png'), dpi=300, bbox_inches='tight')
    plt.close()
    n_viz += 1
    print(f"  {n_viz}. 01_gender_status.png")
    # 02. Borse di Studio
    fig, ax = plt.subplots(figsize=(12, 7))
    schol_target = pd.crosstab(df['Scholarship holder'], df[target_col],
                               normalize='index') * 100
    schol_colors_pre = map_colors_to_labels(schol_target.columns)
    x = np.arange(len(schol_target.index))
    width = 0.25
    for i, col in enumerate(schol_target.columns):
        offset = (i - len(schol_target.columns) / 2 + 0.5) * width
        bars = ax.bar(x + offset, schol_target[col], width,
                      label=col, color=schol_colors_pre[i],
                      edgecolor='black', linewidth=1.5, alpha=0.85)
        for j, (bar, val) in enumerate(zip(bars, schol_target[col])):
            if val > 3:
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                        f'{val:.1f}%',
                        ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax.set_title('Status per Borsa di Studio\n(percentuale all\'interno di ogni gruppo)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Borsa di Studio', fontsize=14, fontweight='bold')
    ax.set_ylabel('Percentuale', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Senza Borsa (0)', 'Con Borsa (1)'], fontsize=12)
    ax.legend(title='Status', fontsize=11, title_fontsize=12, framealpha=0.95)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(schol_target.max()) * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, '02_scholarship_status.png'), dpi=300, bbox_inches='tight')
    plt.close()
    n_viz += 1
    print(f"  {n_viz}. 02_scholarship_status.png")
else:
    # 01. Status per Genere
    fig, ax = plt.subplots(figsize=(12, 7))
    gender_target = pd.crosstab(df['Gender'], df[target_col], normalize='index') * 100
    gender_colors = map_colors_to_labels(gender_target.columns)
    x = np.arange(len(gender_target.index))
    width = 0.25
    for i, col in enumerate(gender_target.columns):
        offset = (i - len(gender_target.columns) / 2 + 0.5) * width
        bars = ax.bar(x + offset, gender_target[col], width,
                      label=col, color=gender_colors[i],
                      edgecolor='black', linewidth=1.5, alpha=0.85)

        for j, (bar, val) in enumerate(zip(bars, gender_target[col])):
            if val > 3:
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                        f'{val:.1f}%',
                        ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax.set_title('Status per Genere\n(percentuale all\'interno di ogni gruppo)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Genere', fontsize=14, fontweight='bold')
    ax.set_ylabel('Percentuale', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Femmine (0)', 'Maschi (1)'], fontsize=12)
    ax.legend(title='Status', fontsize=11, title_fontsize=12, framealpha=0.95)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(gender_target.max()) * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, '01_gender_status.png'), dpi=300, bbox_inches='tight')
    plt.close()
    n_viz += 1
    print(f"  {n_viz}. 01_gender_status.png")
    # 02. Borse di Studio
    fig, ax = plt.subplots(figsize=(12, 7))
    schol_target = pd.crosstab(df['Scholarship holder'], df[target_col], normalize='index') * 100
    schol_colors = map_colors_to_labels(schol_target.columns)
    x = np.arange(len(schol_target.index))
    width = 0.25
    for i, col in enumerate(schol_target.columns):
        offset = (i - len(schol_target.columns) / 2 + 0.5) * width
        bars = ax.bar(x + offset, schol_target[col], width,
                      label=col, color=schol_colors[i],
                      edgecolor='black', linewidth=1.5, alpha=0.85)
        for j, (bar, val) in enumerate(zip(bars, schol_target[col])):
            if val > 3:
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                        f'{val:.1f}%',
                        ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax.set_title('Status per Borsa di Studio\n(percentuale all\'interno di ogni gruppo)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Borsa di Studio', fontsize=14, fontweight='bold')
    ax.set_ylabel('Percentuale', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Senza Borsa (0)', 'Con Borsa (1)'], fontsize=12)
    ax.legend(title='Status', fontsize=11, title_fontsize=12, framealpha=0.95)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(schol_target.max()) * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, '02_scholarship_status.png'), dpi=300, bbox_inches='tight')
    plt.close()
    n_viz += 1
    print(f"  {n_viz}. 02_scholarship_status.png")
   # 03. Tuition fees
    fig, ax = plt.subplots(figsize=(12, 7))
    tuition_target = pd.crosstab(df['Tuition fees up to date'],
                                 df[target_col],
                                 normalize='index') * 100
    tuition_colors = map_colors_to_labels(tuition_target.columns)
    x = np.arange(len(tuition_target.index))
    width = 0.25
    for i, col in enumerate(tuition_target.columns):
        offset = (i - len(tuition_target.columns) / 2 + 0.5) * width
        bars = ax.bar(x + offset, tuition_target[col], width,
                      label=col, color=tuition_colors[i],
                      edgecolor='black', linewidth=1.5, alpha=0.85)
        for j, (bar, val) in enumerate(zip(bars, tuition_target[col])):
            if val > 2:
                ax.text(bar.get_x() + bar.get_width() / 2.,
                        bar.get_height(), f'{val:.1f}%',
                        ha='center', va='bottom',
                        fontweight='bold', fontsize=10)
    ax.set_title('Distribuzione Status per Tasse Universitarie\n(percentuale dei 3 status all\'interno di ogni gruppo)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Stato Tasse', fontsize=14, fontweight='bold')
    ax.set_ylabel('Percentuale', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Non in Regola (0)', 'In Regola (1)'], fontsize=12)
    ax.legend(title='Status', fontsize=11, title_fontsize=12, framealpha=0.95)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(tuition_target.max()) * 1.15)

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, '03_tuition_fees_status.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    n_viz += 1
    print(f"  {n_viz}. 03_tuition_fees_status.png")

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
    print(f"  2. visualizations/ - Cartella con le visualizzazioni")
else:
    print(f"  1. {os.path.basename(original_csv)} - Dataset con valori numerici")
    print(f"  2. {os.path.basename(mapped_csv)} - Dataset con valori mappati")
    print(f"  3. visualizations/ - Cartella con le visualizzazioni")
print(f"\nVisualizzazioni: {viz_dir}")
print("=" * 80)