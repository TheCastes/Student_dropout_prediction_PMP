"""
STRUTTURA CREATA:
=================
project/
├── 01_analysis/                       # Output di student_analysis.py
├── 02_training/                       # Output di train_models.py
├── 01_analysis_preadmission/          # Output di student_analysis.py --preadmission
├── 02_training_preadmission/          # Output di train_models.py --preadmission
└── data/                              # Dataset originale
"""

from pathlib import Path

def create_folder_structure():
    #Crea la struttura di cartelle per il progetto
    print("=" * 80)
    print("SETUP STRUTTURA CARTELLE - Student Dropout Prediction")
    print("=" * 80)
    base_dir = Path.cwd()
    print(f"\nDirectory base: {base_dir}\n")
    folders = {
        '01_analysis': 'Output analisi esplorativa dataset - COMPLETO',
        '02_training': 'Output training modelli ML - COMPLETO',
        '01_analysis_preadmission': 'Output analisi - PRE-IMMATRICOLAZIONE',
        '02_training_preadmission': 'Output training - PRE-IMMATRICOLAZIONE',
        'data': 'Dataset  UCI'
    }
    print("Creazione cartelle...")
    print("-" * 80)
    created = []
    existing = []
    for folder, description in folders.items():
        folder_path = base_dir / folder
        if not folder_path.exists():
            folder_path.mkdir(parents=True, exist_ok=True)
            print(f"{folder:25s} - {description}")
            created.append(folder)
        else:
            print(f"{folder:25s} - Già esistente")
            existing.append(folder)
    # Riepilogo
    print("=" * 80)
    print("SETUP COMPLETATO!")
    print("=" * 80)

    if created:
        print(f"\nCartelle create: {len(created)}")
        for folder in created:
            print(f"  - {folder}/")

    if existing:
        print(f"\nCartelle già esistenti: {len(existing)}")
        for folder in existing:
            print(f"  - {folder}/")
if __name__ == "__main__":
    create_folder_structure()