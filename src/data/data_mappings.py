"""
Data Mappings Module.
Fornisce mappature semantiche per rendere i dati del dataset comprensibili.
"""

# Mappatura Stato Civile (Marital Status)
MARITAL_STATUS = {
    1: "Single",
    2: "Sposato/a",
    3: "Vedovo/a",
    4: "Divorziato/a",
    5: "Convivenza di fatto",
    6: "Separato/a legalmente"
}

# Mappatura Genere (Gender)
GENDER = {
    0: "Femmina",
    1: "Maschio"
}

# Mappatura Sì/No generica
YES_NO = {
    0: "No",
    1: "Sì"
}

# Mappatura Frequenza (Daytime/Evening Attendance)
ATTENDANCE = {
    0: "Serale",
    1: "Diurna"
}

# Mappatura Modalità di Iscrizione (Application Mode)
# Basata sul sistema portoghese di ammissione
APPLICATION_MODE = {
    1: "Esame nazionale di ammissione - 1a fase",
    2: "Cambiamento di corso",
    3: "Cambio di istituzione/corso",
    4: "Titolare di altro corso superiore",
    5: "Ordinanza n. 612/93",
    6: "Esame nazionale di ammissione - 2a fase",
    7: "Esame nazionale di ammissione - 3a fase",
    8: "Ordinanza n. 854-B/99",
    9: "Studente internazionale (Statuto dello studente internazionale)",
    10: "Ordinanza n. 533-A/99 - Titolare di diploma di specializzazione tecnologica",
    15: "Studente maggiore di 23 anni",
    16: "Cambio di corso (studente internazionale)",
    17: "Corsi brevi di laurea",
    18: "Cambiamento di istituzione/corso (studente internazionale)",
    26: "Ordinanza n. 727/99",
    27: "Ordinanza n. 612/93",
    39: "Studente maggiore di 23 anni",
    42: "Trasferimento",
    43: "Cambio di istituzione/corso",
    44: "Titolare di laurea tecnologica",
    51: "Cambio di corso",
    53: "Corsi brevi di laurea",
    57: "Esame nazionale di ammissione - 1a fase"
}

# Mappatura Qualifica Precedente (Previous Qualification)
PREVIOUS_QUALIFICATION = {
    1: "Istruzione secondaria",
    2: "Istruzione superiore - Laurea",
    3: "Istruzione superiore - Laurea magistrale",
    4: "Istruzione superiore - Dottorato",
    5: "Frequenza istruzione superiore",
    6: "12° anno di scolarità - Non completato",
    9: "11° anno di scolarità - Non completato",
    10: "Altro - 11° anno di scolarità",
    12: "10° anno di scolarità",
    14: "10° anno di scolarità - Non completato",
    15: "Corso di formazione di base - 1° ciclo",
    19: "Corso di specializzazione tecnologica",
    38: "Corso tecnico superiore - Laurea professionale",
    39: "Corso tecnico superiore",
    40: "12° anno di scolarità - Non completato",
    42: "Corso di formazione di base - 2° ciclo",
    43: "12° anno di scolarità"
}

# Mappatura Qualifica dei Genitori (Parent's Qualification)
PARENT_QUALIFICATION = {
    1: "Istruzione secondaria - 12° anno o equivalente",
    2: "Istruzione superiore - Laurea (1° ciclo)",
    3: "Istruzione superiore - Laurea magistrale (2° ciclo)",
    4: "Istruzione superiore - Dottorato (3° ciclo)",
    5: "Frequenza istruzione superiore",
    6: "12° anno di scolarità - Non completato",
    9: "11° anno di scolarità - Non completato",
    10: "7° anno (Vecchio)",
    11: "Altra formazione - 11° anno di scolarità",
    12: "10° anno di scolarità",
    13: "Corso generale di commercio",
    14: "10° anno di scolarità - Non completato",
    18: "Corso generale amministrativo",
    19: "Corso complementare di istruzione secondaria",
    22: "Corso tecnico-professionale - 11° anno o equivalente",
    26: "7° anno di scolarità",
    27: "2° ciclo di istruzione di base",
    29: "9° anno di scolarità - Non completato",
    30: "8° anno di scolarità",
    34: "Sconosciuto",
    35: "Non può leggere o scrivere",
    36: "Sa leggere senza aver completato il 1° ciclo",
    37: "Istruzione di base 1° ciclo (4° anno/vecchio)",
    38: "Istruzione di base 2° ciclo (6° anno/vecchio)",
    39: "Corso tecnologico specialistico",
    40: "Istruzione superiore - Laurea (1° ciclo)",
    41: "Corso tecnologico specialistico",
    42: "Istruzione professionale",
    43: "Istruzione complementare di istruzione secondaria",
    44: "Istruzione superiore - Master (2° ciclo)"
}

# Mappatura Occupazione dei Genitori (Parent's Occupation)
# Basata sulla classificazione portoghese delle professioni
PARENT_OCCUPATION = {
    0: "Studente",
    1: "Rappresentanti del potere legislativo e organi esecutivi",
    2: "Specialisti in attività intellettuali e scientifiche",
    3: "Tecnici e professioni di livello intermedio",
    4: "Personale amministrativo",
    5: "Personale dei servizi personali, di protezione e sicurezza",
    6: "Agricoltori e lavoratori qualificati in agricoltura",
    7: "Lavoratori qualificati nell'industria",
    8: "Operatori di impianti e macchinari",
    9: "Lavoratori non qualificati",
    10: "Professioni delle forze armate",
    90: "Altra situazione (pensionato, disoccupato)",
    99: "Non specificato",
    122: "Specialisti delle scienze della salute",
    123: "Insegnanti",
    125: "Specialisti in tecnologie dell'informazione e comunicazione",
    131: "Tecnici e professioni associate delle scienze fisiche",
    132: "Tecnici delle scienze della vita e della salute",
    134: "Tecnici di livello intermedio di servizi legali",
    135: "Tecnici dell'informazione e comunicazione",
    141: "Impiegati d'ufficio, segretari in generale",
    143: "Operatori di elaborazione dati",
    144: "Impiegati materiale-recording",
    151: "Lavoratori dei servizi personali",
    152: "Venditori",
    153: "Lavoratori della cura personale",
    171: "Lavoratori qualificati nell'edilizia",
    173: "Lavoratori qualificati della stampa",
    175: "Lavoratori della trasformazione alimentare",
    191: "Personale di pulizia",
    192: "Lavoratori non qualificati agricoltura, pesca",
    193: "Lavoratori non qualificati nell'industria",
    194: "Assistenti nella preparazione pasti"
}

# Mappatura Nazionalità (Nationality)
NATIONALITY = {
    1: "Portoghese",
    2: "Tedesca",
    6: "Spagnola",
    11: "Italiana",
    13: "Olandese",
    14: "Inglese",
    17: "Lituana",
    21: "Angolana",
    22: "Capoverdiana",
    24: "Guinea-Bissau",
    25: "Mozambicana",
    26: "Santomense",
    32: "Turca",
    41: "Brasiliana",
    62: "Rumena",
    100: "Repubblica Moldava",
    101: "Messicana",
    103: "Ucraina",
    105: "Russa",
    108: "Cubana",
    109: "Colombiana"
}

# Mappatura Corso (Course) - Semplificata per IPP
COURSE = {
    33: "Biofuel Production Technologies",
    171: "Animation and Multimedia Design",
    8014: "Social Service (serale)",
    9003: "Agronomy",
    9070: "Communication Design",
    9085: "Veterinary Nursing",
    9119: "Informatics Engineering",
    9130: "Equiniculture",
    9147: "Management",
    9238: "Social Service",
    9254: "Tourism",
    9500: "Nursing",
    9556: "Oral Hygiene",
    9670: "Advertising and Marketing Management",
    9773: "Journalism and Communication",
    9853: "Basic Education",
    9991: "Management (serale)"
}

def get_readable_value(feature_name: str, value) -> str:
    """
    Converte un valore grezzo in una rappresentazione leggibile.
    
    Args:
        feature_name: Nome della feature
        value: Valore da convertire
        
    Returns:
        Valore in formato leggibile
    """
    feature_lower = feature_name.lower().strip()
    
    # Converti il valore in int se possibile
    try:
        value_int = int(float(value))
    except (ValueError, TypeError):
        return str(value)
    
    # Mappature
    mappings = {
        'marital status': MARITAL_STATUS,
        'gender': GENDER,
        'daytime/evening attendance': ATTENDANCE,
        'application mode': APPLICATION_MODE,
        'previous qualification': PREVIOUS_QUALIFICATION,
        'course': COURSE,
        'nacionality': NATIONALITY,
        'nationality': NATIONALITY,
    }
    
    # Mappature Yes/No
    yes_no_features = [
        'displaced', 'educational special needs', 'debtor',
        'tuition fees up to date', 'scholarship holder',
        'international', 'gender'
    ]
    
    # Mappature per qualifiche/occupazioni dei genitori
    if "mother's qualification" in feature_lower or "father's qualification" in feature_lower:
        return PARENT_QUALIFICATION.get(value_int, f"Codice {value_int}")
    
    if "mother's occupation" in feature_lower or "father's occupation" in feature_lower:
        return PARENT_OCCUPATION.get(value_int, f"Codice {value_int}")
    
    # Cerca nelle mappature standard
    for key, mapping in mappings.items():
        if key in feature_lower:
            return mapping.get(value_int, f"Codice {value_int}")
    
    # Verifica se è una feature Yes/No
    for yn_feature in yes_no_features:
        if yn_feature in feature_lower:
            return YES_NO.get(value_int, str(value))
    
    # Per valori numerici continui
    if 'grade' in feature_lower or 'rate' in feature_lower or 'gdp' in feature_lower:
        return f"{value:.2f}"
    
    if 'age' in feature_lower:
        return f"{value_int} anni"
    
    # Default: ritorna il valore come stringa
    return str(value)


def get_feature_description(feature_name: str) -> str:
    """
    Fornisce una descrizione dettagliata della feature.
    
    Args:
        feature_name: Nome della feature
        
    Returns:
        Descrizione della feature
    """
    descriptions = {
        'marital status': 'Stato civile dello studente al momento dell\'iscrizione',
        'application mode': 'Modalità di ammissione all\'istituto (concorso nazionale, trasferimento, ecc.)',
        'application order': 'Ordine di preferenza del corso (1 = prima scelta, 9 = nona scelta)',
        'course': 'Corso di laurea a cui lo studente è iscritto',
        'daytime/evening attendance': 'Frequenza delle lezioni in orario diurno o serale',
        'previous qualification': 'Titolo di studio precedente posseduto dallo studente',
        'previous qualification (grade)': 'Voto ottenuto nel titolo di studio precedente',
        'nacionality': 'Nazionalità dello studente',
        'mother\'s qualification': 'Livello di istruzione della madre',
        'father\'s qualification': 'Livello di istruzione del padre',
        'mother\'s occupation': 'Professione della madre',
        'father\'s occupation': 'Professione del padre',
        'admission grade': 'Voto di ammissione all\'università',
        'displaced': 'Studente fuori sede (vive lontano da casa)',
        'educational special needs': 'Studente con bisogni educativi speciali',
        'debtor': 'Studente in debito con l\'università',
        'tuition fees up to date': 'Tasse universitarie pagate regolarmente',
        'gender': 'Genere dello studente',
        'scholarship holder': 'Beneficiario di borsa di studio',
        'age at enrollment': 'Età dello studente al momento dell\'iscrizione',
        'international': 'Studente internazionale',
        'unemployment rate': 'Tasso di disoccupazione nazionale al momento dell\'iscrizione',
        'inflation rate': 'Tasso di inflazione nazionale al momento dell\'iscrizione',
        'gdp': 'Variazione del PIL nazionale al momento dell\'iscrizione'
    }
    
    # Features del primo semestre
    if '1st sem' in feature_name.lower():
        base_descriptions = {
            'credited': 'Crediti riconosciuti (da precedenti percorsi)',
            'enrolled': 'Unità curriculari a cui lo studente è iscritto',
            'evaluations': 'Numero di valutazioni sostenute',
            'approved': 'Unità curriculari superate con successo',
            'grade': 'Media dei voti ottenuti',
            'without evaluations': 'Unità curriculari senza valutazione'
        }
        for key, desc in base_descriptions.items():
            if key in feature_name.lower():
                return f"1° Semestre: {desc}"
    
    # Features del secondo semestre
    if '2nd sem' in feature_name.lower():
        base_descriptions = {
            'credited': 'Crediti riconosciuti (da precedenti percorsi)',
            'enrolled': 'Unità curriculari a cui lo studente è iscritto',
            'evaluations': 'Numero di valutazioni sostenute',
            'approved': 'Unità curriculari superate con successo',
            'grade': 'Media dei voti ottenuti',
            'without evaluations': 'Unità curriculari senza valutazione'
        }
        for key, desc in base_descriptions.items():
            if key in feature_name.lower():
                return f"2° Semestre: {desc}"
    
    # Cerca nella mappa delle descrizioni
    for key, desc in descriptions.items():
        if key in feature_name.lower():
            return desc
    
    return f"Feature: {feature_name}"


def format_dataframe_for_display(df, feature_columns=None):
    """
    Formatta un DataFrame sostituendo i valori con versioni leggibili.
    
    Args:
        df: DataFrame da formattare
        feature_columns: Lista di colonne da formattare (None = tutte)
        
    Returns:
        DataFrame formattato
    """
    import pandas as pd
    
    df_display = df.copy()
    
    if feature_columns is None:
        feature_columns = df.columns
    
    for col in feature_columns:
        if col in df.columns:
            df_display[col] = df[col].apply(lambda x: get_readable_value(col, x))
    
    return df_display
