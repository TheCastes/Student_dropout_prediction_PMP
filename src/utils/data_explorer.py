"""
Data Explorer Component.
Componente interattivo per esplorare e comprendere i dati.
"""

import streamlit as st
import pandas as pd
from src.data.data_mappings import (
    get_readable_value,
    get_feature_description,
    format_dataframe_for_display,
    MARITAL_STATUS,
    GENDER,
    APPLICATION_MODE,
    PREVIOUS_QUALIFICATION,
    PARENT_QUALIFICATION,
    PARENT_OCCUPATION,
    NATIONALITY,
    COURSE
)


def show_data_dictionary():
    """
    Mostra un dizionario dati interattivo completo.
    """
    st.header("📖 Dizionario Dati - Guida Completa")
    
    st.markdown("""
    Questa sezione fornisce una guida completa per comprendere tutti i dati 
    utilizzati nel sistema di predizione della performance studentesca.
    """)
    
    # Categorie di feature
    categories = {
        "👤 Informazioni Personali": [
            "Age at enrollment",
            "Marital status",
            "Gender",
            "Displaced",
            "International",
            "Nacionality"
        ],
        "👨‍👩‍👧 Background Familiare": [
            "Mother's qualification",
            "Father's qualification",
            "Mother's occupation",
            "Father's occupation"
        ],
        "🎓 Percorso Accademico": [
            "Previous qualification",
            "Previous qualification (grade)",
            "Admission grade",
            "Application mode",
            "Application order",
            "Course",
            "Daytime/evening attendance"
        ],
        "💰 Situazione Finanziaria": [
            "Scholarship holder",
            "Tuition fees up to date",
            "Debtor",
            "Educational special needs"
        ],
        "📊 Performance 1° Semestre": [
            "Curricular units 1st sem (credited)",
            "Curricular units 1st sem (enrolled)",
            "Curricular units 1st sem (evaluations)",
            "Curricular units 1st sem (approved)",
            "Curricular units 1st sem (grade)",
            "Curricular units 1st sem (without evaluations)"
        ],
        "📈 Performance 2° Semestre": [
            "Curricular units 2nd sem (credited)",
            "Curricular units 2nd sem (enrolled)",
            "Curricular units 2nd sem (evaluations)",
            "Curricular units 2nd sem (approved)",
            "Curricular units 2nd sem (grade)",
            "Curricular units 2nd sem (without evaluations)"
        ],
        "🌍 Indicatori Macroeconomici": [
            "Unemployment rate",
            "Inflation rate",
            "GDP"
        ]
    }
    
    # Crea tab per ogni categoria
    tabs = st.tabs(list(categories.keys()))
    
    for tab, (category_name, features) in zip(tabs, categories.items()):
        with tab:
            for feature in features:
                with st.expander(f"ℹ️ {feature}"):
                    st.markdown(f"**Descrizione:** {get_feature_description(feature)}")
                    
                    # Mostra mapping se disponibile
                    feature_lower = feature.lower()
                    
                    if 'marital status' in feature_lower:
                        st.markdown("**Valori possibili:**")
                        for code, label in MARITAL_STATUS.items():
                            st.write(f"- `{code}`: {label}")
                    
                    elif 'gender' in feature_lower:
                        st.markdown("**Valori possibili:**")
                        for code, label in GENDER.items():
                            st.write(f"- `{code}`: {label}")
                    
                    elif 'application mode' in feature_lower:
                        st.markdown("**Valori possibili:**")
                        st.markdown("*Sistema portoghese di ammissione all'università*")
                        for code, label in sorted(APPLICATION_MODE.items())[:10]:
                            st.write(f"- `{code}`: {label}")
                        st.info(f"... e altri {len(APPLICATION_MODE)-10} codici")
                    
                    elif 'previous qualification' in feature_lower and 'grade' not in feature_lower:
                        st.markdown("**Valori possibili:**")
                        for code, label in sorted(PREVIOUS_QUALIFICATION.items())[:10]:
                            st.write(f"- `{code}`: {label}")
                        st.info(f"... e altri {len(PREVIOUS_QUALIFICATION)-10} codici")
                    
                    elif "qualification" in feature_lower:
                        st.markdown("**Valori possibili:**")
                        st.markdown("*Livelli di istruzione nel sistema portoghese*")
                        for code, label in sorted(PARENT_QUALIFICATION.items())[:10]:
                            st.write(f"- `{code}`: {label}")
                        st.info(f"... e altri {len(PARENT_QUALIFICATION)-10} codici")
                    
                    elif "occupation" in feature_lower:
                        st.markdown("**Valori possibili:**")
                        st.markdown("*Classificazione portoghese delle professioni*")
                        for code, label in sorted(PARENT_OCCUPATION.items())[:10]:
                            st.write(f"- `{code}`: {label}")
                        st.info(f"... e altri {len(PARENT_OCCUPATION)-10} codici")
                    
                    elif 'course' in feature_lower:
                        st.markdown("**Corsi disponibili:**")
                        for code, label in sorted(COURSE.items()):
                            st.write(f"- `{code}`: {label}")
                    
                    elif 'nacionality' in feature_lower or 'nationality' in feature_lower:
                        st.markdown("**Nazionalità principali:**")
                        for code, label in sorted(NATIONALITY.items())[:15]:
                            st.write(f"- `{code}`: {label}")
                    
                    elif any(x in feature_lower for x in ['displaced', 'debtor', 'international', 
                                                           'scholarship', 'tuition', 'special needs']):
                        st.markdown("**Valori possibili:**")
                        st.write("- `0`: No")
                        st.write("- `1`: Sì")
                    
                    elif 'grade' in feature_lower:
                        st.markdown("**Range di valori:** 0 - 200")
                        st.info("Sistema di valutazione portoghese (scala 0-200)")
                    
                    elif 'rate' in feature_lower or 'gdp' in feature_lower:
                        st.markdown("**Tipo:** Valore percentuale")
                        st.info("Indicatore macroeconomico al momento dell'iscrizione")


def show_data_preview_enhanced(df, n_rows=10):
    """
    Mostra un'anteprima avanzata dei dati con toggle tra valori grezzi e leggibili.
    """
    st.subheader("👁️ Anteprima Dati")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        show_mode = st.radio(
            "Modalità visualizzazione:",
            ["📖 Valori Leggibili", "🔢 Codici Numerici"],
            horizontal=True
        )
    
    with col2:
        n_rows = st.number_input("Righe da mostrare:", 5, 100, n_rows)
    
    with col3:
        if st.button("🔄 Aggiorna"):
            st.rerun()
    
    # Mostra dati
    if show_mode == "📖 Valori Leggibili":
        df_display = format_dataframe_for_display(df.head(n_rows))
        st.dataframe(df_display, use_container_width=True, height=400)
        st.info("💡 I valori sono stati tradotti in formato leggibile. "
               "Attiva 'Codici Numerici' per vedere i valori originali.")
    else:
        st.dataframe(df.head(n_rows), use_container_width=True, height=400)
        st.info("💡 Stai visualizzando i codici numerici originali. "
               "Attiva 'Valori Leggibili' per una visualizzazione più chiara.")
    
    # Statistiche rapide
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Totale Righe", f"{len(df):,}")
    with col2:
        st.metric("📋 Totale Colonne", len(df.columns))
    with col3:
        missing = df.isnull().sum().sum()
        st.metric("❌ Valori Mancanti", f"{missing:,}")
    with col4:
        duplicates = df.duplicated().sum()
        st.metric("🔁 Duplicati", f"{duplicates:,}")


def show_feature_explorer(df):
    """
    Esplora singole feature in dettaglio.
    """
    st.subheader("🔍 Esplora Feature")
    
    st.markdown("""
    Seleziona una feature per vedere statistiche dettagliate, 
    distribuzione dei valori e significato.
    """)
    
    # Selezione feature
    feature = st.selectbox(
        "Seleziona una feature da esplorare:",
        df.columns,
        format_func=lambda x: f"{x} - {get_feature_description(x)[:50]}..."
    )
    
    if feature:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown(f"### 📊 {feature}")
            st.info(get_feature_description(feature))
            
            # Statistiche
            st.markdown("**Statistiche:**")
            if df[feature].dtype in ['int64', 'float64']:
                st.write(f"- **Media:** {df[feature].mean():.2f}")
                st.write(f"- **Mediana:** {df[feature].median():.2f}")
                st.write(f"- **Min:** {df[feature].min():.2f}")
                st.write(f"- **Max:** {df[feature].max():.2f}")
                st.write(f"- **Deviazione Standard:** {df[feature].std():.2f}")
            
            st.write(f"- **Valori Unici:** {df[feature].nunique()}")
            st.write(f"- **Valori Mancanti:** {df[feature].isnull().sum()}")
        
        with col2:
            st.markdown("### 📈 Distribuzione")
            
            # Mostra i valori più comuni
            value_counts = df[feature].value_counts().head(10)
            
            st.markdown("**Top 10 valori più frequenti:**")
            
            for val, count in value_counts.items():
                readable = get_readable_value(feature, val)
                pct = (count / len(df)) * 100
                
                if readable != str(val):
                    st.write(f"- **{readable}** (`{val}`): {count:,} ({pct:.1f}%)")
                else:
                    st.write(f"- **{val}**: {count:,} ({pct:.1f}%)")
            
            if len(df[feature].unique()) > 10:
                st.info(f"... e altri {len(df[feature].unique()) - 10} valori unici")


# Funzione principale da aggiungere al training_app.py
def add_data_explorer_tab():
    """
    Aggiunge una tab completa per l'esplorazione dei dati.
    Da inserire in training_app.py dopo il tab "Caricamento Dati".
    """
    with st.expander("📖 Guida ai Dati - Dizionario Completo", expanded=False):
        show_data_dictionary()
    
    if st.session_state.data_loaded:
        st.markdown("---")
        
        # Sub-tabs per diverse visualizzazioni
        subtab1, subtab2 = st.tabs(["👁️ Anteprima", "🔍 Esplora Feature"])
        
        with subtab1:
            show_data_preview_enhanced(st.session_state.X)
        
        with subtab2:
            show_feature_explorer(st.session_state.X)
