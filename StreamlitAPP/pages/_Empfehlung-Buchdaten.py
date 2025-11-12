# Buchempfehlung-Visualisierung
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Empfehlung-Buchdaten", layout="wide")

# ----------------- DATEN LADEN -----------------------

@st.cache_data
def load_data():
    return pd.read_csv('data/final_books_recommend.csv', encoding='ascii')

df = load_data()

# ------------------------ SIDEBAR: DATENSATZ INFO ------------------------------------------

st.sidebar.header("Datensatz Info")
st.sidebar.write("Anzahl Bücher (Zeilen):", df.shape[0])
st.sidebar.write("Anzahl Features (Spalten):", df.shape[1])
st.sidebar.write("Einzigartige Autoren:", df["author"].nunique())
st.sidebar.write("Einzigartige Buchtiteln:", df["title"].nunique())


# Jahrbereich für Slider vorbereiten
min_year = int(df["publication_year"].min())
max_year = int(df["publication_year"].max())

# Sidebar: Jahr-Slider
years_slider = st.sidebar.slider(
    "📅 Jahre auswählen",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year),
    step=1,
)
# Daten filtern nach Jahr
df_filtered = df[
    (df["publication_year"] >= years_slider[0])
    & (df["publication_year"] <= years_slider[1])
]


# Sidebar: Auswahl der Grafik
choose_grafic = st.sidebar.radio(
    "Welche Grafik möchtest du sehen?",
    [
        "Anzahl Bücher pro Jahr",
        "Hauptgenres",
        "Anteil Fiction vs. Non-Fiction",
        "Top Autor:innen",
        "Erscheinungsjahre – Häufigkeitsverteilung", 
        "Verteilung der Durchschnittsbewertungen",
        "Wortanzahl: Original- vs. bereinigte Beschreibungen (Scatterplot)"
    ],
)


# --------------------------- HAUPTTEIL ----------------------------
# Layout: zwei Spalten
col1, col2 = st.columns([2, 2])

# Linke Spalte: Tabelle und Statistik
with col1:
    st.title("📅 BUCHDATEN")
    st.write(f"### 📚 Bereinigte Daten für das Empfehlungssystem")
    st.write(f"### Verlauf über die Jahre {years_slider[0]}–{years_slider[1]}")
    st.dataframe(df_filtered)
    st.markdown("---")
    st.write("**Numerische Basisstatistiken:**")
    st.write(df_filtered.describe())



# Rechte Spalte: Visualisierung
with col2:
    st.write("")
    st.title("📊 Visualisierung")

# -------------------------  ANZAHL BÜCHER PRO JAHR -----------------------------------------

    if choose_grafic == "Anzahl Bücher pro Jahr":
        book_counts = df_filtered["publication_year"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(book_counts.index, book_counts.values, color="#6d597a")
        ax.set_title(" Anzahl Bücher pro Jahr")
        ax.set_xlabel("Erscheinungsjahr")
        ax.set_ylabel("Anzahl Bücher")
        ax.grid(True)
        st.pyplot(fig)


# -------------------------  HAUPTGENRES -----------------------------------------

    elif choose_grafic == "Hauptgenres":
        main_genre_counts = df_filtered["main_genre"].value_counts().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6, 4))
        main_genre_counts.plot(kind="bar", ax=ax, color="#6d597a")

        ax.set_title("Verteilung der Hauptgenres")
        ax.set_xlabel("Hauptgenre")
        ax.set_ylabel("Anzahl Bücher")

        ax.set_xticklabels(main_genre_counts.index, rotation=45, ha="right")  
        st.pyplot(fig)


# -------------------------- FICTION VS. NON-FICTION ----------------------------------------

    elif choose_grafic == "Anteil Fiction vs. Non-Fiction":
        fig, ax = plt.subplots(figsize=(6, 6))
        fiction_counts = df_filtered['is_fiction'].value_counts().rename({1.0:'Fiction', 0.0:'Non-Fiction'})
        fiction_counts.plot(kind='pie', autopct='%1.0f%%', colors=['#6D597A','#E56B6F'])
        ax.set_title('Fiction vs. Non-Fiction Books')
        ax.set_ylabel('')
        st.pyplot(fig)

# -------------------------- TOP AUTOR:INNEN ----------------------------------------

    elif choose_grafic == "Top Autor:innen":
        top_authors = df_filtered["author"].value_counts().sort_values(ascending=True).tail(10)
        fig, ax = plt.subplots(figsize=(6, 4))
        top_authors.plot(kind="barh", ax=ax, color="#6D597A")
        ax.set_title("Top 10 Autor:innen nach Anzahl Bücher")
        ax.set_xlabel("Anzahl Bücher")
        st.pyplot(fig)

     

# ---------------------------- ERSCHEINUNGSJAHRE -------------------------------------

    elif choose_grafic == "Erscheinungsjahre – Häufigkeitsverteilung":
        #years_counts = df_filtered["publication_year"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df_filtered['publication_year'].dropna(), bins=40, color='#6D597A', ax=ax)
        ax.set_title(" Anzahl Bücher pro Jahr")
        ax.set_xlabel("Erscheinungsjahr")
        ax.set_ylabel("Anzahl Bücher")
        st.pyplot(fig)


# ---------------------------- AVERAGE RATING-------------------------------------

    elif choose_grafic == "Verteilung der Durchschnittsbewertungen":
        #avg_rating_counts = df_filtered["publication_year"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df_filtered['avg_rating'].dropna(), bins=30, kde=True, color='#6D597A', ax=ax)
        ax.set_title("Verteilung der Durchschnittsbewertungen")
        ax.set_xlabel("Durchschnittsbewertung")
        ax.set_ylabel("Anzahl Bücher")
        st.pyplot(fig)



# ---------------------------- WORTARZAHL ORIGINAL VS. CLEAN_DESCRIPTIONS --------------------------------

    elif choose_grafic == "Wortanzahl: Original- vs. bereinigte Beschreibungen (Scatterplot)":
        clean_desc = df_filtered['clean_description'].fillna('').str.split().apply(len)
        fig, ax = plt.subplots(figsize=(6,5))
        sns.scatterplot(x=df['word_count_description'], y=clean_desc, alpha=0.4, color='#6D597A', ax=ax)
        ax.set_title("Wortanzahl:\n Original- vs. bereinigte Beschreibungen (Scatterplot)")
        ax.set_xlabel('Original (Wörteranzahl)')
        ax.set_ylabel('Bereinigt (Wörteranzahl)')
        st.pyplot(fig)

