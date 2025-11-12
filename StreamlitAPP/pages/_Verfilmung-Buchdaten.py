# Buchdatenueberblick
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Buchdaten", layout="wide")

# Daten laden
df_ana = pd.read_csv("data/book_data_clean.csv", sep=";", encoding="utf-8")

# FORMATIERUNG: Werte runden (floats)
df_ana["Average_Rating"] = df_ana["Average_Rating"].round(2)
df_ana["Gross_Sales_EUR"] = df_ana["Gross_Sales_EUR"].round(2)
df_ana["Publisher_Revenue_EUR"] = df_ana["Publisher_Revenue_EUR"].round(2)

# FORMATIERUNG: Ganze Zahlen korrekt setzen (nullable Int)
df_ana["Rating_Count"] = df_ana["Rating_Count"].astype(pd.Int64Dtype())
df_ana["Publishing_Year"] = df_ana["Publishing_Year"].astype(pd.Int64Dtype())
df_ana["Adapted_to_Film"] = df_ana["Adapted_to_Film"].astype(pd.Int64Dtype())

df_ana = df_ana.drop(columns=["Publisher_Revenue_EUR"], errors="ignore")

# Jahrbereich für Slider vorbereiten
min_jahr = int(df_ana["Publishing_Year"].min())
max_jahr = int(df_ana["Publishing_Year"].max())

# Sidebar: Jahr-Slider
jahr_slider = st.sidebar.slider(
    "📅 Jahr auswählen",
    min_value=min_jahr,
    max_value=max_jahr,
    value=(min_jahr, max_jahr),
    step=1,
)

# Sidebar: Auswahl der Grafik
auswahl = st.sidebar.radio(
    "🔘 Welche Grafik möchtest du sehen?",
    [
        "Anzahl Bücher pro Jahr",
        "Bewertung vs. Umsatz (Scatterplot)",
        "Verfilmte Bücher pro Jahr",
        "Top Genres",
        "Bewertungsverteilung",
        "Verfilmungsanteil",
        "Top Autor:innen",
        "Ø Bewertung nach Autor:innen-Rating",
        "Bewertungen pro Jahr",
        "Umsatz vs. Jahr (Farbskala: Bewertung)",
        "Bewertungen vs. Jahr (Farbskala: Autor-Rating)",
    ],
)


# Daten filtern nach Jahr
df_filtered = df_ana[
    (df_ana["Publishing_Year"] >= jahr_slider[0])
    & (df_ana["Publishing_Year"] <= jahr_slider[1])
]

# Layout: zwei Spalten
col1, col2 = st.columns([2, 2])

# Linke Spalte: Tabelle und Statistik
with col1:
    st.title("📅 BUCHDATEN")
    st.write(
        f"### 📚 Bereinigte Daten für ML Model, Verlauf über die Jahre ({jahr_slider[0]}–{jahr_slider[1]})"
    )
    st.dataframe(df_filtered)
    st.markdown("---")
    st.write("**Numerische Basisstatistiken:**")
    st.write(df_filtered.describe())

# Rechte Spalte: Visualisierung
with col2:
    st.write("")
    st.title("📊 Visualisierung")

    if auswahl == "Anzahl Bücher pro Jahr":
        counts = df_filtered["Publishing_Year"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(counts.index, counts.values, color="blue")
        ax.set_title(" Anzahl Bücher pro Jahr")
        ax.set_xlabel("Jahr")
        ax.set_ylabel("Anzahl")
        ax.grid(True)
        st.pyplot(fig)

    elif auswahl == "Bewertung vs. Umsatz (Scatterplot)":
        required_cols = {"Average_Rating", "Gross_Sales_EUR", "Author_Rating"}
        if required_cols.issubset(df_filtered.columns):
            rating_map = {"Novice": 1, "Intermediate": 2, "Excellent": 3, "Famous": 4}
            df_filtered["Author_Rating_Num"] = df_filtered["Author_Rating"].map(
                rating_map
            )
            fig, ax = plt.subplots(figsize=(7, 5))
            scatter = ax.scatter(
                df_filtered["Average_Rating"],
                df_filtered["Gross_Sales_EUR"],
                c=df_filtered["Author_Rating_Num"],
                cmap="viridis",
                alpha=0.7,
                edgecolors="k",
            )
            ax.set_xlabel("Average Rating")
            ax.set_ylabel("Gross Sales (EUR)")
            ax.set_title(" Bewertung vs. Umsatz (Farbskala: Author-Rating)")
            ax.grid(True)
            ax.set_yscale("log")
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_ticks([1, 2, 3, 4])
            cbar.set_ticklabels(["Novice", "Intermediate", "Excellent", "Famous"])
            st.pyplot(fig)
        else:
            st.warning(
                "Benötigte Spalten ('Average_Rating', 'Gross_Sales_EUR', 'Author_Rating') fehlen."
            )

    elif auswahl == "Verfilmte Bücher pro Jahr":
        if "Adapted_to_Film" in df_filtered.columns:
            adapted_counts = (
                df_filtered[df_filtered["Adapted_to_Film"] == 1]["Publishing_Year"]
                .value_counts()
                .sort_index()
            )
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(adapted_counts.index, adapted_counts.values, color="red")
            ax.set_title(" Anzahl verfilmter Bücher pro Jahr")
            ax.set_xlabel("Jahr")
            ax.set_ylabel("Verfilmungen")
            ax.grid(True)
            st.pyplot(fig)
        else:
            st.warning("Spalte 'Adapted_to_Film' nicht vorhanden.")

    elif auswahl == "Top Genres":
        genre_counts = df_filtered["Genre"].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(6, 4))
        genre_counts.plot(kind="bar", ax=ax, color="teal")
        ax.set_title("Top 10 Genres nach Anzahl Bücher")
        ax.set_ylabel("Anzahl")
        ax.set_xticklabels(genre_counts.index, rotation=45, ha="right")
        st.pyplot(fig)

    elif auswahl == "Bewertungsverteilung":
        fig, ax = plt.subplots(figsize=(6, 4))
        df_filtered["Average_Rating"].dropna().hist(
            bins=20, ax=ax, color="orange", edgecolor="black"
        )
        ax.set_title("Verteilung der Bewertungen")
        ax.set_xlabel("Durchschnittliche Bewertung")
        ax.set_ylabel("Anzahl Bücher")
        st.pyplot(fig)

    elif auswahl == "Verfilmungsanteil":
        adapted_counts = df_filtered["Adapted_to_Film"].value_counts().sort_index()
        labels = ["Nicht verfilmt", "Verfilmt"]
        fig, ax = plt.subplots()
        ax.pie(
            adapted_counts,
            labels=labels,
            autopct="%1.1f%%",
            colors=["lightgray", "red"],
        )
        ax.set_title("Verfilmungsanteil")
        st.pyplot(fig)

    elif auswahl == "Top Autor:innen":
        top_authors = df_filtered["Author"].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(6, 4))
        top_authors.plot(kind="barh", ax=ax, color="darkgreen")
        ax.set_title("Top 10 Autor:innen nach Anzahl Bücher")
        ax.set_xlabel("Anzahl Bücher")
        st.pyplot(fig)

    elif auswahl == "Ø Bewertung nach Autor:innen-Rating":
        if "Author_Rating" in df_filtered.columns:
            avg_by_auth_rating = (
                df_filtered.groupby("Author_Rating")["Average_Rating"]
                .mean()
                .sort_index()
            )
            fig, ax = plt.subplots(figsize=(6, 4))
            avg_by_auth_rating.plot(kind="bar", ax=ax, color="purple")
            ax.set_title("Ø Bewertung nach Autor:innen-Rating")
            ax.set_ylabel("Durchschnittliche Bewertung")
            st.pyplot(fig)

    elif auswahl == "Bewertungen pro Jahr":
        ratings_by_year = df_filtered.groupby("Publishing_Year")["Rating_Count"].sum()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(
            ratings_by_year.index, ratings_by_year.values, marker="o", color="steelblue"
        )
        ax.set_title("Gesamte Bewertungseinträge pro Jahr")
        ax.set_xlabel("Jahr")
        ax.set_ylabel("Anzahl Bewertungen")
        st.pyplot(fig)

    elif auswahl == "Umsatz vs. Jahr (Farbskala: Bewertung)":
        if {"Publishing_Year", "Gross_Sales_EUR", "Average_Rating"}.issubset(
            df_filtered.columns
        ):
            fig, ax = plt.subplots(figsize=(7, 5))
            scatter = ax.scatter(
                df_filtered["Publishing_Year"],
                df_filtered["Gross_Sales_EUR"],
                c=df_filtered["Average_Rating"],
                cmap="plasma",
                alpha=0.7,
                edgecolors="k",
            )
            ax.set_xlabel("Jahr")
            ax.set_ylabel("Umsatz (EUR)")
            ax.set_title("Umsatz vs. Jahr (Farbskala: Bewertung)")
            ax.set_yscale("log")
            ax.grid(True)
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label("Average Rating")
            st.pyplot(fig)

    elif auswahl == "Bewertungen vs. Jahr (Farbskala: Autor-Rating)":
        if {"Rating_Count", "Publishing_Year", "Author_Rating"}.issubset(
            df_filtered.columns
        ):
            rating_map = {"Novice": 1, "Intermediate": 2, "Excellent": 3, "Famous": 4}
            df_filtered["Author_Rating_Num"] = df_filtered["Author_Rating"].map(
                rating_map
            )

            fig, ax = plt.subplots(figsize=(7, 5))
            scatter = ax.scatter(
                df_filtered["Publishing_Year"],
                df_filtered["Rating_Count"],
                c=df_filtered["Author_Rating_Num"],
                cmap="cool",
                alpha=0.7,
                edgecolors="k",
            )
            ax.set_xlabel("Jahr")
            ax.set_ylabel("Anzahl Bewertungen")
            ax.set_title("Bewertungen vs. Jahr (Farbskala: Autor:innen-Rating)")
            ax.set_yscale("log")
            ax.grid(True)
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_ticks([1, 2, 3, 4])
            cbar.set_ticklabels(["Novice", "Intermediate", "Excellent", "Famous"])
            st.pyplot(fig)
