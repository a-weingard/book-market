from trans_author import AuthorRatingMapper
import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    recall_score,
    confusion_matrix,
    accuracy_score,
    precision_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
)
import seaborn as sns


def show():

    st.markdown(
        "<h2>🎬 Prognose von Buchverfilmungen für Neuerscheinungen 2021–2025</h2>",
        unsafe_allow_html=True,
    )

    try:
        df_pred = pd.read_csv("data/new_books_2024.csv", encoding="utf-8")
        df_pred.columns = df_pred.columns.str.strip()
    except FileNotFoundError:
        st.error("❌ Datei 'data/new_books_2024.csv' wurde nicht gefunden.")
        return
    # WICHTIG: pipeline wird geladen!!!! es wurde im block ML erstellt und als logisti_pipeline.pkl abgespeichrt---> mein gespeichertes MOdel
    try:
        pipeline = joblib.load("logistic_pipeline.pkl")
    except FileNotFoundError:
        st.error("❌ Modell-Datei 'logistic_pipeline.pkl' nicht gefunden.")
        return

    threshold_slider = st.sidebar.slider(
        "🔧 Threshold für Vorhersage",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Ab welcher Wahrscheinlichkeit das Modell eine Verfilmung vorhersagt.",
    )

    autor_list = ["🔽 Bitte wählen..."] + sorted(df_pred["Author"].unique().tolist())
    autor = st.sidebar.selectbox("👤 Wähle einen Autor", autor_list)

    if autor != "🔽 Bitte wählen...":
        buecher_von_autor = df_pred[df_pred["Author"] == autor]
        buch_list = ["🔽 Bitte wählen..."] + sorted(
            buecher_von_autor["Book_Name"].unique().tolist()
        )
        buch = st.sidebar.selectbox("📚 Wähle ein Buch", buch_list)

        if buch != "🔽 Bitte wählen...":
            buchdaten = buecher_von_autor[buecher_von_autor["Book_Name"] == buch].iloc[
                0:1
            ]

            # Buchdetails
            st.write("### 📖 Details zum ausgewählten Buch:")
            st.write(buchdaten)

            # Cover anzeigen, falls ISBN vorhanden
            if "ISBN" in buchdaten.columns and pd.notna(buchdaten["ISBN"].values[0]):
                isbn = str(buchdaten["ISBN"].values[0])
                cover_url = f"https://covers.openlibrary.org/b/isbn/{isbn}-L.jpg"
                st.sidebar.image(
                    cover_url, caption="📕 Buchcover", use_container_width=True
                )

            # Vorhersage
            X_new = buchdaten.drop(columns=["Book_Name"], errors="ignore")
            proba = pipeline.predict_proba(X_new)[:, 1][0]
            pred = "Ja" if proba >= threshold_slider else "Nein"

            buchname = buchdaten["Book_Name"].values[0]
            autorname = buchdaten["Author"].values[0]

            st.markdown(
                f"<h4 style='font-size: 1.2rem;'>📊 <strong>Wahrscheinlichkeit für Verfilmung:</strong> {proba*100:.0f}%</h4>",
                unsafe_allow_html=True,
            )

            if pred == "Ja":
                st.markdown(
                    f"""
                    <div style='background-color:#e6f4ea; padding: 1.2rem; border-left: 6px solid #34a853; border-radius: 6px; font-size: 1.1rem;'>
                        🎬 <strong>Erfolg!</strong> Das Buch <em>{buchname}</em> von <em>{autorname}</em> wird voraussichtlich verfilmt! 🍿<br>
                        <span style='font-size: 0.95rem; color: gray;'>(Schwellenwert: {threshold_slider * 100:.0f}%)</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.balloons()
            else:
                st.markdown(
                    f"""
                    <div style='background-color:#f0f2f6; padding: 1.2rem; border-left: 6px solid #4e79a7; border-radius: 6px; font-size: 1.1rem;'>
                        📘 Das Buch <em>{buchname}</em> von <em>{autorname}</em> wird aktuell wohl nicht verfilmt.<br>
                        <span style='font-size: 0.95rem; color: gray;'>(Schwellenwert: {threshold_slider * 100:.0f}%)</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # === Basis Daten laden ===
    try:
        df_ana = pd.read_csv("data/book_data_clean.csv", sep=";", encoding="utf-8")
        df_ana.columns = df_ana.columns.str.strip()
    except FileNotFoundError:
        st.warning(
            "⚠️ Datei 'data/book_data_clean.csv' für historische Daten nicht gefunden."
        )
        return
    st.write("")  # leere Zeile
    st.markdown("---")  # Trennlinie
    # Titel links, Button rechts
    col1, col2 = st.columns([5, 2])
    with col1:
        st.markdown(
            "<h2 style='margin-bottom: 0;'>📊 Rückblick: Wie gut sagt das Modell echte Verfilmungen voraus?</h2>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(" ")
        if st.button("📅 Zur Machine Learning-Daten ➜"):
            st.switch_page("pages/_Verfilmung-Buchdaten.py")

    st.markdown(" ")

    # === Modell-Performance auf historischen Daten ===
    if "Adapted_to_Film" in df_ana.columns:
        X_hist = df_ana.drop(columns=["Book_Name", "Adapted_to_Film"], errors="ignore")
        y_hist = df_ana["Adapted_to_Film"]

        y_proba_hist = pipeline.predict_proba(X_hist)[:, 1]
        y_pred_hist = (y_proba_hist >= threshold_slider).astype(int)

        # Metriken berechnen
        recall = recall_score(y_hist, y_pred_hist)
        accuracy = accuracy_score(y_hist, y_pred_hist)
        precision = precision_score(y_hist, y_pred_hist)
        f1 = f1_score(y_hist, y_pred_hist)
        roc_auc = roc_auc_score(y_hist, y_proba_hist)
        cm = confusion_matrix(y_hist, y_pred_hist)

        # === TITELZEILE: 3 Spalten ===
        title_col1, title_col2, title_col3 = st.columns([1, 1, 1])
        with title_col1:
            st.markdown(
                f"""
                <h3 style='margin-bottom: 0;'>🤖 Logistisches Regressionsmodell mit geänderten Threshold🎚️ {threshold_slider:.2f}</h3>
                <p style='font-size: 0.9rem; color: gray; margin-top: 0.3rem;'>
                Der Threshold-Schwellenwert bestimmt, **ab welcher Wahrscheinlichkeit ein Buch als „verfilmt“ gilt**.<br>
                Ein niedrigerer Wert erkennt mehr mögliche Verfilmungen (höherer Recall), <br>
                ein höherer Wert reduziert Fehlalarme (höhere Precision).<br>
                (Dynamisch anpassbar über den Schieberegler seitlich)
                </p>
                """,
                unsafe_allow_html=True,
            )
        with title_col2:
            st.markdown(
                """
                <div style='padding-left: 30px;'>
                <h3 style='margin-bottom: 0; margin-top: 0;'>🧮 Verfilmungen: Was das Modell richtig oder falsch erkennt</h3>
                <p style='font-size: 0.9em; margin-top: 0.2em; color: gray;'>
                Confusionsmatrix zeigt, wie viele Bücher das Modell richtig oder falsch als verfilmt bzw. nicht verfilmt erkannt hat.<br>
                So sieht man auf einen Blick, wo das Modell gut ist und wo es Fehler macht.
                </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with title_col3:
            st.markdown(
                """
                <h3 style='margin-bottom: 0; margin-top: 0;'>📉  Threshold-Sensitivität: Trefferquote bei unterschiedlichen Grenzen</h3>
                <p style='font-size: 0.9em; margin-top: 0.2em; color: gray;'>
                Zeigt, wie gut das Modell Verfilmungen erkennt, wenn wir die Entscheidungsschwelle verändern.<br>
                Eine Kurve, die oben rechts liegt, bedeutet bessere Vorhersagen.
                </p>
                """,
                unsafe_allow_html=True,
            )

        # === INHALT: 3 Spalten ===
        col1, col2, col3 = st.columns([1.1, 1, 1])

        # Metriken mit Erklärung
        with col1:
            metrics_data = {
                "Metrik": [
                    "Recall",
                    "Precision",
                    "F1-Score",
                    "Accuracy",
                    "ROC-AUC",
                ],
                "Wert": [recall, precision, f1, accuracy, roc_auc],
                "Erklärung": [
                    "Erkannt: Wie viele echte Verfilmungen korrekt erkannt wurden. **Je näher an 1.00, desto besser.**",
                    "Treffer: Wie viele Vorhersagen für Verfilmung auch wirklich stimmen. **Hoher Wert = wenig Fehlalarme.**",
                    "Balance zwischen Precision & Recall. **Ideal bei unausgeglichenen Klassen.**",
                    "Genauigkeit/Gesamttrefferquote – Anteil der korrekten Vorhersagen an allen Fällen. **Kann bei Ungleichverteilung trügen.**",
                    "Trennschärfe unabhängig vom Schwellenwert. **Über 0.80 = sehr gutes Modell.**",
                ],
            }
            df_metrics = pd.DataFrame(metrics_data)
            st.write("### 📊 Modell-Metriken im Überblick")
            st.table(df_metrics.style.format({"Wert": "{:.2f}"}))

        # Confusion Matrix (Heatmap)
        with col2:
            st.markdown("<div style='padding-left: 100px;'>", unsafe_allow_html=True)
            # Hier Plot-Code
            fig, ax = plt.subplots(figsize=(2, 2), dpi=100)
            sns.heatmap(
                cm,
                annot=True,
                fmt="g",
                cmap="RdYlGn",
                cbar=False,
                xticklabels=["Nicht verfilmt", "Verfilmt"],
                yticklabels=["Nicht verfilmt", "Verfilmt"],
                annot_kws={"size": 6, "color": "black"},
                ax=ax,
            )
            ax.set_xlabel("Vorhergesagte Klasse", fontsize=8)
            ax.set_ylabel("Tatsächliche Klasse", fontsize=8)
            ax.tick_params(axis="x", labelsize=6)
            ax.tick_params(axis="y", labelsize=6)
            plt.tight_layout(pad=0.1)
            # Achsen-Position nach rechts verschieben (x0 verschieben)
            pos = ax.get_position()  # aktueller Position-Box (Bbox)
            new_pos = [
                pos.x0 + 0.4,
                pos.y0,
                pos.width,
                pos.height,
            ]  # x0 nach rechts verschieben
            ax.set_position(new_pos)

            st.pyplot(fig, use_container_width=False)
            st.markdown("</div>", unsafe_allow_html=True)

        # Precision-Recall-Kurve
        with col3:
            prec, rec, _ = precision_recall_curve(y_hist, y_proba_hist)
            fig_pr, ax_pr = plt.subplots(figsize=(2.2, 2.2), dpi=100)
            ax_pr.plot(rec, prec, color="blue")
            ax_pr.set_xlabel("Recall", fontsize=8)
            ax_pr.set_ylabel("Precision", fontsize=8)
            ax_pr.set_title("Precision vs. Recall", fontsize=9)
            ax_pr.grid(True)
            ax_pr.tick_params(axis="both", labelsize=6)
            plt.tight_layout(pad=0.1)
            st.pyplot(fig_pr, use_container_width=False)

    else:
        st.warning("⚠️ Spalte 'Adapted_to_Film' fehlt in den historischen Daten.")
