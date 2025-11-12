import streamlit as st


def show():
    st.markdown(
        """
        <style>
        .title {
            font-size:60px;
            font-weight:bold;
            color:#FFD700;
            text-align:center;
            padding-top:30px;
        }
        .subtitle {
            font-size:32px;
            color:#bbbbbb;
            text-align:center;
            padding-bottom:20px;
        }
        .info-box {
            background-color: #111827;
            padding: 20px;
            border-radius: 10px;
            color: white;
            margin: 10px;
        }
        .section-title {
            font-size: 22px;
            color: #4ade80;
            margin-bottom: 10px;
        }
        .content-text {
            font-size: 18px;
            line-height: 1.6;
            color: black;
        }
        </style>
        <div class='title'>📚 Book Market: Literatur trifft Data Science</div>
        <div class='subtitle'> Alles rund um Bücher.
        Interaktiv. Intelligent. Buchmarkt mit Datenblick. </div>
        """,
        unsafe_allow_html=True,
    )

    col1, spacer1, col2, spacer2, col3 = st.columns([3, 0.5, 3, 0.5, 3])

    with col1:
        st.markdown("#### 📊 Wirtschaftsanalyse", unsafe_allow_html=True)
        st.markdown(
            """
            <div class='content-text'>
            Analyse wirtschaftlicher Einflussfaktoren auf den Buchumsatz: Genre, Bewertung, Verlag und mehr.<br><br>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown("#### 🤖 Buchempfehlungssystem", unsafe_allow_html=True)
        st.markdown(
            """
            <div class='content-text'>
            Personalisierte Buchempfehlungen auf Basis semantischer Ähnlichkeit und benutzerdefinierter Filter.<br><br>

            - Inhaltsbasierte Empfehlungen zu einem gewählten Buch<br>

            - Filterung nach Genre, Autor, Erscheinungsjahr und Bewertung<br>

            - Nutzung von SentenceTransformers & Cosine Similarity

            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown("#### 🎥 🎞️ ⭐ Verfilmungsprognose", unsafe_allow_html=True)
        st.markdown(
            """
            <div class='content-text'>
            Kann man vorhersagen, ob ein frisch erschienenes Buch eines Tages erfolgreich wird – und schließlich auf der Leinwand landet?

            - Analyse von Genre, Sprache, Bewertung, Verkäufen & Verfilmungsstatus

            - Visuelle Darstellung relevanter Zusammenhänge

            - Entwicklung eines Prognosemodells (logistische Regression)

            - Bewertung der Vorhersagekraft mittels Accuracy, Precision & AUC
            
    

            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    credit_col, image_col = st.columns(
        [2, 3.5]
    )  # Links mehr Platz für Text, rechts fürs Bild

    with credit_col:
        st.markdown("#### 📎 Credits")
        st.markdown(
            """
            <div class='content-text'>
            Dieses Projekt wurde im Rahmen des Data Science Institute entwickelt.  
            
            <br><b>Verwendete Modelle und Methoden:</b><br>
            - Explorative Datenanalyse (Pandas, Seaborn, Matplotlib)<br>
            - Lineare Regressionsmodelle zur Untersuchung von Einflussfaktoren auf den Bruttoumsatz<br>
            - Korrelationsanalysen (Pearson)<br>
            - Logistische Regression zur Verfilmungsprognose<br>
            - Modellbewertung mittels Accuracy, Precision, Recall, AUC<br>
            - Semantische Textanalyse mit SentenceTransformers<br>
            - TF-IDF-Vektorisierung & Cosine Similarity<br>
            - Text-Preprocessing inkl. Tokenisierung, Stopword-Filterung (NLTK & scikit-learn)<br>
            - WordClouds zur Visualisierung<br><br>

            <b>Technologien & Bibliotheken:</b><br>
            Streamlit · Pandas · NumPy · scikit-learn · Matplotlib · Seaborn · SentenceTransformers · NLTK · PyTorch · requests · ast<br><br>

            - Projektmanagement mit <a href="https://github.com/Lena-Wow/Datengetriebene-Buchmarktanalyse-NLP-basiertes-Empfehlungssystem-und-Verfilmungsprognose-mittels-ML/tree/main/SCRUM" target="_blank"><b>SCRUM</b></a><br>
            - Zusammenarbeit & Versionskontrolle mit <a href="https://github.com/Lena-Wow/Datengetriebene-Buchmarktanalyse-NLP-basiertes-Empfehlungssystem-und-Verfilmungsprognose-mittels-ML/tree/main" target="_blank"><b>GitHub</b></a><br>
            - Datenquellen: <a href="https://www.kaggle.com/">Kaggle</a>, <a href="https://www.goodreads.com/">Goodreads</a>, <a href="https://openlibrary.org/dev/docs/api/covers">Open Library</a>, <a href="https://developers.google.com/books">Google Books API</a><br><br>

            <b>👩‍💻 Team & GitHub:</b><br>
            - <a href="https://github.com/julia-beispiel" target="_blank">Julia auf GitHub</a><br>
            - <a href="https://github.com/lena-wow" target="_blank">Lena auf GitHub</a><br>
            - <a href="https://github.com/arina-ds" target="_blank">Arina auf GitHub</a><br>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with image_col:
        st.image(
            "data/books.jpg", caption="Projektcover", use_container_width=True
        )
