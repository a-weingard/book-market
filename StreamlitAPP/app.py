import streamlit as st  # als erster Streamlit-Befehl kommen

st.set_page_config(page_title="Book Market", layout="wide")

# Jetzt andere Imports
from trans_author import AuthorRatingMapper
import Verfilmungsprognose
import empfehlung
import start

st.sidebar.title("📘 Projekt-Navigation")
st.sidebar.info("📚 Buchmarkt analysieren – Empfehlungen & Filmchance inklusive.")
page = st.sidebar.radio(
    "Wähle eine Funktion:",
    ["Startseite", "Empfehlungssystem", "Verfilmungsprognose"],
)

if page == "Startseite":
    start.show()
elif page == "Empfehlungssystem":
    empfehlung.show()
elif page == "Verfilmungsprognose":
    Verfilmungsprognose.show()

