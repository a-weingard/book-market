# Book Market – Literatur trifft Data Science

[English version below](#)

Ein datengetriebenes Analyse- und Vorhersageprojekt rund um den internationalen Buchmarkt.  
Die Web-App kombiniert **Data Science**, **Machine Learning** und **Literaturanalyse**, um Bücher datenbasiert zu empfehlen und ihre Verfilmungswahrscheinlichkeit zu prognostizieren.
Interaktiv, intelligent und visuell umgesetzt mit Python, Streamlit und modernen Analysebibliotheken.

## Table of Contents
1. [Projektübersicht](#projektübersicht)
2. [Features der App](#features-der-app)
2. [Verwendete Technologien](#verwendete-technologien)
3. [Installation](#installation)

---

## Projektübersicht

**Book Market** ist eine interaktive Web-App, die Buchdaten analysiert, personalisierte Empfehlungen generiert und mithilfe von Machine-Learning-Methoden vorhersagt, ob ein Buch potenziell verfilmt wird.
Das Ziel war, literarische Trends datenbasiert zu verstehen und gleichzeitig praktische Data-Science-Kompetenzen in einem realistischen Projektkontext zu entwickeln.

Das Projekt wurde im Team nach dem [Scrum](https://www.scrum.org) umgesetzt:
- 3 Sprints à 2 Wochen (agile Planung mit Scrum-Regeln, Rollen, Events und Artefakten)
- Gemeinsame Versionskontrolle über GitHub

---

## Features der App

### Empfehlungssystem

- Empfiehlt ähnliche Bücher auf Basis von Genre, Bewertung und Erscheinungsjahr
- Visualisierung der Ergebnisse mit interaktiven Elementen
- Transparentes Recommender-System mit erklärbaren Ergebnissen

### Verfilmungsprognose

- Logistische Regression zur Vorhersage der Wahrscheinlichkeit, dass ein Buch verfilmt wird
- Input: Buchdaten | Output: Wahrscheinlichkeit der Verfilmung
- Dynamisch einstellbarer Schwellenwert zur Anpassung des Modellverhaltens
- Visualisierte Ausgabe der Prognoseergebnisse

---

## Verwendete Technologien
***
A list of technologies used within the project:
* [Python](https://www.python.org): Version 3.9.6 
* [Pandas](https://pandas.pydata.org): Version 2.2.3
* [NumPy](https://numpy.org): Version 2.2.6
* [Matplotlib](https://matplotlib.org): Version 3.10.1 
* [Seaborn](https://seaborn.pydata.org): Version 0.13.2
* [Scikit-learn](https://scikit-learn.org): Version 1.6.1 (LinearRegression)
* [Streamlit](https://streamlit.io): Version 1.44.1 (App-UI) 
* [Joblib](https://joblib.readthedocs.io): Version 1.5.1 (Modellspeicherung)
* [Requests](https://requests.readthedocs.io/en/latest/): Version 2.32.3 
* [Torch](https://pytorch.org): Version 2.7.1

- **GitHub** (Versionskontrolle & Zusammenarbeit)
- **SCRUM** (agile Projektstruktur)

## Installation

# Repository klonen
git clone https://github.com/deinusername/book-market.git

# In das Projektverzeichnis wechseln
cd book-market

# Abhängigkeiten installieren
pip install -r requirements.txt

# Streamlit-App starten
streamlit run app.py

## Starten

streamlit run streamlit_app.py

### 👥 Team
Lena Wowchik · Arina Weingard · Julia Schober

#### Voraussetzungen

Python 3.9+  
Installiere die benötigten Pakete mit:

```bash
pip install -r requirements.txt

---

## Table of Contents
1. [General Info](#general-info)
2. [Features of the App](#features-of-the-app)
3. [Technologies](#technologies)
4. [Installation](#installation)

## Technologies
***
A list of technologies used within the project:
