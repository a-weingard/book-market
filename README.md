# Book Market – Literatur trifft Data Science

[English version below](#english-version)

Ein datengetriebenes Analyse- und Vorhersageprojekt rund um den internationalen Buchmarkt mit einem inhaltsbasierten Empfehlungssystem.
Die Web-App kombiniert **Data Science**, **Machine Learning** und **Literaturanalyse** und bietet **zwei Hauptfeatures**:
1. Bücher basierend auf ihrer Inhaltsbeschreibung empfohlen
2. Die Wahrscheinlichkeit einer Verfilmung wird prognostiziert
Beide Funktionen basieren auf **separaten Datensätzen und Analysen**.
Die App ist interaktiv, intelligent und visuell umgesetzt mit Python, Streamlit sowie etablierten Bibliotheken für **Datenanalyse und Visualisierung**.

## Table of Contents
1. [Projektübersicht](#projektübersicht)
2. [Features der App](#features-der-app)
3. [Verwendete Technologien](#verwendete-technologien)
4. [Installation (DE)](#installation-(de))
5. [Team (DE)](#team-(de))

---

## Projektübersicht

**Book Market** ist eine interaktive Web-App, die Buchdaten analysiert, personalisierte Empfehlungen generiert und mithilfe von Machine-Learning-Methoden vorhersagt, ob ein Buch potenziell verfilmt wird.
Das Ziel war, literarische Trends datenbasiert zu verstehen und gleichzeitig praktische Data-Science-Kompetenzen in einem realistischen Projektkontext zu entwickeln.

Das Projekt wurde im Team nach dem [Scrum-Framework](https://www.scrum.org) umgesetzt:
- 3 Sprints à 2 Wochen (agile Planung mit Scrum-Regeln, Rollen, Events und Artefakten)
- Gemeinsame Versionskontrolle über [GitHub](https://github.com)

---

## Features der App

### Inhaltbasiertes Empfehlungssystem

- Empfiehlt ähnliche Bücher basierend auf Genre, Bewertung und Erscheinungsjahr
- Visualisierung der Ergebnisse mit interaktiven Elementen
- Transparentes Empfehlungssystem mit erklärbaren Ergebnissen

### Verfilmungsprognose

- Logistische Regression zur Vorhersage der Wahrscheinlichkeit, dass ein Buch verfilmt wird
- Input: Buchdaten | Output: Wahrscheinlichkeit der Verfilmung
- Dynamisch einstellbarer Schwellenwert zur Anpassung des Modellverhaltens
- Visualisierte Ausgabe der Prognoseergebnisse

---

## Verwendete Technologien
***
* [Python](https://www.python.org): Version 3.9.6 
* [Pandas](https://pandas.pydata.org): Version 2.2.3
* [NumPy](https://numpy.org): Version 2.2.6
* [Matplotlib](https://matplotlib.org): Version 3.10.1 
* [Seaborn](https://seaborn.pydata.org): Version 0.13.2
* [Scikit-learn](https://scikit-learn.org): Version 1.6.1
* [Streamlit](https://streamlit.io): Version 1.44.1
* [Joblib](https://joblib.readthedocs.io): Version 1.5.1
* [Requests](https://requests.readthedocs.io/en/latest/): Version 2.32.3 
* [Torch](https://pytorch.org): Version 2.7.1

## Installation (DE)

### Repository klonen
```
git clone https://github.com/a-weingard/book-market.git
```

### In das Projektverzeichnis wechseln
```
cd book-market
```

### Abhängigkeiten installieren
```
pip install -r requirements.txt
```

### Streamlit-App starten
```
streamlit run app.py
```

#### Voraussetzungen

Python 3.9+  
Installiere die benötigten Pakete mit:

```bash
pip install -r requirements.txt

```
---

### Team (DE)
* Lena Wowchik
* Arina Weingard
* Julia Schober

---

English Version

# Book Market – Literature Meets Data Science

A data-driven analysis and prediction learning project focused on the international book market with a content-based recommendation system.
The web app combines Data Science, Machine Learning, and Literary Analysis and consists of **two main features**:
1. Books are recommended based on their content description
2. The likelihood of a book being adapted into a film is predicted
Both features use **separate datasets and analyses**.
Interactive, intelligent, and visually implemented using Python, Streamlit, and widely used libraries for **data analysis and visualization**.

## Table of Contents
1. [Project Overview](#project-overview)
2. [App Features](#app-features)
3. [Technologies](#technologies)
4. [Installation (EN)](#installation-(en))
5. [Team (EN)](#team-(en))

## Project Overview

**Book Market** is an interactive web app that analyzes book data, generates personalized recommendations, and predicts the probability of a book being adapted into a movie using machine learning methods.
The goal of the project is to understand literary trends through data and to develop practical data science skills in a realistic project context.

The project was implemented as a team using [Scrum framework](https://www.scrum.org):
- 3 sprints of 2 weeks each (agile planning with Scrum rules, roles, events, and artifacts)
- Shared version control using [GitHub](https://github.com)

## App Features

### Content-Based Recommendation System

- Recommends similar books based on genre, rating, and publication year
- Visualizes results with interactive elements
- Transparent recommender system with explainable outcomes

### Film Adaptation Prediction

- Logistic regression to predict the probability of a book being adapted into a film
- Input: Book data | Output: Probability of adaptation
- Dynamically adjustable threshold to modify model behavior
- Visual output of prediction results

## Technologies
***
* [Python](https://www.python.org): Version 3.9.6 
* [Pandas](https://pandas.pydata.org): Version 2.2.3
* [NumPy](https://numpy.org): Version 2.2.6
* [Matplotlib](https://matplotlib.org): Version 3.10.1 
* [Seaborn](https://seaborn.pydata.org): Version 0.13.2
* [Scikit-learn](https://scikit-learn.org): Version 1.6.1
* [Streamlit](https://streamlit.io): Version 1.44.1
* [Joblib](https://joblib.readthedocs.io): Version 1.5.1
* [Requests](https://requests.readthedocs.io/en/latest/): Version 2.32.3 
* [Torch](https://pytorch.org): Version 2.7.1

## Installation (EN)

### Clone the repository
```
git clone https://github.com/a-weingard/book-market.git
```

### Navigate to the project directory
```
cd book-market
```

### Install dependencies
```
pip install -r requirements.txt
```

### Run the Streamlit app
```
streamlit run app.py
```

#### Requirements

Python 3.9+  
Install required packages:

```bash
pip install -r requirements.txt

```
---

### Team (EN)
* Lena Wowchik
* Arina Weingard
* Julia Schober