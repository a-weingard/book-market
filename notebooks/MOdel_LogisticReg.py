import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
import seaborn as sns
from trans_author import AuthorRatingMapper
import joblib

# Daten laden
df = pd.read_csv("book_data_clean.csv", sep=";", encoding="utf-8")

y = df["Adapted_to_Film"]
X = df.drop(columns=["Adapted_to_Film", "Book_Name"])

# Top Publisher & Autoren gruppieren
top_publishers = X["Publisher"].value_counts().nlargest(10).index
X["Publisher"] = X["Publisher"].where(
    X["Publisher"].isin(top_publishers), other="other"
)
top_authors = X["Author"].value_counts().nlargest(10).index
X["Author"] = X["Author"].where(X["Author"].isin(top_authors), other="Sonstige")

# Feature-Listen
numerical_features = [
    "Publishing_Year",
    "Author_Rating",
    "Average_Rating",
    "Rating_Count",
    "Gross_Sales_EUR",
]
categorical_features = ["Language_Code", "Genre", "Publisher", "Author"]

# Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# Vollständige Pipeline
pipeline = Pipeline(
    steps=[
        ("rating_mapper", AuthorRatingMapper()),
        ("preprocessing", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ]
)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Modell trainieren
pipeline.fit(X_train, y_train)

# Wahrscheinlichkeiten berechnen und Klassifikation mit Schwellenwert
threshold = 0.4  # Anpassen möglich
y_proba = pipeline.predict_proba(X_test)[:, 1]
y_pred_threshold = (y_proba >= threshold).astype(int)

# Ergebnis ausgeben
print(f"== Auswertung bei Schwellenwert {threshold} ==")
print(classification_report(y_test, y_pred_threshold))

# Confusion Matrix plotten
cm = confusion_matrix(y_test, y_pred_threshold)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Vorhergesagte Klasse")
plt.ylabel("Tatsächliche Klasse")
plt.title(f"Confusion Matrix (Threshold = {threshold})")
plt.show()

# Testdaten speichern (für Streamlit o. Ä.)
# X_test.to_csv("X_test.csv", index=False)
# y_test.to_csv("y_test.csv", index=False, header=True)

# Neue Vorhersagedaten (optional)
df_pred = df.drop(columns=["Adapted_to_Film"]).copy()
df_pred.to_csv("df_pred.csv", index=False)

# Modell speichern
joblib.dump(pipeline, "logistic_pipeline.pkl")
