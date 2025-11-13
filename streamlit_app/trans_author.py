# Basis klassen importieren, für Transformer aus scikit-learn
from sklearn.base import BaseEstimator, TransformerMixin


# Classe individuel definieren für Transformer. Classe ist kompatible mit Pipleleine
class AuthorRatingMapper(BaseEstimator, TransformerMixin):
    """Transformer zur Umwandlung Autor-Rating(Kat) in numerische, um Model die verarbeiten kann"""

    def __init__(self):
        # initialisierung des Mappimhs von text zu Zahl
        self.rating_map = {"Novice": 1, "Intermediate": 2, "Famous": 3, "Excellent": 4}

    def fit(self, X, y=None):
        # nur für kompatibilität mit sckit-learn
        return self

    def transform(self, X):
        # funktion transform macht eigentliche Umwandlung
        X = X.copy()  # kopie von x
        if "Author_Rating" in X.columns:
            # wird rating mit hilfe von dict in Zahlen umgewandelt
            X["Author_Rating"] = X["Author_Rating"].map(self.rating_map)
        return X
