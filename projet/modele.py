import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


class Entraineur:
    """Entraîne une forêt aléatoire et rend ses métriques."""

    def __init__(self, arbres: int = 300, profondeur: int | None = None):
        self.parametres = {"arbres": arbres, "profondeur": profondeur}
        self.modele = RandomForestClassifier(
            n_estimators=arbres, max_depth=profondeur, n_jobs=-1, random_state=0,
        )

    def entrainer(self, X, y) -> float:
        debut = time.perf_counter()
        self.modele.fit(X, y)
        return time.perf_counter() - debut

    def evaluer(self, X, y) -> dict:
        prevu = self.modele.predict(X)
        return {"exactitude": accuracy_score(y, prevu), "f1": f1_score(y, prevu)}
