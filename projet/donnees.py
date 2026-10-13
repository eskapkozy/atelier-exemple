from dataclasses import dataclass

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


@dataclass
class JeuDeDonnees:
    """Un jeu synthétique, assez gros pour que l'entraînement compte."""

    lignes: int = 200_000
    colonnes: int = 40
    graine: int = 42

    def decouper(self):
        X, y = make_classification(
            n_samples=self.lignes, n_features=self.colonnes,
            n_informative=12, random_state=self.graine,
        )
        return train_test_split(X, y, test_size=0.2, random_state=self.graine)
