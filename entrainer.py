"""Le fichier principal : c'est lui que « calcul lancer entrainer.py » envoie sur Colab."""
import json
import os

from projet.donnees import JeuDeDonnees
from projet.modele import Entraineur

X_train, X_test, y_train, y_test = JeuDeDonnees().decouper()
entraineur = Entraineur(arbres=300)
duree = entraineur.entrainer(X_train, y_train)

print(f"Machine : {os.cpu_count()} cœurs")
print(f"Entraînement : {duree:.1f} s")
print("Métriques :", json.dumps(entraineur.evaluer(X_test, y_test), indent=2))
