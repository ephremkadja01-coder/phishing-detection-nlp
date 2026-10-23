import os
import sys
import joblib

# Remontée d'un cran pour cibler le dossier models depuis le sous-dossier tests
dossier_modeles = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

def verifier_fichiers():
    fichiers = [
        "url_model_rf.joblib",
        "url_model_xgb.joblib",
        "email_vectoriseur.joblib",
        "email_model_rf.joblib",
        "email_model_xgb.joblib"
    ]
    
    correct = True
    print("Vérification des composants d'inférence en production :")
    
    for fichier in fichiers:
        chemin = os.path.join(dossier_modeles, fichier)
        if not os.path.exists(chemin):
            print(f"[MANQUANT] : {fichier}")
            correct = False
            continue
        try:
            joblib.load(chemin)
            print(f"[VALIDE]   : {fichier}")
        except Exception as e:
            print(f"[CORROMPU] : {fichier} -> {str(e)}")
            correct = False
            
    if correct:
        print("\nSuccès : Tous les fichiers sont validés pour la production.")
        sys.exit(0)
    else:
        print("\nErreur : Le protocole d'intégrité a échoué.")
        sys.exit(1)

if __name__ == "__main__":
    verifier_fichiers()
