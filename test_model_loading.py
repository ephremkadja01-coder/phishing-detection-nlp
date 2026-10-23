import os
import joblib

dossier_modeles = "models"

def verifier_fichiers():
    fichiers = [
        "url_model_rf.joblib",
        "url_model_xgb.joblib",
        "email_vectoriseur.joblib",
        "email_model_rf.joblib",
        "email_model_xgb.joblib"
    ]
    
    correct = True
    print("Verification des fichiers sauvegardes :")
    
    for fichier in fichiers:
        chemin = os.path.join(dossier_modeles, fichier)
        if not os.path.exists(chemin):
            print(f"Manquant : {fichier}")
            correct = False
            continue
        try:
            joblib.load(chemin)
            print(f"OK : {fichier}")
        except Exception as e:
            print(f"Erreur sur {fichier} : {str(e)}")
            correct = False
            
    if correct:
        print("Tous les fichiers sont prets pour la production.")
    else:
        print("Erreur : Des fichiers manquent ou sont corrompus.")

if __name__ == "__main__":
    verifier_fichiers()
