import os
import joblib
import pandas as pd
from src.features import extraire_indices_url, nettoyer_texte_email

dossier_modeles = "models"

def charger_fichiers():
    mod_url = joblib.load(os.path.join(dossier_modeles, "url_model_xgb.joblib"))
    vec_email = joblib.load(os.path.join(dossier_modeles, "email_vectoriseur.joblib"))
    mod_email = joblib.load(os.path.join(dossier_modeles, "email_model_xgb.joblib"))
    return mod_url, vec_email, mod_email

def lancer_detecteur():
    print("Detecteur de phishing pret.")
    chemin_verif = os.path.join(dossier_modeles, "url_model_xgb.joblib")
    
    if not os.path.exists(chemin_verif):
        print("Erreur : Lancez d'abord train_and_save_models.py pour creer les modeles.")
        return

    mod_url, vec_email, mod_email = charger_fichiers()
    
    print("1 : Tester une URL")
    print("2 : Tester un texte d'email")
    choix = input("Votre choix : ")
    
    if choix == "1":
        url = input("Entrez l'URL : ")
        indices = pd.DataFrame([extraire_indices_url(url)])
        prediction = mod_url.predict(indices)[0]
        probabilite = mod_url.predict_proba(indices)[0][1] * 100
        
        if prediction == 1:
            print(f"ALERT: Phishing Detected (Probability: {probabilite:.2f}%)")
        else:
            print(f"SECURE: Legitimate Link (Probability: {probabilite:.2f}%)")
            
    elif choix == "2":
        email = input("Entrez le texte de l'email : ")
        texte_nettoye = nettoyer_texte_email(email)
        vecteur = vec_email.transform([texte_nettoye])
        prediction = mod_email.predict(vecteur)[0]
        probabilite = mod_email.predict_proba(vecteur)[0][1] * 100
        
        if prediction == 1:
            print(f"ALERT: Phishing Detected (Probability: {probabilite:.2f}%)")
        else:
            print(f"SECURE: Legitimate Email (Probability: {probabilite:.2f}%)")

if __name__ == "__main__":
    lancer_detecteur()
