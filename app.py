import os
import joblib
from src.features import extraire_indices_url, nettoyer_texte_email
from src.pipeline import verifier_menace_url, verifier_menace_email, fusionner_vecteurs_menace

dossier_modeles = "models"

def charger_fichiers():
    mod_url = joblib.load(os.path.join(dossier_modeles, "url_model_xgb.joblib"))
    vec_email = joblib.load(os.path.join(dossier_modeles, "email_vectoriseur.joblib"))
    mod_email = joblib.load(os.path.join(dossier_modeles, "email_model_xgb.joblib"))
    return mod_url, vec_email, mod_email

def lancer_detecteur():
    print("=== SYSTEME DE DETECTION DU PHISHING INDUSTRIEL ===")
    if not os.path.exists(os.path.join(dossier_modeles, "url_model_xgb.joblib")):
        print("Erreur : Les modèles de production ne sont pas compilés.")
        return

    mod_url, vec_email, mod_email = charger_fichiers()
    
    print("\n1 : Tester une URL seule")
    print("2 : Tester le texte d'un e-mail seul")
    print("3 : Tester les deux en corrélation (Fusion Multi-Modale)")
    choix = input("Votre choix (1, 2 ou 3) : ")
    
    if choix == "1":
        url = input("Entrez l'URL : ")
        _, prob = verifier_menace_url(url, mod_url, extraire_indices_url)
        print(f"\nScore de menace URL : {prob:.2f}%")
        
    elif choix == "2":
        email = input("Entrez le texte de l'e-mail : ")
        _, prob = verifier_menace_email(email, vec_email, mod_email, nettoyer_texte_email)
        print(f"\nScore de menace E-mail : {prob:.2f}%")
        
    elif choix == "3":
        url = input("Entrez l'URL du lien contenu : ")
        email = input("Entrez le texte complet du mail : ")
        
        # Calcul individuel des canaux
        _, prob_url = verifier_menace_url(url, mod_url, extraire_indices_url)
        _, prob_email = verifier_menace_email(email, vec_email, mod_email, nettoyer_texte_email)
        
        # Corrélation asymétrique via le module centralise
        pred_fuse, score_fuse = fusionner_vecteurs_menace(prob_url, prob_email)
        
        print("\n--- RAPPORT DE FUSION CYBER ---")
        print(f"Statut Global : {'⚠️ PHISHING DETECTE' if pred_fuse == 1 else '✅ LEGITIME'}")
        print(f"Score Combiné : {score_fuse:.2f}%")
        print(f"|- Composant URL    : {prob_url:.2f}%")
        print(f"|- Composant E-mail : {prob_email:.2f}%")
        print("-------------------------------")

if __name__ == "__main__":
    lancer_detecteur()
