import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from src.features import extraire_indices_url, nettoyer_texte_email
from src.model import preparer_et_decouper_donnees, entrainer_modele_xgb, entrainer_modele_rf

dossier_donnees = "data/raw"
dossier_modeles = "models"

def pipeline_urls():
    print("Pipeline URLs en cours...")
    chemin_csv = os.path.join(dossier_donnees, "PhiUSIIL_Phishing_URL_Dataset.csv")
    
    if not os.path.exists(chemin_csv):
        print(f"Fichier introuvable : {chemin_csv}")
        return
        
    df = pd.read_csv(chemin_csv)
    X_train_brut, X_test_brut, y_train, y_test = preparer_et_decouper_donnees(
        df, colonne_texte='URL', colonne_cible='label'
    )
    
    # Transformation des URLs brutes en tableaux de nombres
    X_train = pd.DataFrame([extraire_indices_url(url) for url in X_train_brut])
    
    modele_rf = entrainer_modele_rf(X_train, y_train)
    modele_xgb = entrainer_modele_xgb(X_train, y_train)
    
    joblib.dump(modele_rf, os.path.join(dossier_modeles, "url_model_rf.joblib"))
    joblib.dump(modele_xgb, os.path.join(dossier_modeles, "url_model_xgb.joblib"))
    print("Modeles URLs sauvegardes.")

def pipeline_emails():
    print("Pipeline Emails en cours...")
    chemin_csv = os.path.join(dossier_donnees, "phishings_emails/phishing_email.csv")
    
    if not os.path.exists(chemin_csv):
        print(f"Fichier introuvable : {chemin_csv}")
        return
        
    df = pd.read_csv(chemin_csv)
    X_train_brut, X_test_brut, y_train, y_test = preparer_et_decouper_donnees(
        df, colonne_texte='text_combined', colonne_cible='label'
    )
    
    # Nettoyage et transformation du texte des emails
    X_train_nettoye = X_train_brut.apply(nettoyer_texte_email)
    vectoriseur = TfidfVectorizer(max_features=2500, stop_words='english')
    X_train_vectorise = vectoriseur.fit_transform(X_train_nettoye)
    
    modele_rf = entrainer_modele_rf(X_train_vectorise, y_train)
    modele_xgb = entrainer_modele_xgb(X_train_vectorise, y_train)
    
    joblib.dump(vectoriseur, os.path.join(dossier_modeles, "email_vectoriseur.joblib"))
    joblib.dump(modele_rf, os.path.join(dossier_modeles, "email_model_rf.joblib"))
    joblib.dump(modele_xgb, os.path.join(dossier_modeles, "email_model_xgb.joblib"))
    print("Modeles Emails sauvegardes.")

if __name__ == "__main__":
    os.makedirs(dossier_modeles, exist_ok=True)
    pipeline_urls()
    pipeline_emails()
    print("Entrainement global termine.")
