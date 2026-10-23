import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import recall_score, f1_score
from src.features import extraire_indices_url, nettoyer_texte_email
from src.model import (
    preparer_et_decouper_donnees, 
    sauvegarder_fichiers_processed, 
    entrainer_modele_xgb, 
    entrainer_modele_rf
)

dossier_donnees = "data/raw"
dossier_modeles = "models"

def evaluer_et_afficher_scores(modele, X_test, y_test, nom_modele):
    predictions = modele.predict(X_test)
    rappel = recall_score(y_test, predictions, pos_label=1) * 100
    f1 = f1_score(y_test, predictions, average='macro') * 100
    print(f"[{nom_modele}] Rappel Phishing : {rappel:.2f}% | Macro-F1 : {f1:.2f}%")

def pipeline_urls():
    print("Pipeline URLs en cours...")
    chemin_csv = os.path.join(dossier_donnees, "PhiUSIIL_Phishing_URL_Dataset.csv")
    
    if not os.path.exists(chemin_csv):
        raise FileNotFoundError(f"Erreur critique : Le fichier requis est introuvable : {chemin_csv}")
        
    df = pd.read_csv(chemin_csv)
    
    df['label_cyber'] = df['label'].apply(lambda x: 1 if x == 0 else 0)
    
    X_train_brut, X_test_brut, y_train, y_test = preparer_et_decouper_donnees(
        df, colonne_texte='URL', colonne_cible='label_cyber'
    )
    
    # SAUVEGARDE DES FILES CSV TEST / TRAIN EN LOCAL
    sauvegarder_fichiers_processed(X_train_brut, X_test_brut, y_train, y_test, "urls")
    
    X_train = pd.DataFrame([extraire_indices_url(url) for url in X_train_brut['URL']])
    X_test = pd.DataFrame([extraire_indices_url(url) for url in X_test_brut['URL']])
    
    modele_rf = entrainer_modele_rf(X_train, y_train)
    modele_xgb = entrainer_modele_xgb(X_train, y_train)
    
    evaluer_et_afficher_scores(modele_rf, X_test, y_test, "Random Forest (URLs)")
    evaluer_et_afficher_scores(modele_xgb, X_test, y_test, "XGBoost (URLs)")
    
    joblib.dump(modele_rf, os.path.join(dossier_modeles, "url_model_rf.joblib"))
    joblib.dump(modele_xgb, os.path.join(dossier_modeles, "url_model_xgb.joblib"))
    print("Modeles URLs sauvegardes.")

def pipeline_emails():
    print("\nPipeline Emails en cours...")
    chemin_csv = os.path.join(dossier_donnees, "phishings_emails/phishing_email.csv")
    
    if not os.path.exists(chemin_csv):
        raise FileNotFoundError(f"Erreur critique : Le fichier requis est introuvable : {chemin_csv}")
        
    df = pd.read_csv(chemin_csv)
    X_train_brut, X_test_brut, y_train, y_test = preparer_et_decouper_donnees(
        df, colonne_texte='text_combined', colonne_cible='label'
    )
    
    # SAUVEGARDE DES FILES CSV TEST / TRAIN EN LOCAL
    sauvegarder_fichiers_processed(X_train_brut, X_test_brut, y_train, y_test, "emails")
    
    X_train_nettoye = X_train_brut['text_combined'].apply(nettoyer_texte_email)
    X_test_nettoye = X_test_brut['text_combined'].apply(nettoyer_texte_email)
    
    vectoriseur = TfidfVectorizer(max_features=2500, stop_words='english')
    X_train_vectorise = vectoriseur.fit_transform(X_train_nettoye)
    X_test_vectorise = vectoriseur.transform(X_test_nettoye)
    
    modele_rf = entrainer_modele_rf(X_train_vectorise, y_train)
    modele_xgb = entrainer_modele_xgb(X_train_vectorise, y_train)
    
    evaluer_et_afficher_scores(modele_rf, X_test_vectorise, y_test, "Random Forest (Emails)")
    evaluer_et_afficher_scores(modele_xgb, X_test_vectorise, y_test, "XGBoost (Emails)")
    
    joblib.dump(vectoriseur, os.path.join(dossier_modeles, "email_vectoriseur.joblib"))
    joblib.dump(modele_rf, os.path.join(dossier_modeles, "email_model_rf.joblib"))
    joblib.dump(modele_xgb, os.path.join(dossier_modeles, "email_model_xgb.joblib"))
    print("Modeles Emails sauvegardes.")

if __name__ == "__main__":
    os.makedirs(dossier_modeles, exist_ok=True)
    pipeline_urls()
    pipeline_emails()
    print("\nEntrainement, division CSV et evaluation termines avec succes.")
