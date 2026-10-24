import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from src.features import extraire_indices_url, nettoyer_texte_email
from src.model import (
    preparer_et_decouper_donnees, 
    sauvegarder_fichiers_processed, 
    entrainer_modele_xgb, 
    entrainer_modele_rf
)
from src.evaluation import calculer_rapport_evaluation_metier, exporter_rapport_markdown

dossier_donnees = "data/raw"
dossier_modeles = "models"

# Liste globale pour compiler l'audit final
registre_audits = []

def pipeline_urls():
    print("Pipeline URLs en cours (Calcul des métriques asymétriques)...")
    chemin_csv = os.path.join(dossier_donnees, "PhiUSIIL_Phishing_URL_Dataset.csv")
    
    if not os.path.exists(chemin_csv):
        raise FileNotFoundError(f"Erreur critique : Fichier introuvable : {chemin_csv}")
        
    df = pd.read_csv(chemin_csv)
    df['label_cyber'] = df['label'].apply(lambda x: 1 if x == 0 else 0)
    
    X_train_brut, X_test_brut, y_train, y_test = preparer_et_decouper_donnees(
        df, colonne_texte='URL', colonne_cible='label_cyber'
    )
    
    sauvegarder_fichiers_processed(X_train_brut, X_test_brut, y_train, y_test, "urls")
    
    X_train = pd.DataFrame([extraire_indices_url(url) for url in X_train_brut['URL']])
    X_test = pd.DataFrame([extraire_indices_url(url) for url in X_test_brut['URL']])
    
    modele_rf = entrainer_modele_rf(X_train, y_train)
    modele_xgb = entrainer_modele_xgb(X_train, y_train)
    
    # Extraction des probabilités réelles de phishing (Classe 1)
    y_proba_rf = modele_rf.predict_proba(X_test)[:, 1]
    y_proba_xgb = modele_xgb.predict_proba(X_test)[:, 1]
    
    # Calcul des rapports d'audit avancés
    registre_audits.append(calculer_rapport_evaluation_metier(y_test, y_proba_rf, "Random Forest (URLs)"))
    registre_audits.append(calculer_rapport_evaluation_metier(y_test, y_proba_xgb, "XGBoost (URLs)"))
    
    joblib.dump(modele_rf, os.path.join(dossier_modeles, "url_model_rf.joblib"))
    joblib.dump(modele_xgb, os.path.join(dossier_modeles, "url_model_xgb.joblib"))
    print("Modèles URLs entraînés et enregistrés.")

def pipeline_emails():
    print("\nPipeline Emails en cours (Calcul des métriques asymétriques)...")
    chemin_csv = os.path.join(dossier_donnees, "phishings_emails/phishing_email.csv")
    
    if not os.path.exists(chemin_csv):
        raise FileNotFoundError(f"Erreur critique : Fichier introuvable : {chemin_csv}")
        
    df = pd.read_csv(chemin_csv)
    
    print("Pré-nettoyage textuel du corpus complet...")
    df['texte_net'] = df['text_combined'].apply(nettoyer_texte_email)
    
    X_train_brut, X_test_brut, y_train, y_test = preparer_et_decouper_donnees(
        df, colonne_texte='texte_net', colonne_cible='label'
    )
    
    sauvegarder_fichiers_processed(X_train_brut, X_test_brut, y_train, y_test, "emails")
    
    vectoriseur = TfidfVectorizer(max_features=2500, stop_words='english')
    X_train_vectorise = vectoriseur.fit_transform(X_train_brut['texte_net'])
    X_test_vectorise = vectoriseur.transform(X_test_brut['texte_net'])
    
    modele_rf = entrainer_modele_rf(X_train_vectorise, y_train)
    modele_xgb = entrainer_modele_xgb(X_train_vectorise, y_train)
    
    # Extraction des probabilités réelles de phishing (Classe 1)
    y_proba_rf = modele_rf.predict_proba(X_test_vectorise)[:, 1]
    y_proba_xgb = modele_xgb.predict_proba(X_test_vectorise)[:, 1]
    
    # Calcul des rapports d'audit avancés
    registre_audits.append(calculer_rapport_evaluation_metier(y_test, y_proba_rf, "Random Forest (Emails)"))
    registre_audits.append(calculer_rapport_evaluation_metier(y_test, y_proba_xgb, "XGBoost (Emails)"))
    
    joblib.dump(vectoriseur, os.path.join(dossier_modeles, "email_vectoriseur.joblib"))
    joblib.dump(modele_rf, os.path.join(dossier_modeles, "email_model_rf.joblib"))
    joblib.dump(modele_xgb, os.path.join(dossier_modeles, "email_model_xgb.joblib"))
    print("Modèles E-mails entraînés et enregistrés.")

if __name__ == "__main__":
    os.makedirs(dossier_modeles, exist_ok=True)
    
    # Exécution des pipelines d'apprentissage
    pipeline_urls()
    pipeline_emails()
    
    # Génération automatique du rapport de gouvernance
    exporter_rapport_markdown(registre_audits)
    print("\n[MLOps] Processus complet terminé. Le cadre d'évaluation d'élite est actif.")
