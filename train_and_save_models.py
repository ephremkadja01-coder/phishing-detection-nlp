import os
import glob
import pandas as pd
import numpy as np
import joblib
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from src.features import extraire_indices_url, nettoyer_texte_email, extraire_indices_semantiques_email
from src.model import (
    preparer_et_decouper_donnees, 
    sauvegarder_fichiers_processed, 
    entrainer_modele_rf, 
    entrainer_modele_xgb
)
from src.evaluation import calculer_rapport_evaluation_metier, exporter_rapport_markdown

dossier_donnees = "data/raw"
dossier_modeles = "models"
registre_audits = []

def purger_anciens_artefacts():
    """
    Purge automatiquement les anciens fichiers modeles .joblib obsoletes (Priorite 2).
    """
    print("[MLOps] Nettoyage et purge du dossier models/...")
    fichiers = glob.glob(os.path.join(dossier_modeles, "*.joblib"))
    for f in fichiers:
        try:
            os.remove(f)
            print(f"|- Supprimé : {os.path.basename(f)}")
        except Exception as e:
            print(f"|- Impossible de supprimer {os.path.basename(f)} : {str(e)}")

def pipeline_urls():
    print("\nPipeline URLs en cours (Calcul des métriques asymétriques)...")
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
    
    y_proba_rf = modele_rf.predict_proba(X_test)[:, 1]
    y_proba_xgb = modele_xgb.predict_proba(X_test)[:, 1]
    
    registre_audits.append(calculer_rapport_evaluation_metier(y_test, y_proba_rf, "Random Forest (URLs)"))
    registre_audits.append(calculer_rapport_evaluation_metier(y_test, y_proba_xgb, "XGBoost (URLs)"))
    
    joblib.dump(modele_rf, os.path.join(dossier_modeles, "url_model_rf.joblib"))
    joblib.dump(modele_xgb, os.path.join(dossier_modeles, "url_model_xgb.joblib"))
    print("Modèles URLs entraînés et enregistrés.")

def pipeline_emails():
    print("\nPipeline Emails en cours (Stratégie Hybride Texte + Signaux Cyber)...")
    chemin_csv = os.path.join(dossier_donnees, "phishings_emails/phishing_email.csv")
    
    if not os.path.exists(chemin_csv):
        raise FileNotFoundError(f"Erreur critique : Fichier introuvable : {chemin_csv}")
        
    df = pd.read_csv(chemin_csv)
    
    X_train_brut, X_test_brut, y_train, y_test = preparer_et_decouper_donnees(
        df, colonne_texte='text_combined', colonne_cible='label'
    )
    
    sauvegarder_fichiers_processed(X_train_brut, X_test_brut, y_train, y_test, "emails")
    
    print("Vectorisation TF-IDF sur le texte nettoyé conservateur...")
    X_train_net = X_train_brut['text_combined'].apply(nettoyer_texte_email)
    X_test_net = X_test_brut['text_combined'].apply(nettoyer_texte_email)
    
    vectoriseur = TfidfVectorizer(max_features=2500, stop_words='english')
    X_train_tfidf = vectoriseur.fit_transform(X_train_net)
    X_test_tfidf = vectoriseur.transform(X_test_net)
    
    print("Extraction des indicateurs sémantiques explicites (Majuscules, !, ?, Urgences)...")
    X_train_features = pd.DataFrame([extraire_indices_semantiques_email(text) for text in X_train_brut['text_combined']])
    X_test_features = pd.DataFrame([extraire_indices_semantiques_email(text) for text in X_test_brut['text_combined']])
    
    # Assemblage horizontal des matrices (TF-IDF + Signaux Cyber)
    X_train_hybride = hstack([X_train_tfidf, csr_matrix(X_train_features.values)])
    X_test_hybride = hstack([X_test_tfidf, csr_matrix(X_test_features.values)])
    
    modele_rf = entrainer_modele_rf(X_train_hybride, y_train)
    modele_xgb = entrainer_modele_xgb(X_train_hybride, y_train)
    
    y_proba_rf = modele_rf.predict_proba(X_test_hybride)[:, 1]
    y_proba_xgb = modele_xgb.predict_proba(X_test_hybride)[:, 1]
    
    registre_audits.append(calculer_rapport_evaluation_metier(y_test, y_proba_rf, "Random Forest (Emails)"))
    registre_audits.append(calculer_rapport_evaluation_metier(y_test, y_proba_xgb, "XGBoost (Emails)"))
    
    joblib.dump(vectoriseur, os.path.join(dossier_modeles, "email_vectoriseur.joblib"))
    joblib.dump(modele_rf, os.path.join(dossier_modeles, "email_model_rf.joblib"))
    joblib.dump(modele_xgb, os.path.join(dossier_modeles, "email_model_xgb.joblib"))
    print("Modèles E-mails hybrides entraînés et enregistrés.")

if __name__ == "__main__":
    os.makedirs(dossier_modeles, exist_ok=True)
    
    # 1. Purge propre automatique des anciens artefacts
    purger_anciens_artefacts()
    
    # 2. Execution des entrainements
    pipeline_urls()
    pipeline_emails()
    
    # 3. Restitution du rapport d'audit
    exporter_rapport_markdown(registre_audits)
    print("\n[MLOps] Entraînement du pipeline hybride complété avec succès.")
