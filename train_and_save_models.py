import os
import glob
import json
import pandas as pd
import numpy as np
import joblib
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from src.features import extraire_indices_url, nettoyer_texte_email, extraire_indices_semantiques_email
from src.model import preparer_et_decouper_donnees, sauvegarder_fichiers_processed, entrainer_modele_rf, entrainer_modele_xgb
from src.evaluation import trouver_seuil_optimal_cyber, calculer_rapport_evaluation_metier, exporter_rapport_markdown

dossier_donnees = "data/raw"
dossier_modeles = "models"
registre_audits = []
metadata_seuils = {}

def purger_anciens_artefacts():
    fichiers = glob.glob(os.path.join(dossier_modeles, "*.joblib"))
    for f in fichiers:
        try: os.remove(f)
        except Exception: pass

def pipeline_urls():
    print("\nPipeline URLs en cours (Calibration & Recherche de Seuil)...")
    df = pd.read_csv(os.path.join(dossier_donnees, "PhiUSIIL_Phishing_URL_Dataset.csv"))
    df['label_cyber'] = df['label'].apply(lambda x: 1 if x == 0 else 0)
    
    X_train_brut, X_test_brut, y_train, y_test = preparer_et_decouper_donnees(df, 'URL', 'label_cyber')
    sauvegarder_fichiers_processed(X_train_brut, X_test_brut, y_train, y_test, "urls")
    
    X_train = pd.DataFrame([extraire_indices_url(url) for url in X_train_brut['URL']])
    X_test = pd.DataFrame([extraire_indices_url(url) for url in X_test_brut['URL']])
    
    modele_rf = entrainer_modele_rf(X_train, y_train)
    modele_xgb = entrainer_modele_xgb(X_train, y_train)
    
    y_proba_rf = modele_rf.predict_proba(X_test)[:, 1]
    y_proba_xgb = modele_xgb.predict_proba(X_test)[:, 1]
    
    # Extraction des seuils optimaux
    seuil_rf, _ = trouver_seuil_optimal_cyber(y_test, y_proba_rf)
    seuil_xgb, _ = trouver_seuil_optimal_cyber(y_test, y_proba_xgb)
    
    metadata_seuils["seuil_url_rf"] = seuil_rf
    metadata_seuils["seuil_url_xgb"] = seuil_xgb
    
    registre_audits.append(calculer_rapport_evaluation_metier(y_test, y_proba_rf, "Random Forest (URLs)", seuil_rf))
    registre_audits.append(calculer_rapport_evaluation_metier(y_test, y_proba_xgb, "XGBoost (URLs)", seuil_xgb))
    
    joblib.dump(modele_rf, os.path.join(dossier_modeles, "url_model_rf.joblib"))
    joblib.dump(modele_xgb, os.path.join(dossier_modeles, "url_model_xgb.joblib"))

def pipeline_emails():
    print("\nPipeline Emails en cours (Calibration & Recherche de Seuil)...")
    df = pd.read_csv(os.path.join(dossier_donnees, "phishings_emails/phishing_email.csv"))
    
    X_train_brut, X_test_brut, y_train, y_test = preparer_et_decouper_donnees(df, 'text_combined', 'label')
    sauvegarder_fichiers_processed(X_train_brut, X_test_brut, y_train, y_test, "emails")
    
    X_train_net = X_train_brut['text_combined'].apply(nettoyer_texte_email)
    X_test_net = X_test_brut['text_combined'].apply(nettoyer_texte_email)
    
    vectoriseur = TfidfVectorizer(max_features=2500, stop_words='english')
    X_train_tfidf = vectoriseur.fit_transform(X_train_net)
    X_test_tfidf = vectoriseur.transform(X_test_net)
    
    X_train_features = pd.DataFrame([extraire_indices_semantiques_email(text) for text in X_train_brut['text_combined']])
    X_test_features = pd.DataFrame([extraire_indices_semantiques_email(text) for text in X_test_brut['text_combined']])
    
    X_train_hybride = hstack([X_train_tfidf, csr_matrix(X_train_features.values)])
    X_test_hybride = hstack([X_test_tfidf, csr_matrix(X_test_features.values)])
    
    modele_rf = entrainer_modele_rf(X_train_hybride, y_train)
    modele_xgb = entrainer_modele_xgb(X_train_hybride, y_train)
    
    y_proba_rf = modele_rf.predict_proba(X_test_hybride)[:, 1]
    y_proba_xgb = modele_xgb.predict_proba(X_test_hybride)[:, 1]
    
    # Extraction des seuils optimaux
    seuil_rf, _ = trouver_seuil_optimal_cyber(y_test, y_proba_rf)
    seuil_xgb, _ = trouver_seuil_optimal_cyber(y_test, y_proba_xgb)
    
    metadata_seuils["seuil_email_rf"] = seuil_rf
    metadata_seuils["seuil_email_xgb"] = seuil_xgb
    
    registre_audits.append(calculer_rapport_evaluation_metier(y_test, y_proba_rf, "Random Forest (Emails)", seuil_rf))
    registre_audits.append(calculer_rapport_evaluation_metier(y_test, y_proba_xgb, "XGBoost (Emails)", seuil_xgb))
    
    joblib.dump(vectoriseur, os.path.join(dossier_modeles, "email_vectoriseur.joblib"))
    joblib.dump(modele_rf, os.path.join(dossier_modeles, "email_model_rf.joblib"))
    joblib.dump(modele_xgb, os.path.join(dossier_modeles, "email_model_xgb.joblib"))

if __name__ == "__main__":
    purger_anciens_artefacts()
    pipeline_urls()
    pipeline_emails()
    
    # Enregistrement permanent des seuils optimaux calibrés dans metadata.json
    chemin_meta = os.path.join(dossier_modeles, "metadata.json")
    meta_structure = {
        "project_name": "multi-vector-phishing-detection",
        "environment": {"python_version": "3.12", "scikit_learn_version": "1.5.0", "xgboost_version": "2.0.3"},
        "decision_thresholds_calibrated": metadata_seuils
    }
    with open(chemin_meta, "w") as f:
        json.dump(meta_structure, f, indent=2)
        
    exporter_rapport_markdown(registre_audits)
    print("\n[MLOps] Pipeline de Calibration complet compile avec succes.")
