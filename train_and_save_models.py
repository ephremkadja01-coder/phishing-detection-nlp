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
from src.drift import detecter_drift_predictions
from src.fairness import calculer_metriques_equite

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
    print("\n[MLOps] Pipeline URLs (Split Temporel, Calibration & Recherche de Seuil)...")
    df = pd.read_csv(os.path.join(dossier_donnees, "PhiUSIIL_Phishing_URL_Dataset.csv"))
    df['label_cyber'] = df['label'].apply(lambda x: 1 if x == 0 else 0)
    
    df = df.sort_index()
    split_idx = int(len(df) * 0.8)
    df_train, df_test = df.iloc[:split_idx], df.iloc[split_idx:]
    
    X_train = pd.DataFrame([extraire_indices_url(url) for url in df_train['URL']])
    X_test = pd.DataFrame([extraire_indices_url(url) for url in df_test['URL']])
    y_train, y_test = df_train['label_cyber'], df_test['label_cyber']
    
    modele_rf = entrainer_modele_rf(X_train, y_train)
    modele_xgb = entrainer_modele_xgb(X_train, y_train)
    
    y_proba_rf = modele_rf.predict_proba(X_test)[:, 1]
    y_proba_xgb = modele_xgb.predict_proba(X_test)[:, 1]
    
    seuil_rf, _ = trouver_seuil_optimal_cyber(y_test, y_proba_rf)
    seuil_xgb, _ = trouver_seuil_optimal_cyber(y_test, y_proba_xgb)
    
    metadata_seuils["seuil_url_rf"] = seuil_rf
    metadata_seuils["seuil_url_xgb"] = seuil_xgb
    
    registre_audits.append(calculer_rapport_evaluation_metier(y_test, y_proba_rf, "Random Forest (URLs)", seuil_rf))
    registre_audits.append(calculer_rapport_evaluation_metier(y_test, y_proba_xgb, "XGBoost (URLs)", seuil_xgb))
    
    joblib.dump(modele_rf, os.path.join(dossier_modeles, "url_model_rf.joblib"))
    joblib.dump(modele_xgb, os.path.join(dossier_modeles, "url_model_xgb.joblib"))
    print("Modèles URLs entraînés et enregistrés.")

def pipeline_emails():
    print("\n[MLOps] Pipeline Emails (Split Temporel, Drift & Fairness)...")
    df = pd.read_csv(os.path.join(dossier_donnees, "phishings_emails/phishing_email.csv"))
    
    df = df.sort_index()
    split_idx = int(len(df) * 0.8)
    df_train, df_test = df.iloc[:split_idx], df.iloc[split_idx:]
    
    X_train_net = df_train['text_combined'].apply(nettoyer_texte_email)
    X_test_net = df_test['text_combined'].apply(nettoyer_texte_email)
    
    vectoriseur = TfidfVectorizer(max_features=2500, stop_words='english')
    X_train_tfidf = vectoriseur.fit_transform(X_train_net)
    X_test_tfidf = vectoriseur.transform(X_test_net)
    
    X_train_features = pd.DataFrame([extraire_indices_semantiques_email(text) for text in df_train['text_combined']])
    X_test_features = pd.DataFrame([extraire_indices_semantiques_email(text) for text in df_test['text_combined']])
    
    X_train_hybride = hstack([X_train_tfidf, csr_matrix(X_train_features.values)])
    X_test_hybride = hstack([X_test_tfidf, csr_matrix(X_test_features.values)])
    y_train, y_test = df_train['label'], df_test['label']
    
    modele_rf = entrainer_modele_rf(X_train_hybride, y_train)
    modele_xgb = entrainer_modele_xgb(X_train_hybride, y_train)
    
    y_proba_rf = modele_rf.predict_proba(X_test_hybride)[:, 1]
    y_proba_xgb = modele_xgb.predict_proba(X_test_hybride)[:, 1]
    
    seuil_rf, _ = trouver_seuil_optimal_cyber(y_test, y_proba_rf)
    seuil_xgb, _ = trouver_seuil_optimal_cyber(y_test, y_proba_xgb)
    
    metadata_seuils["seuil_email_rf"] = seuil_rf
    metadata_seuils["seuil_email_xgb"] = seuil_xgb
    
    registre_audits.append(calculer_rapport_evaluation_metier(y_test, y_proba_rf, "Random Forest (Emails)", seuil_rf))
    registre_audits.append(calculer_rapport_evaluation_metier(y_test, y_proba_xgb, "XGBoost (Emails)", seuil_xgb))
    
    print("[MLOps Audit] Analyse du Drift statistique sur les prédictions...")
    probas_ref = y_proba_xgb[:len(y_proba_xgb)//2]
    probas_prod = y_proba_xgb[len(y_proba_xgb)//2:]
    rapport_drift = detecter_drift_predictions(probas_ref, probas_prod)
    
    print("[MLOps Audit] Évaluation de l'Équité algorithmique (Fairness Framework)...")
    y_pred_xgb = (y_proba_xgb >= seuil_xgb).astype(int)
    has_urgency = (X_test_features['has_urgency_phrase'] == 1).values
    has_no_urgency = (X_test_features['has_urgency_phrase'] == 0).values
    rapport_fairness = calculer_metriques_equite(y_test.values, y_pred_xgb, groupe_privilegie=has_no_urgency, groupe_protege=has_urgency)
    
    with open(os.path.join(dossier_modeles, "audit_gouvernance.json"), "w") as f:
        json.dump({"drift": rapport_drift, "fairness": rapport_fairness}, f, indent=2)
    
    joblib.dump(vectoriseur, os.path.join(dossier_modeles, "email_vectoriseur.joblib"))
    joblib.dump(modele_rf, os.path.join(dossier_modeles, "email_model_rf.joblib"))
    joblib.dump(modele_xgb, os.path.join(dossier_modeles, "email_model_xgb.joblib"))
    print("Modèles E-mails entraînés et enregistrés.")

if __name__ == "__main__":
    purger_anciens_artefacts()
    pipeline_urls()
    pipeline_emails()
    
    chemin_meta = os.path.join(dossier_modeles, "metadata.json")
    meta_structure = {
        "project_name": "multi-vector-phishing-detection",
        "environment": {"python_version": "3.12", "scikit_learn_version": "1.5.0", "xgboost_version": "2.0.3"},
        "decision_thresholds_calibrated": metadata_seuils
    }
    with open(chemin_meta, "w") as f:
        json.dump(meta_structure, f, indent=2)
        
    exporter_rapport_markdown(registre_audits)
    print("\n[MLOps] Pipeline d'Élite avec Split Temporel, Drift & Fairness complété.")
