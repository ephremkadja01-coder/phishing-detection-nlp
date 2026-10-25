import os
import glob
import json
import logging
from scipy.sparse import hstack, csr_matrix
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from src.features import extraire_indices_url, nettoyer_texte_email, extraire_indices_semantiques_email
from src.model import entrainer_modele_rf, entrainer_modele_xgb
from src.evaluation import trouver_seuil_optimal_cyber, calculer_rapport_evaluation_metier, exporter_rapport_markdown
from src.drift import detecter_drift_predictions
from src.fairness import calculer_metriques_equite

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")

DIR_RAW = "data/raw"
DIR_MODELS = "models"

def purger_anciens_artefacts():
    # Evite les chargements fantômes ou conflits de versions locales .joblib
    for f in glob.glob(os.path.join(DIR_MODELS, "*.joblib")):
        try:
            os.remove(f)
        except OSError as e:
            logging.warning(f"Impossible de virer {f} : {e}")

def executer_pipeline_vecteur(nom_pipeline, chemin_csv, col_texte, col_cible, fn_extraction, is_email=False):
    logging.info(f"Demarrage pipeline {nom_pipeline} (Anti-Leakage & Split Stratifie)")
    
    # FIXME: Pas de logs sur fichier volumineux pour l'instant, verifier la RAM dispo en prod locale
    df = pd.read_csv(chemin_csv)
    
    if not is_email:
        df['label_cyber'] = df[col_cible].apply(lambda x: 1 if x == 0 else 0)
        col_cible = 'label_cyber'
        
    assert df[col_cible].isin([0, 1]).all(), f"Labels invalides decouverts dans {nom_pipeline}"

    # [PRIORITÉ 4] Élimination stricte des doublons avant split pour bloquer le leakage massif
    taille_brute = len(df)
    df = df.drop_duplicates(subset=[col_texte]).reset_index(drop=True)
    logging.info(f"|- {nom_pipeline} : {taille_brute - len(df)} doublons vires du dataset.")

    # [PRIORITÉ 1] Pas de donnees de temps fiables -> Split Alatoire Stratifie strict pour l'equilibre des classes
    df_train, df_test = train_test_split(
        df, test_size=0.2, stratify=df[col_cible], random_state=42
    )
    
    # TODO: Refactoriser le calcul des features en multi-processing si le volume explose
    if not is_email:
        X_train = pd.DataFrame([fn_extraction(url) for url in df_train[col_texte]])
        X_test = pd.DataFrame([fn_extraction(url) for url in df_test[col_texte]])
        X_train_final, X_test_final = X_train, X_test
    else:
        # Pipeline specifique e-mail hybride
        X_train_net = df_train[col_texte].apply(nettoyer_texte_email)
        X_test_net = df_test[col_texte].apply(nettoyer_texte_email)
        
        vectoriseur = TfidfVectorizer(max_features=2500, stop_words='english')
        X_train_tfidf = vectoriseur.fit_transform(X_train_net)
        X_test_tfidf = vectoriseur.transform(X_test_net)
        
        X_train_feats = pd.DataFrame([extraire_indices_semantiques_email(text) for text in df_train[col_texte]])
        X_test_feats = pd.DataFrame([extraire_indices_semantiques_email(text) for text in df_test[col_texte]])
        
        X_train_final = hstack([X_train_tfidf, csr_matrix(X_train_feats.values)])
        X_test_final = hstack([X_test_tfidf, csr_matrix(X_test_feats.values)])
        joblib.dump(vectoriseur, os.path.join(DIR_MODELS, "email_vectoriseur.joblib"))

    y_train, y_test = df_train[col_cible].values, df_test[col_cible].values
    
    modele_rf = entrainer_modele_rf(X_train_final, y_train)
    modele_xgb = entrainer_modele_xgb(X_train_final, y_train)
    
    y_proba_rf = modele_rf.predict_proba(X_test_final)[:, 1]
    y_proba_xgb = modele_xgb.predict_proba(X_test_final)[:, 1]
    
    # NOTE: FN=10xFP car rater un phishing coute infiniment plus cher a la boite que de bloquer un flux sain
    seuil_rf, _ = trouver_seuil_optimal_cyber(y_test, y_proba_rf)
    seuil_xgb, _ = trouver_seuil_optimal_cyber(y_test, y_proba_xgb)
    
    prefixe_binaire = "email" if is_email else "url"
    joblib.dump(modele_rf, os.path.join(DIR_MODELS, f"{prefixe_binaire}_model_rf.joblib"))
    joblib.dump(modele_xgb, os.path.join(DIR_MODELS, f"{prefixe_binaire}_model_xgb.joblib"))
    
    # HACK: Retour de dictionnaires d'evaluation pour eviter les effets de bord de variables globales
    return {
        f"seuil_{prefixe_binaire}_rf": seuil_rf,
        f"seuil_{prefixe_binaire}_xgb": seuil_xgb,
        "audits": [
            calculer_rapport_evaluation_metier(y_test, y_proba_rf, f"Random Forest ({nom_pipeline})", seuil_rf),
            calculer_rapport_evaluation_metier(y_test, y_proba_xgb, f"XGBoost ({nom_pipeline})", seuil_xgb)
        ],
        "meta_inf": (y_test, y_proba_xgb, X_test_feats if is_email else None)
    }

if __name__ == "__main__":
    os.makedirs(DIR_MODELS, exist_ok=True)
    purger_anciens_artefacts()
    
    res_url = executer_pipeline_vecteur(
        "URLs", os.path.join(DIR_RAW, "PhiUSIIL_Phishing_URL_Dataset.csv"), 
        'URL', 'label', extraire_indices_url, is_email=False
    )
    
    res_email = executer_pipeline_vecteur(
        "Emails", os.path.join(DIR_RAW, "phishings_emails/phishing_email.csv"), 
        'text_combined', 'label', None, is_email=True
    )
    
    # Extraction des metadonnées d'inférence pour compilation finale
    y_test_em, y_proba_em, X_test_feats_em = res_email["meta_inf"]
    
    # [PRIORITÉ 9 & 10] Calculs de robustesse SOC post-training
    probas_ref = y_proba_em[:len(y_proba_em)//2]
    probas_prod = y_proba_em[len(y_proba_em)//2:]
    rapport_drift = detecter_drift_predictions(probas_ref, probas_prod)
    
    y_pred_em_xgb = (y_proba_em >= res_email["seuil_email_xgb"]).astype(int)
    has_urgency = (X_test_feats_em['has_urgency_phrase'] == 1).values
    has_no_urgency = (X_test_feats_em['has_urgency_phrase'] == 0).values
    rapport_fairness = calculer_metriques_equite(y_test_em, y_pred_em_xgb, groupe_privilegie=has_no_urgency, groupe_protege=has_urgency)
    
    with open(os.path.join(DIR_MODELS, "audit_gouvernance.json"), "w") as f:
        json.dump({"drift": rapport_drift, "fairness": rapport_fairness}, f, indent=2)
        
    # Assemblage de la structure metadata.json pour l'inférence
    seuils_calibres = {**{k: v for k, v in res_url.items() if "seuil" in k}, **{k: v for k, v in res_email.items() if "seuil" in k}}
    with open(os.path.join(DIR_MODELS, "metadata.json"), "w") as f:
        json.dump({
            "project_name": "multi-vector-phishing-detection",
            "environment": {"python_version": "3.12", "scikit_learn_version": "1.5.0", "xgboost_version": "2.0.3"},
            "decision_thresholds_calibrated": seuils_calibres
        }, f, indent=2)
        
    exporter_rapport_markdown(res_url["audits"] + res_email["audits"])
    logging.info("[MLOps] Pipeline d'entrainement et audits de conformite acheves.")
