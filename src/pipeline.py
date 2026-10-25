import os
import json
import logging
import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix

PATH_META = "models/metadata.json"
SEUIL_DEF = 0.50

def charger_seuil_metadata(cle_seuil):
    if os.path.exists(PATH_META):
        try:
            with open(PATH_META, "r") as f:
                meta = json.load(f)
            return float(meta.get("decision_thresholds_calibrated", {}).get(cle_seuil, SEUIL_DEF))
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logging.warning(f"Erreur lecture metadata.json, fallback a {SEUIL_DEF}: {e}")
    return SEUIL_DEF

def verifier_menace_url(url_brute, modele_url, fonction_extraire_indices):
    indices = fonction_extraire_indices(url_brute)
    X_entree = pd.DataFrame([indices])
    
    if modele_url is None:
        return 0, 0.0
        
    probabilites = modele_url.predict_proba(X_entree)
    # NOTE: On force l'extraction du premier element [0, 1] de la matrice pour eviter le bug de dimension sous sklearn 1.5+
    probabilite_phishing = float(probabilites[0][1]) * 100
    
    seuil_optimal = charger_seuil_metadata("seuil_url_xgb")
    prediction = 1 if (probabilite_phishing / 100.0) >= seuil_optimal else 0
    
    return prediction, probabilite_phishing

def verifier_menace_email(texte_brut, vectoriseur_tfidf, modele_email, fonction_nettoyer_texte):
    from src.features import extraire_indices_semantiques_email
    
    texte_nettoye = fonction_nettoyer_texte(texte_brut)
    if vectoriseur_tfidf is None or modele_email is None:
        return 0, 0.0
        
    X_tfidf = vectoriseur_tfidf.transform([texte_nettoye])
    features_explicites = extraire_indices_semantiques_email(texte_brut)
    X_features = pd.DataFrame([features_explicites])
    X_hybride = hstack([X_tfidf, csr_matrix(X_features.values)])
    
    probabilites = modele_email.predict_proba(X_hybride)
    # NOTE: Extraction de la classe positive [0, 1] securisee contre les tableaux numpy multi-dimensionnels
    probabilite_phishing = float(probabilites[0][1]) * 100
    
    seuil_optimal = charger_seuil_metadata("seuil_email_xgb")
    prediction = 1 if (probabilite_phishing / 100.0) >= seuil_optimal else 0
    
    return prediction, probabilite_phishing

def fusionner_vecteurs_menace(score_url, score_email, seuil_cyber=80.0):
    if score_url >= seuil_cyber or score_email >= seuil_cyber:
        score_fusionne = max(score_url, score_email)
        prediction_fusionnee = 1
    else:
        score_fusionne = (score_url * 0.6) + (score_email * 0.4)
        prediction_fusionnee = 1 if score_fusionne >= 50.0 else 0
        
    return prediction_fusionnee, round(score_fusionne, 2)
