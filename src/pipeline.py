import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix

def verifier_menace_url(url_brute, modele_url, fonction_extraire_indices):
    indices = fonction_extraire_indices(url_brute)
    X_entree = pd.DataFrame([indices])
    
    prediction = int(modele_url.predict(X_entree)[0])
    probabilite = float(modele_url.predict_proba(X_entree)[0][1]) * 100
    
    return prediction, probabilite

def verifier_menace_email(texte_brut, vectoriseur_tfidf, modele_email, fonction_nettoyer_texte):
    # Pour s'aligner avec le script de training, on importe dynamiquement l'extracteur sémantique
    from src.features import extraire_indices_semantiques_email
    
    # 1. Pipeline TF-IDF
    texte_nettoye = fonction_nettoyer_texte(texte_brut)
    X_tfidf = vectoriseur_tfidf.transform([texte_nettoye])
    
    # 2. Pipeline Signaux Cyber
    features_explicites = extraire_indices_semantiques_email(texte_brut)
    X_features = pd.DataFrame([features_explicites])
    
    # 3. Fusion des axes d'inférence
    X_hybride = hstack([X_tfidf, csr_matrix(X_features.values)])
    
    prediction = int(modele_email.predict(X_hybride)[0])
    probabilite = float(modele_email.predict_proba(X_hybride)[0][1]) * 100
    
    return prediction, probabilite

def fusionner_vecteurs_menace(score_url, score_email, seuil_cyber=80.0):
    if score_url >= seuil_cyber or score_email >= seuil_cyber:
        score_fusionne = max(score_url, score_email)
        prediction_fusionnee = 1
    else:
        score_fusionne = (score_url * 0.6) + (score_email * 0.4)
        prediction_fusionnee = 1 if score_fusionne >= 50.0 else 0
        
    return prediction_fusionnee, round(score_fusionne, 2)
