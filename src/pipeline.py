import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix

def verifier_menace_url(url_brute, modele_url, fonction_extraire_indices):
    """
    Extrait les indices d'une URL et predit la menace via le modele calibre Isotonic.
    """
    indices = fonction_extraire_indices(url_brute)
    X_entree = pd.DataFrame([indices])
    
    # Avec CalibratedClassifierCV, predict_proba renvoie un tableau standard [prob_0, prob_1]
    probabilites = modele_url.predict_proba(X_entree)[0]
    probabilite_phishing = float(probabilites[1]) * 100
    
    # Seuil operationnel persistant ou par defaut a 50%
    prediction = 1 if probabilite_phishing >= 50.0 else 0
    
    return prediction, probabilite_phishing

def verifier_menace_email(texte_brut, vectoriseur_tfidf, modele_email, fonction_nettoyer_texte):
    """
    Nettoie le texte d'un email et predit la menace via le pipeline hybride calibre.
    """
    from src.features import extraire_indices_semantiques_email
    
    # 1. Reconstruction de la matrice hybride d'inférence
    texte_nettoye = fonction_nettoyer_texte(texte_brut)
    X_tfidf = vectoriseur_tfidf.transform([texte_nettoye])
    
    features_explicites = extraire_indices_semantiques_email(texte_brut)
    X_features = pd.DataFrame([features_explicites])
    X_hybride = hstack([X_tfidf, csr_matrix(X_features.values)])
    
    # 2. Inference calibree Isotonic
    probabilites = modele_email.predict_proba(X_hybride)[0]
    probabilite_phishing = float(probabilites[1]) * 100
    
    prediction = 1 if probabilite_phishing >= 50.0 else 0
    
    return prediction, probabilite_phishing

def fusionner_vecteurs_menace(score_url, score_email, seuil_cyber=80.0):
    """
    Meta-Regle de fusion asymetrique basee sur les probabilites physiques calibrees.
    """
    if score_url >= seuil_cyber or score_email >= seuil_cyber:
        score_fusionne = max(score_url, score_email)
        prediction_fusionnee = 1
    else:
        score_fusionne = (score_url * 0.6) + (score_email * 0.4)
        prediction_fusionnee = 1 if score_fusionne >= 50.0 else 0
        
    return prediction_fusionnee, round(score_fusionne, 2)
