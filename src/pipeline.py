import pandas as pd
import numpy as np

def verifier_menace_url(url_brute, modele_url, fonction_extraire_indices):
    """
    Extrait les indices d'une URL et predit la menace avec sa probabilite cyber.
    """
    indices = fonction_extraire_indices(url_brute)
    X_entree = pd.DataFrame([indices])
    
    prediction = int(modele_url.predict(X_entree)[0])
    probabilite = float(modele_url.predict_proba(X_entree)[0][1]) * 100
    
    return prediction, probabilite

def verifier_menace_email(texte_brut, vectoriseur_tfidf, modele_email, fonction_nettoyer_texte):
    """
    Nettoie le texte d'un email et predit la menace avec sa probabilite cyber.
    """
    texte_nettoye = fonction_nettoyer_texte(texte_brut)
    X_entree = vectoriseur_tfidf.transform([texte_nettoye])
    
    prediction = int(modele_email.predict(X_entree)[0])
    probabilite = float(modele_email.predict_proba(X_entree)[0][1]) * 100
    
    return prediction, probabilite

def fusionner_vecteurs_menace(score_url, score_email, seuil_cyber=80.0):
    """
    Applique une Meta-Regle de Securite asymetrique (OR logique calibre)
    pour combiner les signaux heterogenes URL + Email sans affaiblir le detecteur.
    """
    # Strategie Cyber : Si l'un des deux canaux est certain d'une menace, on applique l'alerte maximale
    if score_url >= seuil_cyber or score_email >= seuil_cyber:
        score_fusionne = max(score_url, score_email)
        prediction_fusionnee = 1
    else:
        # Sinon, calcul de la moyenne ponderee cyber
        score_fusionne = (score_url * 0.6) + (score_email * 0.4)
        prediction_fusionnee = 1 if score_fusionne >= 50.0 else 0
        
    return prediction_fusionnee, round(score_fusionne, 2)
