import sys
import os

# Ajout du dossier racine pour l'importation de nos modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from features import extraire_indices_url, nettoyer_texte_email

def verifier_menace_url(url_brute, modele_url):
    """
    Extrait les indices d'une URL et predit si c'est une menace.
    """
    # Extraction des indices
    indices = extraire_indices_url(url_brute)
    
    # Transformation des indices en format exploitable par le modele (DataFrame a 1 ligne)
    import pandas as pd
    X_entree = pd.DataFrame([indices])
    
    # Prediction de la probabilite (classe 1 = Phishing)
    probabilite = modele_url.predict_proba(X_entree)[0][1]
    prediction = 1 if probabilite >= 0.5 else 0
    
    return prediction, probabilite

def verifier_menace_email(texte_brut, vectoriseur_tfidf, modele_email):
    """
    Nettoie le texte d'un email et predit si c'est une menace.
    """
    # Nettoyage NLP initial
    texte_nettoye = nettoyer_texte_email(texte_brut)
    
    # Vectorisation TF-IDF
    X_entree = vectoriseur_tfidf.transform([texte_nettoye])
    
    # Prediction de la probabilite
    probabilite = modele_email.predict_proba(X_entree)[0][1]
    prediction = 1 if probabilite >= 0.5 else 0
    
    return prediction, probabilite
