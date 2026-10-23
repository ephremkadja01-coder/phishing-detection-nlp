import pandas as pd

def verifier_menace_url(url_brute, modele_url, fonction_extraire_indices):
    """
    Extrait les indices d'une URL et predit si c'est un phishing.
    """
    indices = fonction_extraire_indices(url_brute)
    X_entree = pd.DataFrame([indices])
    
    prediction = modele_url.predict(X_entree)[0]
    # Extraction de la probabilite de la classe 1 (Phishing)
    probabilite = modele_url.predict_proba(X_entree)[0][1] * 100
    
    return prediction, probabilite

def verifier_menace_email(texte_brut, vectoriseur_tfidf, modele_email, fonction_nettoyer_texte):
    """
    Nettoie le texte d'un email et predit si c'est un phishing.
    """
    texte_nettoye = fonction_nettoyer_texte(texte_brut)
    X_entree = vectoriseur_tfidf.transform([texte_nettoye])
    
    prediction = modele_email.predict(X_entree)[0]
    # Extraction de la probabilite de la classe 1 (Phishing)
    probabilite = modele_email.predict_proba(X_entree)[0][1] * 100
    
    return prediction, probabilite
