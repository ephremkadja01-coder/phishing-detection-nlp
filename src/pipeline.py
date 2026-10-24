import pandas as pd

def verifier_menace_url(url_brute, modele_url, fonction_extraire_indices):
    indices = fonction_extraire_indices(url_brute)
    X_entree = pd.DataFrame([indices])
    
    prediction = modele_url.predict(X_entree)[0]
    probabilite = modele_url.predict_proba(X_entree)[0][1] * 100
    return prediction, probabilite

def verifier_menace_email(texte_brut, vectoriseur_tfidf, modele_email, fonction_nettoyer_texte):
    texte_nettoye = fonction_nettoyer_texte(texte_brut)
    X_entree = vectoriseur_tfidf.transform([texte_nettoye])
    
    prediction = modele_email.predict(X_entree)[0]
    probabilite = modele_email.predict_proba(X_entree)[0][1] * 100
    return prediction, probabilite
