import pandas as pd
import numpy as np
import shap

def extraire_top_contributions_url(url, modele_url_calibre, fonction_extraire_indices, features_noms):
    """
    Calcule les contributions SHAP sur l'estimateur XGBoost sous-jacent du bloc calibre.
    """
    indices = fonction_extraire_indices(url)
    df_entree = pd.DataFrame([indices])[features_noms]
    
    # On extrait le premier estimateur de la liste de cross-validation pour SHAP
    xgb_interne = modele_url_calibre.calibrated_classifiers_[0].estimator
    
    explainer = shap.TreeExplainer(xgb_interne)
    shap_values = explainer.shap_values(df_entree)
    valeurs_shap = np.array(shap_values).flatten()
    
    indices_tries = np.argsort(np.abs(valeurs_shap))[::-1]
    contributions = []
    for idx in indices_tries[:3]:
        feature_nom = features_noms[idx]
        val_shap = valeurs_shap[idx]
        signe = "+" if val_shap > 0 else "-"
        if np.abs(val_shap) > 1e-4:
            contributions.append(f"{feature_nom} ({signe}{np.abs(val_shap):.3f})")
            
    return contributions if contributions else ["Structure comportementale globale"]

def extraire_top_contributions_email(texte_brut, vectoriseur_tfidf, modele_email_calibre, fonction_nettoyer_texte):
    """
    Calcule les contributions SHAP sur l'estimateur XGBoost email sous-jacent du bloc calibre.
    """
    from src.features import extraire_indices_semantiques_email
    from scipy.sparse import hstack, csr_matrix
    
    texte_net = fonction_nettoyer_texte(texte_brut)
    X_tfidf = vectoriseur_tfidf.transform([texte_net])
    
    features_explicites = extraire_indices_semantiques_email(texte_brut)
    X_features = pd.DataFrame([features_explicites])
    X_hybride = hstack([X_tfidf, csr_matrix(X_features.values)]).toarray()
    
    mots_vocabulaire = vectoriseur_tfidf.get_feature_names_out()
    features_explicites_noms = list(features_explicites.keys())
    tous_les_noms = list(mots_vocabulaire) + features_explicites_noms
    
    # On extrait le premier estimateur de la liste de cross-validation pour SHAP
    xgb_interne = modele_email_calibre.calibrated_classifiers_[0].estimator
    
    explainer = shap.TreeExplainer(xgb_interne)
    shap_values = explainer.shap_values(X_hybride)
    valeurs_shap = np.array(shap_values).flatten()
    
    indices_tries = np.argsort(np.abs(valeurs_shap))[::-1]
    contributions = []
    for idx in indices_tries[:3]:
        token_nom = tous_les_noms[idx]
        val_shap = valeurs_shap[idx]
        signe = "+" if val_shap > 0 else "-"
        if np.abs(val_shap) > 1e-4:
            contributions.append(f"'{token_nom}' ({signe}{np.abs(val_shap):.3f})")
            
    return contributions if contributions else ["Lexique suspect général"]
