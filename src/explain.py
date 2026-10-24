import pandas as pd
import numpy as np
import shap

def extraire_top_contributions_url(url, modele_url, fonction_extraire_indices, features_noms):
    """
    Calcule les contributions SHAP exactes pour les variables d'une URL sur XGBoost.
    """
    # 1. Extraction des indices
    indices = fonction_extraire_indices(url)
    df_entree = pd.DataFrame([indices])[features_noms]
    
    # 2. Initialisation de l'explicateurs d'arbres natif de SHAP
    explainer = shap.TreeExplainer(modele_url)
    shap_values = explainer.shap_values(df_entree)
    
    # Prise en charge des structures de tableaux NumPy
    valeurs_shap = np.array(shap_values).flatten()
    
    # 3. Tri par valeur absolue pour isoler les 3 plus grands déclencheurs
    indices_tries = np.argsort(np.abs(valeurs_shap))[::-1]
    
    contributions = []
    for idx in indices_tries[:3]:
        feature_nom = features_noms[idx]
        val_shap = valeurs_shap[idx]
        signe = "+" if val_shap > 0 else "-"
        # On n'affiche la variable que si son impact SHAP est significatif
        if np.abs(val_shap) > 1e-4:
            contributions.append(f"{feature_nom} ({signe}{np.abs(val_shap):.3f})")
            
    return contributions if contributions else ["Structure comportementale globale"]

def extraire_top_contributions_email(texte_brut, vectoriseur_tfidf, modele_email, fonction_nettoyer_texte):
    """
    Calcule les contributions SHAP exactes pour les tokens TF-IDF et indicateurs hybrides d'un e-mail.
    """
    from src.features import extraire_indices_semantiques_email
    from scipy.sparse import hstack, csr_matrix
    
    # 1. Reconstruction de la matrice hybride d'inférence
    texte_net = fonction_nettoyer_texte(texte_brut)
    X_tfidf = vectoriseur_tfidf.transform([texte_net])
    
    features_explicites = extraire_indices_semantiques_email(texte_brut)
    X_features = pd.DataFrame([features_explicites])
    X_hybride = hstack([X_tfidf, csr_matrix(X_features.values)]).toarray()
    
    # 2. Construction de la liste des noms de colonnes pour SHAP
    mots_vocabulaire = vectoriseur_tfidf.get_feature_names_out()
    features_explicites_noms = list(features_explicites.keys())
    tous_les_noms = list(mots_vocabulaire) + features_explicites_noms
    
    # 3. Extraction SHAP
    explainer = shap.TreeExplainer(modele_email)
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
