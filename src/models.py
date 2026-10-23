import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

def preparer_et_decouper_donnees(df, colonne_texte, colonne_cible, est_email=False):
    """
    Supprime les doublons, applique le nettoyage et separe en Train (80%) / Test (20%).
    """
    # Suppression des doublons
    df_nettoye = df.drop_duplicates(subset=[colonne_texte]).copy()
    
    X = df_nettoye[colonne_texte]
    y = df_nettoye[colonne_cible]
    
    # Decoupage stratifie 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def entrainer_modele_xgb(X_train, y_train):
    """
    Entraine un modele XGBoost sur les donnees fournies.
    """
    modele = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    modele.fit(X_train, y_train)
    return modele

def entrainer_modele_rf(X_train, y_train):
    """
    Entraine un modele Random Forest sur les donnees fournies.
    """
    modele = RandomForestClassifier(random_state=42, n_jobs=-1)
    modele.fit(X_train, y_train)
    return modele
