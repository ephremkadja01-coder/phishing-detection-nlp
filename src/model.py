import os
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

def preparer_et_decouper_donnees(df, colonne_texte, colonne_cible):
    # Suppression des doublons
    df_nettoye = df.drop_duplicates(subset=[colonne_texte]).copy()
    
    X = df_nettoye[colonne_texte]
    y = df_nettoye[colonne_cible]
    
    # Separation 80% entrainement et 20% test avec stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def entrainer_modele_xgb(X_train, y_train):
    modele = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    modele.fit(X_train, y_train)
    return modele

def entrainer_modele_rf(X_train, y_train):
    modele = RandomForestClassifier(random_state=42, n_jobs=-1)
    modele.fit(X_train, y_train)
    return modele
