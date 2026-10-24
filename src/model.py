import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def preparer_et_decouper_donnees(df, colonne_texte, colonne_cible, test_size=0.2, random_state=42):
    """
    Nettoie les doublons bruts et partitionne le dataset de maniere stratifiee.
    """
    df_propre = df.drop_duplicates(subset=[colonne_texte]).copy()
    
    X = df_propre[[colonne_texte]]
    y = df_propre[colonne_cible]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def sauvegarder_fichiers_processed(X_train, X_test, y_train, y_test, prefixe, dossier="data/processed"):
    """
    Enregistre physiquement les splits sur le disque pour archivage et audit.
    """
    os.makedirs(dossier, exist_ok=True)
    
    train_df = X_train.copy()
    train_df['target'] = y_train
    
    test_df = X_test.copy()
    test_df['target'] = y_test
    
    train_df.to_csv(os.path.join(dossier, f"{prefixe}_train.csv"), index=False)
    test_df.to_csv(os.path.join(dossier, f"{prefixe}_test.csv"), index=False)
    print(f"Fichiers processed sauvegardes pour : {prefixe}")

def entrainer_modele_rf(X_train, y_train, random_state=42):
    model = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def entrainer_modele_xgb(X_train, y_train, random_state=42):
    model = XGBClassifier(n_estimators=100, random_state=random_state, eval_metric='logloss', n_jobs=-1)
    model.fit(X_train, y_train)
    return model
