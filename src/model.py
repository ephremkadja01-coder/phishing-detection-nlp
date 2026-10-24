import os
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

def preparer_et_decouper_donnees(df, colonne_texte, colonne_cible):
    """
    Supprime les doublons et separe le dataset en Train (80%) et Test (20%) de maniere stratifiee.
    """
    df_nettoye = df.drop_duplicates(subset=[colonne_texte]).copy()
    
    X = df_nettoye[[colonne_texte]]
    y = df_nettoye[colonne_cible]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def sauvegarder_fichiers_processed(X_train, X_test, y_train, y_test, prefixe_nom):
    """
    Sauvegarde physiquement les splits Train/Test en fichiers CSV dans data/processed/.
    """
    dossier_cible = "data/processed"
    os.makedirs(dossier_cible, exist_ok=True)
    
    # Association du texte et de sa cible pour l'historique
    df_train = X_train.copy()
    df_train['label'] = y_train
    
    df_test = X_test.copy()
    df_test['label'] = y_test
    
    df_train.to_csv(os.path.join(dossier_cible, f"{prefixe_nom}_train.csv"), index=False)
    df_test.to_csv(os.path.join(dossier_cible, f"{prefixe_nom}_test.csv"), index=False)
    print(f"Fichiers processed sauvegardes pour : {prefixe_nom}")

def entrainer_modele_xgb(X_train, y_train):
    modele = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    modele.fit(X_train, y_train)
    return modele

def entrainer_modele_rf(X_train, y_train):
    modele = RandomForestClassifier(random_state=42, n_jobs=-1)
    modele.fit(X_train, y_train)
    return modele
