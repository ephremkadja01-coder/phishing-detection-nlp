import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc

def calculer_rapport_evaluation_metier(y_reel, y_proba, nom_modele, poids_fn=10, poids_fp=1):
    """
    Génère un audit de performance approfondi et asymétrique orienté Cybersécurité.
    """
    # 1. Extraction de la prédiction classique (seuil 0.5)
    y_pred = (y_proba >= 0.5).astype(int)
    
    # 2. Calcul de la Matrice de Confusion
    tn, fp, fn, tp = confusion_matrix(y_reel, y_pred).ravel()
    
    # 3. Calcul de la métrique reine en déséquilibre : PR-AUC
    precision, rappel, _ = precision_recall_curve(y_reel, y_proba)
    score_pr_auc = auc(rappel, precision) * 100
    
    # 4. Calcul du coût financier / opérationnel espéré (Asymmetric Cyber Cost)
    cout_total = (fn * poids_fn) + (fp * poids_fp)
    cout_moyen = cout_total / len(y_reel)
    
    # 5. Audit des "Far Errors" (Erreurs extrêmes à haut risque)
    # Rater un phishing avec une probabilité quasi-nulle de menace
    far_fn = np.sum((y_reel == 1) & (y_proba < 0.10))
    # Bloquer un site sain avec une probabilité maximale de menace
    far_fp = np.sum((y_reel == 0) & (y_proba > 0.90))
    
    # 6. Structuration du rapport sous forme de dictionnaire propre
    rapport = {
        "Modèle": nom_modele,
        "Vrais Négatifs (Sains)": int(tn),
        "Faux Positifs (Fausses Alertes)": int(fp),
        "Faux Négatifs (Ratés Cyber)": int(fn),
        "Vrais Positifs (Capturés)": int(tp),
        "PR-AUC (%)": round(score_pr_auc, 2),
        "Coût Asymétrique Total": int(cout_total),
        "Coût Moyen par Requête": round(cout_moyen, 4),
        "Far Faux Négatifs (Critiques)": int(far_fn),
        "Far Faux Positifs (Bloquants)": int(far_fp)
    }
    
    return rapport

def exporter_rapport_markdown(liste_rapports, chemin_sortie="doc/EVALUATION.md"):
    """
    Compile automatiquement les résultats dans un rapport Markdown auditable.
    """
    os.makedirs(os.path.dirname(chemin_sortie), exist_ok=True)
    df = pd.DataFrame(liste_rapports)
    
    with open(chemin_sortie, "w", encoding="utf-8") as f:
        f.write("# 📊 Rapport d'Évaluation d'Élite & Audit Métier\n\n")
        f.write("Ce document centralise l'analyse étanche de nos classificateurs face au modèle de coût asymétrique.\n\n")
        f.write("## 1. Tableau Comparatif des Performances\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n## 2. Guide d'Interprétation Cyber\n")
        f.write("* **PR-AUC** : Plus il est proche de 100%, plus le modèle sépare efficacement le phishing sans générer de fausses alertes.\n")
        f.write("* **Coût Asymétrique** : Calculé sur l'hypothèse qu'un Faux Négatif (FN) coûte **10x plus cher** qu'un Faux Positif (FP).\n")
        f.write("* **Far Faux Négatifs** : Représente le pire scénario (le modèle est sûr à 90%+ qu'une URL est saine alors qu'elle est malveillante).\n")
    
    print(f"[MLOps] Rapport d'évaluation compilé et sauvegardé dans : {chemin_sortie}")
