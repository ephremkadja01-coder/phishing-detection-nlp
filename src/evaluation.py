import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc

def trouver_seuil_optimal_cyber(y_reel, y_proba, poids_fn=10, poids_fp=1):
    """
    Balaie les seuils de decision pour identifier celui qui minimise le cout asymetrique global.
    """
    seuils = np.linspace(0.01, 0.99, 99)
    couts = []
    
    for seuil in seuils:
        y_pred = (y_proba >= seuil).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_reel, y_pred).ravel()
        cout = (fn * poids_fn) + (fp * poids_fp)
        couts.append(cout)
        
    idx_opt = np.argmin(couts)
    return float(seuils[idx_opt]), int(couts[idx_opt])

def calculer_rapport_evaluation_metier(y_reel, y_proba, nom_modele, seuil_decision=0.50, poids_fn=10, poids_fp=1):
    """
    Genere un audit adaptatif aligne sur le seuil operationnel optimal du modele.
    """
    y_pred = (y_proba >= seuil_decision).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_reel, y_pred).ravel()
    
    precision, rappel, _ = precision_recall_curve(y_reel, y_proba)
    score_pr_auc = auc(rappel, precision) * 100
    
    cout_total = (fn * poids_fn) + (fp * poids_fp)
    cout_moyen = cout_total / len(y_reel)
    
    far_fn = np.sum((y_reel == 1) & (y_proba < 0.10))
    far_fp = np.sum((y_reel == 0) & (y_proba > 0.90))
    
    rapport = {
        "Modèle": nom_modele,
        "Seuil Appliqué": round(seuil_decision, 2),
        "Vrais Négatifs (Sains)": int(tn),
        "Faux Positifs (Fausses Alertes)": int(fp),
        "Faux Négatifs (Ratés Cyber)": int(fn),
        "Vrais Positifs (Capturés)": int(tp),
        "PR-AUC (%)": round(score_pr_auc, 2),
        "Coût Asymétrique Total": int(cout_total),
        "Coût Moyen": round(cout_moyen, 4),
        "Far Faux Négatifs": int(far_fn),
        "Far Faux Positifs": int(far_fp)
    }
    return rapport

def exporter_rapport_markdown(liste_rapports, chemin_sortie="doc/EVALUATION.md"):
    os.makedirs(os.path.dirname(chemin_sortie), exist_ok=True)
    df = pd.DataFrame(liste_rapports)
    
    with open(chemin_sortie, "w", encoding="utf-8") as f:
        f.write("# 📊 Rapport d'Évaluation d'Élite, Calibration & Seuil Optimal\n\n")
        f.write("Ce document centralise l'analyse étanche de nos classificateurs après étalonnage Isotonic.\n\n")
        f.write("## 1. Tableau Comparatif des Performances (Seuils Métiers Optimisés)\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n## 2. Note Technique MLOps\n")
        f.write("* Le **Seuil Appliqué** n'est plus fixé arbitrairement à 0.50. Il est calculé algorithmiquement pour minimiser les ratés de phishing (FN) qui portent un poids de risque 10x supérieur aux fausses alertes (FP).\n")
