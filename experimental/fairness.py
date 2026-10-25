import numpy as np
from sklearn.metrics import confusion_matrix

def calculer_metriques_equite(y_reel, y_pred, groupe_privilegie, groupe_protege):
    """
    Evalue si le modele discrimine un sous-groupe d'utilisateurs
    en calculant le ratio d'impact disparate sur les Faux Positifs.
    """
    # Matrice de confusion pour le groupe privilégie (ex: messages standards)
    cm_priv = confusion_matrix(y_reel[groupe_privilegie], y_pred[groupe_privilegie], labels=[0, 1])
    tn_priv, fp_priv, fn_priv, tp_priv = cm_priv.ravel() if cm_priv.size == 4 else (0,0,0,0)
    taux_fp_priv = fp_priv / (tn_priv + fp_priv + 1e-6)
    
    # Matrice de confusion pour le groupe protégé (ex: messages urgents)
    cm_prot = confusion_matrix(y_reel[groupe_protege], y_pred[groupe_protege], labels=[0, 1])
    tn_prot, fp_prot, fn_prot, tp_prot = cm_prot.ravel() if cm_prot.size == 4 else (0,0,0,0)
    taux_fp_prot = fp_prot / (tn_prot + fp_prot + 1e-6)
    
    # Calcul du Disparate Impact Ratio sur les fausses alertes
    disparate_impact = taux_fp_prot / (taux_fp_priv + 1e-6)
    
    return {
        "taux_fausses_alertes_groupe_privilegie": round(float(taux_fp_priv), 4),
        "taux_fausses_alertes_groupe_protege": round(float(taux_fp_prot), 4),
        "disparate_impact_ratio": round(float(disparate_impact), 4),
        "equitable": int(0.8 <= disparate_impact <= 1.25)
    }
