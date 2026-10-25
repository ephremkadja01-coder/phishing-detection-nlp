import numpy as np
from scipy.stats import ks_2samp

def detecter_drift_predictions(probas_reference, probas_production, seuil_alpha=0.05):
    """
    Compare les distributions de probabilites via le test de Kolmogorov-Smirnov
    pour identifier un glissement de concept (Model/Data Drift).
    """
    statistique_ks, p_value = ks_2samp(probas_reference, probas_production)
    drift_detecte = p_value < seuil_alpha
    
    rapport = {
        "statistique_ks": round(float(statistique_ks), 4),
        "p_value": float(p_value),
        "drift_detecte": int(drift_detecte),
        "statut": "🚨 DRIFT DETECTE : Planifier un réentraînement immédiat." if drift_detecte else "✅ STABLE : Les distributions de production sont conformes."
    }
    return rapport
