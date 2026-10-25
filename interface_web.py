import os
import joblib
import gradio as gr
from src.features import extraire_indices_url, nettoyer_texte_email
from src.pipeline import verifier_menace_url, verifier_menace_email, fusionner_vecteurs_menace, charger_seuil_metadata
from src.explain import extraire_top_contributions_url, extraire_top_contributions_email

dossier_modeles = "models"
mod_url, vec_email, mod_email = None, None, None
FEATURES_URL_NOMS = []

def charger_les_composants_cyber():
    global mod_url, vec_email, mod_email, FEATURES_URL_NOMS
    chemin_mod_url = os.path.join(dossier_modeles, "url_model_xgb.joblib")
    if os.path.exists(chemin_mod_url):
        mod_url = joblib.load(chemin_mod_url)
        vec_email = joblib.load(os.path.join(dossier_modeles, "email_vectoriseur.joblib"))
        mod_email = joblib.load(os.path.join(dossier_modeles, "email_model_xgb.joblib"))
        
        if hasattr(mod_url, "calibrated_classifiers_") and len(mod_url.calibrated_classifiers_) > 0:
            xgb_interne_url = mod_url.calibrated_classifiers_[0].estimator
            FEATURES_URL_NOMS = xgb_interne_url.get_booster().feature_names
        else:
            FEATURES_URL_NOMS = []

def analyser_url_interface(url):
    try:
        if mod_url is None:
            return 0.0, "⚠️ Modèle URL indisponible."
        prediction, pourcentage = verifier_menace_url(url, mod_url, extraire_indices_url)
        score_val = float(pourcentage)
        top_declencheurs = extraire_top_contributions_url(url, mod_url, extraire_indices_url, FEATURES_URL_NOMS)
        explicability_txt = ", ".join(top_declencheurs)
        
        seuil_txt = f"| Seuil métrique optimisé : {charger_seuil_metadata('seuil_url_xgb'):.2f} (FN=10xFP)"
        shap_notice = "\n\n*Note SOC : SHAP explique l'estimateur brut XGBoost. Les probabilités finales sont calibrées Isotonic."
        
        if int(prediction) == 1:
            return score_val, f"⚠️ ALERTE : URL de Phishing Détectée !\nRisque : {score_val:.2f}% {seuil_txt}\n\n[SHAP Déclencheurs] : {explicability_txt}{shap_notice}"
        return score_val, f"✅ SÉCURISÉ : Ce lien semble légitime.\nRisque : {score_val:.2f}% {seuil_txt}\n\n[SHAP Déclencheurs] : {explicability_txt}{shap_notice}"
    except Exception as e:
        return 0.0, f"Erreur URL : {str(e)}"

def analyser_email_interface(email):
    try:
        if mod_email is None:
            return 0.0, "⚠️ Modèle sémantique indisponible."
        prediction, pourcentage = verifier_menace_email(email, vec_email, mod_email, nettoyer_texte_email)
        score_val = float(pourcentage)
        top_mots = extraire_top_contributions_email(email, vec_email, mod_email, nettoyer_texte_email)
        explicability_txt = ", ".join(top_mots)
        
        seuil_txt = f"| Seuil métrique optimisé : {charger_seuil_metadata('seuil_email_xgb'):.2f} (FN=10xFP)"
        shap_notice = "\n\n*Note SOC : SHAP explique l'estimateur brut XGBoost. Les probabilités finales sont calibrées Isotonic."
        
        if int(prediction) == 1:
            return score_val, f"⚠️ ALERTE : Contenu Suspect Détecté !\nIndice : {score_val:.2f}% {seuil_txt}\n\n[Vrais SHAP Actifs] : {explicability_txt}{shap_notice}"
        return score_val, f"✅ SÉCURISÉ : Cet e-mail semble sain.\nRisque : {score_val:.2f}% {seuil_txt}\n\n[Vrais SHAP Actifs] : {explicability_txt}{shap_notice}"
    except Exception as e:
        return 0.0, f"Erreur E-mail : {str(e)}"

def analyser_fusion_interface(url, email):
    try:
        score_url, msg_url = analyser_url_interface(url)
        score_email, msg_email = analyser_email_interface(email)
        prediction_fusionnee, score_fusionne = fusionner_vecteurs_menace(score_url, score_email)
        
        verdict = "🚨 ALERTE CRITIQUE : Phishing Combiné Détecté !" if prediction_fusionnee == 1 else "🟢 SÉCURISÉ : Les signaux combinés sont stables."
        
        # NOTE : Formatage propre des blocs multilingues pour eviter les plantages de remplacement
        str_url = str(msg_url).replace("\n", " ")
        str_email = str(msg_email).replace("\n", " ")
        
        return (
            f"{verdict}\n"
            f"-----------------------------------------\n"
            f"Score de Menace Global  : {score_fusionne:.2f}%\n"
            f"|- {str_url}\n"
            f"|- {str_email}\n"
            f"-----------------------------------------\n"
            f"Stratégie de décision   : Barrière asymétrique (OR Gate active à 80%)"
        )
    except Exception as e:
        return f"Erreur lors du calcul de fusion : {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("# 🛡️ Multi-Vector Phishing Detection Hub — Operational Edition")
    
    with gr.Tabs():
        with gr.TabItem("🔍 Analyse de Liens (URLs)"):
            url_input = gr.Textbox(label="Collez l'URL suspecte à vérifier")
            url_button = gr.Button("Lancer l'Analyse", variant="primary")
            url_output = gr.Textbox(label="Verdict")
            url_button.click(analyser_url_interface, inputs=url_input, outputs=url_output)
            
        with gr.TabItem("📧 Analyse d'E-mails (NLP)"):
            email_input = gr.Textbox(label="Collez le texte de l'e-mail", lines=4)
            email_button = gr.Button("Scanner le Contenu", variant="primary")
            email_output = gr.Textbox(label="Verdict")
            email_button.click(analyser_email_interface, inputs=email_input, outputs=email_output)
            
        with gr.TabItem("🔀 Fusion Multi-Modale (URL + Texte)"):
            fuse_url = gr.Textbox(label="URL contenue dans le message")
            fuse_email = gr.Textbox(label="Corps de texte de l'e-mail reçu", lines=5)
            fuse_button = gr.Button("Analyser la Menace Combinée", variant="stop")
            fuse_output = gr.TextArea(label="Rapport d'Audit Inter-Canaux SOC", lines=10)
            fuse_button.click(analyser_fusion_interface, inputs=[fuse_url, fuse_email], outputs=fuse_output)

if __name__ == "__main__":
    charger_les_composants_cyber()
    demo.launch()
