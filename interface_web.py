import os
import joblib
import gradio as gr
from src.features import extraire_indices_url, nettoyer_texte_email
from src.pipeline import verifier_menace_url, verifier_menace_email

dossier_modeles = "models"

# 1. Chargement des composants de production
mod_url = joblib.load(os.path.join(dossier_modeles, "url_model_xgb.joblib"))
vec_email = joblib.load(os.path.join(dossier_modeles, "email_vectoriseur.joblib"))
mod_email = joblib.load(os.path.join(dossier_modeles, "email_model_xgb.joblib"))

# 2. Fonctions de raccordement pour l'interface graphique
def analyser_url_interface(url):
    prediction, probabilite = verifier_menace_url(url, mod_url, extraire_indices_url)
    confiance = probabilite[0][1] if prediction == 1 else probabilite[0][0]
    
    if prediction == 1:
        return f"⚠️ ALERTE : URL de Phishing Détectée !\nNiveau de risque : {confiance:.2f}%"
    return f"✅ SÉCURISÉ : Ce lien semble légitime.\nProbabilité de phishing : {probabilite[0][1]:.2f}%"

def analyser_email_interface(email):
    prediction, probabilite = verifier_menace_email(email, vec_email, mod_email, nettoyer_texte_email)
    confiance = probabilite[0][1] if prediction == 1 else probabilite[0][0]
    
    if prediction == 1:
        return f"⚠️ ALERTE : Contenu Suspect (Phishing d'E-mail) !\nIndice de menace : {confiance:.2f}%"
    return f"✅ SÉCURISÉ : Cet e-mail semble sain.\nProbabilité de phishing : {probabilite[0][1]:.2f}%"

# 3. Construction de l'interface visuelle moderne
with gr.Blocks(theme=gr.themes.Soft(), title="Cyber-Detector App") as demo:
    gr.Markdown("# 🛡️ Multi-Vector Phishing Detection Hub")
    gr.Markdown("Analyse comportementale en temps réel par Intelligence Artificielle (XGBoost Engines).")
    
    with gr.Tabs():
        with gr.TabItem("🔍 Analyse de Liens (URLs)"):
            url_input = gr.Textbox(label="Collez l'URL suspecte à vérifier", placeholder="https://secure-login-update...")
            url_button = gr.Button("Lancer l'Analyse Cyber", variant="primary")
            url_output = gr.Textbox(label="Verdict du Détecteur")
            url_button.click(analyser_url_interface, inputs=url_input, outputs=url_output)
            
        with gr.TabItem("📧 Analyse d'E-mails (NLP)"):
            email_input = gr.Textbox(label="Collez le texte intégral du mail suspect", lines=6, placeholder="Cher client, votre compte est suspendu...")
            email_button = gr.Button("Scanner le Contenu Sémantique", variant="primary")
            email_output = gr.Textbox(label="Verdict du Détecteur")
            email_button.click(analyser_email_interface, inputs=email_input, outputs=email_output)

# 4. Lancement du serveur local
if __name__ == "__main__":
    demo.launch()
