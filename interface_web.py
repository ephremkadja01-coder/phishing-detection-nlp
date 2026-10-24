import os
import joblib
import numpy as np
from urllib.parse import urlparse
import gradio as gr
from src.features import extraire_indices_url, nettoyer_texte_email
from src.pipeline import verifier_menace_url, verifier_menace_email

dossier_modeles = "models"

# 1. Chargement des composants de production
mod_url = joblib.load(os.path.join(dossier_modeles, "url_model_xgb.joblib"))
vec_email = joblib.load(os.path.join(dossier_modeles, "email_vectoriseur.joblib"))
mod_email = joblib.load(os.path.join(dossier_modeles, "email_model_xgb.joblib"))

# 2. Liste blanche cyber des domaines institutionnels et majeurs
DOMAINES_SURS = {
    "github.com", "github.io", "laposte.fr", "laposte.com", "google.com", 
    "google.fr", "microsoft.com", "apple.com", "linkedin.com", "facebook.com",
    "twitter.com", "amazon.fr", "amazon.com", "netflix.com", "iua.ci"
}

def verifier_liste_blanche(url):
    """
    Verifie si l'URL appartient nativement a un domaine de confiance.
    """
    try:
        url_chaine = str(url).lower().strip()
        if not url_chaine.startswith(('http://', 'https://')):
            url_chaine = 'https://' + url_chaine
        analyse = urlparse(url_chaine)
        domaine = analyse.netloc
        if domaine.startswith('www.'):
            domaine = domaine[4:]
        return domaine in DOMAINES_SURS
    except Exception:
        return False

def analyser_url_interface(url):
    try:
        # SECURITÉ INDUSTRIELLE : Court-circuit immédiat si le domaine est de confiance
        if verifier_liste_blanche(url):
            return "✅ SÉCURISÉ : Ce lien appartient à un domaine de confiance certifié.\nProbabilité de phishing : 0.00%"

        prediction, pourcentage = verifier_menace_url(url, mod_url, extraire_indices_url)
        
        # Securisation de l'affichage des types complexes renvoyes par le pipeline
        pred_val = int(np.array(prediction).flatten()[0])
        score_val = float(np.array(pourcentage).flatten()[1]) if len(np.array(pourcentage).flatten()) > 1 else float(np.array(pourcentage).flatten()[0])
        
        if pred_val == 1:
            return f"⚠️ ALERTE : URL de Phishing Détectée !\nNiveau de risque : {score_val:.2f}%"
        return f"✅ SÉCURISÉ : Ce lien semble légitime.\nProbabilité de phishing : {score_val:.2f}%"
    except Exception as e:
        return f"Erreur technique lors de l'analyse : {str(e)}"

def analyser_email_interface(email):
    try:
        prediction, pourcentage = verifier_menace_email(email, vec_email, mod_email, nettoyer_texte_email)
        
        pred_val = int(np.array(prediction).flatten()[0])
        score_val = float(np.array(pourcentage).flatten()[1]) if len(np.array(pourcentage).flatten()) > 1 else float(np.array(pourcentage).flatten()[0])
        
        if pred_val == 1:
            return f"⚠️ ALERTE : Contenu Suspect (Phishing d'E-mail) !\nIndice de menace : {score_val:.2f}%"
        return f"✅ SÉCURISÉ : Cet e-mail semble sain.\nProbabilité de phishing : {score_val:.2f}%"
    except Exception as e:
        return f"Erreur technique lors de l'analyse : {str(e)}"

# 3. Construction de l'interface visuelle
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

if __name__ == "__main__":
    demo.launch()
