import os
import joblib
import numpy as np
from urllib.parse import urlparse
import gradio as gr
from src.features import extraire_indices_url, nettoyer_texte_email
from src.pipeline import verifier_menace_url, verifier_menace_email, fusionner_vecteurs_menace

dossier_modeles = "models"

# 1. Chargement des composants de production
mod_url = joblib.load(os.path.join(dossier_modeles, "url_model_xgb.joblib"))
vec_email = joblib.load(os.path.join(dossier_modeles, "email_vectoriseur.joblib"))
mod_email = joblib.load(os.path.join(dossier_modeles, "email_model_xgb.joblib"))

# Liste blanche cyber des domaines institutionnels majeurs
DOMAINES_SURS = {
    "github.com", "github.io", "laposte.fr", "laposte.com", "google.com", 
    "google.fr", "microsoft.com", "apple.com", "linkedin.com", "facebook.com",
    "twitter.com", "amazon.fr", "amazon.com", "netflix.com", "iua.ci"
}

def verifier_liste_blanche(url):
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
        if verifier_liste_blanche(url):
            return 0.0, "✅ SÉCURISÉ : Ce lien appartient à un domaine de confiance certifié.\nProbabilité de phishing : 0.00%\n\n[Explicabilité SOC] : Whitelist institutionnelle active."

        prediction, pourcentage = verifier_menace_url(url, mod_url, extraire_indices_url)
        score_val = float(pourcentage)
        
        if int(prediction) == 1:
            return score_val, f"⚠️ ALERTE : URL de Phishing Détectée !\nNiveau de risque : {score_val:.2f}%"
        return score_val, f"✅ SÉCURISÉ : Ce lien semble légitime.\nProbabilité de phishing : {score_val:.2f}%"
    except Exception as e:
        return 0.0, f"Erreur technique lors de l'analyse : {str(e)}"

def analyser_email_interface(email):
    try:
        prediction, pourcentage = verifier_menace_email(email, vec_email, mod_email, nettoyer_texte_email)
        score_val = float(pourcentage)
        
        if int(prediction) == 1:
            return score_val, f"⚠️ ALERTE : Contenu Suspect (Phishing d'E-mail) !\nIndice de menace : {score_val:.2f}%"
        return score_val, f"✅ SÉCURISÉ : Cet e-mail semble sain.\nProbabilité de phishing : {score_val:.2f}%"
    except Exception as e:
        return 0.0, f"Erreur technique lors de l'analyse : {str(e)}"

def analyser_fusion_interface(url, email):
    try:
        # Extraction propre du score numérique (index 0 du tuple)
        score_url, _ = analyser_url_interface(url)
        score_email, _ = analyser_email_interface(email)
        
        # Corrélation asymétrique
        prediction_fusionnee, score_fusionne = fusionner_vecteurs_menace(score_url, score_email)
        
        if prediction_fusionnee == 1:
            verdict = "🚨 ALERTE CRITIQUE : Phishing Combiné Détecté !"
        else:
            verdict = "🟢 SÉCURISÉ : Les signaux combinés sont stables."
        
        rapport_soc = (
            f"{verdict}\n"
            f"-----------------------------------------\n"
            f"Score de Menace Global  : {score_fusionne:.2f}%\n"
            f"|- Contribution URL     : {score_url:.2f}%\n"
            f"|- Contribution E-mail  : {score_email:.2f}%\n"
            f"-----------------------------------------\n"
            f"Stratégie de décision   : Barrière asymétrique (OR Gate active à 80%)"
        )
        return rapport_soc
    except Exception as e:
        return f"Erreur lors du calcul de fusion : {str(e)}"

# 3. Construction de l'interface visuelle
with gr.Blocks(theme=gr.themes.Soft(), title="Cyber-Detector App Hub") as demo:
    gr.Markdown("# 🛡️ Multi-Vector Phishing Detection Hub — Operational Edition")
    gr.Markdown("Analyse comportementale et fusion de signaux hétérogènes pour analystes SOC.")
    
    with gr.Tabs():
        with gr.TabItem("🔍 Analyse de Liens (URLs)"):
            url_input = gr.Textbox(label="Collez l'URL suspecte à vérifier", placeholder="https://secure-login-update...")
            url_button = gr.Button("Lancer l'Analyse", variant="primary")
            url_output = gr.Textbox(label="Verdict")
            # Association à l'affichage textuel (index 1 du tuple de retour)
            url_button.click(lambda x: analyser_url_interface(x)[1], inputs=url_input, outputs=url_output)
            
        with gr.TabItem("📧 Analyse d'E-mails (NLP)"):
            email_input = gr.Textbox(label="Collez le texte de l'e-mail", lines=4)
            email_button = gr.Button("Scanner le Contenu", variant="primary")
            email_output = gr.Textbox(label="Verdict")
            # Association à l'affichage textuel (index 1 du tuple de retour)
            email_button.click(lambda x: analyser_email_interface(x)[1], inputs=email_input, outputs=email_output)
            
        with gr.TabItem("🔀 Fusion Multi-Modale (URL + Texte)"):
            gr.Markdown("### Analyse conjointe des surfaces d'attaques structurelles et sémantiques.")
            fuse_url = gr.Textbox(label="URL contenue dans le message", placeholder="https://...")
            fuse_email = gr.Textbox(label="Corps de texte de l'e-mail reçu", lines=5)
            fuse_button = gr.Button("Analyser la Menace Combinée", variant="stop")
            fuse_output = gr.TextArea(label="Rapport d'Audit Inter-Canaux SOC", lines=8)
            fuse_button.click(analyser_fusion_interface, inputs=[fuse_url, fuse_email], outputs=fuse_output)

if __name__ == "__main__":
    demo.launch()
