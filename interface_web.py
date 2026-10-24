import os
import joblib
import gradio as gr
from src.features import extraire_indices_url, nettoyer_texte_email
from src.pipeline import verifier_menace_url, verifier_menace_email, fusionner_vecteurs_menace

dossier_modeles = "models"

# 1. Chargement des composants de production
mod_url = joblib.load(os.path.join(dossier_modeles, "url_model_xgb.joblib"))
vec_email = joblib.load(os.path.join(dossier_modeles, "email_vectoriseur.joblib"))
mod_email = joblib.load(os.path.join(dossier_modeles, "email_model_xgb.joblib"))

# Récupération des noms de variables pour l'explicabilité des URLs
FEATURES_URL_NOMS = mod_url.get_booster().feature_names

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
            return 0.0, "✅ SÉCURISÉ : Whitelist institutionnelle active."
        prediction, pourcentage = verifier_menace_url(url, mod_url, extraire_indices_url)
        return float(pourcentage), f"Verdict : {'⚠️ Phishing' if prediction == 1 else '✅ Sain'}"
    except Exception as e:
        return 0.0, f"Erreur : {str(e)}"

def analyser_email_interface(email):
    try:
        prediction, pourcentage = verifier_menace_email(email, vec_email, mod_email, nettoyer_texte_email)
        return float(pourcentage), f"Verdict : {'⚠️ Phishing' if prediction == 1 else '✅ Sain'}"
    except Exception as e:
        return 0.0, f"Erreur : {str(e)}"

def analyser_fusion_interface(url, email):
    try:
        # 1. Calcul individuel des canaux
        score_url, _ = analyser_url_interface(url)
        score_email, _ = analyser_email_interface(email)
        
        # 2. Appel de la Meta-Règle de fusion présente dans src/pipeline.py
        prediction_fusionnee, score_fusionne = fusionner_vecteurs_menace(score_url, score_email)
        
        verdict = "🚨 ALERTE CRITIQUE : Phishing Combiné Détecté !" if prediction_fusionnee == 1 else "🟢 SÉCURISÉ : Les signaux combinés sont stables."
        
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

# 3. Construction de l'interface visuelle multicouche
with gr.Blocks(theme=gr.themes.Soft(), title="Cyber-Detector App Hub") as demo:
    gr.Markdown("# 🛡️ Multi-Vector Phishing Detection Hub — Operational Edition")
    gr.Markdown("Analyse comportementale et fusion de signaux hétérogènes pour analystes SOC.")
    
    with gr.Tabs():
        with gr.TabItem("🔍 Analyse de Liens (URLs)"):
            url_input = gr.Textbox(label="Collez l'URL suspecte à vérifier", placeholder="https://secure-login-update...")
            url_button = gr.Button("Lancer l'Analyse", variant="primary")
            url_output = gr.Textbox(label="Verdict")
            url_button.click(lambda x: analyser_url_interface(x)[1], inputs=url_input, outputs=url_output)
            
        with gr.TabItem("📧 Analyse d'E-mails (NLP)"):
            email_input = gr.Textbox(label="Collez le texte de l'e-mail", lines=4)
            email_button = gr.Button("Scanner le Contenu", variant="primary")
            email_output = gr.Textbox(label="Verdict")
            email_button.click(lambda x: analyser_email_interface(x)[1], inputs=email_input, outputs=email_output)
            
        # NOUVEAUTÉ ADAPTÉE À L'AUDIT : L'onglet de fusion opérationnelle
        with gr.TabItem("🔀 Fusion Multi-Modale (URL + Texte)"):
            gr.Markdown("### Analyse conjointe des surfaces d'attaques structurelles et sémantiques.")
            fuse_url = gr.Textbox(label="URL contenue dans le message", placeholder="https://...")
            fuse_email = gr.Textbox(label="Corps de texte de l'e-mail reçu", lines=5, placeholder="Copier le texte ici...")
            fuse_button = gr.Button("Analyser la Menace Combinée", variant="stop")
            fuse_output = gr.TextArea(label="Rapport d'Audit Inter-Canaux SOC", lines=8)
            fuse_button.click(analyser_fusion_interface, inputs=[fuse_url, fuse_email], outputs=fuse_output)

if __name__ == "__main__":
    demo.launch()
