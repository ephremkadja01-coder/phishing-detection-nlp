import sys
import os

# Raccordement chirurgical au dossier racine pour les imports src/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import extraire_indices_url, calculer_entropie_shannon, nettoyer_texte_email
from src.pipeline import fusionner_vecteurs_menace

def test_extracteur_url_retourne_cles_attendues():
    """
    Vérifie que l'extracteur produit l'intégralité des 14 signatures requises par XGBoost.
    """
    url_test = "https://laposte.fr"
    indices = extraire_indices_url(url_test)
    
    cles_requises = [
        "longueur_url", "nb_points", "nb_tirets", "nb_arobases", "nb_interrogations",
        "nb_egaux", "nb_slashs", "profondeur_sous_domaine", "ratio_chiffres_lettres",
        "ratio_caracteres_speciaux", "contient_mot_suspect", "entropie_url",
        "contient_ip", "utilise_https"
    ]
    
    for cle in cles_requises:
        assert cle in indices, f"Erreur MLOps : La clé critique {cle} est absente de l'extraction."

def test_extracteur_url_gere_cas_malformes():
    """
    Valide la résilience de l'extracteur face aux chaînes vides ou corrompues.
    """
    indices_vides = extraire_indices_url("")
    assert indices_vides["longueur_url"] == 0
    assert indices_vides["contient_ip"] == 0
    assert indices_vides["utilise_https"] == 0

def test_entropie_shannon_coherente():
    """
    Vérifie mathématiquement que l'entropie détecte l'ordre (basse) vs le désordre (haute).
    """
    entropie_ordonnee = calculer_entropie_shannon("aaaaa")
    # Une chaîne uniforme doit avoir une surprise / entropie de 0.0
    assert entropie_ordonnee == 0.0
    
    entropie_aleatoire = calculer_entropie_shannon("hZ78qD29x")
    assert entropie_aleatoire > 2.0

def test_nettoyage_email_preserve_signaux_lexicaux():
    """
    S'assure que le pré-nettoyage NLP conserve les mots sans les détruire par agressivité.
    """
    texte_brut = "⚠️ URGENT !! Accès SUSPENDU à votre compte.. !!"
    texte_nettoye = nettoyer_texte_email(texte_brut)
    
    assert "urgent" in texte_nettoye
    assert "suspendu" in texte_nettoye
    assert "compte" in texte_nettoye
    assert "⚠️" not in texte_nettoye
    assert "!!" not in texte_nettoye

def test_fusionneur_applique_meta_regle_or():
    """
    Vérifie la robustesse de la barrière asymétrique "OU" globale.
    """
    # Cas 1 : Un canal est ultra-suspect (URL), l'autre est sain. La fusion doit bloquer.
    pred_1, score_1 = fusionner_vecteurs_menace(score_url=95.0, score_email=10.0, seuil_cyber=80.0)
    assert pred_1 == 1
    assert score_1 == 95.0
    
    # Cas 2 : Les deux canaux sont en dessous du seuil, calcul de la moyenne pondérée cyber
    # (30 * 0.6) + (20 * 0.4) = 18 + 8 = 26.0
    pred_2, score_2 = fusionner_vecteurs_menace(score_url=30.0, score_email=20.0, seuil_cyber=80.0)
    assert pred_2 == 0
    assert score_2 == 26.0
