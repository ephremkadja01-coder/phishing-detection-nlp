import sys
import os
import pytest

# Forcer l'arborescence racine dans le path systeme de Python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import extraire_indices_url, calculer_distance_levenshtein, nettoyer_texte_email
from src.pipeline import fusionner_vecteurs_menace

def test_extracteur_url_retourne_toutes_les_18_cles():
    indices = extraire_indices_url("https://xn--ggle-cca.xyz")
    assert indices["tld_suspect"] == 1
    assert indices["has_punycode"] == 1
    assert "min_edit_dist_marques" in indices
    assert len(indices) == 18

def test_distance_levenshtein_exacte():
    assert calculer_distance_levenshtein("g00gle", "google") == 2
    assert calculer_distance_levenshtein("github", "github") == 0

def test_clean_text_preserves_indicators():
    texte_net = nettoyer_texte_email("Urgent call to click here http://test.com with $500 !!")
    assert "<URL>" in texte_net
    assert "<MONEY>" in texte_net
    assert "!!" in texte_net

def test_fusion_or_gate():
    pred, score = fusionner_vecteurs_menace(score_url=90.0, score_email=5.0)
    assert pred == 1
    assert score == 90.0
