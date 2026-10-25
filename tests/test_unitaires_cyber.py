import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import extraire_indices_url, calculer_distance_levenshtein, nettoyer_texte_email
from src.pipeline import fusionner_vecteurs_menace

def test_extracteur_url_retourne_toutes_les_18_cles():
    indices = extraire_indices_url("https://example.com")
    assert "tld_suspect" in indices
    assert len(indices) == 18

def test_isolation_has_port_non_standard_cyber():
    # Cas 1 : URL avec un port non-standard explicite
    ind_non_standard = extraire_indices_url("http://example.com:8080")
    assert ind_non_standard["has_port_non_standard"] == 1

    ind_sans_port = extraire_indices_url("http://example.com")
    assert ind_sans_port["has_port_non_standard"] == 0
    
    # Cas 2 : URL Punycode pure sans aucun port reseau
    ind_punycode = extraire_indices_url("http://xn--mnchen-3ya.de")
    assert ind_punycode["has_punycode"] == 1
    assert ind_punycode["has_port_non_standard"] == 0
    
    # Cas 3 : URL utilisant un port standardise
    ind_standard = extraire_indices_url("https://example.com:443")
    assert ind_standard["has_port_non_standard"] == 0

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
