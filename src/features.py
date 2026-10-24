import re
import math
from urllib.parse import urlparse

def calculer_entropie_shannon(chaine):
    if not chaine:
        return 0.0
    frequences = {}
    for caractere in chaine:
        frequences[caractere] = frequences.get(caractere, 0) + 1
    entropie = 0.0
    total = len(chaine)
    for nb in frequences.values():
        p = nb / total
        entropie -= p * math.log2(p)
    return entropie

def extraire_indices_url(url_brute):
    indices = {}
    url_chaine = str(url_brute).lower()
    
    # Variables d'origine alignees
    indices['longueur_url'] = len(url_chaine)
    indices['nb_points'] = url_chaine.count('.')
    indices['nb_tirets'] = url_chaine.count('-')
    indices['nb_arobases'] = url_chaine.count('@')
    indices['nb_interrogations'] = url_chaine.count('?')
    indices['nb_egaux'] = url_chaine.count('=')
    
    # NOUVEAUTÉS : Enrichissement Cyber pour casser le chevauchement (Audit Copilot & Google AI)
    indices['nb_slashs'] = url_chaine.count('/')
    indices['entropie_url'] = calculer_entropie_shannon(url_chaine)
    
    # Verification des domaines de confiance majeurs (Securisation anti-faux positifs)
    domaines_confiance = ['github', 'google', 'microsoft', 'apple', 'linkedin', 'facebook']
    indices['est_domaine_connu'] = 1 if any(dom in url_chaine for dom in domaines_confiance) else 0
    
    # Direct IP Bypass detection
    motif_ip = r'(?:\d{1,3}\.){3}\d{1,3}'
    try:
        analyse_url = urlparse(url_chaine)
        domaine = analyse_url.netloc if analyse_url.netloc else url_chaine.split('/')
        indices['contient_ip'] = 1 if re.search(motif_ip, domaine) else 0
    except Exception:
        indices['contient_ip'] = 0
        
    indices['utilise_https'] = 1 if url_chaine.startswith('https') else 0
    
    return indices

def nettoyer_texte_email(texte_brut):
    texte = str(texte_brut).lower()
    texte = re.sub(r'[^a-zA-Z\s]', '', texte)
    texte = re.sub(r'\s+', ' ', texte).strip()
    return texte
