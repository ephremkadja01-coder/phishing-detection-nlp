import re
from urllib.parse import urlparse

def extraire_indices_url(url_brute):
    indices = {}
    url_chaine = str(url_brute)
    
    # 1. Calcul de la longueur
    indices['longueur_url'] = len(url_chaine)
    
    # 2. Comptage des caracteres suspects
    indices['nb_points'] = url_chaine.count('.')
    indices['nb_tirets'] = url_chaine.count('-')
    indices['nb_arobases'] = url_chaine.count('@')
    indices['nb_interrogations'] = url_chaine.count('?')
    indices['nb_egaux'] = url_chaine.count('=')
    
    # 3. Detection d'une adresse IP directe
    motif_ip = r'(?:\d{1,3}\.){3}\d{1,3}'
    try:
        analyse_url = urlparse(url_chaine)
        domaine = analyse_url.netloc if analyse_url.netloc else url_chaine.split('/')[0]
        indices['contient_ip'] = 1 if re.search(motif_ip, domaine) else 0
    except Exception:
        indices['contient_ip'] = 0
        
    # 4. Verification du protocole HTTPS
    indices['utilise_https'] = 1 if url_chaine.lower().startswith('https') else 0
    
    return indices

def nettoyer_texte_email(texte_brut):
    # 1. Passage en minuscules
    texte = str(texte_brut).lower()
    
    # 2. Suppression de la ponctuation et des caracteres speciaux
    texte = re.sub(r'[^a-zA-Z\s]', '', texte)
    
    # 3. Nettoyage des espaces en trop
    texte = re.sub(r'\s+', ' ', texte).strip()
    
    return texte
