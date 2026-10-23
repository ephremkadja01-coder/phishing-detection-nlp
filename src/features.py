import re
from urllib.parse import urlparse

def extraire_indices_url(url_brute):
    """
    Extrait les caracteristiques numeriques d'une URL pour le Machine Learning.
    """
    indices = {}
    url_chaine = str(url_brute)
    
    # 1. Caracteristiques de longueur
    indices['longueur_url'] = len(url_chaine)
    
    # 2. Comptage des caracteres suspects
    indices['nb_points'] = url_chaine.count('.')
    indices['nb_tirets'] = url_chaine.count('-')
    indices['nb_arobases'] = url_chaine.count('@')
    indices['nb_interrogations'] = url_chaine.count('?')
    indices['nb_egaux'] = url_chaine.count('=')
    
    # 3. Detection d'une adresse IP dans le nom de domaine
    motif_ip = r'(?:\d{1,3}\.){3}\d{1,3}'
    try:
        analyse_url = urlparse(url_chaine)
        domaine = analyse_url.netloc if analyse_url.netloc else url_chaine.split('/')[0]
        indices['contient_ip'] = 1 if re.search(motif_ip, domaine) else 0
    except Exception:
        indices['contient_ip'] = 0
        
    # 4. Verification de la presence de HTTPS
    indices['utilise_https'] = 1 if url_chaine.lower().startswith('https') else 0
    
    return indices

# continuer avec cette fonction de nettoyage de texte pour les emails
import re

def nettoyer_texte_email(texte_brut):
    """
    Nettoie le texte d'un email (minuscules, suppression ponctuation et caracteres speciaux).
    """
    # 1. Passage en minuscules
    texte = str(texte_brut).lower()
    
    # 2. Suppression des caracteres speciaux et de la ponctuation (conservation des lettres et espaces)
    texte = re.sub(r'[^a-zA-Z\s]', '', texte)
    
    # 3. Suppression des espaces multiples
    texte = re.sub(r'\s+', ' ', texte).strip()
    
    return texte
