import re
import math
from urllib.parse import urlparse

def calculer_entropie_shannon(chaine):
    """
    Calcule l'entropie de Shannon pour detecter l'aleatoire (DGA) dans une URL.
    """
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
    """
    Extrait un vecteur etendu de caractéristiques structurelles et cyber d'une URL.
    """
    indices = {}
    url_chaine = str(url_brute).lower().strip()
    
    # 1. Base lexicale et longueur
    longueur = len(url_chaine)
    indices['longueur_url'] = longueur
    
    nb_pts = url_chaine.count('.')
    indices['nb_points'] = nb_pts
    indices['nb_tirets'] = url_chaine.count('-')
    indices['nb_arobases'] = url_chaine.count('@')
    indices['nb_interrogations'] = url_chaine.count('?')
    indices['nb_egaux'] = url_chaine.count('=')
    indices['nb_slashs'] = url_chaine.count('/')
    
    # 2. ENRICHISSEMENT CYBER AVANCÉ (Priorité 3)
    # Calcul de la profondeur de sous-domaine
    try:
        analyse_url = urlparse(url_chaine if url_chaine.startswith(('http://', 'https://')) else 'https://' + url_chaine)
        domaine = analyse_url.netloc
        indices['profondeur_sous_domaine'] = domaine.count('.')
    except Exception:
        indices['profondeur_sous_domaine'] = nb_pts
        domaine = url_chaine
        
    # Calcul des ratios statistiques
    nb_chiffres = sum(c.isdigit() for c in url_chaine)
    nb_lettres = sum(c.isalpha() for c in url_chaine)
    indices['ratio_chiffres_lettres'] = round(nb_chiffres / (nb_lettres + 1), 4)
    
    caracteres_speciaux = url_chaine.count('.') + url_chaine.count('-') + url_chaine.count('?') + url_chaine.count('=') + url_chaine.count('/')
    indices['ratio_caracteres_speciaux'] = round(caracteres_speciaux / (longueur + 1), 4)
    
    # Analyse lexicale des mots-cles d'ingénierie sociale
    mots_suspects = ['login', 'verify', 'secure', 'account', 'banking', 'update', 'signin', 'compte']
    indices['contient_mot_suspect'] = 1 if any(mot in url_chaine for mot in mots_suspects) else 0
    
    # Statistiques de desordre (Shannon)
    indices['entropie_url'] = round(calculer_entropie_shannon(url_chaine), 4)
    
    # 3. Verifications de l'hote et protocoles
    motif_ip = r'(?:\d{1,3}\.){3}\d{1,3}'
    indices['contient_ip'] = 1 if re.search(motif_ip, domaine) else 0
    indices['utilise_https'] = 1 if url_chaine.startswith('https') else 0
    
    return indices

def nettoyer_texte_email(texte_brut):
    """
    Normalise le texte pour le traitement NLP.
    """
    texte = str(texte_brut).lower()
    texte = re.sub(r'[^a-zA-Z\s]', '', texte)
    texte = re.sub(r'\s+', ' ', texte).strip()
    return texte
