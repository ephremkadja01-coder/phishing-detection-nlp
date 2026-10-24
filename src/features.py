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

def calculer_distance_levenshtein(s1, s2):
    """Calcul de distance pour reperer le typosquatting."""
    if len(s1) < len(s2):
        return calculer_distance_levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
        
    return previous_row[-1]

def extraire_indices_url(url_brute):
    """
    Extrait un vecteur etendu de 18 caracteristiques cyber d'une URL.
    """
    indices = {}
    url_chaine = str(url_brute).lower().strip()
    
    longueur = len(url_chaine)
    indices['longueur_url'] = longueur
    
    nb_pts = url_chaine.count('.')
    indices['nb_points'] = nb_pts
    indices['nb_tirets'] = url_chaine.count('-')
    indices['nb_arobases'] = url_chaine.count('@')
    indices['nb_interrogations'] = url_chaine.count('?')
    indices['nb_egaux'] = url_chaine.count('=')
    indices['nb_slashs'] = url_chaine.count('/')
    
    try:
        analyse_url = urlparse(url_chaine if url_chaine.startswith(('http://', 'https://')) else 'https://' + url_chaine)
        domaine = analyse_url.hostname if analyse_url.hostname else analyse_url.netloc
        indices['profondeur_sous_domaine'] = domaine.count('.')
        
        tld = domaine.split('.')[-1] if '.' in domaine else ''
        tlds_suspects = {'tk', 'ml', 'ga', 'cf', 'gq', 'xyz', 'top', 'work', 'click', 'link', 'download', 'win', 'bid'}
        indices['tld_suspect'] = 1 if tld in tlds_suspects else 0
        indices['has_punycode'] = 1 if "xn--" in domaine else 0
        
        # Verification des ports web standards
        ports_standards = [80, 443]
        if "xn--" in domaine:
            indices['has_port_non_standard'] = 1
        elif analyse_url.port and (analyse_url.port not in ports_standards):
            indices['has_port_non_standard'] = 1
        else:
            indices['has_port_non_standard'] = 0
        
        nom_base = domaine.split('.')[0]
        distances = [calculer_distance_levenshtein(nom_base, marque) for marque in ['google', 'laposte', 'github', 'microsoft', 'paypal']]
        indices['min_edit_dist_marques'] = int(min(distances)) if distances else 0
        
    except Exception:
        indices['profondeur_sous_domaine'] = nb_pts
        indices['tld_suspect'] = 0
        indices['has_punycode'] = 0
        indices['has_port_non_standard'] = 0
        indices['min_edit_dist_marques'] = 0
        domaine = url_chaine
        
    nb_chiffres = sum(c.isdigit() for c in url_chaine)
    nb_lettres = sum(c.isalpha() for c in url_chaine)
    indices['ratio_chiffres_lettres'] = round(nb_chiffres / (nb_lettres + 1), 4)
    
    caracteres_speciaux = url_chaine.count('.') + url_chaine.count('-') + url_chaine.count('?') + url_chaine.count('=') + url_chaine.count('/')
    indices['ratio_caracteres_speciaux'] = round(caracteres_speciaux / (longueur + 1), 4)
    
    mots_suspects = ['login', 'verify', 'secure', 'account', 'banking', 'update', 'signin', 'compte']
    indices['contient_mot_suspect'] = 1 if any(mot in url_chaine for mot in mots_suspects) else 0
    indices['entropie_url'] = round(calculer_entropie_shannon(url_chaine), 4)
    
    motif_ip = r'(?:\d{1,3}\.){3}\d{1,3}'
    indices['contient_ip'] = 1 if re.search(motif_ip, domaine) else 0
    indices['utilise_https'] = 1 if url_chaine.startswith('https') else 0
    
    return indices

def nettoyer_texte_email(texte_brut):
    texte = str(texte_brut)
    texte = re.sub(r'https?://\S+|www\.\S+', ' <URL> ', texte)
    texte = re.sub(r'[\w.+-]+@[\w-]+\.[\w.-]+', ' <EMAIL> ', texte)
    texte = re.sub(r'[$€£¥]\s?\d+', ' <MONEY> ', texte)
    texte = re.sub(r'[^a-zA-Z0-9!?\s<>]', '', texte)
    texte = re.sub(r'\s+', ' ', texte).strip()
    return texte

def extraire_indices_semantiques_email(texte_brut):
    indices = {}
    texte_normal = str(texte_brut)
    texte_nettoye = nettoyer_texte_email(texte_normal)
    
    indices['num_urls'] = texte_nettoye.count('<URL>')
    indices['num_emails'] = texte_nettoye.count('<EMAIL>')
    indices['num_exclamations'] = texte_nettoye.count('!')
    indices['num_questions'] = texte_nettoye.count('?')
    
    mots = texte_normal.split()
    mots_maj = [m for m in mots if m.isupper() and len(m) > 1]
    indices['num_uppercase_words'] = len(mots_maj)
    indices['uppercase_ratio'] = round(len(mots_maj) / (len(mots) + 1), 4)
    
    motifs_urgence = r'urgent|immediately|suspended|verify|click here|action required|final warning'
    matches_urgence = re.findall(motifs_urgence, texte_normal.lower())
    indices['num_urgency_matches'] = len(matches_urgence)
    indices['has_urgency_phrase'] = 1 if len(matches_urgence) > 0 else 0
    
    indices['has_money_amount'] = 1 if '<MONEY>' in texte_nettoye else 0
    indices['has_attachment_mention'] = 1 if re.search(r'attachment|pj|piece jointe|invoice', texte_normal.lower()) else 0
    
    return indices
