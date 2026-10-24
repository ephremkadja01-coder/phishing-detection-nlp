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
    Extrait un vecteur etendu de caracteristiques structurelles et cyber d'une URL.
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
        domaine = analyse_url.netloc
        indices['profondeur_sous_domaine'] = domaine.count('.')
    except Exception:
        indices['profondeur_sous_domaine'] = nb_pts
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
    """
    Nettoie le texte de l'email en preservant et en tokenisant les signatures cyber cles.
    """
    texte = str(texte_brut)
    
    # Preservation des jetons structurels
    texte = re.sub(r'https?://\S+|www\.\S+', ' <URL> ', texte)
    texte = re.sub(r'[\w.+-]+@[\w-]+\.[\w.-]+', ' <EMAIL> ', texte)
    texte = re.sub(r'[$€£¥]\s?\d+', ' <MONEY> ', texte)
    
    # Nettoyage conservateur
    texte = re.sub(r'[^a-zA-Z0-9!?\s<>]', '', texte)
    texte = re.sub(r'\s+', ' ', texte).strip()
    return texte

def extraire_indices_semantiques_email(texte_brut):
    """
    Extrait un vecteur de metriques explicites issues du contenu textuel de l'email (Priorite 2).
    """
    indices = {}
    texte_normal = str(texte_brut)
    texte_nettoye = nettoyer_texte_email(texte_normal)
    
    # Comptages explicites des signaux cyber
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
