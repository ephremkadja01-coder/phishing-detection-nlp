# 🛡️ Threat Modeling & Robustness Manifesto

Ce document formalise l'analyse des menaces et la surface d'attaque de l'architecture du projet de détection multi-vectorielle du phishing, conformément aux exigences de la Cyber Data Science.

## 1. Profil de l'Attaquant et Scénarios d'Attaque

Le pipeline de détection intercepte deux profils de menaces distincts :
*   **Attaquant Opportuniste (Mass-Phishing) :** Utilise des outils d'automatisation de masse pour générer des vagues d'URLs et d'e-mails. Ce profil est intercepté à plus de 99% par l'extraction de l'Entropie de Shannon (DGA) et le vocabulaire d'urgence NLP.
*   **Attaquant Ciblé (Spear-Phishing / APT) :** Conçoit des vecteurs sur mesure pour contourner les défenses périmétriques. Ce profil utilise des techniques avancées d'évasion sémantique et structurelle.

## 2. Analyse de la Surface d'Attaque du Modèle (Adversarial Machine Learning)

Comme tout système basé sur l'apprentissage automatique, ce pipeline présente des vulnérabilités inhérentes que les attaquants peuvent chercher à exploiter :

### A. Attaques par Évasion (Adversarial Evasion)
L'adversaire modifie la structure de l'URL ou du texte pour passer sous les radars des seuils de décision statistiques :
*   **Contournement URL (Homographes et Typosquatting) :** Utilisation de caractères Punycode (`xn--`) ou de substitutions de caractères (`o` par `0`, `l` par `1`) pour imiter des domaines de confiance tout en maintenant une entropie basse.
*   **Contournement Sémantique (NLP Dilution) :** Injection de mots neutres, professionnels ou hautement légitimes au sein du texte d'un e-mail malveillant pour "diluer" la densité des termes d'urgence extraits par la matrice TF-IDF et forcer un Faux Négatif.

### B. Risque d'Empoisonnement du Modèle (Data Poisoning)
*   Si le pipeline est réentraîné de manière automatisée à partir de flux de données de production ou de collectes de passerelles de messagerie, un attaquant peut délibérément soumettre des milliers d'URLs malveillantes contenant des caractéristiques de sites légitimes pour corrompre la frontière de décision des arbres XGBoost lors de la prochaine compilation.

## 3. Contre-Mesures Métier et Barrières de Sécurité Implémentées

Pour atténuer ces risques et garantir la robustesse opérationnelle, l'architecture intègre des mécanismes de défense spécifiques :
1.  **Méta-Règle Asymétrique de Sécurité (OU Logique) :** Empêche l'évasion sur un seul canal. Si l'un des deux vecteurs (URL ou Email) franchit la barrière cyber de 80.0%, l'alerte maximale est levée, indépendamment de la neutralité de l'autre canal.
2.  **Sanctuarisation par Liste Blanche :** Protection absolue contre le typosquatting inversé ou les collisions statistiques pour les domaines d'autorité critique, garantissant un taux d'erreur nul sur les infrastructures certifiées.
3.  **Contrainte Dimensionnelle Stricte (`max_features=2500`) :** Empêche les attaques par déni de service (DoS) sur la mémoire vive (RAM) du processeur lors de l'évaluation de matrices creuses massives.

## 4. Usage Interdit et Gouvernance
*   Ce prototype ne doit en aucun cas être déployé comme l'unique point de contrôle d'une infrastructure corporate. Il doit impérativement être couplé à des analyses dynamiques de réputation DNS (WHOIS/ASN), des bacs à sable (sandboxing) et des vérifications de certificats TLS.
