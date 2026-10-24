# 📊 Rapport d'Évaluation d'Élite & Audit Métier

Ce document centralise l'analyse étanche de nos classificateurs face au modèle de coût asymétrique.

## 1. Tableau Comparatif des Performances

| Modèle                 |   Vrais Négatifs (Sains) |   Faux Positifs (Fausses Alertes) |   Faux Négatifs (Ratés Cyber) |   Vrais Positifs (Capturés) |   PR-AUC (%) |   Coût Asymétrique Total |   Coût Moyen par Requête |   Far Faux Négatifs (Critiques) |   Far Faux Positifs (Bloquants) |
|:-----------------------|-------------------------:|----------------------------------:|------------------------------:|----------------------------:|-------------:|-------------------------:|-------------------------:|--------------------------------:|--------------------------------:|
| Random Forest (URLs)   |                    26918 |                                52 |                           173 |                       19931 |        99.76 |                     1782 |                   0.0379 |                             143 |                               6 |
| XGBoost (URLs)         |                    26951 |                                19 |                           175 |                       19929 |        99.82 |                     1769 |                   0.0376 |                             140 |                               2 |
| Random Forest (Emails) |                     7453 |                               151 |                           113 |                        8434 |        99.79 |                     1281 |                   0.0793 |                               0 |                              17 |
| XGBoost (Emails)       |                     7317 |                               287 |                           106 |                        8441 |        99.7  |                     1347 |                   0.0834 |                              13 |                              39 |

## 2. Guide d'Interprétation Cyber
* **PR-AUC** : Plus il est proche de 100%, plus le modèle sépare efficacement le phishing sans générer de fausses alertes.
* **Coût Asymétrique** : Calculé sur l'hypothèse qu'un Faux Négatif (FN) coûte **10x plus cher** qu'un Faux Positif (FP).
* **Far Faux Négatifs** : Représente le pire scénario (le modèle est sûr à 90%+ qu'une URL est saine alors qu'elle est malveillante).
