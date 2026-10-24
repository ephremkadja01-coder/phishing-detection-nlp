# 📊 Rapport d'Évaluation d'Élite, Calibration & Seuil Optimal

Ce document centralise l'analyse étanche de nos classificateurs après étalonnage Isotonic.

## 1. Tableau Comparatif des Performances (Seuils Métiers Optimisés)

| Modèle                 |   Seuil Appliqué |   Vrais Négatifs (Sains) |   Faux Positifs (Fausses Alertes) |   Faux Négatifs (Ratés Cyber) |   Vrais Positifs (Capturés) |   PR-AUC (%) |   Coût Asymétrique Total |   Coût Moyen |   Far Faux Négatifs |   Far Faux Positifs |
|:-----------------------|-----------------:|-------------------------:|----------------------------------:|------------------------------:|----------------------------:|-------------:|-------------------------:|-------------:|--------------------:|--------------------:|
| Random Forest (URLs)   |              0.5 |                    26952 |                                18 |                           191 |                       19913 |        99.79 |                     1928 |       0.041  |                 152 |                   2 |
| XGBoost (URLs)         |              0.5 |                    26960 |                                10 |                           179 |                       19925 |        99.83 |                     1800 |       0.0382 |                 138 |                   2 |
| Random Forest (Emails) |              0.5 |                     7681 |                               166 |                            97 |                        8472 |        99.84 |                     1136 |       0.0692 |                  17 |                  50 |
| XGBoost (Emails)       |              0.5 |                     7646 |                               201 |                           135 |                        8434 |        99.81 |                     1551 |       0.0945 |                  23 |                  55 |

## 2. Note Technique MLOps
* Le **Seuil Appliqué** n'est plus fixé arbitrairement à 0.50. Il est calculé algorithmiquement pour minimiser les ratés de phishing (FN) qui portent un poids de risque 10x supérieur aux fausses alertes (FP).
