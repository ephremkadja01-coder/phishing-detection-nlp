# 📊 Rapport d'Évaluation d'Élite, Calibration & Seuil Optimal

Ce document centralise l'analyse étanche de nos classificateurs après étalonnage Isotonic.

## 1. Tableau Comparatif des Performances (Seuils Métiers Optimisés)

| Modèle                 |   Seuil Appliqué |   Vrais Négatifs (Sains) |   Faux Positifs (Fausses Alertes) |   Faux Négatifs (Ratés Cyber) |   Vrais Positifs (Capturés) |   PR-AUC (%) |   Coût Asymétrique Total |   Coût Moyen |   Far Faux Négatifs |   Far Faux Positifs |
|:-----------------------|-----------------:|-------------------------:|----------------------------------:|------------------------------:|----------------------------:|-------------:|-------------------------:|-------------:|--------------------:|--------------------:|
| Random Forest (URLs)   |             0.12 |                    26834 |                               136 |                           139 |                       19965 |        99.83 |                     1526 |       0.0324 |                 137 |                   1 |
| XGBoost (URLs)         |             0.08 |                    26738 |                               232 |                           119 |                       19985 |        99.85 |                     1422 |       0.0302 |                 126 |                   2 |
| Random Forest (Emails) |             0.08 |                     7396 |                               451 |                            10 |                        8559 |        99.84 |                      551 |       0.0336 |                  17 |                  50 |
| XGBoost (Emails)       |             0.22 |                     7469 |                               378 |                            38 |                        8531 |        99.81 |                      758 |       0.0462 |                  23 |                  55 |

## 2. Note Technique MLOps
* Le **Seuil Appliqué** n'est plus fixé arbitrairement à 0.50. Il est calculé algorithmiquement pour minimiser les ratés de phishing (FN) qui portent un poids de risque 10x supérieur aux fausses alertes (FP).
