# 📊 Rapport d'Évaluation d'Élite, Calibration & Seuil Optimal

Ce document centralise l'analyse étanche de nos classificateurs après étalonnage Isotonic.

## 1. Tableau Comparatif des Performances (Seuils Métiers Optimisés)

| Modèle                 |   Seuil Appliqué |   Vrais Négatifs (Sains) |   Faux Positifs (Fausses Alertes) |   Faux Négatifs (Ratés Cyber) |   Vrais Positifs (Capturés) |   PR-AUC (%) |   Coût Asymétrique Total |   Coût Moyen |   Far Faux Négatifs |   Far Faux Positifs |
|:-----------------------|-----------------:|-------------------------:|----------------------------------:|------------------------------:|----------------------------:|-------------:|-------------------------:|-------------:|--------------------:|--------------------:|
| Random Forest (URLs)   |             0.06 |                    26758 |                                62 |                           106 |                       20233 |        99.85 |                     1122 |       0.0238 |                 109 |                   2 |
| XGBoost (URLs)         |             0.06 |                    26756 |                                64 |                           105 |                       20234 |        99.86 |                     1114 |       0.0236 |                 108 |                   0 |
| Random Forest (Emails) |             0.03 |                     3644 |                              3131 |                            70 |                        9653 |        93.94 |                     3831 |       0.2322 |                1554 |                  85 |
| XGBoost (Emails)       |             0.03 |                     3539 |                              3236 |                            89 |                        9634 |        97.09 |                     4126 |       0.2501 |                 272 |                  80 |

## 2. Note Technique MLOps
* Le **Seuil Appliqué** n'est plus fixé arbitrairement à 0.50. Il est calculé algorithmiquement pour minimiser les ratés de phishing (FN) qui portent un poids de risque 10x supérieur aux fausses alertes (FP).
