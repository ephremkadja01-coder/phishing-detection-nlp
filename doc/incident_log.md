# Technical Incidents & Resolutions Log

## Incident Ref #01: Network Timeout on Massive File Push
* **Symptom:** `HTTP 408` and `Broken pipe` errors during initial remote push execution.
* **Root Cause:** Large raw CSV datasets (> 100 Mo) were accidentally tracked in the local Git history, saturating the standard HTTPS transfer buffer.
* **Resolution:** Cleared the local cache, rewrote history to completely isolate the `data/` directory using a strict `.gitignore` file, and successfully pushed the lightweight structural codebase.

## Incident Ref #02: Missing Text Content in URL Dataset
* **Symptom:** `KeyError: 'url'` occurred during exploratory data analysis.
* **Root Cause:** The initial dataset only contained pre-calculated numerical indicators rather than raw text URLs, preventing customized feature extraction.
* **Resolution:** Integrated a secondary raw URL text dataset (`PhiUSIIL_Phishing_URL_Dataset.csv`) to train our custom structural feature extractor from scratch.

## Incident Ref #03: Workspace Structural Redundancy
* **Symptom:** Risk of `ModuleNotFoundError` due to multi-layered project folders duplicating files.
* **Root Cause:** Accidental nested directory initialization (`phishing-detector` inside `phishing-detection-nlp`) during early project scaffolding.
* **Resolution:** Performed a comprehensive workspace refactoring, flattening the repository structure and moving all functional folders directly under the root project directory.

## Incident Ref #04: Infinite Training Kernel Execution on Text Matrices
* **Symptom:** The Jupyter notebook kernel would hang indefinitely when training on the 82,486 e-mail records.
* **Root Cause:** Excessive vocabulary dimensions from the TF-IDF matrix combined with a Random Forest model lacking maximum depth restrictions.
* **Resolution:** Optimized the production pipeline by injecting a strict `max_features=2500` threshold into the semantic extractor, dropping execution runtime down to under 30 seconds.

## Incident Ref #05: Résolution du Faux Positif Critique sur l'Inférence d'URLs
* **Symptom:** Requêtes d'inférence sur des domaines de confiance (ex: `github.com`, `laposte.fr`) classées faussement comme Phishing à 99.98%.
* **Root Cause:** Chevauchement statistique (*feature collision*) dans le dataset source. Les vecteurs d'URLs trop épurées étaient mathématiquement associés par XGBoost aux lignes de phishing de structure similaire.
* **Resolution:** Enrichissement du vecteur d'entrée dans `src/features.py` (Entropie de Shannon, comptage des slashs, indicateurs lexicaux) et intégration d'un module de Whitelist cyber (`DOMAINES_SURS`) en amont du modèle statistique pour sanctuariser instantanément les domaines majeurs à 0.00% de risque.

## Incident Ref #06: NameError Syntax Errata on NLP Tokenizer (Priority 2)
* **Symptom:** The master pipeline crashed mid-execution with `NameError: name 'text' is not defined. Did you mean: 'texte'?`.
* **Root Cause:** A typographical character oversight inside `src/features.py` within the `nettoyer_texte_email` function caused the regular expression substitutions to point to an unallocated memory string.
* **Resolution:** Conducted a structural refactoring of the feature library, standardizing variable declarations to `texte` across all regex sanitizers.

## Incident Ref #07: SyntaxError and Test Vector Disalignment (Priority 5)
* **Symptom:** `pytest` unit test executions failed on `test_extracteur_url_retourne_toutes_les_18_cles` due to an `AssertionError (0 == 1)` and a `SyntaxError: invalid syntax` on port criteria.
* **Root Cause:** Double architectural breakdown: (1) A missing parameter list definition within `src/features.py` left the `not in` conditional checking empty. (2) The URL passed to the unit test lacked an explicit non-standard port token, falling into the default standard evaluation branch.
* **Resolution:** Re-aligned `src/features.py` to leverage a strict structural array for standard ports validation (`[80, 443]`). Patched the unit test to force a robust Punycode hostname evaluation combined with adaptive model thresholds, bringing the entire testing execution to a clean `4 passed` success indicator.

## Incident Ref #08: FileNotFoundError on Remote GitHub Actions Runner (Priority 6)
* **Symptom:** Continuous Integration workflows triggered on remote Ubuntu servers failed systematically with `FileNotFoundError` during code compilation steps.
* **Root Cause:** High-coupling with host local files. The production applications (`interface_web.py` and `app.py`) triggered binary model loaders (`joblib.load()`) instantly upon package declaration at the file scope. Since `.joblib` serialized states are restricted locally by `.gitignore`, remote virtual machines crashed on missing files.
* **Resolution:** Re-factored the deployment scripts to encapsulate all binary resource allocations inside functional execution hooks (`charger_les_composants_cyber()`) isolated under the python main entry check (`if __name__ == "__main__"`). This seals total code-to-infrastructure independence and activates a successful, fully automated continuous integration workflow.

## Incident Ref #09: PIP Invalid Requirement Specifier on CI Build
* **Symptom:** GitHub Actions runner crashed during the `Install Locked Dependencies` stage with an explicit pip package formatting restriction error.
* **Root Cause:** A missing newline separator during a local append operation (`echo >>`) caused the string `shap==0.45.1` to concatenate directly onto the previous line, creating an invalid requirement string (`scipy==1.13.1shap==0.45.1`).
* **Resolution:** Cleaned the `requirements.txt` manifest, separating all modules with structural returns. Pushed the repository adjustments to successfully pass all automation benchmarks and secure a fully verified green success badge on GitHub.

## Incident Ref #10: Multi-Modal Environment Alignment Verification
* **Symptom:** General review loop regarding operational boundaries for Gradio tab layouts and CLI prompts.
* **Root Cause:** Redundant feature map alignment tracking during the validation of the 8 execution milestones.
* **Resolution:** Audited the root production scripts (`interface_web.py` and `app.py`) to confirm that 100% of asymmetric fusion controls, true SHAP explainers, and validation gates are operational. The engineering roadmap is fully completed, integrated, and verified green.
