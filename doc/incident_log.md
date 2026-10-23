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
