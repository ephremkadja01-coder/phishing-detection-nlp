# Multi-Vector Phishing Detection

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-used-orange?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-used-green)](https://xgboost.readthedocs.io/)
[![Gradio](https://img.shields.io/badge/Gradio-used-orange)](https://www.gradio.app/)
[![pytest](https://img.shields.io/badge/pytest-used-purple?logo=pytest&logoColor=white)](https://pytest.org/)
[![Docker](https://img.shields.io/badge/Docker-supported-blue?logo=docker&logoColor=white)](https://www.docker.com/)

A machine-learning prototype for classifying suspicious URLs and phishing emails. It includes an 18-feature URL extractor, a hybrid TF-IDF and explicit-feature email pipeline, isotonic-calibrated Random Forest and XGBoost models, asymmetric evaluation, governance reports, SHAP explanations, URL/email score fusion, a command-line interface, and a Gradio dashboard.

> **Important:** This is an educational and experimental prototype, not a production security product. Predictions, explanations, and probabilities can be wrong. The project does not check live domain reputation, DNS, certificates, redirects, or page contents. Do not use it as the only basis for blocking links, handling email, or making security decisions.

## Business Context

Phishing can arrive through links or deceptive email content. This project explores two different detection approaches:

- A structural model looks for patterns in a URL, such as its length, punctuation, entropy, and HTTPS usage.
- A text model uses TF-IDF features from an email body to identify patterns associated with phishing.

The two detection paths are trained independently. The CLI and web dashboard can analyze a URL and an email separately or combine their scores with an asymmetric fusion rule. The project is intended for learning and experimentation; it does not query live domain-reputation services, inspect web page contents, or provide a complete email security workflow.

## Repository Structure

```text
phishing-detection-nlp/
├── app.py                       # Command-line inference interface
├── interface_web.py             # Gradio web interface
├── train_and_save_models.py     # Data preparation, training, and model saving
├── requirements.txt             # Environment export with local paths; not portable
├── Dockerfile                   # Container image for the Gradio interface
├── Makefile                     # Install, train, test, serve, and clean shortcuts
├── src/
│   ├── features.py              # 18 URL features and hybrid email features
│   ├── model.py                 # Isotonic-calibrated model training
│   ├── pipeline.py              # Inference and URL/email score fusion
│   ├── explain.py               # SHAP contribution calculations
│   ├── evaluation.py            # Asymmetric metrics and threshold search
│   └── drift.py                 # KS-based prediction-distribution comparison
├── experimental/
│   └── fairness.py              # Experimental false-positive-rate group comparison
├── tests/
│   ├── test_fairness_audit.py   # Experimental urgency-group fairness test
│   ├── test_unitaires_cyber.py  # URL features, text cleaning, and fusion tests
│   └── test_model_loading.py    # Smoke check for saved model artifacts
├── .github/workflows/ci.yml     # GitHub Actions dependency audit and test job
├── doc/                         # Business logic, threat model, evaluation, incident log
├── notebooks/                   # Data understanding, preparation, modeling, evaluation
├── data/
│   ├── README.md                # Data governance and checksum ledger
│   └── raw/                     # Local source datasets (not tracked by Git)
└── models/
    ├── metadata.json            # Environment and model-threshold metadata
    └── audit_gouvernance.json   # Latest checked-in drift and fairness audit output
```

The contents of `data/` and `models/*.joblib` are excluded from version control. The repository includes metadata and a sample governance audit, but a fresh clone does not include the datasets or trained model artifacts. Supply the required data and train the models before running inference or building a usable Docker image.

## End-to-End Pipeline

1. Load the URL and consolidated email CSVs from `data/raw/`.
2. For URLs, map source labels to the project convention (`1` = phishing); for email, use the supplied labels as-is (`1` = phishing).
3. Remove duplicate URL/email text rows, then create an 80/20 stratified random train/test split with `random_state=42`. The code does not use timestamps or perform a temporal split.
4. Extract 18 URL features. For email, clean text, fit a training-only TF-IDF vocabulary capped at 2,500 terms, and append 10 explicit structural and urgency features.
5. Train Random Forest and XGBoost classifiers for each path, with isotonic probability calibration using five-fold cross-validation.
6. Search candidate thresholds from 0.01 to 0.99 to minimize a cost that weights false negatives 10 times more than false positives, then export evaluation metrics to `doc/EVALUATION.md`.
7. For email holdout predictions, compare the two halves of the returned score array with a KS test and compare false-positive rates for messages with and without a detected urgency phrase. Save these exploratory results to `models/audit_gouvernance.json`; this is not production monitoring.
8. Save the trained model artifacts under `models/`, along with declared environment and threshold metadata.
9. Load the XGBoost artifacts in the CLI or web interface. The interfaces can analyze either input independently or fuse URL and email scores.

The training script processes both datasets in sequence and requires both source files, even if you only plan to use one of the interfaces. Training rewrites the evaluation report, governance audit, metadata, and local model artifacts. It does not currently export processed train/test CSV files.

## Data and Label Requirements

The datasets are not bundled with this repository. Obtain them from their respective sources, review their terms of use, and place them in the following locations:

```text
data/
└── raw/
    ├── PhiUSIIL_Phishing_URL_Dataset.csv
    └── phishings_emails/
        └── phishing_email.csv
```

The training script expects these columns and label conventions:

| Dataset file | Required columns | Notes |
|---|---|---|
| `data/raw/PhiUSIIL_Phishing_URL_Dataset.csv` | `URL`, `label` | Source `label=0` is mapped to phishing (`1`); all other values are mapped to legitimate (`0`). |
| `data/raw/phishings_emails/phishing_email.csv` | `text_combined`, `label` | Must already be consolidated. Labels are used as-is: `1` means phishing and `0` legitimate. |

The email script does not combine the other email corpus files in `data/raw/phishings_emails/`. Prepare the consolidated CSV with the required schema before training. The training code removes duplicate rows based on the text column before splitting. Email text is normalized before the TF-IDF vectorizer is fitted on the training split only. Incorrect or inconsistent labels will make the resulting evaluation and predictions unreliable. See [data/README.md](data/README.md) for the data governance notes; independently verify any supplied checksums against the dataset source before relying on them.

## Feature Engineering

### URLs

`src/features.py` extracts 18 numeric indicators from each URL:

- URL length and counts of `.`, `-`, `@`, `?`, `=`, and `/` (7 features);
- parsed hostname dot count as a subdomain-depth proxy;
- suspect-TLD and Punycode indicators;
- a non-standard-port indicator and minimum Levenshtein distance to a small set of trusted brand names;
- digit-to-letter and selected-special-character ratios;
- presence of selected social-engineering terms (`login`, `verify`, `secure`, `account`, `banking`, `update`, `signin`, `compte`);
- Shannon entropy of the URL string;
- an IPv4-shaped string check; and
- whether the input starts with HTTPS.

These features describe the URL string, not the actual safety, ownership, or reputation of a website. The IP check is a pattern match, not full address validation. They do not validate TLS certificates, verify whether a port is reachable, or resolve redirects/domains. `has_port_non_standard` is set only when parsing finds an explicit port other than 80 or 443; Punycode is represented by a separate feature.

### Email text

The cleaner replaces URLs, email addresses, and currency amounts with markers, removes other unsupported punctuation while preserving `!` and `?`, and normalizes whitespace. The training pipeline combines a `TfidfVectorizer` representation (English stop-word removal, at most 2,500 terms) with 10 explicit features: URL and email counts, exclamation and question counts, uppercase-word count and ratio, urgency-match count and flag, money-amount presence, and attachment/invoice mention.

### Score fusion and dashboard explanations

`src/pipeline.py` exposes `fusionner_vecteurs_menace`, used by both the CLI and web dashboard: if either input score is at least 80%, the result is flagged and uses the higher score; otherwise, it uses a 60% URL / 40% email weighted score and a 50% fusion decision threshold.

`src/explain.py` uses SHAP `TreeExplainer` with an XGBoost estimator inside the calibrated model to calculate signed feature contributions for the dashboard. SHAP values describe the selected underlying estimator, not the entire calibrated ensemble, and are not causal explanations. They should be treated as supporting signals rather than definitive reasons for a verdict.

## Models and Evaluation

For both URLs and emails, the training script fits:

- `RandomForestClassifier`
- `XGBClassifier`

The target class `1` represents phishing at inference time. Isotonic calibration is applied to the classifiers. Training searches thresholds from 0.01 to 0.99 to minimize false-negative cost (10x false-positive cost by default) and records the selected thresholds in metadata. URL and email inference read their corresponding XGBoost thresholds from that metadata; if metadata is unavailable or invalid, inference falls back to 0.50. The evaluation report includes PR-AUC, confusion-matrix counts, asymmetric cost, and high-confidence error counts. These metrics use the supplied holdout partition and are not an independent benchmark or a guarantee of performance on new data. The report is regenerated by training and reflects the most recent run.

Training saves five artifacts; the applications use the XGBoost models and email vectorizer, while the smoke check attempts to load all five:

```text
models/
├── url_model_rf.joblib
├── url_model_xgb.joblib
├── email_model_rf.joblib
├── email_model_xgb.joblib
└── email_vectoriseur.joblib
```

The web and command-line applications use the XGBoost models. The Random Forest artifacts are saved and checked by the smoke test but are not used for inference by the current interfaces. Model files are generated locally and are not committed to this repository.

## Installation and Usage

### Requirements

- Git
- Python 3.12 (the version recorded in `models/metadata.json` and used by the Docker image)
- The two input datasets described above
- Docker, only if using the container workflow

Training loads the datasets into memory. Runtime and memory use depend on the machine and dataset files.

### 1. Clone the repository

```bash
git clone https://github.com/ephremkadja01-coder/phishing-detection-nlp.git
cd phishing-detection-nlp
```

### 2. Create and activate a virtual environment

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Add the datasets

Place the files in the paths listed in [Data and Label Requirements](#data-and-label-requirements). Confirm that the CSV headers match the required column names exactly.

### 5. Train and save the models

Run from the repository root:

```bash
python train_and_save_models.py
```

This trains both URL and email models, exports the evaluation report, and saves the five `.joblib` artifacts in `models/`. The current script does not write train/test split CSV files.

### 6. Use the Makefile shortcuts (optional)

```bash
make install
make train
make test
make serve
```

`make train` removes existing `models/*.joblib` files before training. `make clean` removes local Python/test caches and notebook checkpoints; it does not remove datasets or models.

### 7. Check that model artifacts load

```bash
python tests/test_model_loading.py
```

This is an artifact-loading smoke check and requires the five generated artifacts. It does not test predictive quality or validate a model against fresh data.

### 8. Run tests

```bash
pytest tests/ -v
```

The current unit tests cover URL feature indicators, Levenshtein distance, email cleaning, score fusion, and the experimental urgency-group fairness calculation. The model-loading check is a standalone script, not a pytest test.

### 9. Run an inference interface

Command-line interface:

```bash
python app.py
```

Choose `1` to analyze a URL, `2` to analyze email text, or `3` to analyze and fuse both inputs.

Gradio web interface:

```bash
python interface_web.py
```

Open the local address printed by Gradio, usually `http://127.0.0.1:7860`.

Run both applications from the repository root so their relative paths to `models/` resolve correctly.

### 10. Build and run with Docker (optional)

Train the models first so that the required local artifacts exist in `models/`, then build and run from the repository root:

```bash
docker build -t phishing-detection-nlp .
docker run --rm -p 7860:7860 phishing-detection-nlp
```

The image runs only the Gradio interface and exposes port `7860`. The current Dockerfile does not copy datasets, notebooks, tests, or the CLI into the image.

## Reproducibility

The model constructors and the stratified 80/20 train/test split specify `random_state=42`. The split is random, not temporal, and does not use an explicit timestamp. Reproducing identical results still depends on using the same input data and compatible dependency versions; the datasets and model artifacts are local and are not included in Git. `models/metadata.json` records the declared Python and key library versions plus the thresholds found during training.

`requirements.txt` is an environment-specific export: it mixes version-pinned entries with unpinned dependencies and local `file:///...` package references. It is not a portable lockfile and does not guarantee cross-environment reproducibility; some local references may not exist on another machine. The Python 3.12 badge reflects the versions configured in CI and the Docker base image, not a package lock. Joblib model artifacts are Python-library-version sensitive, so retrain locally rather than copying `.joblib` files between environments.

## Technical Stack

- **Python** for data processing and application code
- **pandas** for CSV loading and data preparation
- **scikit-learn** for train/test splitting, Random Forest, TF-IDF, and evaluation metrics
- **XGBoost** for gradient-boosted classification
- **joblib** for saving and loading fitted models
- **tabulate** (through pandas `to_markdown`) for rendering the generated evaluation report as Markdown
- **Gradio** for the browser-based interface
- **Docker** for an optional containerized Gradio deployment
- **pytest** for focused unit tests
- **SHAP** for model contribution calculations
- **SciPy** for the KS drift comparison
- **GitHub Actions** for dependency-audit and test automation
- **Experimental fairness audit** for comparing false-positive rates between email urgency groups

## Operational Troubleshooting & Post-Mortem

This project records historical development incidents and their resolutions in [doc/incident_log.md](doc/incident_log.md). The main changes delivered so far include:

- **Data and target handling:** Standardized URL labels to `1 = phishing` and email labels to `1 = phishing`, removed duplicate text rows before splitting, and changed evaluation to a reproducible stratified random holdout (80/20, seed 42). No timestamp-based split is implemented.
- **URL and email signals:** Expanded URL analysis to 18 string-derived features, separated Punycode detection from non-standard-port detection, and combined email TF-IDF with 10 explicit structural/urgency indicators. The active inference path has no exact-host allowlist.
- **Model evaluation and inference:** Added isotonic calibration for Random Forest and XGBoost, asymmetric threshold search (false negatives weighted 10x false positives), and use of the selected XGBoost thresholds during URL/email inference, with a 0.50 fallback when metadata is unavailable.
- **Multi-modal interface and explanations:** Added URL-only, email-only, and combined analysis to the CLI and Gradio dashboard, plus SHAP contribution output based on an underlying XGBoost estimator.
- **Governance experiments:** Added a KS comparison between halves of email holdout scores and false-positive-rate comparisons for email urgency groups. These are offline training-time analyses, not live drift monitoring or a comprehensive fairness certification. The fairness helper currently lives under `experimental/`.
- **CI and dependency hygiene:** Added a GitHub Actions job that installs `requirements.txt`, runs `pip-audit`, and runs pytest. The audit command is wrapped so findings produce a warning rather than failing the workflow. The current dependency file is an environment-specific export, not a portable lockfile.
- **Operational fixes:** Resolved dataset schema, URL feature, tokenizer, NumPy scalar, module-path, and CI dependency-format issues documented in the incident log. Added the `tabulate` runtime dependency needed to export evaluation reports and corrected the fairness import after its move to `experimental/`.

## Project documentation

- [Business logic and cyber threat modeling](doc/business_logic.md)
- [Threat model and robustness notes](doc/THREAT_MODEL.md)
- [Evaluation report](doc/EVALUATION.md) (regenerated by training)
- [Technical incident log](doc/incident_log.md)
- [Data governance and checksum ledger](data/README.md)
- [Governance audit output](models/audit_gouvernance.json) (a checked-in snapshot, overwritten by training)
- [Exploratory and modeling notebooks](notebooks/)

## Limitations and Safety Notes

- URL inference uses a small set of hand-engineered string features. It does not check domain age, DNS, certificates, redirects, page contents, or live reputation.
- A legitimate URL may be classified as phishing, and a phishing URL may be classified as legitimate. Treat predictions as experimental signals, not a verdict.
- The non-standard-port indicator is distinct from Punycode detection, but it only reflects whether parsing found an explicit port other than 80 or 443; it does not establish whether that port is suspicious or reachable.
- Inference uses the XGBoost thresholds in `models/metadata.json`, or 50% if the relevant metadata is unavailable or invalid. Fusion applies its separate 80% OR gate and weighted-score rule.
- The governance report compares portions/groups within one email holdout; it is not ongoing production monitoring or a comprehensive fairness assessment.
- SHAP contributions use an underlying XGBoost estimator and do not explain the full isotonic-calibrated ensemble or establish causality.
- The email classifier uses TF-IDF and does not understand meaning in the way a contextual language model does.
- Reported metrics are calculated on a stratified random holdout from the supplied datasets. They may not represent performance on current, unseen traffic, and removing exact duplicate text does not remove related or near-duplicate samples.
- The threat-model document describes risks and design assumptions; it is not a security certification or evidence that the listed mitigations have been independently validated.
- Never use this prototype as the sole control for email filtering, URL blocking, or incident response.

## Virtual Environment Troubleshooting

Use Python 3.12, matching the project metadata and Docker image. Create the virtual environment in the repository after navigating to its root, and activate it in the same terminal:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -c "import sys; print(sys.executable)"
```

The printed executable should be the repository's `.venv/bin/python`. If it points to Conda or another Python installation, open a fresh terminal, deactivate Conda, and activate the project's `.venv` again. Do not move or copy an existing virtual environment; recreate it in the new location because its scripts can retain the old absolute path.

## References

- [scikit-learn documentation](https://scikit-learn.org/stable/)
- [XGBoost documentation](https://xgboost.readthedocs.io/)
- [Gradio documentation](https://www.gradio.app/docs)
