# Multi-Vector Phishing Detection

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.0-orange?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-green)](https://xgboost.readthedocs.io/)
[![Gradio](https://img.shields.io/badge/Gradio-4.44.1-orange)](https://www.gradio.app/)
[![pytest](https://img.shields.io/badge/pytest-8.2.2-purple?logo=pytest&logoColor=white)](https://pytest.org/)
[![Docker](https://img.shields.io/badge/Docker-supported-blue?logo=docker&logoColor=white)](https://www.docker.com/)

A machine-learning prototype for classifying suspicious URLs and phishing emails. It includes URL feature engineering, an email TF-IDF pipeline, Random Forest and XGBoost training, asymmetric evaluation reports, a score-fusion helper, a command-line interface, and a Gradio web dashboard.

> **Important:** This is an educational and experimental prototype, not a production security product. Predictions and probabilities can be wrong. The web allowlist is a manually maintained shortcut, not a domain verification service. Do not use this project as the only basis for blocking links, handling email, or making security decisions.

## Business Context

Phishing can arrive through links or deceptive email content. This project explores two different detection approaches:

- A structural model looks for patterns in a URL, such as its length, punctuation, entropy, and HTTPS usage.
- A text model uses TF-IDF features from an email body to identify patterns associated with phishing.

The two detection paths are trained independently. A helper can combine their scores using an asymmetric rule, but neither the CLI nor the current web dashboard invokes that helper: each interface analyzes a URL or an email separately. The project is intended for learning and experimentation; it does not query live domain-reputation services, inspect web page contents, or provide a complete email security workflow.

## Repository Structure

```text
phishing-detection-nlp/
├── app.py                       # Command-line inference interface
├── interface_web.py             # Gradio web interface
├── train_and_save_models.py     # Data preparation, training, and model saving
├── requirements.txt             # Pinned Python dependencies
├── Dockerfile                   # Container image for the Gradio interface
├── Makefile                     # Install, train, test, serve, and clean shortcuts
├── src/
│   ├── features.py              # URL feature extraction and email text cleaning
│   ├── model.py                 # Data splitting, model training, and split saving
│   ├── pipeline.py              # Inference and URL/email score-fusion helper
│   └── evaluation.py            # Asymmetric evaluation metrics and Markdown report
├── tests/
│   ├── test_unitaires_cyber.py  # URL features, cleaning, entropy, and fusion tests
│   └── test_model_loading.py    # Smoke check for saved model artifacts
├── doc/                         # Business logic, threat model, evaluation, incident log
├── notebooks/                   # Data understanding, preparation, modeling, evaluation
├── data/
│   ├── README.md                # Data governance and checksum ledger
│   ├── raw/                     # Local source datasets (not tracked by Git)
│   └── processed/               # Locally generated train/test CSV files
└── models/
    └── metadata.json            # Environment, split, and URL feature metadata
```

The contents of `data/` and `models/*.joblib` are excluded from version control. The repository includes metadata, but a fresh clone does not include the datasets or trained models. Supply the required data and train the models before running inference or building the Docker image.

## End-to-End Pipeline

1. Load the URL and consolidated email CSVs from `data/raw/`.
2. For URLs, map source labels to the project convention (`1` = phishing); for email, use the supplied labels as-is (`1` = phishing).
3. Remove duplicate URL strings or cleaned email text, then create stratified 80/20 train/test splits with `random_state=42`.
4. Extract 14 URL features or clean email text and fit a training-only TF-IDF vocabulary capped at 2,500 terms.
5. Train a Random Forest and an XGBoost classifier for each detection path.
6. Calculate PR-AUC, confusion-matrix counts, asymmetric cost (FN weighted 10x FP), and high-confidence error counts; export the report to `doc/EVALUATION.md`.
7. Save the split CSVs under `data/processed/` and the trained artifacts under `models/`.
8. Load the XGBoost artifacts in the CLI or web interface to classify new inputs.

The training script processes both datasets in sequence and requires both source files, even if you only plan to use one of the interfaces. Training rewrites the evaluation report and locally generated split/model files.

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

The email script does not combine the other email corpus files in `data/raw/phishings_emails/`. Prepare the consolidated CSV with the required schema before training. Email text is normalized before duplicate removal and splitting; the TF-IDF vectorizer is fitted on the training split only. Incorrect or inconsistent labels will make the resulting evaluation and predictions unreliable. See [data/README.md](data/README.md) for the data governance notes; independently verify any supplied checksums against the dataset source before relying on them.

## Feature Engineering

### URLs

`src/features.py` extracts 14 numeric indicators from each URL:

- URL length and counts of `.`, `-`, `@`, `?`, `=`, and `/`;
- parsed hostname dot count as subdomain-depth proxy;
- digit-to-letter and selected-special-character ratios;
- presence of selected social-engineering terms (`login`, `verify`, `secure`, `account`, `banking`, `update`, `signin`, `compte`);
- Shannon entropy of the URL string;
- an IPv4-shaped string check; and
- whether the input starts with HTTPS.

These features describe the URL string, not the actual safety, ownership, or reputation of a website. The IP check is a pattern match, not full address validation. They do not validate TLS certificates or resolve redirects/domains.

The Gradio interface has a hard-coded exact-host allowlist that bypasses URL model inference for listed hosts. This behavior exists only in `interface_web.py`; the CLI does not use it. The list is limited and manually maintained. It does not verify ownership, safety, or certificates, and a whitelist match must not be treated as a security guarantee.

### Email text

Email text is converted to lowercase, non-letter characters are removed, and repeated whitespace is normalized. A `TfidfVectorizer` with English stop-word removal and a maximum vocabulary size of 2,500 transforms the cleaned text before classification.

### Score fusion and dashboard explanations

`src/pipeline.py` exposes `fusionner_vecteurs_menace`: if either input score is at least 80%, the result is flagged and uses the higher score; otherwise, it uses a 60% URL / 40% email weighted score and a 50% decision threshold. This helper is covered by unit tests but is not currently called from `app.py` or `interface_web.py`.

The Gradio dashboard returns heuristic explanation text for flagged inputs. URL output lists up to three positive-valued features in model feature order; email output lists up to three matching words from a fixed vocabulary. These are not SHAP values, feature-importance rankings for the specific prediction, nor causal explanations.

## Models and Evaluation

For both URLs and emails, the training script fits:

- `RandomForestClassifier`
- `XGBClassifier`

The target class `1` represents phishing at inference time. Evaluation uses the fixed 0.5 probability threshold for confusion-matrix and cost metrics; false negatives are assigned 10 times the cost of false positives by default. The report also includes PR-AUC and counts of false negatives below 10% phishing probability and false positives above 90%. These are metrics on the supplied, deduplicated random holdout split, not an independent benchmark or a guarantee of performance on new data. The report is regenerated by training and may reflect the data and model run used most recently.

The split CSVs are written to:

```text
data/processed/
├── urls_train.csv
├── urls_test.csv
├── emails_train.csv
└── emails_test.csv
```

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

This trains both URL and email models, writes processed split files, exports the evaluation report, and saves the five `.joblib` artifacts in `models/`.

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

The unit tests cover URL feature keys and edge cases, Shannon entropy, email cleaning, and score fusion. The model-loading check is a standalone script, not a pytest test.

### 9. Run an inference interface

Command-line interface:

```bash
python app.py
```

Choose `1` to analyze a URL or `2` to analyze email text, then enter the requested input.

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

The train/test split uses `random_state=42` and stratifies on the target label. The model constructors also specify `random_state=42`. Reproducing identical results still depends on using the same input data and compatible dependency versions; the saved data and models are intentionally local and are not included in Git. `models/metadata.json` records the Python and key library versions, split settings, email vocabulary limit, URL feature names, and operational threshold notes.

The pinned dependencies are listed in `requirements.txt`. Joblib model artifacts are Python-library-version sensitive, so for a different machine or environment, the recommended workflow is to install the requirements and retrain locally rather than copy `.joblib` files between environments.

## Technical Stack

- **Python** for data processing and application code
- **pandas** for CSV loading and data preparation
- **scikit-learn** for train/test splitting, Random Forest, TF-IDF, and evaluation metrics
- **XGBoost** for gradient-boosted classification
- **joblib** for saving and loading fitted models
- **tabulate** for rendering the generated evaluation report as Markdown
- **Gradio** for the browser-based interface
- **Docker** for an optional containerized Gradio deployment
- **pytest** for focused unit tests

## Operational Troubleshooting & Post-Mortem

This project records historical development incidents and their resolutions in the auditable ledger [doc/incident_log.md](doc/incident_log.md). Key engineering interventions include:

- **UCI target alignment:** Corrected the PhiUSIIL source-label mapping used by the training pipeline so that the project consistently uses `1` for phishing and `0` for legitimate. Verify the source dataset's label convention before training on a different version.
- **URL feature collision and false positives:** Added Shannon entropy and URL path-delimiter features, and introduced an exact-host allowlist in the Gradio interface to bypass model inference for selected hosts. The allowlist is local to the web interface, manually maintained, and is not proof that a URL is safe.
- **NumPy scalar handling:** Simplified the web interface to consume native scalar prediction and probability values returned by the shared inference functions, avoiding the earlier array-shape handling issue.

## Project documentation

- [Business logic and cyber threat modeling](doc/business_logic.md)
- [Threat model and robustness notes](doc/THREAT_MODEL.md)
- [Evaluation report](doc/EVALUATION.md) (regenerated by training)
- [Technical incident log](doc/incident_log.md)
- [Data governance and checksum ledger](data/README.md)
- [Exploratory and modeling notebooks](notebooks/)

## Limitations and Safety Notes

- URL inference uses a small set of hand-engineered string features. It does not check domain age, DNS, certificates, redirects, page contents, or live reputation.
- A legitimate URL may be classified as phishing, and a phishing URL may be classified as legitimate. Treat predictions as experimental signals, not a verdict.
- The web allowlist is a local shortcut for exact hostnames, not proof that a URL or page is safe. The CLI does not apply the allowlist.
- The score-fusion helper is not integrated into either user interface; URL and email checks remain separate workflows.
- Dashboard explanation labels are simple heuristics and are not model attribution methods.
- The email classifier uses TF-IDF and does not understand meaning in the way a contextual language model does.
- Reported metrics are calculated on a random split from the supplied datasets. They may not represent performance on current, unseen traffic.
- Duplicate removal is applied before splitting (on raw URLs and normalized email text). This does not rule out all forms of dataset leakage or overlap between related source records.
- The threat-model document describes risks and design assumptions; it is not a security certification or evidence that the listed mitigations have been independently validated.
- Never use this prototype as the sole control for email filtering, URL blocking, or incident response.

## References

- [scikit-learn documentation](https://scikit-learn.org/stable/)
- [XGBoost documentation](https://xgboost.readthedocs.io/)
- [Gradio documentation](https://www.gradio.app/docs)
