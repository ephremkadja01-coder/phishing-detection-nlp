# Phishing Detection with URLs and Email Text

A machine-learning prototype for classifying suspicious URLs and phishing emails. It trains separate Random Forest and XGBoost models for URL features and email text, and provides command-line and Gradio interfaces for trying the saved models.

> **Important:** This is an educational prototype, not a production security product. A model prediction can be wrong, including for well-known legitimate sites. Do not use it as the only basis for blocking links, handling email, or making security decisions.

## Business Context

Phishing can arrive through links or deceptive email content. This project explores two different detection approaches:

- A structural model looks for patterns in a URL, such as its length, punctuation, entropy, and HTTPS usage.
- A text model uses TF-IDF features from an email body to identify patterns associated with phishing.

The two detection paths are trained and run independently. The project is intended for learning and experimentation; it does not query live domain-reputation services, inspect web page contents, or provide a complete email security workflow.

## Repository Structure

```text
phishing-detection-nlp/
├── app.py                       # Command-line inference interface
├── interface_web.py             # Gradio web interface
├── train_and_save_models.py     # Data preparation, training, and model saving
├── requirements.txt             # Pinned Python dependencies
├── src/
│   ├── features.py              # URL feature extraction and email text cleaning
│   ├── model.py                 # Data splitting, model training, and split saving
│   └── pipeline.py              # Shared inference functions
├── tests/
│   └── test_model_loading.py    # Checks that the saved model files can be loaded
├── doc/                         # Project notes and incident log
├── notebooks/                   # Exploratory and evaluation notebooks
├── data/
│   ├── raw/                     # Local source datasets (not tracked by Git)
│   └── processed/               # Locally generated train/test CSV files
└── models/                      # Locally generated .joblib artifacts
```

The contents of `data/` and `models/*.joblib` are excluded from version control. Therefore, cloning the repository alone is not enough to run training or inference: you must supply the required datasets and train the models first.

## End-to-End Pipeline

1. Load the URL and email CSV files from `data/raw/`.
2. Remove duplicate rows based on the raw URL or email text, then create stratified 80/20 train/test splits with a fixed split seed.
3. Extract URL features or clean and vectorize email text.
4. Train a Random Forest and an XGBoost classifier for each detection path.
5. Print phishing-class recall and macro-F1 on each test split.
6. Save the split CSVs under `data/processed/` and the trained artifacts under `models/`.
7. Load the saved artifacts in the CLI or web interface to classify new inputs.

The training script processes both datasets in sequence. It expects both source files to be present, even if you only plan to use one of the interfaces.

## Data and Label Requirements

The datasets are not bundled with this repository. Obtain them from their respective sources, review their terms of use, and place them in the following locations:

```text
data/
└── raw/
    ├── PhiUSIIL_Phishing_URL_Dataset.csv
    └── phishings_emails/
        └── phishing_email.csv
```

The training script expects these columns:

| Dataset file | Required columns | Notes |
|---|---|---|
| `data/raw/PhiUSIIL_Phishing_URL_Dataset.csv` | `URL`, `label` | The script maps source `label=0` to phishing (`1`) and source `label=1` to legitimate (`0`). |
| `data/raw/phishings_emails/phishing_email.csv` | `text_combined`, `label` | This must already be a consolidated CSV. The script uses `label` as-is: `1` must mean phishing and `0` legitimate. |

The email script does not combine the other email corpus files in `data/raw/phishings_emails/`. Prepare the consolidated CSV with the required schema before training. Incorrect or inconsistent labels will make the resulting evaluation and predictions unreliable.

## Feature Engineering

### URLs

`src/features.py` extracts numeric indicators from each URL, including:

- URL length and counts of `.`, `-`, `@`, `?`, `=`, and `/`;
- Shannon entropy of the URL string;
- a simple indicator for selected brand-name strings;
- an IPv4 address pattern check; and
- whether the string starts with HTTPS.

These features describe the string, not the actual safety or ownership of the website. The brand-name feature is a simple substring heuristic and should not be interpreted as domain verification.

The Gradio interface also has a hard-coded exact-domain allowlist that can bypass URL model inference for listed hosts. This behavior exists only in `interface_web.py`; the CLI does not use it. The list is limited, manually maintained, and not a trusted-domain service.

### Email text

Email text is converted to lowercase, non-letter characters are removed, and repeated whitespace is normalized. A `TfidfVectorizer` with a maximum vocabulary size of 2,500 transforms the cleaned text before classification.

## Models and Evaluation

For both URLs and emails, the training script fits:

- `RandomForestClassifier`
- `XGBClassifier`

The target class `1` represents phishing at inference time. During training, the script prints recall for class `1` and macro-F1 for the held-out split. These are local split metrics, not an independent benchmark or a guarantee of performance on new data.

The split CSVs are written to:

```text
data/processed/
├── urls_train.csv
├── urls_test.csv
├── emails_train.csv
└── emails_test.csv
```

The saved inference artifacts are:

```text
models/
├── url_model_rf.joblib
├── url_model_xgb.joblib
├── email_model_rf.joblib
├── email_model_xgb.joblib
└── email_vectoriseur.joblib
```

The web and command-line applications use the XGBoost models. Model files are generated locally and are not committed to this repository.

## Installation and Usage

### Requirements

- Git
- Python 3.10–3.12 recommended
- The two input datasets described above

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

This trains both URL and email models, writes processed split files, prints evaluation metrics, and saves the five `.joblib` artifacts in `models/`.

### 6. Check that model artifacts load

```bash
python tests/test_model_loading.py
```

This is an artifact-loading smoke check. It does not test predictive quality or validate the model against fresh data.

### 7. Run an inference interface

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

## Reproducibility

The train/test split uses `random_state=42` and stratifies on the target label. The model constructors also specify `random_state=42`. Reproducing identical results still depends on using the same input data and compatible dependency versions; the saved data and models are intentionally local and are not included in Git.

The pinned dependencies are listed in `requirements.txt`. Joblib model artifacts are Python-library-version sensitive, so for a different machine or environment, the recommended workflow is to install the requirements and retrain locally rather than copy `.joblib` files between environments.

## Technical Stack

- **Python** for data processing and application code
- **pandas** for CSV loading and data preparation
- **scikit-learn** for train/test splitting, Random Forest, TF-IDF, and evaluation metrics
- **XGBoost** for gradient-boosted classification
- **joblib** for saving and loading fitted models
- **Gradio** for the browser-based interface

## Limitations and Safety Notes

- URL inference uses a small set of hand-engineered string features. It does not check domain age, DNS, certificates, redirects, page contents, or live reputation.
- A legitimate URL may be classified as phishing, and a phishing URL may be classified as legitimate. Treat predictions as experimental signals, not a verdict.
- The web allowlist is a local shortcut for exact hostnames, not proof that a URL or page is safe. The CLI does not apply the allowlist.
- The email classifier uses TF-IDF and does not understand meaning in the way a contextual language model does.
- Reported metrics are calculated on a random split from the supplied datasets. They may not represent performance on current, unseen traffic.
- Training-time duplicate removal is performed on the raw input text before splitting. The email cleaning/vectorization is applied after the split; this is not a guarantee against all forms of dataset leakage.
- Never use this prototype as the sole control for email filtering, URL blocking, or incident response.

## References

- [scikit-learn documentation](https://scikit-learn.org/stable/)
- [XGBoost documentation](https://xgboost.readthedocs.io/)
- [Gradio documentation](https://www.gradio.app/docs)
