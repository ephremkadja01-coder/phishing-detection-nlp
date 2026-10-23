# Multi-Vector Phishing Detection Pipeline

A modular machine learning prototype designed to evaluate the classification of malicious web links (URLs) and e-mail textual contents using custom structural feature engineering and Natural Language Processing (NLP) models.

## Project Context & Operational Reality

This repository contains a local pipeline developed to study the behavior of tree-based machine learning classifiers against online threat vectors. It operates concurrently on two raw surfaces: parsing URL strings for structural anomalies and evaluating text mutations within email bodies.

This system is built around two key constraints observed during our development phase:
1. **Asymmetric Security Metric:** In phishing detection, missing a threat (False Negative) can lead to a security breach, while blocking a clean item (False Positive) creates minor disruption. Therefore, our models are structurally optimized to prioritize **Recall on the Phishing class**.
2. **Computational Scale Limits:** Processing massive arrays of text variables can easily saturate standard hardware. The local pipeline is heavily restricted to lightweight feature spaces to avoid machine freezes and optimize execution times on standard CPU architectures.

## Repository Layout

The current architecture separates data files, R&D environments, and core code logic. The layout on disk is structured as follows:

```text
phishing-detection-nlp/
├── src/
│   ├── __init__.py
│   ├── features.py            # Custom string cleaning and cyber feature extractors
│   ├── model.py               # Data splitting and tree-based model architectures
│   └── pipeline.py            # Shared functions for production inference
├── doc/
│   ├── business_logic.md      # Cyber threat modeling and metric optimization notes
│   └── incident_log.md        # Technical incidents ledger and troubleshooting records
├── data/
│   ├── raw/                   # Immutable local data files (PhiUSIIL URL & Combined Email datasets)
│   └── processed/             # Empty directory (Reserved for future persistent batch outputs)
├── models/                    # Reserved directory for persisted binary model artifacts (.joblib)
├── notebooks/                 # CRISP-DM experimental phase tracking
│   ├── 01_data_understanding.ipynb       # Data exploration logs (Pending content population)
│   ├── 02_data_preparation.ipynb         # EDA and text tokenization research entry point
│   ├── 03_modeling_experiments.ipynb      # R&D modeling tests (Pending content population)
│   └── 04_model_evaluation.ipynb           # Local metrics curves (Pending content population)
├── app.py                     # Command-line prediction interface
├── train_and_save_models.py   # Master script automating local data processing and training
└── test_model_loading.py      # Pre-deployment artifact validation smoke test
```

*   `src/` holds the operational codebase imported concurrently by experimental components and root-level automation scripts to maintain environment consistency.
*   `notebooks/` records our research journey. To avoid repository bloat, three notebooks (`01`, `03`, `04`) act as placeholders for incoming documentation, while `02` centers on active raw data preparation.

## Data Pipeline & Modeling Strategy

**1. Data Ingestion & Target Construction.** The pipeline ingests two independent datasets: the PhiUSIIL URL collection (**235,795 rows**) and a consolidated email corpus (**82,486 instances** compiled across Enron, CEAS, and Nazario records). Target classes are mapped into binary indicators where `1` represents a malicious threat and `0` signifies a legitimate element.

**2. Target & Feature Engineering (URLs).** Raw URL strings pass through `src/features.py` to extract specific structural features: total string length, character counting flags (`.`, `-`, `@`, `?`, `=`), checking for direct IPv4 bypasses, and secure protocol initiation.

**3. Text Preprocessing Boundaries (Emails).** Email text body strings are forced to lower-case and stripped of non-alphabetic indicators. Features are then extracted via a `TfidfVectorizer` limited to its `max_features=2500` most frequent tokens. This dimensional restriction was introduced to prevent kernel memory saturation during high-volume matrix evaluations.

**4. Separation & Modelling.** An 80/20 stratified split is calculated on both sets. To maintain strict environment replication, data partitions are held in memory during the process instead of being written back to disk inside `data/processed/`. Two tree-based families are compiled and saved in parallel: an ensemble **Random Forest Classifier** and an optimized **XGBoost Classifier**.

## Version Control & Persistence Policy

*   **Heavy Data Files Exclusion:** Local datasets (`data/raw/` totaling ~371 MB) are heavily restricted and excluded from Git tracking via `.gitignore` rules to maintain repository lightness and prevent transfer buffer errors.
*   **Artifact Isolation:** Serialized binary files (`models/*.joblib` totaling ~67 MB) represent transient environment states and are strictly locked locally. They are designed to be recompiled deterministically by executing `train_and_save_models.py` directly inside the host environment.
*   **Predictive Validation:** Pipeline stability is mechanically guarded by `test_model_loading.py`, which spawns a verification process to ensure serialized files unpack cleanly from disk without system regression.

## Installation & Replication

The system requires an isolated Python 3.12+ environment.

```bash
# Clone the repository and position yourself at the root directory
cd phishing-detection-nlp

# Instantiate and activate the isolated virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install explicitly required dependencies
pip install -r requirements.txt

# Run the master pipeline to process local data and train models
python train_and_save_models.py

# Execute the automated pre-deployment verification smoke test
python test_model_loading.py
```
