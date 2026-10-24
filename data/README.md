# Data Governance & Checksums Ledger

This directory structure anchors the source and intermediate tables used for model compilation. Due to file constraints (~371 MB), data remains strictly local.

## Local Reference Checksums (SHA-256)

These hashes were measured from the dataset files present in the workspace on 2026-10-24. They are local reference values, not independently authenticated checksums published by the dataset providers. Recompute them after obtaining the data and compare only when using the same dataset versions.

*   **File:** `data/raw/PhiUSIIL_Phishing_URL_Dataset.csv`
    *   **SHA-256:** `a236549cd369cd80bd478ff8e1779cbf44c58d5c3f79f7a51a1adbed7d06d1c6`
*   **File:** `data/raw/phishings_emails/phishing_email.csv`
    *   **SHA-256:** `73a5c5410baf46ed69517fedfd37910f2e93777dc504b73ea483bc16be3a69d1`

## Data Mapping Assertions
*   **URL Pipeline:** Explicit inversion mapping (`1 if x == 0 else 0`) is programmatically forced to match cyber standards where `1` equals confirmed Phishing.
*   **Email Pipeline:** Standard mapping where `1` equals Phishing and `0` equals Legitimate.
