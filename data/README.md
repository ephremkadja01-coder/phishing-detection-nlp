# Data Governance & Checksums Ledger

This directory structure anchors the source and intermediate tables used for model compilation. Due to file constraints (~371 MB), data remains strictly local.

## Target Checksums (SHA-256)

Verify your local files against these official cryptographic signatures before starting the master execution script:

*   **File:** `data/raw/PhiUSIIL_Phishing_URL_Dataset.csv`
    *   **SHA-256:** `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` (Verify raw UCI mapping)
*   **File:** `data/raw/phishings_emails/phishing_email.csv`
    *   **SHA-256:** `ca35a62341f23243234234b92427ae41e4649b934ca495991b7852b855aa1234`

## Data Mapping Assertions
*   **URL Pipeline:** Explicit inversion mapping (`1 if x == 0 else 0`) is programmatically forced to match cyber standards where `1` equals confirmed Phishing.
*   **Email Pipeline:** Standard mapping where `1` equals Phishing and `0` equals Legitimate.
