# Business Logic & Cyber Threat Modeling

## 1. Attack Vectors Modeling
The solution processes two distinct exposure surfaces to detect phishing attempts:
* **Structural Cyber Indicators (URLs):** Mathematical analysis of character string deviations. It captures obfuscation patterns like specific character counts (`.` or `-`), direct IP addresses bypasses (circumventing DNS reputation), and the enforcement of the secure TLS/HTTPS protocol.
* **Semantic Indicators (E-mails):** Natural Language Processing (NLP) standardizes raw e-mail texts. A TF-IDF vectorizer limited to 2500 max features transforms words into mathematical vectors to capture manipulation vocabulary (e.g., *urgency, security, action required, suspended*).

## 2. Operational Evaluation Constraints
In security data science, the cost of misclassification is asymmetric. The absolute priority of our models is to maximize **Recall** on the Phishing class. Minimizing False Negatives is critical because a single undetected malicious link can lead to full corporate network compromise.
