# Variables de configuration de l'infrastructure
PYTHON = python
PIP = pip
PYTEST = pytest

.PHONY: install train test serve clean help

help:
	@echo "=== Commandes d'automatisation MLOps disponibles ==="
	@echo "  make install  : Installe et verrouille les dependances du projet"
	@echo "  make train    : Nettoie les anciens artefacts et relance l'apprentissage"
	@echo "  make test     : Lance la suite complete de tests unitaires pytest"
	@echo "  make serve    : Demarre l'interface graphique interactive web Gradio"
	@echo "  make clean    : Purge les dossiers de cache et les checkpoints locaux"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PIP) install -r requirements.txt

train:
	rm -f models/*.joblib
	$(PYTHON) train_and_save_models.py

test:
	$(GRADIO_SERVER_NAME)="" $(PYTEST) tests/ -v

serve:
	$(PYTHON) interface_web.py

clean:
	rm -rf .pytest_cache .ipynb_checkpoints
	find . -type d -name "__pycache__" -exec rm -rf {} +
