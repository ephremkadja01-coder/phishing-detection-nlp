# Utilisation d'une image de base Python ultra-legere et stable
FROM python:3.12-slim

# Configuration des variables d'environnement pour optimiser l'execution de Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Definition du dossier de travail a l'interieur du conteneur
WORKDIR /app

# Installation des dependances systeme minimales si necessaire
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copie exclusive des dependances verrouillees pour optimiser la mise en cache Docker
COPY requirements.txt .

# Installation des bibliotheques Python sans cache pour alleger l'image finale
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copie de l'intégralite du code source structure du projet
COPY src/ ./src/
COPY models/ ./models/
COPY interface_web.py .

# Exposition du port reseau natif de Gradio pour l'interface graphique
EXPOSE 7860

# Force Gradio a écouter sur toutes les interfaces reseau du conteneur
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Commande de lancement automatique du conteneur vers l'interface Web
CMD ["python", "interface_web.py"]
