# syntax=docker/dockerfile:1

# Image de base légère avec Python 3.11
FROM python:3.11-slim-bookworm

# Empêche Python d'écrire des .pyc et force les logs "non bufferisés"
# (on voit les logs immédiatement, important en production).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Répertoire de travail : tout ce qui suit s'exécute dans /app
WORKDIR /app

# Étape 1 : copier UNIQUEMENT les dépendances et les installer.
# On le fait AVANT de copier le code pour profiter du cache des couches Docker :
# tant que requirements.txt ne change pas, l'installation n'est pas refaite.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Étape 2 : copier le code applicatif.
COPY app ./app
COPY scripts ./scripts
COPY frontend ./frontend

# Créer un utilisateur non-root et l'utiliser (bonne pratique de sécurité :
# le processus ne tourne pas en "root" dans le conteneur).
RUN useradd --create-home appuser
USER appuser

# Port exposé par FastAPI (documentation + liaison possible).
EXPOSE 8000

# Vérification de santé : Docker interroge /health pour savoir si le
# conteneur est prêt. S'il échoue trop souvent, le conteneur est marqué
# "unhealthy".
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Commande de démarrage du conteneur.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
