# syntax=docker/dockerfile:1

# =============================================================================
# Étape 1 — Compilation de la feuille de styles
# =============================================================================
# Node n'existe QUE dans cette étape. L'image finale reste du Python pur : on
# ne paie le coût de Node ni en taille, ni en surface d'attaque.
#
# C'est le principe du « build multi-étapes » : compiler avec les outils lourds,
# livrer sans eux. Le conteneur de production ne sait même pas que Tailwind
# existe.
FROM node:22-alpine AS styles

WORKDIR /build

# Cache des couches Docker : tant que package.json ne change pas, `npm ci`
# n'est pas rejoué (comme pour requirements.txt plus bas).
COPY package.json package-lock.json ./
RUN npm ci

# Sources nécessaires à Tailwind : la feuille d'entrée, le balisage et la
# logique (Tailwind lit les noms de classes dans ces deux derniers).
COPY frontend ./frontend
RUN npm run build:css

# =============================================================================
# Étape 2 — Application
# =============================================================================
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

# Le frontend servi est fait de deux fichiers seulement : le balisage/logique
# et la feuille COMPILÉE. Les sources de la feuille (frontend/css/) et Node
# n'entrent pas dans l'image de production.
COPY frontend/index.html frontend/app.js ./frontend/
COPY --from=styles /build/frontend/app.css ./frontend/app.css

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
