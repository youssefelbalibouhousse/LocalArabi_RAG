"""Configuration centralisée du projet (source unique de vérité).

Toutes les valeurs sont surchargeable par variables d'environnement, ce qui
permet de changer de modèle ou d'URL Ollama sans toucher au code.
"""

import os
from pathlib import Path

# Racine du projet, indépendante du répertoire courant.
BASE_DIR = Path(__file__).resolve().parent.parent

# Emplacements
DOCUMENTS_DIR = BASE_DIR / "data" / "documents"
CHROMA_DB_PATH = str(BASE_DIR / "chroma_db")
FRONTEND_DIR = BASE_DIR / "frontend"

# ChromaDB / embeddings
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "arabic_documents")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")

# Ollama / génération
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1")

# Paramètres RAG
N_RESULTS = int(os.getenv("N_RESULTS", "5"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))

# Seuil de pertinence : distance maximale acceptée pour un chunk renvoyé par
# ChromaDB (métrique de distance L2). Une valeur <= 0 désactive le filtrage
# (comportement par défaut, sans risque de casser le chatbot si les distances
# ne sont pas calibrées). Exemple d'activation : DISTANCE_THRESHOLD=1.0
DISTANCE_THRESHOLD = float(os.getenv("DISTANCE_THRESHOLD", "-1"))

# CORS : origines autorisées par le navigateur.
# "*" en développement ; en production, mettre l'URL du front
# (ex. CORS_ALLOW_ORIGINS="https://mon-site.com,https://www.mon-site.com").
CORS_ALLOW_ORIGINS = [
    o.strip() for o in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",") if o.strip()
]

# --- Authentification ---
# Base de données des comptes (SQLite par défaut, fichier data/app.db).
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'data' / 'app.db'}")

# Environnement d'exécution : "development" (défaut) ou "production".
# En production, l'application refuse de démarrer avec la clé de développement.
ENVIRONMENT = os.getenv("ENVIRONMENT", "development").strip().lower()

# Clé de signature des jetons JWT.
# Contraintes : au moins 32 octets (RFC 7518, algorithme HS256).
# Générer une vraie clé avec : python -c "import secrets; print(secrets.token_hex(32))"
#
# La clé ci-dessous est PUBLIQUE et ne sert qu'au développement local.
# Sa longueur respecte volontairement la contrainte RFC pour éviter les
# avertissements de bibliothèque, mais elle ne protège rien.
DEV_SECRET_KEY = "dev-only-secret-key-change-me-in-production"
SECRET_KEY = os.getenv("SECRET_KEY", DEV_SECRET_KEY)

JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))


def using_default_secret_key() -> bool:
    """Indique si la clé JWT n'a pas été fournie (clé de développement)."""
    return SECRET_KEY == DEV_SECRET_KEY
