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

# Clé de signature des jetons JWT. DOIT être secrète et changée en production
# (générer avec : python -c "import secrets; print(secrets.token_hex(32))").
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-me")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
