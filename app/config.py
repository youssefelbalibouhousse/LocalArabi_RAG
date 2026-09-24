"""Configuration centralisée du projet (source unique de vérité).

Toutes les valeurs sont surchargeable par variables d'environnement, ce qui
permet de changer de modèle ou d'URL Ollama sans toucher au code.
"""

import os
from pathlib import Path

# Racine du projet, indépendante du répertoire courant.
BASE_DIR = Path(__file__).resolve().parent.parent


# --- Utilitaires ---------------------------------------------------------

def _env_flag(name: str, default: bool) -> bool:
    """Lit une variable d'environnement booléenne.

    Valeurs considérées comme vraies : 1, true, yes, on (insensible à la casse).
    Une variable absente OU vide renvoie la valeur par défaut (ce qui permet
    d'écrire ``ALLOW_REGISTRATION=`` dans un fichier .env sans changer le comportement).
    """
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")

# Emplacements
DOCUMENTS_DIR = BASE_DIR / "data" / "documents"
CHROMA_DB_PATH = str(BASE_DIR / "chroma_db")
FRONTEND_DIR = BASE_DIR / "frontend"

# ChromaDB / embeddings
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "arabic_documents")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")

# Ollama / génération
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# --- Fournisseur du modèle de génération ---------------------------------
# "ollama" : serveur Ollama auto-hébergé (local ou VM GPU louée) — défaut.
# "openai" : toute API compatible OpenAI (Groq, Together, vLLM, OpenRouter,
#            LM Studio, serveur d'inférence maison...).
# Les embeddings (bge-m3) restent servis par Ollama dans tous les cas.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()

# URL de base de l'API (utile uniquement si LLM_PROVIDER="openai").
# Exemples : https://api.groq.com/openai/v1
#            https://api.together.xyz/v1
#            http://localhost:8000/v1  (serveur local compatible OpenAI)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").strip()

# Clé d'API du fournisseur (utile uniquement si LLM_PROVIDER="openai").
# Certains serveurs locaux n'en exigent pas : la valeur par défaut suffit alors.
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()

# Modèle utilisé pour la génération.
# Exemples : "llama3.1" (Ollama), "llama-3.1-8b-instant" (Groq),
#            "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo" (Together).
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

# Niveau de journalisation de l'application : DEBUG, INFO, WARNING, ERROR.
# Passer à DEBUG pour diagnostiquer un problème en production.
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").strip().upper()

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

# Inscription publique : autoriser n'importe qui sur Internet à créer un compte.
#
# « Secure by default » : ouverte en développement (confort), FERMÉE en production
# (sinon n'importe qui peut consommer vos ressources et votre budget GPU).
#
# Pour un pilote avec quelques testeurs :
#   1. laisser fermé (ALLOW_REGISTRATION=false, ou rien en production) ;
#   2. créer les comptes un par un avec : python scripts/create_user.py <nom>
ALLOW_REGISTRATION = _env_flag("ALLOW_REGISTRATION", ENVIRONMENT != "production")


def using_default_secret_key() -> bool:
    """Indique si la clé JWT n'a pas été fournie (clé de développement)."""
    return SECRET_KEY == DEV_SECRET_KEY
