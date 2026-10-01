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


def _parse_rate_limit(name: str, default: tuple[int, int]) -> tuple[int, int]:
    """Analyse un réglage « N/FENÊTRE » : ex. « 5/60 » = 5 requêtes par 60 secondes.

    Une valeur illisible fait échouer le démarrage. C'est volontaire : mieux vaut
    une erreur explicite qu'une protection silencieusement différente de celle
    que l'exploitant croit avoir configurée.
    """
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default

    try:
        maximum, fenetre = (int(part) for part in raw.split("/", 1))
    except ValueError:
        raise ValueError(
            f"{name}={raw!r} est invalide. Format attendu : « 5/60 » "
            "(5 requêtes par 60 secondes)."
        ) from None

    if maximum < 1 or fenetre < 1:
        raise ValueError(f"{name}={raw!r} : les deux valeurs doivent être >= 1.")

    return maximum, fenetre

# --- Emplacements ---------------------------------------------------------
# Le corpus est rangé en DEUX dossiers, et ce n'est pas un détail de rangement :
#   · `documents/` — corpus de démonstration, versionné donc public ;
#   · `shamela/`   — EPUB de shamela.ws, IGNORÉ par Git. Le texte y est
#     numérique et propre, mais les éditions modernes restent sous droits
#     d'éditeurs tiers : elles ne doivent jamais être poussées (voir .gitignore).
# L'ingestion traite les deux de la même façon — le dossier n'est qu'un rangement.
DOCUMENTS_DIR = Path(os.getenv("DOCUMENTS_DIR", BASE_DIR / "data" / "documents"))
SHAMELA_DIR = Path(os.getenv("SHAMELA_DIR", BASE_DIR / "data" / "shamela"))

# Dossiers parcourus par l'ingestion, dans cet ordre.
CORPUS_DIRS = (DOCUMENTS_DIR, SHAMELA_DIR)

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

# Ingestion : nombre de chunks écrits par appel à ChromaDB.
#
# Cette valeur n'est pas libre : elle se lit CONJOINTEMENT avec
# `EMBEDDING_TIMEOUT`, parce qu'un lot est embarqué en UNE SEULE requête.
# Le serveur d'embeddings est ici le facteur limitant, et de loin.
#
# Mesuré sur ce projet (bge-m3, chunks de ~520 caractères) : ~0,55 s par
# chunk. D'où :
#     32 chunks ≈  18 s   (6 fois sous le délai de 120 s)
#    256 chunks ≈ 140 s   (DÉPASSE le délai — l'écriture échoue)
# La valeur 256 a réellement échoué en production locale : ChromaDB annonçait
# « timed out in add », le client attendant 60 s par défaut.
#
# La règle : garder « INGEST_BATCH_SIZE × 0,6 s » très en dessous de
# EMBEDDING_TIMEOUT. Un lot plus petit coûte quelques requêtes de plus, mais
# rend la reprise plus fine — c'est le bon compromis.
INGEST_BATCH_SIZE = int(os.getenv("INGEST_BATCH_SIZE", "32"))

if INGEST_BATCH_SIZE < 1:
    raise ValueError(
        f"INGEST_BATCH_SIZE={INGEST_BATCH_SIZE} est invalide : la valeur doit être >= 1."
    )

# Délai d'expiration d'un appel d'embedding, en secondes.
#
# Le défaut de la bibliothèque ChromaDB est de 60 s — trop court dès qu'un lot
# contient quelques centaines de chunks. Le délai s'applique aussi bien à un
# lot d'ingestion qu'à l'embedding d'une seule question : il doit donc rester
# borné, d'où une valeur généreuse mais pas infinie.
EMBEDDING_TIMEOUT = int(os.getenv("EMBEDDING_TIMEOUT", "120"))

if EMBEDDING_TIMEOUT < 1:
    raise ValueError(
        f"EMBEDDING_TIMEOUT={EMBEDDING_TIMEOUT} est invalide : la valeur doit être >= 1."
    )

# Seuil de pertinence : distance maximale acceptée pour un chunk renvoyé par
# ChromaDB (métrique de distance L2). Une valeur <= 0 désactive le filtrage
# (comportement par défaut, sans risque de casser le chatbot si les distances
# ne sont pas calibrées). Exemple d'activation : DISTANCE_THRESHOLD=1.0
DISTANCE_THRESHOLD = float(os.getenv("DISTANCE_THRESHOLD", "-1"))

# --- Citation verbatim (EXPÉRIMENTAL, désactivé par défaut) ---------------
#
# POURQUOI. Trois vérifications déterministes ont été essayées pour empêcher le
# modèle d'inventer, et les trois ont échoué (voir README.md, « Trois
# vérifications déterministes essayées, trois échecs »). La raison est constante :
# les inventions sont faites du VOCABULAIRE du corpus, donc indistinguables
# lexicalement. Ce qui distingue une citation d'une invention est sémantique.
#
# D'où le renversement : ne plus vérifier une réponse libre, mais exiger du
# modèle une CITATION verbatim, et vérifier cette citation (`app/fidelite.py`).
# La vérification redevient exacte — la citation est dans le contexte, ou elle
# n'y est pas — et le modèle ne peut pas tricher.
#
# Pourquoi un drapeau : pour mesurer les deux régimes avec le MÊME code, comme
# `HYBRID_ENABLED`. Un réglage qu'on ne peut pas comparer ne se démontre pas.
CITATIONS_OBLIGATOIRES = _env_flag("CITATIONS_OBLIGATOIRES", False)

# --- Recherche hybride (lexicale + vectorielle) ---------------------------
#
# POURQUOI elle existe, mesuré le 29/09 sur les 59 questions du jeu d'or :
#
#   | recherche            | hit@10 | MRR@10 |
#   |----------------------|--------|--------|
#   | vectorielle seule    | 78,0 % | 0,52   |
#   | lexicale seule       | 88,1 % | 0,69   |
#   | fusion (les deux)    | 98,3 % | 0,69   |
#
# La cause est identifiable question par question. Dans `تفسير ابن المنذر`,
# 85 % des chunks sont des chaînes de transmetteurs (« حدّثنا… عن… ») : un chunk
# qui contient le verset cherché PLUS deux cents mots de formule ressemble, vu
# du vecteur, à n'importe quel autre chunk de transmission. Quatre questions
# citaient un verset mot pour mot ; l'index contenait ce verset, à la page
# attendue, et la recherche vectorielle ne l'a JAMAIS renvoyé — la recherche
# lexicale le trouve au rang 1.
#
# Les deux recherches échouent différemment, et c'est ce qui justifie de les
# FUSIONNER plutôt que d'en choisir une : BM25 ne comprend pas les paraphrases,
# le vecteur noie les termes rares.
HYBRID_ENABLED = _env_flag("HYBRID_ENABLED", True)

# Candidats demandés à chaque recherche avant fusion. La fusion ne peut pas
# classer ce qu'on ne lui a pas donné : borner à `N_RESULTS` (5) reviendrait à
# ne fusionner que les 5 premiers de chaque liste, et à perdre le passage que
# l'un des deux trouve au rang 40.
HYBRID_CANDIDATES = int(os.getenv("HYBRID_CANDIDATES", "50"))

# Constante d'aplatissement de la fusion RRF, et poids relatifs.
#
# ⚠️ CES TROIS VALEURS NE SONT PAS INDÉPENDANTES DE `HYBRID_CANDIDATES`, et
# l'ignorer a produit une erreur de conception réelle. Avec une constante de 60
# et un poids de 3 pour le vecteur, le rang 1 LEXICAL ne vaut que 1/(60+1) =
# 0,0164, quand le 50e candidat VECTORIEL vaut déjà 3/(60+50) = 0,0273 : la
# recherche lexicale ne pouvait alors que RECLASSER ce que le vecteur avait déjà
# vu, jamais introduire un chunk que le vecteur avait manqué. Quatre questions
# dont la page attendue était classée 1re sur 3 355 par BM25 restaient
# introuvables. Le gain de la fusion était réel, mais son explication fausse.
#
# Grille mesurée à poids égaux (59 questions, 3 355 chunks) :
#
#   constante   hit@5   hit@10   MRR
#        5      89,8 %   96,6 %  0,686
#       10      89,8 %   98,3 %  0,687   <- retenu (10 à 20 donnent le même hit@10)
#       20      86,4 %   98,3 %  0,667
#       60      84,7 %   94,9 %  0,648
#
# `hit@5` est ce que le MODÈLE voit (N_RESULTS=5) ; il plafonne tant que la
# constante reste <= 10. La constante 10 est le seul point qui satisfasse les
# deux métriques — et le résultat ne dépend pas de la troncature (50, 100 et 200
# candidats donnent la même chose), ce qui est la meilleure garantie contre un
# réglage qui ne vaudrait que pour ces 59 questions.
#
# ⚠️ Poids ÉGAUX (1:1) et non pondérés. Toutes les configurations pondérées
# mesurées sont erratiques (76,3 % à 94,9 % selon la constante, sans régularité),
# là où toutes les configurations à poids égaux se tiennent entre 89,8 % et
# 96,6 %. La mise en forme standard de RRF est symétrique : introduire un poids
# revient à décider à l'avance quelle recherche a raison, et la mesure ne le
# confirme pas.
HYBRID_RRF_K = int(os.getenv("HYBRID_RRF_K", "10"))
HYBRID_VECTOR_WEIGHT = float(os.getenv("HYBRID_VECTOR_WEIGHT", "1.0"))
HYBRID_LEXICAL_WEIGHT = float(os.getenv("HYBRID_LEXICAL_WEIGHT", "1.0"))

if HYBRID_CANDIDATES < 1:
    raise ValueError(
        f"HYBRID_CANDIDATES={HYBRID_CANDIDATES} est invalide : la valeur doit être >= 1."
    )

if HYBRID_RRF_K < 1:
    raise ValueError(
        f"HYBRID_RRF_K={HYBRID_RRF_K} est invalide : la valeur doit être >= 1."
    )

if HYBRID_VECTOR_WEIGHT <= 0 or HYBRID_LEXICAL_WEIGHT <= 0:
    raise ValueError(
        "Les poids de la recherche hybride doivent être strictement positifs "
        f"(reçu : {HYBRID_VECTOR_WEIGHT} et {HYBRID_LEXICAL_WEIGHT})."
    )

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


# --- Limitation de débit (rate limiting) ---------------------------------
# Format « N/FENÊTRE » : N requêtes maximum par client sur FENÊTRE secondes.
# Les limites sont comptées par client (adresse IP) ET par route.
RATE_LIMIT_ENABLED = _env_flag("RATE_LIMIT_ENABLED", True)

# Connexion : strict, contre les attaques par force brute. 5 essais par minute.
LOGIN_RATE_LIMIT = _parse_rate_limit("LOGIN_RATE_LIMIT", (5, 60))

# Inscription : très strict. 5 comptes par heure.
REGISTER_RATE_LIMIT = _parse_rate_limit("REGISTER_RATE_LIMIT", (5, 3600))

# Question au chatbot : 20 par minute. Chaque appel coûte du temps GPU (ou de
# l'argent en API cloud) : cette limite protège votre budget.
ASK_RATE_LIMIT = _parse_rate_limit("ASK_RATE_LIMIT", (20, 60))


# --- Sauvegardes ----------------------------------------------------------
# Dossier de destination des archives (ignoré par Git : voir .gitignore).
BACKUP_DIR = Path(os.getenv("BACKUP_DIR", BASE_DIR / "backups"))

# Nombre d'archives conservées : au-delà, les plus anciennes sont supprimées.
#
# Garder PLUSIEURS versions est indispensable : une corruption découverte
# aujourd'hui est déjà présente dans la sauvegarde d'aujourd'hui. Il faut
# pouvoir remonter à un état antérieur au problème.
BACKUP_RETENTION = int(os.getenv("BACKUP_RETENTION", "7"))

if BACKUP_RETENTION < 1:
    raise ValueError(
        f"BACKUP_RETENTION={BACKUP_RETENTION} est invalide : la valeur doit être >= 1 "
        "(sinon toutes les sauvegardes seraient supprimées)."
    )


def sqlite_database_path() -> Path | None:
    """Chemin du fichier SQLite décrit par `DATABASE_URL`, ou None si autre moteur.

    Permet au script de sauvegarde de localiser la base à copier SANS coder
    « data/app.db » en dur : si `DATABASE_URL` change demain, la sauvegarde suit.

    Renvoie None pour les URL non-SQLite (PostgreSQL, MySQL...) et pour les bases
    en mémoire (`sqlite://`), qui n'ont par définition aucun fichier à copier.
    """
    prefix = "sqlite:///"
    if not DATABASE_URL.startswith(prefix):
        return None

    chemin = DATABASE_URL[len(prefix):]
    return Path(chemin) if chemin else None


# --- Respect de la langue de la réponse -----------------------------------
# Le modèle suit imparfaitement la consigne de langue, surtout quand les
# extraits fournis sont en arabe : il « continue » dans la langue du contexte.
# Deux protections complémentaires, à des niveaux différents (défense en
# profondeur) :
#   1. l'invite place la consigne de langue EN FIN de texte (voir build_prompt) ;
#   2. la langue RÉELLEMENT produite est vérifiée, et une réécriture est
#      demandée si elle est fausse (voir rag.answer_question).
#
# Le contrôle est déterministe (comptage d'alphabet, voir app/language.py) :
# il ne dépend pas du bon vouloir du modèle.

# Désactiver la vérification (utile pour diagnostiquer ou maîtriser la facture :
# chaque reprise est un appel LLM supplémentaire).
LANGUAGE_ENFORCEMENT_ENABLED = _env_flag("LANGUAGE_ENFORCEMENT_ENABLED", True)

# Nombre maximum de réécritures demandées pour une seule question.
# 0 = on constate le problème dans les journaux, sans relancer le modèle.
LANGUAGE_MAX_RETRIES = int(os.getenv("LANGUAGE_MAX_RETRIES", "1"))

if LANGUAGE_MAX_RETRIES < 0:
    raise ValueError(
        f"LANGUAGE_MAX_RETRIES={LANGUAGE_MAX_RETRIES} est invalide : la valeur doit être >= 0."
    )
