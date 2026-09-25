"""API FastAPI du chatbot RAG arabe (avec authentification)."""

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from sqlmodel import Session, select

from app import auth, config, rag
from app.database import create_db_and_tables, get_session
from app.models import User
from app.ratelimit import RateLimit
from app.schemas import Token, UserCreate, UserRead

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Journalisation : sans handler sur le logger racine, les messages INFO de
    # l'application seraient ignorés (seuls WARNING et au-dessus passent).
    # Uvicorn configure ses propres loggers séparément : pas de doublon.
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Au démarrage : crée le fichier de base et les tables si nécessaire.
    create_db_and_tables()

    # Sécurité : en production, on refuse de démarrer avec la clé de développement.
    # Principe « fail fast » : mieux vaut une erreur explicite au démarrage
    # qu'un service silencieusement vulnérable.
    if config.using_default_secret_key():
        if config.ENVIRONMENT == "production":
            raise RuntimeError(
                "SECRET_KEY n'est pas définie alors que ENVIRONMENT=production. "
                "Générez une clé avec : "
                'python -c "import secrets; print(secrets.token_hex(32))"'
            )
        logger.warning(
            "SECRET_KEY utilise la clé de développement par défaut. "
            "Définissez SECRET_KEY avant tout déploiement en production."
        )

    # Trace le fournisseur de génération utilisé (indispensable en exploitation).
    if config.LLM_PROVIDER == "openai":
        logger.info(
            "Fournisseur LLM : API compatible OpenAI (%s, modèle %s)",
            config.LLM_BASE_URL or "URL par défaut",
            config.LLM_MODEL,
        )
    else:
        if config.LLM_PROVIDER != "ollama":
            logger.warning(
                "LLM_PROVIDER=%r inconnu : repli sur Ollama. "
                "Valeurs acceptées : 'ollama', 'openai'.",
                config.LLM_PROVIDER,
            )
        logger.info(
            "Fournisseur LLM : Ollama (%s, modèle %s)",
            config.OLLAMA_URL,
            config.LLM_MODEL,
        )

    # Trace les limites de débit actives (permet de vérifier un réglage).
    if config.RATE_LIMIT_ENABLED:
        logger.info(
            "Limitation de débit : login %d/%ds, register %d/%ds, ask %d/%ds",
            *config.LOGIN_RATE_LIMIT,
            *config.REGISTER_RATE_LIMIT,
            *config.ASK_RATE_LIMIT,
        )
    else:
        logger.warning(
            "Limitation de débit DÉSACTIVÉE (RATE_LIMIT_ENABLED=false) : "
            "l'API est exposée aux attaques par force brute et à l'épuisement de budget."
        )

    yield
    # (rien à nettoyer à l'arrêt pour l'instant)


app = FastAPI(title="Arabic RAG Chatbot", lifespan=lifespan)

# Autorise le navigateur (ou un fichier HTML local) à interroger l'API.
# Les origines sont configurables (voir config.CORS_ALLOW_ORIGINS).
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Limiteurs de débit, un par route sensible (chaque limiteur a son propre
# compteur, donc les limites ne se « mélangent » pas entre les routes).
login_rate_limit = RateLimit(*config.LOGIN_RATE_LIMIT)
register_rate_limit = RateLimit(*config.REGISTER_RATE_LIMIT)
ask_rate_limit = RateLimit(*config.ASK_RATE_LIMIT)


# --- Santé / info --------------------------------------------------------

@app.get("/health")
def health():
    """Route de santé : utilisée par Docker et les moniteurs en production."""
    return {
        "app": "Arabic RAG Chatbot",
        "status": "ok",
        "docs": "/docs",
        "endpoints": ["/register", "/login", "/me", "/ask"],
    }


@app.get("/config")
def public_config():
    """Paramètres publics (non sensibles) permettant au frontend de s'adapter.

    Exposé volontairement sans authentification : ne jamais y mettre de secret.
    """
    return {
        "allow_registration": config.ALLOW_REGISTRATION,
        "languages": ["ar", "fr"],
    }


# --- Authentification ----------------------------------------------------

@app.post(
    "/register",
    response_model=UserRead,
    dependencies=[Depends(register_rate_limit)],
)
def register(data: UserCreate, session: Session = Depends(get_session)):
    """Crée un nouveau compte (si l'inscription publique est autorisée)."""
    if not config.ALLOW_REGISTRATION:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "L'inscription publique est désactivée. "
                "Contactez l'administrateur pour obtenir un compte."
            ),
        )

    existing = session.exec(
        select(User).where(User.username == data.username)
    ).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Ce nom d'utilisateur est déjà pris.",
        )

    user = User(
        username=data.username,
        hashed_password=auth.hash_password(data.password),
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


@app.post(
    "/login",
    response_model=Token,
    dependencies=[Depends(login_rate_limit)],
)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    session: Session = Depends(get_session),
):
    """Vérifie les identifiants et renvoie un jeton JWT."""
    user = session.exec(
        select(User).where(User.username == form_data.username)
    ).first()
    if not user or not auth.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Nom d'utilisateur ou mot de passe incorrect.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return Token(access_token=auth.create_access_token(user.username))


@app.get("/me", response_model=UserRead)
def read_me(user: User = Depends(auth.get_current_user)):
    """Renvoie l'utilisateur actuellement connecté."""
    return user


# --- Chatbot (protégé) ---------------------------------------------------

@app.get("/ask", dependencies=[Depends(ask_rate_limit)])
def ask(
    question: str,
    language: str = "ar",
    user: User = Depends(auth.get_current_user),
):
    # Langue de la réponse : "fr" ou "ar" (toute autre valeur → arabe).
    language = "fr" if language.lower() == "fr" else "ar"

    # 1. RETRIEVE — chunks les plus proches + leurs sources (fichier/page/lignes).
    # La collection est chargée paresseusement (au premier appel, pas au démarrage).
    docs, sources = rag.retrieve(rag.get_collection(), question)

    no_info = (
        "Désolé, il n'y a pas assez d'informations dans les documents fournis."
        if language == "fr"
        else "عذرًا، لا توجد معلومات كافية في الوثائق المرفقة."
    )

    # Si aucun chunk pertinent n'a été trouvé, on n'appelle pas le LLM.
    if not docs:
        return {
            "question": question,
            "answer": no_info,
            "sources": sources,
            "context_used": docs,
            "language": language,
        }

    context = "\n\n".join(docs)

    # 2. GENERATE — réponse basée uniquement sur le contexte, dans la langue demandée
    answer = rag.generate(question, context, language)

    # 3. Ajoute la mention des sources à la réponse
    if sources:
        label = "📄 Sources : " if language == "fr" else "📄 المصادر: "
        answer = f"{answer}\n\n{label}{rag.format_sources(sources, language)}"
        # 4. Ajoute les extraits (phrases) cités, avec leur référence
        answer = f"{answer}\n\n{rag.format_excerpts(docs, sources, language)}"

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "context_used": docs,
        "language": language,
    }


# --- Frontend ------------------------------------------------------------

# Sert les fichiers statiques (index.html) pour tout chemin non géré par une
# route ci-dessus. Doit être déclaré EN DERNIER, sinon il masquerait les routes.
app.mount("/", StaticFiles(directory=str(config.FRONTEND_DIR), html=True), name="frontend")
