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
from app.schemas import Token, UserCreate, UserRead

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Au démarrage : crée le fichier de base et les tables si nécessaire.
    create_db_and_tables()
    # Avertit si la clé de signature des jetons est encore celle de développement.
    if config.SECRET_KEY == "dev-secret-change-me":
        logger.warning(
            "SECRET_KEY utilise la valeur de développement par défaut. "
            "Définissez la variable d'environnement SECRET_KEY en production."
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


# --- Authentification ----------------------------------------------------

@app.post("/register", response_model=UserRead)
def register(data: UserCreate, session: Session = Depends(get_session)):
    """Crée un nouveau compte."""
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


@app.post("/login", response_model=Token)
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

@app.get("/ask")
def ask_arabic(question: str, user: User = Depends(auth.get_current_user)):
    # 1. RETRIEVE — chunks les plus proches + leurs sources (fichier/page/lignes).
    # La collection est chargée paresseusement (au premier appel, pas au démarrage).
    docs, sources = rag.retrieve(rag.get_collection(), question)

    # Si aucun chunk pertinent n'a été trouvé, on n'appelle pas le LLM.
    if not docs:
        return {
            "question": question,
            "answer": "عذرًا، لا توجد معلومات كافية في الوثائق المرفقة.",
            "sources": sources,
            "context_used": docs,
        }

    context = "\n\n".join(docs)

    # 2. GENERATE — réponse en arabe basée uniquement sur le contexte
    answer = rag.generate(question, context)

    # 3. Ajoute la mention des sources à la réponse
    if sources:
        answer = f"{answer}\n\n📄 المصادر: {rag.format_sources(sources)}"

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "context_used": docs,
    }


# --- Frontend ------------------------------------------------------------

# Sert les fichiers statiques (index.html) pour tout chemin non géré par une
# route ci-dessus. Doit être déclaré EN DERNIER, sinon il masquerait les routes.
app.mount("/", StaticFiles(directory=str(config.FRONTEND_DIR), html=True), name="frontend")
