"""Cœur de la sécurité : hachage des mots de passe et jetons JWT.

Concepts clés :
- On stocke une *empreinte* (hash) du mot de passe, jamais le mot de passe lui-même.
  bcrypt ajoute automatiquement un « sel » aléatoire, donc deux comptes avec le même
  mot de passe ont des empreintes différentes.
- Après connexion, l'utilisateur reçoit un JWT signé avec SECRET_KEY. À chaque requête
  protégée, il renvoie ce jeton ; on vérifie la signature pour l'identifier sans stocker
  de session côté serveur.
"""

from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlmodel import Session, select

from app import config
from app.database import get_session
from app.models import User

# Indique à FastAPI où obtenir un jeton (route POST /login) — sert aussi à /docs.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


# --- Mots de passe -------------------------------------------------------

def hash_password(password: str) -> str:
    """Retourne l'empreinte bcrypt d'un mot de passe (avec sel aléatoire)."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    """Vérifie qu'un mot de passe correspond à son empreinte."""
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))


# --- Jetons JWT ----------------------------------------------------------

def create_access_token(username: str) -> str:
    """Crée un JWT identifiant l'utilisateur, avec une date d'expiration."""
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES
    )
    payload = {"sub": username, "exp": expire}  # 'sub' = sujet (l'utilisateur)
    return jwt.encode(payload, config.SECRET_KEY, algorithm=config.JWT_ALGORITHM)


def get_current_user(
    token: str = Depends(oauth2_scheme),
    session: Session = Depends(get_session),
) -> User:
    """Dépendance qui protège une route : décode le JWT et retourne l'utilisateur.

    Lève 401 si le jeton est absent, invalide, expiré, ou si le compte n'existe plus.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Identifiants invalides ou session expirée.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(
            token, config.SECRET_KEY, algorithms=[config.JWT_ALGORITHM]
        )
        username = payload.get("sub")
        if username is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception

    user = session.exec(select(User).where(User.username == username)).first()
    if user is None:
        raise credentials_exception
    return user
