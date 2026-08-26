"""Schémas Pydantic : ce que l'API reçoit et renvoie (distinct des tables SQL)."""

from pydantic import BaseModel


class UserCreate(BaseModel):
    """Corps de la requête d'inscription."""
    username: str
    password: str


class UserRead(BaseModel):
    """Représentation publique d'un utilisateur (sans mot de passe)."""
    id: int
    username: str


class Token(BaseModel):
    """Jeton renvoyé après connexion réussie."""
    access_token: str
    token_type: str = "bearer"
