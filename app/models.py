"""Modèles de base de données (tables SQL via SQLModel)."""

from datetime import datetime, timezone

from sqlmodel import Field, SQLModel


class User(SQLModel, table=True):
    """Un compte utilisateur.

    On ne stocke JAMAIS le mot de passe en clair, seulement son empreinte
    (hash bcrypt) dans `hashed_password`.
    """

    id: int | None = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True)
    hashed_password: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
