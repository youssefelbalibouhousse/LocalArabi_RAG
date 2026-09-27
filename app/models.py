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


class IngestedDocument(SQLModel, table=True):
    """État d'ingestion d'un document — le « registre ».

    Vit dans `data/app.db`, PAS dans `chroma_db/` : l'index vectoriel est
    DÉRIVÉ (il se reconstruit), le registre ne l'est pas — c'est le seul
    souvenir de ce qui a été ingéré, et avec quel modèle d'embedding.

    `fingerprint` est l'empreinte SHA-256 du CONTENU du fichier source. On
    compare des empreintes et jamais des dates de modification : une copie ou
    une restauration change la date sans changer le contenu, ce qui
    déclencherait une ré-ingestion pour rien.
    """

    id: int | None = Field(default=None, primary_key=True)
    source: str = Field(index=True, unique=True)
    fingerprint: str
    chunk_count: int
    embedding_model: str
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
