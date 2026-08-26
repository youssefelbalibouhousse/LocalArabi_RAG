"""Connexion à la base de données des comptes (SQLite via SQLModel)."""

from sqlmodel import Session, SQLModel, create_engine

from app import config

# `check_same_thread=False` est nécessaire pour SQLite avec FastAPI
# (plusieurs requêtes peuvent toucher la base depuis des threads différents).
engine = create_engine(
    config.DATABASE_URL,
    connect_args={"check_same_thread": False},
)


def create_db_and_tables():
    """Crée le fichier de base et les tables si elles n'existent pas encore."""
    SQLModel.metadata.create_all(engine)


def get_session():
    """Dépendance FastAPI : fournit une session DB, fermée automatiquement."""
    with Session(engine) as session:
        yield session
