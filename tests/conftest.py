"""Fixtures partagées par tous les tests.

Principe d'isolation : chaque test s'exécute sur une base de données EN MÉMOIRE
et n'appelle jamais le vrai Ollama. Les tests sont donc rapides, déterministes,
et sans aucun effet de bord sur les données réelles du projet
(data/app.db, chroma_db/).
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from app import main
from app.database import get_session
from app.ratelimit import RateLimit


@pytest.fixture(autouse=True)
def reset_rate_limits():
    """Remet les compteurs de limitation de débit à zéro avant chaque test.

    Sans cela, les tests s'épuiseraient les uns les autres : la fixture
    `auth_headers` consomme une inscription et une connexion par test.
    """
    RateLimit.reset_all()
    yield
    RateLimit.reset_all()


@pytest.fixture(name="engine")
def engine_fixture():
    """Un moteur SQLite EN MÉMOIRE, tout neuf pour chaque test.

    StaticPool est indispensable : sans lui, chaque connexion créerait sa propre
    base vide en mémoire, et le test ne verrait pas ses propres écritures.
    """
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture(name="session")
def session_fixture(engine):
    """Une session ouverte sur la base de test (fermée automatiquement après)."""
    with Session(engine) as session:
        yield session


@pytest.fixture(name="client")
def client_fixture(session):
    """Un client HTTP de test branché sur la base en mémoire.

    On remplace la dépendance `get_session` par notre session de test :
    c'est l'injection de dépendances de FastAPI (dependency_overrides)
    qui rend cette substitution possible, sans toucher au code de production.
    """
    def get_session_override():
        yield session

    main.app.dependency_overrides[get_session] = get_session_override

    # TestClient utilisé SANS "with" : le lifespan (démarrage de l'app)
    # n'est pas exécuté, donc la vraie base data/app.db n'est pas créée.
    client = TestClient(main.app)
    yield client

    main.app.dependency_overrides.clear()


@pytest.fixture(name="auth_headers")
def auth_headers_fixture(client):
    """Crée un compte, se connecte, et renvoie l'en-tête Authorization prêt à l'emploi."""
    client.post("/register", json={"username": "tester", "password": "secret123"})
    response = client.post(
        "/login",
        data={"username": "tester", "password": "secret123"},
    )
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}
