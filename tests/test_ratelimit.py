"""Tests de la limitation de débit (app/ratelimit.py).

Deux niveaux :
  - UNITAIRES    : l'algorithme de fenêtre glissante, testé sur des clés choisies ;
  - INTÉGRATION  : le comportement HTTP observable (429 + en-tête Retry-After).
"""

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from app import config, main, rag, ratelimit
from app.ratelimit import RateLimit, client_key

# --- Outils de test ------------------------------------------------------

class _FakeTime:
    """Horloge contrôlable : évite de dormir réellement dans les tests."""

    def __init__(self):
        self.value = 1000.0

    def monotonic(self):
        return self.value


def _requete(headers=None, client=("1.2.3.4", 1234)):
    """Construit une requête minimale pour tester le calcul de la clé client."""
    scope = {
        "type": "http",
        "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
        "client": client,
    }
    return Request(scope)


# --- Algorithme : fenêtre glissante --------------------------------------

def test_autorise_jusqu_a_la_limite():
    limit = RateLimit(max_requests=3, window_seconds=60)

    for _ in range(3):
        limit.check("client")  # aucune exception attendue


def test_refuse_la_requete_de_trop():
    limit = RateLimit(max_requests=3, window_seconds=60)
    for _ in range(3):
        limit.check("client")

    with pytest.raises(HTTPException) as erreur:
        limit.check("client")

    assert erreur.value.status_code == 429


def test_les_clients_ont_des_compteurs_independants():
    """Le client B ne doit pas être pénalisé par les requêtes du client A."""
    limit = RateLimit(max_requests=1, window_seconds=60)
    limit.check("client-A")

    limit.check("client-B")  # doit passer : compteur distinct


def test_retry_after_indique_le_temps_restant(monkeypatch):
    horloge = _FakeTime()
    monkeypatch.setattr(ratelimit, "time", horloge)

    limit = RateLimit(max_requests=1, window_seconds=60)
    limit.check("client")
    horloge.value += 20  # 20 secondes se sont écoulées

    with pytest.raises(HTTPException) as erreur:
        limit.check("client")

    # Il reste 40 secondes sur la fenêtre de 60.
    assert erreur.value.headers["Retry-After"] == "40"


def test_la_fenetre_se_vide_avec_le_temps(monkeypatch):
    """Après la fenêtre, le client est de nouveau autorisé (comportement glissant)."""
    horloge = _FakeTime()
    monkeypatch.setattr(ratelimit, "time", horloge)

    limit = RateLimit(max_requests=1, window_seconds=60)
    limit.check("client")
    with pytest.raises(HTTPException):
        limit.check("client")

    horloge.value += 61  # la fenêtre est dépassée
    limit.check("client")  # de nouveau autorisé


def test_reset_remet_le_compteur_a_zero():
    limit = RateLimit(max_requests=1, window_seconds=60)
    limit.check("client")

    limit.reset()

    limit.check("client")  # doit passer


# --- Identification du client -------------------------------------------

def test_client_key_utilise_x_forwarded_for():
    """Derrière un proxy, l'IP réelle du client est dans X-Forwarded-For."""
    requete = _requete(headers={"X-Forwarded-For": "203.0.113.7"})

    assert client_key(requete) == "203.0.113.7"


def test_client_key_prend_la_premiere_ip_de_la_liste():
    """X-Forwarded-For peut contenir une chaîne de proxys : on prend l'origine."""
    requete = _requete(headers={"X-Forwarded-For": "203.0.113.7, 10.0.0.1"})

    assert client_key(requete) == "203.0.113.7"


def test_client_key_sans_proxy_utilise_l_adresse_de_connexion():
    requete = _requete(client=("198.51.100.3", 5555))

    assert client_key(requete) == "198.51.100.3"


# --- Intégration HTTP ---------------------------------------------------

def test_login_refuse_apres_trop_de_tentatives(client):
    """Protection anti force brute : au-delà de la limite, on répond 429."""
    client.post("/register", json={"username": "alice", "password": "secret123"})
    maximum = config.LOGIN_RATE_LIMIT[0]

    for _ in range(maximum):
        reponse = client.post("/login", data={"username": "alice", "password": "faux"})
        assert reponse.status_code == 401  # mauvais mot de passe, mais autorisé

    reponse = client.post("/login", data={"username": "alice", "password": "faux"})

    assert reponse.status_code == 429
    assert "Retry-After" in reponse.headers


def test_register_refuse_apres_trop_de_creations(client):
    maximum = config.REGISTER_RATE_LIMIT[0]

    for index in range(maximum):
        reponse = client.post(
            "/register",
            json={"username": f"utilisateur{index}", "password": "secret123"},
        )
        assert reponse.status_code == 200

    reponse = client.post(
        "/register",
        json={"username": "un_de_trop", "password": "secret123"},
    )

    assert reponse.status_code == 429


def test_ask_refuse_apres_trop_de_questions(client, auth_headers, monkeypatch):
    """Protection du budget : /ask consomme du GPU ou des tokens d'API."""
    monkeypatch.setattr(main.ask_rate_limit, "max_requests", 2)
    monkeypatch.setattr(rag, "get_collection", lambda: "collection-simulee")
    monkeypatch.setattr(rag, "retrieve", lambda *args, **kwargs: ([], []))

    for _ in range(2):
        assert client.get("/ask", params={"question": "q"}, headers=auth_headers).status_code == 200

    reponse = client.get("/ask", params={"question": "q"}, headers=auth_headers)

    assert reponse.status_code == 429


def test_limitation_desactivee_laisse_tout_passer(client, monkeypatch):
    """Un interrupteur global permet de désactiver la limitation."""
    monkeypatch.setattr(config, "RATE_LIMIT_ENABLED", False)
    monkeypatch.setattr(main.login_rate_limit, "max_requests", 1)
    client.post("/register", json={"username": "alice", "password": "secret123"})

    for _ in range(5):
        reponse = client.post("/login", data={"username": "alice", "password": "faux"})
        assert reponse.status_code == 401  # jamais 429
