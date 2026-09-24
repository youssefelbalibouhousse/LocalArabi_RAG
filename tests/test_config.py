"""Tests de la configuration liée à la sécurité (app/config.py).

Ces tests verrouillent une faille réelle détectée par la suite de tests :
la clé JWT par défaut était trop courte (20 octets) pour l'algorithme HS256,
ce qui la rendait attaquable par force brute (RFC 7518 §3.2 exige >= 32 octets).
"""

import warnings

import jwt
import pytest
from fastapi.testclient import TestClient

from app import auth, config, main

# --- Longueur de la clé --------------------------------------------------

def test_cle_de_developpement_respecte_la_longueur_minimale():
    """RFC 7518 §3.2 : HS256 exige une clé d'au moins 32 octets."""
    assert len(config.DEV_SECRET_KEY.encode("utf-8")) >= 32


def test_signature_sans_avertissement_de_longueur(monkeypatch):
    """Signer et décoder un jeton ne doit plus lever InsecureKeyLengthWarning."""
    monkeypatch.setattr(config, "SECRET_KEY", config.DEV_SECRET_KEY)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # transforme tout avertissement en erreur
        token = auth.create_access_token("alice")
        jwt.decode(token, config.SECRET_KEY, algorithms=[config.JWT_ALGORITHM])


# --- Détection de la clé de développement --------------------------------

def test_detecte_la_cle_de_developpement(monkeypatch):
    monkeypatch.setattr(config, "SECRET_KEY", config.DEV_SECRET_KEY)

    assert config.using_default_secret_key() is True


def test_ne_signale_pas_une_vraie_cle(monkeypatch):
    monkeypatch.setattr(config, "SECRET_KEY", "une-vraie-cle-secrete-de-plus-de-32-octets")

    assert config.using_default_secret_key() is False


# --- Refus de démarrer en production sans vraie clé ----------------------

def test_refuse_de_demarrer_en_production_sans_cle(monkeypatch):
    """« Fail fast » : mieux vaut un démarrage raté qu'un service vulnérable."""
    # On neutralise la création de la vraie base : ce test ne teste que la sécurité.
    monkeypatch.setattr(main, "create_db_and_tables", lambda: None)
    monkeypatch.setattr(config, "SECRET_KEY", config.DEV_SECRET_KEY)
    monkeypatch.setattr(config, "ENVIRONMENT", "production")

    with pytest.raises(RuntimeError, match="SECRET_KEY"), TestClient(main.app):
        pass


def test_demarre_en_developpement_avec_la_cle_par_defaut(monkeypatch):
    """En développement, la clé par défaut est tolérée (simple avertissement)."""
    monkeypatch.setattr(main, "create_db_and_tables", lambda: None)
    monkeypatch.setattr(config, "SECRET_KEY", config.DEV_SECRET_KEY)
    monkeypatch.setattr(config, "ENVIRONMENT", "development")

    with TestClient(main.app) as client:
        assert client.get("/health").status_code == 200


# --- Lecture des variables d'environnement booléennes --------------------

@pytest.mark.parametrize("valeur", ["1", "true", "TRUE", " yes ", "on"])
def test_env_flag_reconnait_les_valeurs_vraies(monkeypatch, valeur):
    monkeypatch.setenv("TEST_FLAG", valeur)

    assert config._env_flag("TEST_FLAG", False) is True


@pytest.mark.parametrize("valeur", ["0", "false", "no", "off", "n'importe quoi"])
def test_env_flag_reconnait_les_valeurs_fausses(monkeypatch, valeur):
    monkeypatch.setenv("TEST_FLAG", valeur)

    assert config._env_flag("TEST_FLAG", True) is False


def test_env_flag_absente_renvoie_le_defaut(monkeypatch):
    monkeypatch.delenv("TEST_FLAG", raising=False)

    assert config._env_flag("TEST_FLAG", True) is True


def test_env_flag_vide_renvoie_le_defaut(monkeypatch):
    """Une variable vide (ex. `${VAR:-}` dans compose) ne doit pas écraser le défaut."""
    monkeypatch.setenv("TEST_FLAG", "")

    assert config._env_flag("TEST_FLAG", True) is True
