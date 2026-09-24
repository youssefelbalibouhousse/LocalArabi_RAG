"""Tests UNITAIRES de la logique de sécurité (app/auth.py).

On teste ici des fonctions isolées : pas d'API, pas de base, pas de réseau.
C'est le niveau le plus bas de la pyramide des tests : rapide et précis.
"""

import jwt
import pytest

from app import auth, config

# --- Hachage des mots de passe -------------------------------------------

def test_hash_ne_stocke_pas_le_mot_de_passe_en_clair():
    hashed = auth.hash_password("secret123")

    assert hashed != "secret123"
    assert hashed.startswith("$2")  # préfixe d'une empreinte bcrypt


def test_verify_password_accepte_le_bon_mot_de_passe():
    hashed = auth.hash_password("secret123")

    assert auth.verify_password("secret123", hashed) is True


def test_verify_password_refuse_un_mauvais_mot_de_passe():
    hashed = auth.hash_password("secret123")

    assert auth.verify_password("mauvais-mot-de-passe", hashed) is False


def test_deux_empreintes_du_meme_mot_de_passe_sont_differentes():
    """Le « sel » aléatoire de bcrypt garantit deux empreintes distinctes."""
    assert auth.hash_password("secret123") != auth.hash_password("secret123")


# --- Jetons JWT ----------------------------------------------------------

def test_access_token_identifie_l_utilisateur():
    token = auth.create_access_token("alice")

    payload = jwt.decode(token, config.SECRET_KEY, algorithms=[config.JWT_ALGORITHM])

    assert payload["sub"] == "alice"
    assert "exp" in payload  # une date d'expiration est bien présente


def test_access_token_invalide_si_mauvaise_signature():
    """Un jeton signé avec une autre clé doit être rejeté."""
    token = auth.create_access_token("alice")

    # La fausse clé respecte aussi la longueur minimale : on teste bien
    # le rejet par signature invalide, pas la longueur de la clé.
    with pytest.raises(jwt.PyJWTError):
        jwt.decode(
            token,
            "mauvaise-cle-secrete-mais-assez-longue-pour-hs256",
            algorithms=[config.JWT_ALGORITHM],
        )
