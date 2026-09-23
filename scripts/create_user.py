"""Crée un compte utilisateur en ligne de commande.

Utile lorsque l'inscription publique est désactivée (`ALLOW_REGISTRATION=false`),
typiquement pour ouvrir l'application à quelques testeurs choisis.

Usage :
    python scripts/create_user.py <nom_utilisateur>
    python scripts/create_user.py <nom_utilisateur> --password <mot_de_passe>

En Docker :
    docker compose run --rm api python scripts/create_user.py <nom_utilisateur>
"""

import argparse
import getpass
import sys
from pathlib import Path

# Permet d'exécuter le script directement (python scripts/create_user.py)
# en rendant le package `app` importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlmodel import Session, select  # noqa: E402

from app import auth  # noqa: E402
from app.database import create_db_and_tables, engine  # noqa: E402
from app.models import User  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Crée un compte utilisateur dans la base du projet.",
    )
    parser.add_argument("username", help="nom d'utilisateur à créer")
    parser.add_argument(
        "--password",
        help=(
            "mot de passe. À éviter : il resterait dans l'historique du shell. "
            "Préférez la saisie interactive (masquée)."
        ),
    )
    args = parser.parse_args()

    # Crée le fichier de base et les tables si nécessaire.
    create_db_and_tables()

    with Session(engine) as session:
        existing = session.exec(
            select(User).where(User.username == args.username)
        ).first()
        if existing:
            print(f"❌ Le compte « {args.username} » existe déjà.")
            return 1

        password = args.password or getpass.getpass("Mot de passe : ")
        if not password:
            print("❌ Le mot de passe ne peut pas être vide.")
            return 1

        session.add(
            User(
                username=args.username,
                hashed_password=auth.hash_password(password),
            )
        )
        session.commit()

    print(f"✅ Compte « {args.username} » créé.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
