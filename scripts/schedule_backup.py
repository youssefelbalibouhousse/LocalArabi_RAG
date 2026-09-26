"""Installe (ou retire) une sauvegarde quotidienne automatique du projet.

Une sauvegarde qu'on doit lancer à la main est une sauvegarde qu'on finira par
oublier. Ce script délègue la répétition au système d'exploitation.

Usage :
    python scripts/schedule_backup.py                 # montre le plan (AUCUN effet)
    python scripts/schedule_backup.py --install       # installe la tâche quotidienne
    python scripts/schedule_backup.py --uninstall     # retire la tâche
    python scripts/schedule_backup.py --status        # état actuel de la tâche

Options :
    --hour 3 --minute 30     heure d'exécution (défaut : 03:00, heure creuse)
    --mode host|docker       où tourne l'application (défaut : host)
    --output-dir DOSSIER     transmis à scripts/backup.py
    --keep N                 nombre d'archives à conserver

Deux planificateurs sont gérés selon le système :

| Système        | Outil                    | Idempotence            |
|----------------|--------------------------|------------------------|
| Linux / macOS  | `crontab`                | ligne repérée par un **marqueur** |
| Windows        | Planificateur de tâches  | option `/F` (écrase)   |

⚠️ Sur Windows, `--install` peut exiger un terminal **« en tant
qu'administrateur »**. Ce script ne tente jamais d'élever les privilèges lui
même : il signale l'échec et laisse l'utilisateur décider.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Permet d'exécuter le script directement (python scripts/schedule_backup.py)
# en rendant le package `app` importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config

# Nom de la tâche Windows (visible dans le Planificateur de tâches).
TASK_NAME = "ArabicRAG-Backup"

# Marqueur qui identifie NOTRE ligne dans un crontab.
#
# C'est la clé de l'idempotence : on repère la ligne par ce commentaire, jamais
# par sa position. Réinstaller ne crée donc jamais de doublon, même si
# l'utilisateur a réorganisé son crontab entre-temps.
CRON_MARKER = "# arabic-rag-backup"

DEFAULT_HOUR = 3
DEFAULT_MINUTE = 0


# --- Construction de la commande ------------------------------------------

def python_executable() -> str:
    """Interpréteur Python à inscrire dans la tâche planifiée.

    `sys.executable` désigne le Python COURANT, donc l'environnement virtuel du
    projet si c'est lui qui a lancé ce script. C'est précisément ce qu'il faut :
    la tâche doit retrouver les dépendances installées dans le venv, pas celles
    du Python système.
    """
    return sys.executable or ("python" if os.name == "nt" else "python3")


def backup_arguments(output_dir=None, keep=None) -> list[str]:
    """Arguments transmis à `scripts/backup.py`.

    `--verify` et `--log` ne sont pas optionnels dans un contexte planifié :
      - `--verify` : chaque archive est prouvée valide dès sa création ;
      - `--log`    : le planificateur ne conserve pas la sortie standard, donc
                     sans journal une sauvegarde ratée échouerait en silence.
    """
    arguments = ["--verify", "--log"]

    if output_dir:
        arguments += ["--output-dir", str(output_dir)]
    if keep is not None:
        arguments += ["--keep", str(keep)]

    return arguments


def backup_command(mode="host", project_dir=None, output_dir=None, keep=None) -> str:
    """Commande complète exécutée par le planificateur.

    - mode « host »   : le Python du projet (venv) ;
    - mode « docker » : la même sauvegarde, exécutée DANS le conteneur `api`
      (c'est le mode de la production, où le venv n'existe pas sur l'hôte).

    Les chemins sont absolus en mode host : le Planificateur de tâches Windows
    n'a pas de « dossier de démarrage », contrairement à un `cd` en shell.
    """
    projet = Path(project_dir) if project_dir else config.BASE_DIR
    arguments = " ".join(backup_arguments(output_dir, keep))

    if mode == "docker":
        # `docker compose` doit être lancé depuis le dossier du projet :
        # c'est là que se trouve compose.yaml.
        return (
            f'cd "{projet}" && '
            f"docker compose exec -T api python scripts/backup.py {arguments}"
        )

    script = projet / "scripts" / "backup.py"
    return f'"{python_executable()}" "{script}" {arguments}'


# --- Validation -----------------------------------------------------------

def validate_time(hour: int, minute: int) -> tuple[int, int]:
    """Vérifie l'heure demandée. « Fail fast » : mieux vaut une erreur claire
    qu'une tâche planifiée pour une heure impossible (donc jamais exécutée)."""
    if not 0 <= hour <= 23:
        raise ValueError(f"heure invalide : {hour} (attendu : 0-23).")
    if not 0 <= minute <= 59:
        raise ValueError(f"minute invalide : {minute} (attendu : 0-59).")
    return hour, minute


def validate_mode(mode: str, platform: str | None = None) -> None:
    """Vérifie que le mode demandé est réalisable sur ce système.

    Le mode « docker » s'appuie sur un enchaînement `cd ... && docker compose`,
    qui est une syntaxe de shell POSIX. Le Planificateur de tâches Windows
    n'exécute PAS de shell : la tâche échouerait silencieusement.
    """
    if mode not in ("host", "docker"):
        raise ValueError(f"mode invalide : « {mode} » (attendu : host ou docker).")

    systeme = platform or os.name
    if mode == "docker" and systeme == "nt":
        raise ValueError(
            "le mode docker n'est pas compatible avec le Planificateur de tâches "
            "Windows (pas de shell). Utilisez --mode host, ou --mode docker sur le "
            "serveur Linux via cron."
        )


# --- Construction de la planification ------------------------------------

def build_cron_line(project_dir=None, command=None, hour=DEFAULT_HOUR,
                    minute=DEFAULT_MINUTE) -> str:
    """Ligne de crontab, au format « minute heure * * * commande # marqueur ».

    `project_dir` est conservé TEL QUEL (pas de `Path()`) : un crontab est par
    nature POSIX, et faire transiter « /srv/app » par `Path` sous Windows le
    transformerait en « \\srv\\app » — chemin qui n'existe pas sur le serveur.
    """
    projet = project_dir if project_dir else config.BASE_DIR
    commande = command or backup_command()

    return f'{minute} {hour} * * * cd "{projet}" && {commande} {CRON_MARKER}'


def crontab_with_schedule(crontab_text: str, line: str) -> str:
    """Ajoute la ligne au crontab en remplaçant l'éventuelle précédente.

    IDEMPOTENCE : réinstaller ne doit pas créer un doublon. C'est exactement le
    même problème que pour un `CREATE TABLE` sans `IF NOT EXISTS` — on identifie
    notre ligne par un marqueur, jamais par sa position.
    """
    lignes = [ligne_existante for ligne_existante in crontab_text.splitlines()
              if CRON_MARKER not in ligne_existante]
    lignes.append(line)

    return "\n".join(lignes).strip("\n") + "\n"


def crontab_without_schedule(crontab_text: str) -> str:
    """Retire notre ligne du crontab (les autres tâches sont préservées)."""
    lignes = [ligne for ligne in crontab_text.splitlines() if CRON_MARKER not in ligne]

    if not lignes:
        return ""
    return "\n".join(lignes).strip("\n") + "\n"


def build_schtasks_create_argv(task_name: str, command: str, hour: int,
                               minute: int) -> list[str]:
    """ARGV de création de la tâche Windows.

    On construit une LISTE, jamais une chaîne : chaque élément reste un argument
    distinct, ce qui évite tout problème de guillemets avec des chemins
    contenant des espaces.

    `/F` écrase la tâche du même nom — c'est notre idempotence côté Windows.
    """
    return [
        "schtasks",
        "/Create",
        "/TN", task_name,
        "/TR", command,
        "/SC", "DAILY",
        "/ST", f"{hour:02d}:{minute:02d}",
        "/F",
    ]


def build_schtasks_delete_argv(task_name: str) -> list[str]:
    return ["schtasks", "/Delete", "/TN", task_name, "/F"]


def build_schtasks_query_argv(task_name: str) -> list[str]:
    return ["schtasks", "/Query", "/TN", task_name]


# --- Exécution (fine couche dépendante du système) ------------------------

def _read_crontab() -> str:
    """Crée le crontab de l'utilisateur s'il n'existe pas encore."""
    resultat = subprocess.run(
        ["crontab", "-l"], capture_output=True, text=True, check=False
    )
    # « no crontab for <user> » n'est PAS une erreur : c'est le cas normal d'un
    # utilisateur qui n'a encore aucune tâche planifiée.
    return resultat.stdout if resultat.returncode == 0 else ""


def _write_crontab(contenu: str) -> None:
    subprocess.run(["crontab", "-"], input=contenu, text=True, check=True)


def show_plan(mode: str, hour: int, minute: int, output_dir=None, keep=None) -> int:
    """Affiche ce qui SERAIT installé, sans rien modifier. Action par défaut."""
    commande = backup_command(mode, output_dir=output_dir, keep=keep)

    print("Plan de sauvegarde automatique (aucune modification effectuée)\n")
    print(f"  Heure      : tous les jours à {hour:02d}:{minute:02d}")
    print(f"  Mode       : {mode}")
    print(f"  Commande   : {commande}")
    print(f"  Journal    : {output_dir or config.BACKUP_DIR}{os.sep}backup.log")

    if os.name == "nt":
        argv = build_schtasks_create_argv(TASK_NAME, commande, hour, minute)
        print("\n  Tâche Windows qui serait créée :")
        print(f"    {argv[0]}")
        for element in argv[1:]:
            # Les valeurs contenant des espaces sont affichées entre guillemets :
            # c'est ainsi que le système les recevra.
            affichage = f'"{element}"' if " " in element else element
            print(f"      {affichage}")
        print("\n  Pour l'installer :")
        print("    python scripts/schedule_backup.py --install")
        print("  ⚠️ Si l'accès est refusé, relancez la commande dans un terminal")
        print("     « en tant qu'administrateur ».")
    else:
        print("\n  Ligne de crontab qui serait ajoutée :")
        print("    " + build_cron_line(command=commande, hour=hour, minute=minute))
        print("\n  Pour l'installer :")
        print("    python scripts/schedule_backup.py --install")

    return 0


def install_schedule(mode: str, hour: int, minute: int, output_dir=None,
                     keep=None) -> int:
    """Installe la tâche quotidienne (remplace celle du même nom si besoin)."""
    validate_mode(mode)
    commande = backup_command(mode, output_dir=output_dir, keep=keep)

    if os.name == "nt":
        argv = build_schtasks_create_argv(TASK_NAME, commande, hour, minute)
        resultat = subprocess.run(argv, capture_output=True, text=True, check=False)

        if resultat.returncode != 0:
            print(f"❌ Échec de la création de la tâche « {TASK_NAME} ».")
            print(f"   {resultat.stderr.strip() or resultat.stdout.strip()}")
            print("   Piste : relancer ce script dans un terminal « administrateur ».")
            return 1

        print(f"✅ Tâche Windows « {TASK_NAME} » installée (tous les jours à "
              f"{hour:02d}:{minute:02d}).")
    else:
        ligne = build_cron_line(command=commande, hour=hour, minute=minute)
        _write_crontab(crontab_with_schedule(_read_crontab(), ligne))
        print(f"✅ Tâche cron installée (tous les jours à {hour:02d}:{minute:02d}).")
        print(f"   {ligne}")

    print(f"   Journal : {output_dir or config.BACKUP_DIR}{os.sep}backup.log")
    return 0


def remove_schedule() -> int:
    """Retire la tâche planifiée, sans toucher aux AUTRES tâches."""
    if os.name == "nt":
        resultat = subprocess.run(
            build_schtasks_delete_argv(TASK_NAME),
            capture_output=True, text=True, check=False,
        )
        if resultat.returncode != 0:
            print(f"❌ Aucune tâche « {TASK_NAME} » à supprimer.")
            return 1
        print(f"✅ Tâche Windows « {TASK_NAME} » supprimée.")
        return 0

    actuel = _read_crontab()
    if CRON_MARKER not in actuel:
        print("❌ Aucune tâche de sauvegarde dans le crontab.")
        return 1

    _write_crontab(crontab_without_schedule(actuel))
    print("✅ Tâche cron supprimée (les autres tâches sont intactes).")
    return 0


def schedule_status() -> int:
    """Affiche l'état de la tâche planifiée."""
    if os.name == "nt":
        resultat = subprocess.run(
            build_schtasks_query_argv(TASK_NAME),
            capture_output=True, text=True, check=False,
        )
        if resultat.returncode != 0:
            print(f"❌ Tâche « {TASK_NAME} » absente (aucune sauvegarde automatique).")
            return 1
        print(resultat.stdout.strip())
        return 0

    actuel = _read_crontab()
    lignes = [ligne for ligne in actuel.splitlines() if CRON_MARKER in ligne]
    if not lignes:
        print("❌ Aucune sauvegarde automatique dans le crontab.")
        return 1

    print("Tâche de sauvegarde active :")
    for ligne in lignes:
        print(f"  {ligne}")
    return 0


# --- Interface en ligne de commande ---------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Planifie une sauvegarde quotidienne automatique du projet.",
        epilog=(
            "Exemples :\n"
            "  python scripts/schedule_backup.py                    # montre le plan\n"
            "  python scripts/schedule_backup.py --install --hour 4 # installe à 4 h\n"
            "  python scripts/schedule_backup.py --mode docker --install  # (Linux)\n"
            "  python scripts/schedule_backup.py --status\n"
        ),
    )
    parser.add_argument("--install", action="store_true", help="installe la tâche")
    parser.add_argument("--uninstall", action="store_true", help="retire la tâche")
    parser.add_argument("--status", action="store_true", help="affiche l'état actuel")
    parser.add_argument("--hour", type=int, default=DEFAULT_HOUR, metavar="H",
                        help=f"heure d'exécution (défaut : {DEFAULT_HOUR})")
    parser.add_argument("--minute", type=int, default=DEFAULT_MINUTE, metavar="M",
                        help=f"minute d'exécution (défaut : {DEFAULT_MINUTE})")
    parser.add_argument("--mode", default="host", choices=["host", "docker"],
                        help="où tourne l'application (défaut : host)")
    parser.add_argument("--output-dir", metavar="DOSSIER", help="dossier des archives")
    parser.add_argument("--keep", type=int, metavar="N", help="archives à conserver")
    args = parser.parse_args()

    try:
        validate_time(args.hour, args.minute)
        validate_mode(args.mode)
    except ValueError as exc:
        print(f"❌ {exc}")
        return 2

    if args.uninstall:
        return remove_schedule()
    if args.status:
        return schedule_status()
    if args.install:
        return install_schedule(
            args.mode, args.hour, args.minute, args.output_dir, args.keep
        )

    return show_plan(args.mode, args.hour, args.minute, args.output_dir, args.keep)


if __name__ == "__main__":
    raise SystemExit(main())
