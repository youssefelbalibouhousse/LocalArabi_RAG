"""Sauvegarde et restauration des données du projet (règle 3-2-1).

Ce script fournit les trois opérations inséparables d'une vraie stratégie :
créer une archive, la VÉRIFIER, et la RESTAURER. Sans la troisième, les deux
premières ne sont qu'un espoir.

Usage :
    python scripts/backup.py                        # crée une archive
    python scripts/backup.py --list                 # liste les archives
    python scripts/backup.py --no-vectors           # sans la base vectorielle
    python scripts/backup.py --restore <archive>    # restaure (après vérification)
    python scripts/backup.py --restore <archive> --dry-run   # vérifie seulement

En Docker :
    docker compose exec api python scripts/backup.py

Ce qui est sauvegardé, par ordre d'irremplaçabilité :
    data/app.db        les comptes (mots de passe hachés = non réversibles)
    data/documents/    les PDF sources (sans eux, plus rien à indexer)
    .env               la SECRET_KEY (la perdre invalide tous les jetons)
    chroma_db/         la base vectorielle — DÉRIVÉE des PDF, donc secondaire
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import sys
import tarfile
import tempfile
from contextlib import closing, redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath

# Permet d'exécuter le script directement (python scripts/backup.py)
# en rendant le package `app` importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config

ARCHIVE_PREFIX = "backup_"
ARCHIVE_SUFFIX = ".tar.gz"
MANIFEST_NAME = "MANIFEST.json"
LOG_NAME = "backup.log"


@dataclass(frozen=True)
class BackupItem:
    """Un élément à sauvegarder.

    Attributs :
        source   : chemin réel sur le disque.
        relative : chemin dans l'archive (toujours portable, séparateurs « / »).
        kind     : « sqlite », « dir » ou « file » — détermine la méthode de copie.
    """

    source: Path
    relative: Path
    kind: str


# --- Utilitaires ----------------------------------------------------------

def sha256(path: Path) -> str:
    """Empreinte SHA-256 d'un fichier, lue par blocs (les gros fichiers ne
    sont jamais chargés entièrement en mémoire)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for bloc in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(bloc)
    return digest.hexdigest()


def safe_copy_sqlite(source: Path, destination: Path) -> None:
    """Copie une base SQLite de façon COHÉRENTE, même si l'application écrit.

    Pourquoi pas un simple `shutil.copy2` ? Parce que SQLite écrit dans le
    fichier principal ET dans un journal (mode WAL). Une copie brute peut donc
    capturer un état intermédiaire — la « torn copy » — et produire une base
    illisible au moment où l'on en aura le plus besoin.

    `Connection.backup()` délègue la copie à SQLite lui-même : verrouillage et
    recopie page par page, ce qui garantit un instantané cohérent.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    # mode=ro : la source est ouverte en LECTURE SEULE (aucun risque d'écriture).
    source_uri = f"file:{source.as_posix()}?mode=ro"
    with (
        closing(sqlite3.connect(source_uri, uri=True)) as src,
        closing(sqlite3.connect(str(destination))) as dst,
    ):
        src.backup(dst)


def collect(include_vectors: bool = True) -> list[BackupItem]:
    """Renvoie les éléments à sauvegarder, du plus critique au moins critique.

    Le classement suit la question : « cette donnée est-elle régénérable ? ».
    Un index (chroma_db) se reconstruit ; un mot de passe haché, non.
    """
    items: list[BackupItem] = []

    # 1. Les comptes : irremplaçables (un hachage bcrypt ne se « dé-hache » pas).
    database = config.sqlite_database_path()
    if database is not None and database.is_file():
        items.append(BackupItem(database, Path("data/app.db"), "sqlite"))

    # 2. Les PDF sources : irremplaçables, et nécessaires pour reconstruire l'index.
    if config.DOCUMENTS_DIR.is_dir():
        items.append(BackupItem(config.DOCUMENTS_DIR, Path("data/documents"), "dir"))

    # 3. La base vectorielle : DÉRIVÉE des PDF (scripts/build_kb.py la régénère).
    #    Sauvegardée par confort (évite une réindexation de plusieurs minutes),
    #    pas par nécessité. --no-vectors l'exclut et allège fortement l'archive.
    if include_vectors:
        chroma = Path(config.CHROMA_DB_PATH)
        if chroma.is_dir():
            items.append(BackupItem(chroma, Path("chroma_db"), "dir"))

    # 4. Le fichier .env : contient la SECRET_KEY, unique elle aussi.
    env_file = config.BASE_DIR / ".env"
    if env_file.is_file():
        items.append(BackupItem(env_file, Path(".env"), "file"))

    return items


def _stage_item(item: BackupItem, staging: Path) -> None:
    """Recopie un élément dans le dossier temporaire, prêt à être archivé."""
    target = staging / item.relative
    target.parent.mkdir(parents=True, exist_ok=True)

    if item.kind == "sqlite":
        safe_copy_sqlite(item.source, target)
    elif item.kind == "dir":
        shutil.copytree(item.source, target, dirs_exist_ok=True)
    else:
        shutil.copy2(item.source, target)


def build_manifest(staging: Path, include_vectors: bool) -> dict:
    """Décrit le contenu de l'archive : date, sources, taille et empreinte SHA-256.

    L'empreinte permet de PROUVER, des mois plus tard, que l'archive est intacte.
    Une sauvegarde dont on ne peut rien vérifier n'inspire aucune confiance.
    """
    fichiers = [
        {
            "path": path.relative_to(staging).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(p for p in staging.rglob("*") if p.is_file())
    ]

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "environment": config.ENVIRONMENT,
        "includes_vectors": include_vectors,
        "files": fichiers,
    }


# --- Création -------------------------------------------------------------

def apply_retention(
    destination: Path | str | None = None, keep: int | None = None
) -> list[Path]:
    """Supprime les archives les plus anciennes au-delà de `keep`.

    L'horodatage « AAAA-MM-JJ_HHMMSS » a été choisi pour que le tri
    ALPHABÉTIQUE des noms soit aussi le tri CHRONOLOGIQUE : inutile de lire la
    date de modification de chaque fichier, et l'ordre reste juste même si l'on
    déplace les archives.
    """
    destination = Path(destination) if destination is not None else config.BACKUP_DIR
    keep = config.BACKUP_RETENTION if keep is None else keep

    if keep < 1:
        raise ValueError(f"keep={keep} est invalide : la valeur doit être >= 1.")

    archives = sorted(destination.glob(f"{ARCHIVE_PREFIX}*{ARCHIVE_SUFFIX}"))

    supprimees: list[Path] = []
    # `archives[:0]` est vide : si l'on est sous la limite, rien n'est supprimé.
    for archive in archives[: max(len(archives) - keep, 0)]:
        archive.unlink()
        supprimees.append(archive)

    return supprimees


def list_backups(destination: Path | str | None = None) -> list[Path]:
    """Liste les archives présentes, de la plus ancienne à la plus récente."""
    destination = Path(destination) if destination is not None else config.BACKUP_DIR
    return sorted(destination.glob(f"{ARCHIVE_PREFIX}*{ARCHIVE_SUFFIX}"))


def create_backup(
    destination: Path | str | None = None,
    keep: int | None = None,
    include_vectors: bool = True,
) -> Path | None:
    """Crée une archive horodatée, puis applique la politique de rétention.

    Renvoie le chemin de l'archive créée, ou None s'il n'y avait rien à sauvegarder.
    """
    destination = Path(destination) if destination is not None else config.BACKUP_DIR
    destination.mkdir(parents=True, exist_ok=True)

    items = collect(include_vectors)
    if not items:
        print("❌ Rien à sauvegarder : aucune donnée trouvée.")
        return None

    archive = destination / (
        f"{ARCHIVE_PREFIX}{datetime.now():%Y-%m-%d_%H%M%S}{ARCHIVE_SUFFIX}"
    )

    # Dossier temporaire : on y prépare une copie PROPRE (et cohérente pour
    # SQLite) avant d'empaqueter, plutôt que d'archiver des fichiers vivants.
    with tempfile.TemporaryDirectory() as temporaire:
        staging = Path(temporaire)

        for item in items:
            _stage_item(item, staging)
            print(f"   • {item.relative.as_posix()}")

        # Le manifeste est calculé AVANT d'être écrit : il ne se liste pas lui-même.
        manifest = build_manifest(staging, include_vectors)
        (staging / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        with tarfile.open(archive, "w:gz") as tar:
            tar.add(staging / MANIFEST_NAME, arcname=MANIFEST_NAME)
            for item in items:
                tar.add(staging / item.relative, arcname=item.relative.as_posix())

    taille_mo = archive.stat().st_size / (1024 * 1024)
    print(f"✅ Archive créée : {archive.name} ({taille_mo:.1f} Mo)")
    print(f"   {len(manifest['files'])} fichier(s) sauvegardé(s).")

    for supprimee in apply_retention(destination, keep):
        print(f"🗑️  Ancienne archive supprimée : {supprimee.name}")

    return archive


# --- Restauration ---------------------------------------------------------

def _validate_members(tar: tarfile.TarFile, destination: Path) -> None:
    """Refuse toute archive tentant d'écrire en dehors du dossier cible.

    Protection contre le « tar slip » (path traversal) : une archive piégée peut
    contenir « ../../etc/passwd » ou un lien symbolique pointant hors du dossier,
    et écraser des fichiers système au moment de l'extraction.

    On ne fait donc JAMAIS confiance à une archive — même la sienne : elle peut
    avoir transité par un disque défaillant ou un service de stockage tiers.
    """
    racine = destination.resolve()

    for member in tar.getmembers():
        # Les archives utilisent toujours « / », quel que soit l'OS d'origine.
        nom = PurePosixPath(member.name)

        if nom.is_absolute() or ".." in nom.parts:
            raise ValueError(f"Archive refusée : chemin dangereux « {member.name} ».")

        if member.issym() or member.islnk():
            raise ValueError(f"Archive refusée : lien non autorisé « {member.name} ».")

        if not (racine / nom).resolve().is_relative_to(racine):
            raise ValueError(
                f"Archive refusée : chemin hors du dossier cible « {member.name} »."
            )


def _extract(tar: tarfile.TarFile, destination: Path) -> None:
    """Extrait l'archive dans `destination` (membres déjà validés).

    `filter="data"` (Python >= 3.11.4) ajoute une SECONDE couche de protection,
    appliquée par la bibliothèque standard elle-même. Le repli silencieux sert
    aux versions plus anciennes, où ce paramètre n'existe pas encore.
    """
    try:
        tar.extractall(destination, filter="data")
    except TypeError:  # Python < 3.11.4 : le paramètre `filter` n'existe pas
        tar.extractall(destination)


def verify_manifest(staging: Path, manifest: dict) -> None:
    """Vérifie que chaque fichier extrait correspond à l'empreinte enregistrée.

    C'est LE contrôle qui distingue une sauvegarde d'un espoir : taille et
    SHA-256 sont comparés fichier par fichier AVANT de toucher aux données
    réelles. Une seule anomalie interrompt la restauration.
    """
    anomalies: list[str] = []

    for entree in manifest.get("files", []):
        chemin = staging / entree["path"]

        if not chemin.is_file():
            anomalies.append(f"fichier absent : {entree['path']}")
        elif chemin.stat().st_size != entree["size"]:
            anomalies.append(f"taille différente : {entree['path']}")
        elif sha256(chemin) != entree["sha256"]:
            anomalies.append(f"empreinte invalide : {entree['path']}")

    if anomalies:
        raise ValueError("Archive corrompue :\n  - " + "\n  - ".join(anomalies))


def restore_backup(
    archive: Path | str,
    project_root: Path | str | None = None,
    dry_run: bool = False,
) -> list[str]:
    """Restaure une archive après l'avoir intégralement VÉRIFIÉE.

    L'ordre des étapes est le cœur de la sécurité ici :
        1. valider les chemins (aucune écriture hors du dossier cible) ;
        2. extraire dans un dossier TEMPORAIRE (jamais dans les données réelles) ;
        3. vérifier les empreintes SHA-256 ;
        4. seulement alors, copier vers le projet.

    Rien n'est écrit si une seule vérification échoue : une restauration
    partielle est bien pire que pas de restauration du tout.
    """
    archive = Path(archive)
    if not archive.is_file():
        raise FileNotFoundError(f"Archive introuvable : {archive}")

    project_root = Path(project_root) if project_root is not None else config.BASE_DIR

    restaures: list[str] = []

    with tempfile.TemporaryDirectory() as temporaire:
        staging = Path(temporaire)

        with tarfile.open(archive, "r:gz") as tar:
            _validate_members(tar, staging)
            _extract(tar, staging)

        manifest_path = staging / MANIFEST_NAME
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            verify_manifest(staging, manifest)
            print("🔎 Empreintes SHA-256 vérifiées : archive intacte.")
        else:
            print("⚠️  Manifeste absent : archive produite par une version antérieure.")

        for entree in sorted(staging.iterdir()):
            if entree.name == MANIFEST_NAME:
                continue

            target = project_root / entree.name

            if not dry_run:
                if entree.is_dir():
                    # Remplacement EXACT : un dossier fusionné conserverait des
                    # fichiers obsolètes (ex. anciens segments ChromaDB pointant
                    # vers un index qui n'existe plus).
                    if target.exists():
                        shutil.rmtree(target)
                    shutil.copytree(entree, target)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(entree, target)

            print(f"   ♻️  {entree.name}")
            restaures.append(entree.name)

    return restaures


# --- Interface en ligne de commande ---------------------------------------

def _resolve_archive(name: str, destination: Path) -> Path:
    """Accepte un chemin complet, ou un simple nom d'archive du dossier par défaut."""
    candidat = Path(name)
    return candidat if candidat.is_file() else destination / candidat.name


def _afficher_liste(destination: Path) -> int:
    archives = list_backups(destination)
    if not archives:
        print(f"Aucune archive dans {destination}")
        return 0

    print(f"Archives dans {destination} :")
    for archive in archives:
        stat = archive.stat()
        taille_mo = stat.st_size / (1024 * 1024)
        modifie = datetime.fromtimestamp(stat.st_mtime)
        print(f"  {archive.name:<38} {taille_mo:>7.1f} Mo   {modifie:%Y-%m-%d %H:%M}")
    return 0


class _Tee:
    """Écrit dans PLUSIEURS flux à la fois (ici : la console et le journal).

    Sert à l'option `--log`. Le nom vient du « T » des plomberies : un flux
    d'entrée, deux sorties — comme un raccord en T.
    """

    def __init__(self, *flux):
        self.flux = flux

    def write(self, texte):
        for flux in self.flux:
            flux.write(texte)
        return len(texte)

    def flush(self):
        for flux in self.flux:
            flux.flush()


def _executer(args, destination: Path) -> int:
    """Effectue l'action demandée : lister, restaurer ou créer."""
    if args.list:
        return _afficher_liste(destination)

    if args.restore:
        archive = _resolve_archive(args.restore, destination)
        print(f"♻️  Restauration depuis {archive.name}…")
        try:
            restaures = restore_backup(archive, dry_run=args.dry_run)
        except (ValueError, FileNotFoundError, tarfile.TarError) as exc:
            print(f"❌ {exc}")
            return 1

        if args.dry_run:
            print(f"✅ Vérification réussie ({len(restaures)} élément(s)). Rien n'a été écrit.")
        else:
            print(f"✅ {len(restaures)} élément(s) restauré(s). Redémarrez l'application.")
        return 0

    print(f"📦 Création d'une sauvegarde dans {destination}…")
    archive = create_backup(
        destination, keep=args.keep, include_vectors=not args.no_vectors
    )
    if archive is None:
        return 1

    if not args.verify:
        return 0

    # On PROUVE immédiatement que l'archive est exploitable. C'est ce qui rend
    # automatique la règle « une sauvegarde jamais restaurée n'existe pas » :
    # plus besoin d'y penser, la vérification fait partie de la sauvegarde.
    print("🔎 Vérification de l'archive qui vient d'être créée…")
    try:
        restore_backup(archive, dry_run=True)
    except (ValueError, FileNotFoundError, tarfile.TarError) as exc:
        print(f"❌ L'archive créée est invalide : {exc}")
        return 1

    print("✅ Sauvegarde créée ET vérifiée.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sauvegarde et restauration des données du projet (règle 3-2-1).",
        epilog=(
            "Exemples :\n"
            "  python scripts/backup.py                       # crée une archive\n"
            "  python scripts/backup.py --verify --log        # crée, vérifie, journalise\n"
            "  python scripts/backup.py --list                # liste les archives\n"
            "  python scripts/backup.py --no-vectors          # sans la base vectorielle\n"
            "  python scripts/backup.py --restore <archive>   # restaure\n"
        ),
    )
    parser.add_argument("--list", action="store_true", help="liste les archives existantes")
    parser.add_argument("--restore", metavar="ARCHIVE", help="restaure une archive")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="avec --restore : vérifie l'archive sans écrire aucun fichier",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="après la création, vérifie l'archive (empreintes SHA-256) — usage planifié",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help=f"ajoute la sortie à {LOG_NAME} (indispensable en exécution planifiée)",
    )
    parser.add_argument(
        "--no-vectors",
        action="store_true",
        help="exclut la base vectorielle (donnée dérivée, reconstruite par build_kb.py)",
    )
    parser.add_argument(
        "--output-dir",
        metavar="DOSSIER",
        help=f"dossier des archives (défaut : {config.BACKUP_DIR})",
    )
    parser.add_argument(
        "--keep",
        type=int,
        metavar="N",
        help=f"nombre d'archives conservées (défaut : {config.BACKUP_RETENTION})",
    )
    args = parser.parse_args()

    destination = Path(args.output_dir) if args.output_dir else config.BACKUP_DIR

    if not args.log:
        return _executer(args, destination)

    # Journalisation : tout ce qui s'affiche part aussi dans le journal, horodaté,
    # y compris le CODE DE SORTIE. Sans trace écrite, une sauvegarde planifiée qui
    # échoue échoue en silence — le pire scénario pour un mécanisme de sécurité.
    destination.mkdir(parents=True, exist_ok=True)
    with (
        (destination / LOG_NAME).open("a", encoding="utf-8") as journal,
        redirect_stdout(_Tee(sys.stdout, journal)),
    ):
        print(f"\n=== {datetime.now():%Y-%m-%d %H:%M:%S} ===")
        code = _executer(args, destination)
        print(f"--- code de sortie : {code} ---")

    return code


if __name__ == "__main__":
    raise SystemExit(main())
