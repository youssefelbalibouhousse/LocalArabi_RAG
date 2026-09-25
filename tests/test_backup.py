"""Tests de scripts/backup.py (sauvegarde, vérification, restauration).

Deux propriétés sont critiques et donc verrouillées explicitement :

1. une copie de base SQLite doit être COHÉRENTE, même si des écritures sont en
   cours — c'est tout l'intérêt de l'API de sauvegarde de SQLite, et c'est ce
   qui la distingue d'un simple `shutil.copy2` ;
2. une archive piégée ne doit JAMAIS pouvoir écrire hors du dossier cible
   (protection contre le « tar slip »), et une archive corrompue doit être
   détectée AVANT que la moindre donnée réelle ne soit écrasée.

Comme pour le reste de la suite, aucun test ne touche aux vraies données :
tout se passe dans un dossier temporaire (tmp_path).
"""

import io
import json
import shutil
import sqlite3
import tarfile
from pathlib import Path

import backup
import pytest

from app import config


@pytest.fixture(name="projet")
def projet_fixture(tmp_path, monkeypatch):
    """Un faux projet isolé contenant toutes les catégories de données."""
    base = tmp_path / "projet"
    (base / "data" / "documents").mkdir(parents=True)
    (base / "chroma_db").mkdir()

    (base / "data" / "documents" / "cours.pdf").write_bytes(b"%PDF-1.4 faux pdf")
    (base / "chroma_db" / "chroma.sqlite3").write_text("index", encoding="utf-8")
    (base / ".env").write_text("SECRET_KEY=abc\n", encoding="utf-8")

    database = base / "data" / "app.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE user (id INTEGER PRIMARY KEY, username TEXT)")
        connection.execute("INSERT INTO user (username) VALUES ('alice')")

    monkeypatch.setattr(config, "BASE_DIR", base)
    monkeypatch.setattr(config, "DOCUMENTS_DIR", base / "data" / "documents")
    monkeypatch.setattr(config, "CHROMA_DB_PATH", str(base / "chroma_db"))
    monkeypatch.setattr(config, "DATABASE_URL", f"sqlite:///{database}")

    return base


def _archive_falsifiee(chemin: Path, taille: int, empreinte: str) -> Path:
    """Fabrique une archive dont le manifeste annonce une empreinte erronée."""
    contenu = b"contenu reel"
    manifeste = {
        "created_at": "2026-01-01T00:00:00",
        "environment": "test",
        "includes_vectors": False,
        "files": [{"path": "data/app.db", "size": taille, "sha256": empreinte}],
    }

    with tarfile.open(chemin, "w:gz") as tar:
        blob = json.dumps(manifeste).encode("utf-8")
        info = tarfile.TarInfo(backup.MANIFEST_NAME)
        info.size = len(blob)
        tar.addfile(info, io.BytesIO(blob))

        info = tarfile.TarInfo("data/app.db")
        info.size = len(contenu)
        tar.addfile(info, io.BytesIO(contenu))

    return chemin


# --- Empreintes et copie SQLite -------------------------------------------

def test_sha256_correspond_a_la_valeur_de_reference(tmp_path):
    """Verrouille l'algorithme : si le calcul change, toutes les archives
    existantes deviendraient invérifiables."""
    fichier = tmp_path / "exemple.txt"
    fichier.write_bytes(b"abc")

    assert backup.sha256(fichier) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_copie_sqlite_est_coherente(projet, tmp_path):
    destination = tmp_path / "copie.db"

    backup.safe_copy_sqlite(projet / "data" / "app.db", destination)

    with sqlite3.connect(destination) as connection:
        assert connection.execute("SELECT username FROM user").fetchall() == [("alice",)]


def test_copie_sqlite_reste_valide_pendant_une_ecriture(projet, tmp_path):
    """La copie doit fonctionner alors qu'une transaction est ouverte sur la source.

    C'est précisément le cas où un `shutil.copy2` produirait une « torn copy ».
    La transaction non validée de 'bob' ne doit PAS apparaître : la sauvegarde
    est un instantané, pas une copie approximative.
    """
    destination = tmp_path / "copie.db"

    with sqlite3.connect(projet / "data" / "app.db") as ecrivain:
        ecrivain.execute("BEGIN")
        ecrivain.execute("INSERT INTO user (username) VALUES ('bob')")

        backup.safe_copy_sqlite(projet / "data" / "app.db", destination)

    with sqlite3.connect(destination) as connection:
        lignes = connection.execute("SELECT username FROM user").fetchall()

    assert lignes == [("alice",)]


# --- Création d'archive ----------------------------------------------------

def test_sauvegarde_cree_une_archive_avec_manifeste(projet, tmp_path):
    archive = backup.create_backup(destination=tmp_path / "archives")

    assert archive is not None
    assert archive.name.startswith("backup_") and archive.name.endswith(".tar.gz")

    with tarfile.open(archive) as tar:
        noms = tar.getnames()

    assert backup.MANIFEST_NAME in noms
    assert "data/app.db" in noms
    assert "data/documents/cours.pdf" in noms
    assert "chroma_db/chroma.sqlite3" in noms
    assert ".env" in noms


def test_sauvegarde_sans_vecteurs_exclut_l_index(projet, tmp_path):
    """chroma_db est une donnée DÉRIVÉE : on doit pouvoir l'exclure pour
    alléger fortement l'archive."""
    archive = backup.create_backup(destination=tmp_path / "archives", include_vectors=False)

    with tarfile.open(archive) as tar:
        noms = tar.getnames()

    assert not any(nom.startswith("chroma_db") for nom in noms)
    assert "data/app.db" in noms
    assert "data/documents/cours.pdf" in noms


def test_manifeste_contient_les_empreintes_de_chaque_fichier(projet, tmp_path):
    archive = backup.create_backup(destination=tmp_path / "archives")

    with tarfile.open(archive) as tar:
        manifeste = json.load(tar.extractfile(backup.MANIFEST_NAME))

    chemins = {entree["path"] for entree in manifeste["files"]}
    assert chemins == {
        "data/app.db",
        "data/documents/cours.pdf",
        "chroma_db/chroma.sqlite3",
        ".env",
    }
    for entree in manifeste["files"]:
        assert len(entree["sha256"]) == 64
        assert entree["size"] > 0


def test_sauvegarde_sans_donnees_ne_produit_rien(tmp_path, monkeypatch):
    vide = tmp_path / "vide"
    vide.mkdir()
    monkeypatch.setattr(config, "BASE_DIR", vide)
    monkeypatch.setattr(config, "DOCUMENTS_DIR", vide / "data" / "documents")
    monkeypatch.setattr(config, "CHROMA_DB_PATH", str(vide / "chroma_db"))
    monkeypatch.setattr(
        config, "DATABASE_URL", f"sqlite:///{vide / 'data' / 'app.db'}"
    )

    assert backup.create_backup(destination=tmp_path / "archives") is None


# --- Rétention -------------------------------------------------------------

def _fausses_archives(dossier: Path, nombre: int) -> None:
    dossier.mkdir(parents=True, exist_ok=True)
    for jour in range(1, nombre + 1):
        (dossier / f"backup_2026-01-{jour:02d}_120000.tar.gz").write_bytes(b"x")


def test_retention_supprime_les_plus_anciennes(tmp_path):
    _fausses_archives(tmp_path, 4)

    supprimees = backup.apply_retention(tmp_path, keep=2)

    assert [chemin.name for chemin in supprimees] == [
        "backup_2026-01-01_120000.tar.gz",
        "backup_2026-01-02_120000.tar.gz",
    ]
    assert len(backup.list_backups(tmp_path)) == 2


def test_retention_ne_supprime_rien_sous_la_limite(tmp_path):
    _fausses_archives(tmp_path, 1)

    assert backup.apply_retention(tmp_path, keep=7) == []
    assert len(backup.list_backups(tmp_path)) == 1


def test_retention_refuse_une_valeur_invalide(tmp_path):
    _fausses_archives(tmp_path, 1)

    with pytest.raises(ValueError, match="keep=0"):
        backup.apply_retention(tmp_path, keep=0)

    assert len(backup.list_backups(tmp_path)) == 1


# --- Restauration (chemin nominal) ----------------------------------------

def test_restauration_restitue_toutes_les_donnees(projet, tmp_path):
    """Le test qui donne sa valeur à tout le reste : on simule un sinistre,
    puis on vérifie que les données reviennent réellement à l'identique."""
    archive = backup.create_backup(destination=tmp_path / "archives")

    # 💥 Sinistre : suppression des données du projet.
    shutil.rmtree(projet / "data")
    shutil.rmtree(projet / "chroma_db")
    (projet / ".env").unlink()

    elements = backup.restore_backup(archive, project_root=projet)

    assert sorted(elements) == [".env", "chroma_db", "data"]
    assert (projet / "data" / "documents" / "cours.pdf").read_bytes() == b"%PDF-1.4 faux pdf"
    assert (projet / "chroma_db" / "chroma.sqlite3").read_text(encoding="utf-8") == "index"
    assert (projet / ".env").read_text(encoding="utf-8") == "SECRET_KEY=abc\n"

    with sqlite3.connect(projet / "data" / "app.db") as connection:
        assert connection.execute("SELECT username FROM user").fetchall() == [("alice",)]


def test_restauration_remplace_exactement_les_dossiers(projet, tmp_path):
    """Après restauration, aucun fichier parasite ne doit subsister : un dossier
    fusionné garderait des segments d'index devenus invalides."""
    archive = backup.create_backup(destination=tmp_path / "archives")
    (projet / "chroma_db" / "segment_obsolete.bin").write_bytes(b"obsolete")

    backup.restore_backup(archive, project_root=projet)

    assert not (projet / "chroma_db" / "segment_obsolete.bin").exists()


def test_dry_run_verifie_sans_rien_ecrire(projet, tmp_path):
    archive = backup.create_backup(destination=tmp_path / "archives")
    (projet / "data" / "app.db").unlink()

    elements = backup.restore_backup(archive, project_root=projet, dry_run=True)

    assert "data" in elements
    assert not (projet / "data" / "app.db").exists()


def test_restauration_archive_introuvable(tmp_path, projet):
    with pytest.raises(FileNotFoundError, match="introuvable"):
        backup.restore_backup(tmp_path / "inexistant.tar.gz", project_root=projet)


# --- Sécurité : archives piégées et archives corrompues -------------------

def test_restauration_refuse_un_chemin_qui_sort_du_dossier(tmp_path, projet):
    """« tar slip » : une archive contenant « ../evade.txt » doit être rejetée."""
    contenu = b"contenu malveillant"
    archive = tmp_path / "backup_2026-01-01_120000.tar.gz"

    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("../evade.txt")
        info.size = len(contenu)
        tar.addfile(info, io.BytesIO(contenu))

    with pytest.raises(ValueError, match="chemin dangereux"):
        backup.restore_backup(archive, project_root=projet)

    assert not (projet.parent / "evade.txt").exists()


def test_restauration_refuse_un_chemin_absolu(tmp_path, projet):
    archive = tmp_path / "backup_2026-01-02_120000.tar.gz"

    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("/etc/absolu.txt")
        info.size = 0
        tar.addfile(info, io.BytesIO(b""))

    with pytest.raises(ValueError, match="chemin dangereux"):
        backup.restore_backup(archive, project_root=projet)


def test_restauration_refuse_un_lien_symbolique(tmp_path, projet):
    """Un lien symbolique vers /etc/passwd est aussi une évasion possible."""
    archive = tmp_path / "backup_2026-01-03_120000.tar.gz"

    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("lien")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tar.addfile(info)

    with pytest.raises(ValueError, match="lien non autorisé"):
        backup.restore_backup(archive, project_root=projet)


def test_restauration_detecte_une_empreinte_invalide(tmp_path, projet):
    archive = _archive_falsifiee(
        tmp_path / "backup_2026-01-04_120000.tar.gz",
        taille=len(b"contenu reel"),
        empreinte="0" * 64,
    )

    with pytest.raises(ValueError, match="Archive corrompue"):
        backup.restore_backup(archive, project_root=projet)


def test_restauration_detecte_une_taille_incoherente(tmp_path, projet):
    archive = _archive_falsifiee(
        tmp_path / "backup_2026-01-05_120000.tar.gz", taille=999, empreinte="0" * 64
    )

    with pytest.raises(ValueError, match="taille différente"):
        backup.restore_backup(archive, project_root=projet)


def test_restauration_detecte_un_fichier_manquant(tmp_path, projet):
    """Un fichier annoncé au manifeste mais absent de l'archive = archive tronquée."""
    archive = tmp_path / "backup_2026-01-06_120000.tar.gz"
    manifeste = {
        "created_at": "2026-01-01T00:00:00",
        "environment": "test",
        "includes_vectors": False,
        "files": [{"path": "data/app.db", "size": 1, "sha256": "0" * 64}],
    }

    with tarfile.open(archive, "w:gz") as tar:
        blob = json.dumps(manifeste).encode("utf-8")
        info = tarfile.TarInfo(backup.MANIFEST_NAME)
        info.size = len(blob)
        tar.addfile(info, io.BytesIO(blob))

    with pytest.raises(ValueError, match="fichier absent"):
        backup.restore_backup(archive, project_root=projet)


# --- Configuration ---------------------------------------------------------

def test_sqlite_database_path_detecte_le_fichier(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATABASE_URL", f"sqlite:///{tmp_path / 'app.db'}")

    assert config.sqlite_database_path() == tmp_path / "app.db"


def test_sqlite_database_path_ignore_les_autres_moteurs(monkeypatch):
    monkeypatch.setattr(config, "DATABASE_URL", "postgresql://user@db:5432/app")

    assert config.sqlite_database_path() is None


def test_sqlite_database_path_ignore_une_base_en_memoire(monkeypatch):
    monkeypatch.setattr(config, "DATABASE_URL", "sqlite://")

    assert config.sqlite_database_path() is None
