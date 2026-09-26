"""Tests de scripts/schedule_backup.py (sauvegarde quotidienne automatique).

Le cœur du module est fait de fonctions **pures** qui construisent la
planification sans rien exécuter : ce sont elles qui sont testées ici. Aucun
test ne modifie le système de la machine (ni crontab, ni Planificateur de
tâches) : les appels système sont interceptés.

Trois propriétés sont critiques :

1. **Idempotence** — réinstaller la tâche ne doit pas créer de doublon, sinon
   deux sauvegardes tourneraient en parallèle et écriraient dans la même base ;
2. **Préservation** — les AUTRES tâches planifiées de l'utilisateur doivent
   survivre intactes à une installation comme à une désinstallation ;
3. **Absence d'effet de bord par défaut** — lancer le script sans option ne
   doit rien installer.
"""

import subprocess

import pytest
import schedule_backup as plan

from app import config

# --- Construction de la commande planifiée -------------------------------

def test_les_arguments_imposent_la_verification_et_le_journal():
    """Une sauvegarde PLANIFIÉE doit toujours être vérifiée et tracée.

    Personne ne regarde l'écran à 3 h du matin : sans `--verify`, une archive
    corrompue passerait inaperçue ; sans `--log`, un échec serait invisible.
    """
    arguments = plan.backup_arguments()

    assert "--verify" in arguments
    assert "--log" in arguments


def test_les_arguments_transmettent_le_dossier_et_la_retention(tmp_path):
    arguments = plan.backup_arguments(output_dir=tmp_path, keep=3)

    assert arguments[arguments.index("--output-dir") + 1] == str(tmp_path)
    assert arguments[arguments.index("--keep") + 1] == "3"


def test_commande_en_mode_host_utilise_des_chemins_absolus():
    """Le Planificateur de tâches Windows n'a pas de « dossier de démarrage ».

    Un chemin relatif comme « scripts/backup.py » échouerait donc : la tâche
    s'exécuterait depuis C:\\Windows\\System32.
    """
    commande = plan.backup_command("host")

    assert str(config.BASE_DIR / "scripts" / "backup.py") in commande
    assert "--verify" in commande


def test_commande_en_mode_docker_passe_par_le_conteneur():
    commande = plan.backup_command("docker")

    assert 'cd "' in commande
    assert "docker compose exec -T api" in commande
    assert "--log" in commande


# --- Validation des paramètres -------------------------------------------

def test_heure_valide_acceptee():
    assert plan.validate_time(3, 30) == (3, 30)


@pytest.mark.parametrize("heure,minute", [(24, 0), (-1, 0), (0, 60), (0, -1)])
def test_heure_invalide_refusee(heure, minute):
    """« Fail fast » : une heure impossible donne une tâche jamais exécutée."""
    with pytest.raises(ValueError):
        plan.validate_time(heure, minute)


def test_mode_inconnu_refuse():
    with pytest.raises(ValueError, match="mode invalide"):
        plan.validate_mode("kubernetes", platform="posix")


def test_mode_docker_refuse_sous_windows():
    """Le Planificateur de tâches n'exécute pas de shell : « cd ... && ... » échoue.

    Mieux vaut refuser clairement que créer une tâche qui échouera en silence.
    """
    with pytest.raises(ValueError, match="Windows"):
        plan.validate_mode("docker", platform="nt")


def test_mode_docker_accepte_sous_linux():
    plan.validate_mode("docker", platform="posix")


# --- Ligne de crontab ----------------------------------------------------

def test_ligne_de_crontab_contient_l_heure_la_commande_et_le_marqueur():
    ligne = plan.build_cron_line(
        project_dir="/srv/app", command="python backup.py", hour=3, minute=30
    )

    assert ligne.startswith("30 3 * * * ")
    assert 'cd "/srv/app"' in ligne
    assert "python backup.py" in ligne
    assert ligne.endswith(plan.CRON_MARKER)


# --- Idempotence et préservation du crontab ------------------------------

CRONTAB_EXISTANT = """\
# mes autres tâches
0 5 * * * /usr/bin/autre-sauvegarde.sh
30 3 * * * cd "/srv/app" && python backup.py # arabic-rag-backup
"""


def test_installer_remplace_la_ligne_existante_sans_creer_de_doublon():
    """IDEMPOTENCE : le test le plus important du module.

    Sans le marqueur, deux `--install` successifs créeraient deux tâches
    concurrentes — donc deux sauvegardes simultanées écrivant dans la même
    base. C'est exactement le genre de bug qui ne se voit qu'au pire moment.
    """
    ligne = plan.build_cron_line(
        project_dir="/srv/app", command="python backup.py", hour=4, minute=0
    )

    resultat = plan.crontab_with_schedule(CRONTAB_EXISTANT, ligne)

    assert resultat.count(plan.CRON_MARKER) == 1
    assert "0 4 * * * " in resultat
    assert "30 3 * * * " not in resultat  # l'ancienne ligne a disparu


def test_installer_preserve_les_autres_taches():
    ligne = plan.build_cron_line(hour=4, minute=0)

    resultat = plan.crontab_with_schedule(CRONTAB_EXISTANT, ligne)

    assert "0 5 * * * /usr/bin/autre-sauvegarde.sh" in resultat
    assert "# mes autres tâches" in resultat


def test_installer_sur_un_crontab_vide():
    resultat = plan.crontab_with_schedule("", plan.build_cron_line(hour=3, minute=0))

    assert resultat.count(plan.CRON_MARKER) == 1


def test_desinstaller_retire_seulement_notre_ligne():
    resultat = plan.crontab_without_schedule(CRONTAB_EXISTANT)

    assert plan.CRON_MARKER not in resultat
    assert "0 5 * * * /usr/bin/autre-sauvegarde.sh" in resultat


def test_desinstaller_un_crontab_deja_propre_ne_casse_rien():
    assert plan.crontab_without_schedule("") == ""


def test_le_crontab_produit_se_termine_par_un_saut_de_ligne():
    """`crontab` ignore une dernière ligne dépourvue de saut de ligne final."""
    assert plan.crontab_with_schedule("", "0 3 * * * x").endswith("\n")
    assert plan.crontab_without_schedule(CRONTAB_EXISTANT).endswith("\n")


# --- Tâche Windows -------------------------------------------------------

def test_argv_schtasks_est_une_liste_et_non_une_chaine():
    """Une LISTE garde chaque argument distinct : aucun piège de guillemets."""
    argv = plan.build_schtasks_create_argv("MaTache", "python backup.py", 3, 30)

    assert isinstance(argv, list)
    assert argv[0] == "schtasks"
    assert argv[argv.index("/TN") + 1] == "MaTache"
    assert argv[argv.index("/TR") + 1] == "python backup.py"


def test_argv_schtasks_planifie_quotidiennement_a_l_heure_demandee():
    argv = plan.build_schtasks_create_argv("MaTache", "cmd", 4, 5)

    assert argv[argv.index("/SC") + 1] == "DAILY"
    assert argv[argv.index("/ST") + 1] == "04:05"


def test_argv_schtasks_ecrase_la_tache_existante():
    """/F = idempotence côté Windows, équivalent du marqueur côté cron."""
    assert "/F" in plan.build_schtasks_create_argv("MaTache", "cmd", 3, 0)


def test_argv_schtasks_pour_supprimer_et_interroger():
    assert plan.build_schtasks_delete_argv("MaTache")[:2] == ["schtasks", "/Delete"]
    assert plan.build_schtasks_query_argv("MaTache")[:2] == ["schtasks", "/Query"]


# --- Aucun effet de bord sans demande explicite --------------------------

def test_le_plan_par_defaut_ne_modifie_rien(capsys, monkeypatch):
    """Lancer le script sans option AFFICHE, il n'installe pas.

    Principe de moindre surprise : modifier le système doit toujours être un
    geste explicite (`--install`).
    """

    def interdit(*args, **kwargs):
        raise AssertionError("aucune commande système ne doit être exécutée")

    monkeypatch.setattr(plan.subprocess, "run", interdit)

    code = plan.show_plan("host", 3, 0)

    assert code == 0
    assert "aucune modification effectuée" in capsys.readouterr().out


# --- Dialogue avec le planificateur (intercepté) -------------------------

def _faux_run(appels, crontab_initial=""):
    """Remplace `subprocess.run` : enregistre les appels sans rien exécuter."""
    def run(argv, **kwargs):
        appels.append({"argv": argv, "input": kwargs.get("input")})
        sortie = crontab_initial if "-l" in argv else ""
        return subprocess.CompletedProcess(argv, 0, stdout=sortie, stderr="")
    return run


def test_installer_ecrit_le_crontab_sous_linux(monkeypatch):
    """Vérifie le dialogue avec `crontab` sans jamais toucher au vrai crontab."""
    appels = []
    monkeypatch.setattr(plan.os, "name", "posix")
    monkeypatch.setattr(
        plan.subprocess, "run", _faux_run(appels, crontab_initial="0 5 * * * autre\n")
    )

    code = plan.install_schedule("host", 3, 30)

    assert code == 0
    ecritures = [appel for appel in appels if appel["input"] is not None]
    assert len(ecritures) == 1
    assert plan.CRON_MARKER in ecritures[0]["input"]
    assert "0 5 * * * autre" in ecritures[0]["input"]


def test_desinstaller_sans_tache_ne_modifie_pas_le_crontab(monkeypatch):
    appels = []
    monkeypatch.setattr(plan.os, "name", "posix")
    monkeypatch.setattr(plan.subprocess, "run", _faux_run(appels))

    assert plan.remove_schedule() == 1
    assert [appel for appel in appels if appel["input"] is not None] == []


def test_l_etat_signale_l_absence_de_tache(monkeypatch, capsys):
    appels = []
    monkeypatch.setattr(plan.os, "name", "posix")
    monkeypatch.setattr(plan.subprocess, "run", _faux_run(appels))

    assert plan.schedule_status() == 1
    assert "Aucune sauvegarde automatique" in capsys.readouterr().out


def test_l_etat_affiche_la_ligne_installee(monkeypatch, capsys):
    appels = []
    monkeypatch.setattr(plan.os, "name", "posix")
    monkeypatch.setattr(plan.subprocess, "run", _faux_run(appels, CRONTAB_EXISTANT))

    assert plan.schedule_status() == 0
    assert "30 3 * * *" in capsys.readouterr().out
