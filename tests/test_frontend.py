"""Tests du frontend statique (frontend/index.html et frontend/app.js).

Ces tests ne remplacent PAS une revue visuelle : ils verrouillent les erreurs
qu'un refactor de balisage provoque **sans aucun signal visible** — un
identifiant renommé, un fichier oublié, une dépendance externe réintroduite.

C'est exactement la catégorie de régression qu'on ne découvre qu'en production,
devant un utilisateur, et qui donne l'impression que « le site est cassé ».
"""

import json
import re
from pathlib import Path

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"


def _lire(nom: str) -> str:
    return (FRONTEND / nom).read_text(encoding="utf-8")


# --- Cohérence entre le balisage et la logique ---------------------------

def test_tous_les_identifiants_utilises_par_le_js_existent_dans_le_html():
    """LE test qui protège ce refactor.

    Un `id` renommé dans le HTML casse l'application SANS erreur visible :
    `getElementById` renvoie simplement `null`. L'échec n'apparaît qu'au
    premier clic de l'utilisateur — ou jamais, si l'élément est secondaire.
    """
    html = _lire("index.html")
    js = _lire("app.js")

    ids_html = set(re.findall(r'\bid="([^"]+)"', html))
    ids_js = set(re.findall(r"(?:byId|getElementById)\(\s*'([^']+)'\s*\)", js))

    assert ids_js, "aucun identifiant détecté : le motif de recherche est faux"

    manquants = sorted(ids_js - ids_html)
    assert not manquants, f"identifiants absents du HTML : {manquants}"


# --- Absence de dépendance externe ---------------------------------------

def test_le_html_ne_charge_aucun_script_externe():
    """Aucun CDN : la page doit s'afficher hors ligne, et sans tiers."""
    html = _lire("index.html")

    externes = re.findall(r'<script[^>]+src="(https?://[^"]+)"', html)
    assert not externes, f"script(s) externe(s) réintroduit(s) : {externes}"

    # Et la feuille doit être la version COMPILÉE, servie localement.
    assert 'href="app.css"' in html


def test_le_html_charge_la_logique_locale():
    assert 'src="app.js"' in _lire("index.html")


# --- Direction et accessibilité ------------------------------------------

def test_le_document_declare_sa_langue_et_sa_direction():
    """L'état initial est l'arabe : le HTML doit être complet et lisible tel quel."""
    html = _lire("index.html")

    assert re.search(r'<html[^>]+lang="ar"', html)
    assert re.search(r'<html[^>]+dir="rtl"', html)


def test_chaque_champ_de_saisie_possede_un_libelle_accessible():
    """Un `placeholder` n'est PAS un libellé : il disparaît à la première frappe.

    Un lecteur d'écran n'a donc plus rien à annoncer pour ce champ.
    """
    champs = re.findall(r"<input\b[^>]*>", _lire("index.html"))

    assert champs, "aucun champ détecté : le motif de recherche est faux"

    sans_libelle = [champ for champ in champs if "aria-label=" not in champ]
    assert not sans_libelle, f"champ(s) sans aria-label : {sans_libelle}"


# --- Configuration du build ----------------------------------------------

def test_la_feuille_d_entree_declare_explicitement_ses_sources():
    """Tailwind doit scanner le balisage ET la logique : sinon les classes
    utilisées uniquement dans le JS disparaîtraient de la feuille compilée."""
    css = _lire("css/input.css")

    assert '@source "../index.html"' in css
    assert '@source "../app.js"' in css


def test_la_detection_automatique_de_sources_est_desactivee():
    """Sans `source(none)`, Tailwind scanne TOUT le dossier du projet et
    ramasse des chaînes qui ressemblent à des classes dans la documentation,
    les tests ou les notes.

    Le symptôme mesuré : le build local produisait 20,7 Ko là où le build
    Docker en produisait 13,3. Les deux ne devaient donc pas contenir les
    mêmes classes — une différence capable de faire fonctionner un style en
    développement puis de le faire disparaître en production.
    """
    css = _lire("css/input.css")

    assert '@import "tailwindcss" source(none);' in css


def test_le_paquet_declare_le_script_de_compilation():
    """Sans ce script, ni le poste de développement ni l'image Docker ne
    pourraient construire la feuille — et le frontend s'afficherait sans style."""
    paquet = json.loads(
        (FRONTEND.parent / "package.json").read_text(encoding="utf-8")
    )

    assert "build:css" in paquet["scripts"]
