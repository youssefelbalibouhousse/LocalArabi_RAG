"""Tests de app/epub.py (extraction des EPUB de shamela.ws).

Quatre propriétés sont critiques. Chacune a d'abord été MESURÉE sur un vrai
fichier de shamela.ws (`12445.epub`, non versionné) avant d'être écrite ici :

1. **L'ordre de lecture vient du `spine`** — dans l'archive réelle, les
   fichiers sont rangés dans le désordre (`P117.xhtml` avant `P11.xhtml`) ;
2. **La page est la page IMPRIMÉE** — `P80.xhtml` porte la page 81 : le nom du
   fichier ment, et s'y fier produirait une citation introuvable ;
3. **Une page découpée en plusieurs fichiers est refusionnée** — 17 pages sur
   154 sont dans ce cas dans l'archive réelle ;
4. **Le texte est propre** — les marqueurs de corruption du PDF (`ا،عتاال`,
   `الشرا`) y sont totalement absents.

Les tests construisent leurs propres EPUB : aucun fichier de shamela.ws n'est
versionné (éditions sous droits), donc la suite ne dépend d'aucune donnée
externe. Un dernier test, ignoré si le fichier est absent, vérifie les
invariants sur l'archive réelle.
"""

import zipfile
from itertools import pairwise
from pathlib import Path

import build_kb
import pytest

from app import config, epub

# --- Fabrique d'EPUB de test ----------------------------------------------

CONTENEUR = (
    '<?xml version="1.0"?>\n'
    '<container version="1.0" '
    'xmlns="urn:oasis:names:tc:opendocument:xmlns:container">\n'
    '  <rootfiles>\n'
    '    <rootfile full-path="OEBPS/content.opf" '
    'media-type="application/oebps-package+xml"/>\n'
    "  </rootfiles>\n"
    "</container>\n"
)


def _xhtml(titre: str, etiquette: str | None, texte: str) -> str:
    """Un document XHTML de la forme produite par shamela.ws, en plus petit.

    L'étiquette de page est un bloc VOISIN du texte, comme dans la réalité :
    c'est cette structure qui permet de lire le texte sans l'étiquette.
    """
    corps = "<br />".join(texte.split("\n"))
    bloc = f'<div class="center">{etiquette}</div>' if etiquette else ""
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" '
        '"http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">\n'
        '<html xml:lang="ar" lang="ar" dir="rtl" '
        'xmlns="http://www.w3.org/1999/xhtml">\n'
        f"<head><title>{titre}</title></head>\n"
        '<body class="rtl">\n'
        '<div dir="rtl" id="book-container">\n'
        f"{corps}\n"
        "</div>\n"
        f"{bloc}\n"
        "</body>\n</html>\n"
    )


def _opf(titre: str, auteur: str, langue: str, ordre: list[str]) -> str:
    """Le paquet OPF : métadonnées, manifeste et spine dans l'ordre voulu."""
    items = "\n".join(
        f'<item id="{identifiant}" href="xhtml/{identifiant}.xhtml" '
        'media-type="application/xhtml+xml"/>'
        for identifiant in ordre
    )
    refs = "\n".join(f'<itemref idref="{identifiant}"/>' for identifiant in ordre)
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<package xmlns="http://www.idpf.org/2007/opf" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" version="3.0" '
        'unique-identifier="BookID" dir="rtl">\n'
        '  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:opf="http://www.idpf.org/2007/opf">\n'
        f"    <dc:title>{titre}</dc:title>\n"
        f'    <dc:creator opf:role="aut">{auteur}</dc:creator>\n'
        f"    <dc:language>{langue}</dc:language>\n"
        "    <dc:publisher>shamela.ws</dc:publisher>\n"
        '    <dc:identifier id="BookID">urn:uuid:test</dc:identifier>\n'
        "  </metadata>\n"
        f"  <manifest>\n{items}\n  </manifest>\n"
        f"  <spine>\n{refs}\n  </spine>\n"
        "</package>\n"
    )


def ecrire_epub(
    chemin: Path,
    pages,
    *,
    titre: str = "كتاب الاختبار",
    auteur: str = "المؤلف",
    langue: str = "ar",
    ordre: list[str] | None = None,
    ordre_du_zip: list[str] | None = None,
) -> Path:
    """Écrit un EPUB minimal et renvoie son chemin.

    `pages` est une liste de triplets `(identifiant, étiquette, texte)`. Le
    `spine` suit `ordre` (par défaut l'ordre de `pages`), tandis que l'archive
    ZIP est écrite dans `ordre_du_zip`. Séparer les deux permet de vérifier que
    le lecteur suit le spine et non le hasard du rangement de l'archive.
    """
    par_identifiant = {
        identifiant: (etiquette, texte)
        for identifiant, etiquette, texte in pages
    }
    identifiants = list(par_identifiant)
    ordre = ordre if ordre is not None else identifiants
    ordre_du_zip = ordre_du_zip if ordre_du_zip is not None else identifiants

    with zipfile.ZipFile(chemin, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip")
        archive.writestr("META-INF/container.xml", CONTENEUR)
        archive.writestr("OEBPS/content.opf", _opf(titre, auteur, langue, ordre))
        for identifiant in ordre_du_zip:
            etiquette, texte = par_identifiant[identifiant]
            archive.writestr(
                f"OEBPS/xhtml/{identifiant}.xhtml", _xhtml(titre, etiquette, texte)
            )
    return chemin


@pytest.fixture(name="simple")
def simple_fixture(tmp_path) -> Path:
    """Un EPUB de deux pages d'une ligne chacune."""
    return ecrire_epub(
        tmp_path / "simple.epub",
        [
            ("P1", "الصفحة: 1", "أول سطر"),
            ("P2", "الصفحة: 2", "ثاني سطر"),
        ],
    )


# --- Lecture nominale -----------------------------------------------------

def test_lit_les_pages_et_leur_texte(simple):
    metadonnees, pages = epub.lire_epub(simple)

    assert [page.numero for page in pages] == [1, 2]
    assert pages[0].texte == "أول سطر"
    assert metadonnees.langue == "ar"


def test_les_metadonnees_sont_lues(simple):
    metadonnees, _ = epub.lire_epub(simple)

    assert metadonnees.titre == "كتاب الاختبار"
    assert metadonnees.auteur == "المؤلف"
    assert metadonnees.editeur == "shamela.ws"


def test_pour_index_n_emet_que_les_champs_presents(tmp_path):
    """ChromaDB refuse une valeur None : un champ absent doit être omis."""
    chemin = ecrire_epub(
        tmp_path / "vide.epub",
        [("P1", "الصفحة: 1", "نص")],
        titre="",
        auteur="",
        langue="ar",
    )
    metadonnees, _ = epub.lire_epub(chemin)

    index = metadonnees.pour_index()
    assert index["language"] == "ar"
    assert "title" not in index
    assert "author" not in index


def test_un_epub_sans_metadonnees_ne_fait_pas_echouer_la_lecture(tmp_path):
    chemin = tmp_path / "sans-meta.epub"
    with zipfile.ZipFile(chemin, "w") as archive:
        archive.writestr("META-INF/container.xml", CONTENEUR)
        archive.writestr(
            "OEBPS/content.opf",
            '<?xml version="1.0" encoding="utf-8"?>\n'
            '<package xmlns="http://www.idpf.org/2007/opf" version="3.0">\n'
            "  <metadata/>\n"
            '  <manifest><item id="P1" href="xhtml/P1.xhtml" '
            'media-type="application/xhtml+xml"/></manifest>\n'
            '  <spine><itemref idref="P1"/></spine>\n'
            "</package>\n",
        )
        archive.writestr("OEBPS/xhtml/P1.xhtml", _xhtml("t", "الصفحة: 3", "نص"))

    metadonnees, pages = epub.lire_epub(chemin)

    assert metadonnees.pour_index() == {}
    assert [page.numero for page in pages] == [3]


# --- Numéro de page -------------------------------------------------------

def test_la_page_est_l_etiquette_pas_le_nom_du_fichier(tmp_path):
    """Sur l'archive réelle, P80.xhtml porte la page 81 : le nom ment."""
    chemin = ecrire_epub(
        tmp_path / "decalage.epub",
        [("P80", "الجزء: 1 - الصفحة: 81", "نص الصفحة")],
    )

    _, pages = epub.lire_epub(chemin)

    assert pages[0].numero == 81
    assert pages[0].etiquetee is True


def test_les_chiffres_arabo_indiens_sont_reconnus(tmp_path):
    """Selon l'édition, le numéro s'écrit ٨١ plutôt que 81."""
    chemin = ecrire_epub(tmp_path / "indiens.epub", [("P1", "الصفحة: ٨١", "نص")])

    _, pages = epub.lire_epub(chemin)

    assert pages[0].numero == 81


def test_sans_etiquette_la_page_retombe_sur_sa_position(tmp_path):
    chemin = ecrire_epub(
        tmp_path / "sans-etiquette.epub",
        [("P1", None, "un"), ("P2", None, "deux"), ("P3", None, "trois")],
    )

    _, pages = epub.lire_epub(chemin)

    assert [page.numero for page in pages] == [1, 2, 3]
    assert all(page.etiquetee is False for page in pages)


def test_l_etiquette_ne_pollue_pas_le_texte(tmp_path):
    chemin = ecrire_epub(
        tmp_path / "etiquette.epub",
        [("P1", "الجزء: 1 - الصفحة: 5", "السطر الأول")],
    )

    _, pages = epub.lire_epub(chemin)

    assert pages[0].texte == "السطر الأول"
    assert "الصفحة" not in pages[0].texte


# --- Ordre de lecture et fusion -------------------------------------------

def test_l_ordre_de_lecture_suit_le_spine_pas_l_archive(tmp_path):
    """Dans l'archive réelle, P117 est rangé avant P11 : l'ordre n'est pas fiable."""
    chemin = ecrire_epub(
        tmp_path / "desordre.epub",
        [
            ("P1", "الصفحة: 1", "premier"),
            ("P2", "الصفحة: 2", "deuxieme"),
            ("P3", "الصفحة: 3", "troisieme"),
        ],
        ordre_du_zip=["P3", "P1", "P2"],
    )

    _, pages = epub.lire_epub(chemin)

    assert [page.texte for page in pages] == ["premier", "deuxieme", "troisieme"]
    assert [page.numero for page in pages] == [1, 2, 3]


def test_deux_fichiers_de_la_meme_page_sont_refusionnes(tmp_path):
    """Sans fusion, `(source, page)` ne désignerait pas un endroit unique."""
    chemin = ecrire_epub(
        tmp_path / "fusion.epub",
        [
            ("P1", "الصفحة: 34", "début de la page"),
            ("P2", "الصفحة: 34", "suite de la page"),
        ],
    )

    _, pages = epub.lire_epub(chemin)

    assert len(pages) == 1
    assert pages[0].numero == 34
    assert pages[0].parties == 2
    assert pages[0].texte.split("\n") == ["début de la page", "suite de la page"]


def test_la_fusion_rend_les_numeros_de_page_strictement_croissants(tmp_path):
    """Invariant dont dépend la citation : une page = un endroit."""
    chemin = ecrire_epub(
        tmp_path / "croissant.epub",
        [
            ("P1", "الصفحة: 10", "a"),
            ("P2", "الصفحة: 10", "b"),
            ("P3", "الصفحة: 11", "c"),
            ("P4", "الصفحة: 11", "d"),
            ("P5", "الصفحة: 11", "e"),
            ("P6", "الصفحة: 12", "f"),
        ],
    )

    _, pages = epub.lire_epub(chemin)

    numeros = [page.numero for page in pages]
    assert numeros == [10, 11, 12]
    assert all(b > a for a, b in pairwise(numeros))
    assert [page.parties for page in pages] == [2, 3, 1]


def test_deux_pages_differentes_ne_sont_pas_fusionnees(tmp_path):
    chemin = ecrire_epub(
        tmp_path / "distinctes.epub",
        [("P1", "الصفحة: 1", "un"), ("P2", "الصفحة: 2", "deux")],
    )

    _, pages = epub.lire_epub(chemin)

    assert len(pages) == 2


# --- Mise en forme du texte -----------------------------------------------

def test_les_br_deviennent_des_lignes(tmp_path):
    """Les numéros de ligne d'une citation reposent sur ces coupures."""
    chemin = ecrire_epub(
        tmp_path / "lignes.epub",
        [("P1", "الصفحة: 1", "ligne un\nligne deux\nligne trois")],
    )

    _, pages = epub.lire_epub(chemin)

    assert pages[0].texte.split("\n") == ["ligne un", "ligne deux", "ligne trois"]


def test_les_paragraphes_produisent_aussi_des_lignes(tmp_path):
    """Un EPUB qui découpe en <p> ne doit pas donner une seule ligne géante."""
    chemin = tmp_path / "paragraphes.epub"
    document = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<html xmlns="http://www.w3.org/1999/xhtml"><body>'
        '<div id="book-container"><p>premier</p><p>second</p></div>'
        '<div class="center">الصفحة: 1</div>'
        "</body></html>"
    )
    with zipfile.ZipFile(chemin, "w") as archive:
        archive.writestr("META-INF/container.xml", CONTENEUR)
        archive.writestr(
            "OEBPS/content.opf", _opf("t", "a", "ar", ["P1"])
        )
        archive.writestr("OEBPS/xhtml/P1.xhtml", document)

    _, pages = epub.lire_epub(chemin)

    assert pages[0].texte.split("\n") == ["premier", "second"]


def test_le_texte_hors_du_conteneur_est_ignore(tmp_path):
    """Chez Shamela, l'étiquette est un bloc voisin : elle ne doit pas entrer."""
    chemin = tmp_path / "voisin.epub"
    document = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<html xmlns="http://www.w3.org/1999/xhtml"><body>'
        '<div id="book-container">texte retenu</div>'
        '<div class="autre">texte écarté</div>'
        '<div class="center">الصفحة: 7</div>'
        "</body></html>"
    )
    with zipfile.ZipFile(chemin, "w") as archive:
        archive.writestr("META-INF/container.xml", CONTENEUR)
        archive.writestr("OEBPS/content.opf", _opf("t", "a", "ar", ["P1"]))
        archive.writestr("OEBPS/xhtml/P1.xhtml", document)

    _, pages = epub.lire_epub(chemin)

    assert pages[0].texte == "texte retenu"
    assert pages[0].numero == 7


def test_les_entites_html_nommees_sont_resolues(tmp_path):
    """`&nbsp;` n'est pas une entité XML : ElementTree échouerait sans ce travail."""
    chemin = ecrire_epub(
        tmp_path / "entites.epub",
        [("P1", "الصفحة: 1", "un&nbsp;mot &amp; un autre &laquo;sens&raquo;")],
    )

    _, pages = epub.lire_epub(chemin)

    assert pages[0].texte == "un mot & un autre «sens»"


def test_les_espaces_insecables_sont_normalises(tmp_path):
    """U+00A0 ressemble à un espace sans en être : il casserait les comparaisons."""
    chemin = ecrire_epub(
        tmp_path / "nbsp.epub",
        [("P1", "الصفحة: 1", "قبل&nbsp;بعد")],
    )

    _, pages = epub.lire_epub(chemin)

    assert "\u00a0" not in pages[0].texte
    assert pages[0].texte == "قبل بعد"


def test_les_notes_de_bas_de_page_restent_dans_le_texte(tmp_path):
    """Les notes en clair sont l'un des apports de l'EPUB sur le PDF océrisé."""
    chemin = ecrire_epub(
        tmp_path / "notes.epub",
        [("P1", "الصفحة: 1", "قال عثمان (1) وعلي (2) رضي الله عنهما")],
    )

    _, pages = epub.lire_epub(chemin)

    assert "(1)" in pages[0].texte
    assert "(2)" in pages[0].texte


def test_une_page_sans_texte_est_ecartee(tmp_path):
    """Un texte vide ne doit pas devenir un chunk : ChromaDB le refuse."""
    chemin = ecrire_epub(
        tmp_path / "vide.epub",
        [("P1", "الصفحة: 1", "du texte"), ("P2", "الصفحة: 2", "")],
    )

    _, pages = epub.lire_epub(chemin)

    assert [page.numero for page in pages] == [1]


# --- Diagnostic -----------------------------------------------------------

def test_pages_absentes_signale_les_trous(tmp_path):
    """Le PDF « scanne » 6, 30-32 : l'électronique doit pouvoir le dire."""
    chemin = ecrire_epub(
        tmp_path / "trous.epub",
        [
            ("P1", "الصفحة: 1", "a"),
            ("P2", "الصفحة: 2", "b"),
            ("P3", "الصفحة: 5", "c"),
            ("P4", "الصفحة: 6", "d"),
        ],
    )

    _, pages = epub.lire_epub(chemin)

    assert epub.pages_absentes(pages) == [3, 4]


def test_pages_absentes_sans_etiquette_ne_signale_rien(tmp_path):
    """Sans étiquette, `numero` est une position : parler de trou n'aurait pas de sens."""
    chemin = ecrire_epub(
        tmp_path / "positions.epub",
        [("P1", None, "a"), ("P2", None, "b")],
    )

    _, pages = epub.lire_epub(chemin)

    assert epub.pages_absentes(pages) == []


@pytest.mark.parametrize(
    ("nombres", "attendu"),
    [
        ([], ""),
        ([6], "6"),
        ([3, 4, 5], "3-5"),
        ([6, 30, 31, 32, 137, 138], "6, 30-32, 137-138"),
        ([2, 1, 2, 3], "1-3"),
    ],
)
def test_formater_plages(nombres, attendu):
    assert epub.formater_plages(nombres) == attendu


# --- EPUB invalides -------------------------------------------------------

def test_un_fichier_qui_n_est_pas_une_archive_est_refuse(tmp_path):
    chemin = tmp_path / "faux.epub"
    chemin.write_text("ceci n'est pas un zip", encoding="utf-8")

    with pytest.raises(epub.EpubError, match="archive ZIP"):
        epub.lire_epub(chemin)


def test_un_conteneur_sans_chemin_d_opf_est_refuse(tmp_path):
    chemin = tmp_path / "sans-opf.epub"
    with zipfile.ZipFile(chemin, "w") as archive:
        archive.writestr(
            "META-INF/container.xml",
            '<?xml version="1.0"?>\n<container version="1.0"><rootfiles/></container>',
        )

    with pytest.raises(epub.EpubError, match="aucun fichier OPF"):
        epub.lire_epub(chemin)


def test_un_spine_qui_reference_un_inconnu_est_refuse(tmp_path):
    """Mieux vaut échouer que d'indexer un livre amputé sans le dire."""
    chemin = tmp_path / "spine-casse.epub"
    with zipfile.ZipFile(chemin, "w") as archive:
        archive.writestr("META-INF/container.xml", CONTENEUR)
        archive.writestr(
            "OEBPS/content.opf",
            '<?xml version="1.0" encoding="utf-8"?>\n'
            '<package xmlns="http://www.idpf.org/2007/opf" version="3.0">\n'
            "  <metadata/>\n"
            '  <manifest><item id="P1" href="xhtml/P1.xhtml" '
            'media-type="application/xhtml+xml"/></manifest>\n'
            '  <spine><itemref idref="P9"/></spine>\n'
            "</package>\n",
        )

    with pytest.raises(epub.EpubError, match="P9"):
        epub.lire_epub(chemin)


def test_un_document_declare_mais_absent_est_refuse(tmp_path):
    chemin = tmp_path / "manquant.epub"
    with zipfile.ZipFile(chemin, "w") as archive:
        archive.writestr("META-INF/container.xml", CONTENEUR)
        archive.writestr("OEBPS/content.opf", _opf("t", "a", "ar", ["P1"]))

    with pytest.raises(epub.EpubError, match="absent"):
        epub.lire_epub(chemin)


def test_un_xhtml_mal_forme_est_refuse(tmp_path):
    chemin = tmp_path / "malforme.epub"
    with zipfile.ZipFile(chemin, "w") as archive:
        archive.writestr("META-INF/container.xml", CONTENEUR)
        archive.writestr("OEBPS/content.opf", _opf("t", "a", "ar", ["P1"]))
        archive.writestr("OEBPS/xhtml/P1.xhtml", "<html><body>oublié de fermer")

    with pytest.raises(epub.EpubError, match="XML illisible"):
        epub.lire_epub(chemin)


# --- Intégration avec la chaîne d'ingestion -------------------------------

def test_le_texte_produit_la_meme_forme_que_les_pdf(simple):
    """`chunk_pages` doit pouvoir traiter les pages sans savoir d'où elles viennent."""
    _, pages = epub.lire_epub(simple)

    chunks = build_kb.chunk_pages([(page.numero, page.texte) for page in pages])

    assert [sorted(chunk) for chunk in chunks] == [
        ["line_end", "line_start", "page", "text"],
        ["line_end", "line_start", "page", "text"],
    ]
    assert [chunk["page"] for chunk in chunks] == [1, 2]


# --- L'archive réelle, quand elle est présente ----------------------------

REEL = config.SHAMELA_DIR / "12445.epub"


@pytest.mark.skipif(not REEL.exists(), reason="EPUB Shamela non téléchargé")
def test_l_archive_reelle_respecte_les_invariants():
    """Vérifie sur le vrai fichier ce que les tests ci-dessus simulent.

    Ignoré quand le fichier est absent : il n'est pas versionné (droits
    d'éditeur), donc la suite reste exécutable partout.
    """
    metadonnees, pages = epub.lire_epub(REEL)

    assert metadonnees.titre
    assert metadonnees.auteur
    assert pages

    numeros = [page.numero for page in pages]
    assert all(b > a for a, b in pairwise(numeros)), "pages non uniques"
    assert all(page.etiquetee for page in pages), "étiquettes manquantes"
    assert sum(page.parties for page in pages) >= len(pages)

    tout = "\n".join(page.texte for page in pages)
    assert tout.count("،عتا") == 0, "texte corrompu : ce n'est pas l'EPUB attendu"
    assert "\u00a0" not in tout, "espaces insécables non normalisées"
