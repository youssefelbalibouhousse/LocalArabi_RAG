"""Tests de scripts/build_kb.py (câblage de l'ingestion, PDF et EPUB).

Ce que ces tests protègent, et que `test_epub.py` ne peut pas protéger : le
RACCORDEMENT. Un extracteur juste mais mal branché indexe un livre sans ses
métadonnées, ou ignore un dossier du corpus en silence — deux pannes qui ne
produisent aucune erreur, seulement des résultats faux.

Aucun test ne touche au vrai Ollama, à `chroma_db/` ni à `data/` : la
collection est le substitut en mémoire de `test_ingest`, et la session vient de
`conftest.py` (base SQLite en mémoire).
"""

import argparse
import zipfile

import build_kb
import pytest
from test_epub import ecrire_epub
from test_ingest import CollectionFactice

from app import config, ingest


def args(force: bool = False) -> argparse.Namespace:
    """Les options telles que `argparse` les fournit au script."""
    return argparse.Namespace(force=force, only=None, status=False)


# --- Collecte des documents -----------------------------------------------

@pytest.fixture(name="corpus")
def corpus_fixture(tmp_path, monkeypatch):
    """Un faux corpus : deux dossiers, deux formats, et du bruit.

    Les deux fichiers du second dossier portent des noms DIFFÉRENTS, et l'un des
    deux une extension en majuscules. Attention au piège : sur un système de
    fichiers insensible à la casse (Windows, macOS), `livre.epub` et
    `livre.EPUB` sont le MÊME fichier — les écrire tous les deux n'en crée qu'un,
    et un test qui les attend séparément échoue pour une raison sans rapport
    avec le code testé.
    """
    premier = tmp_path / "documents"
    second = tmp_path / "shamela"
    premier.mkdir()
    second.mkdir()

    (premier / "cours.pdf").write_bytes(b"%PDF-1.4 faux")
    (premier / "notes.txt").write_text("pas un document", encoding="utf-8")
    (second / "livre.epub").write_bytes(b"PK faux")
    (second / "traite.EPUB").write_bytes(b"PK faux aussi")

    monkeypatch.setattr(config, "CORPUS_DIRS", (premier, second))
    return premier, second


def test_collecte_les_deux_dossiers_et_les_deux_formats(corpus):
    trouves = build_kb.collecter_documents()

    assert sorted(chemin.name for chemin in trouves) == [
        "cours.pdf",
        "livre.epub",
        "traite.EPUB",
    ]


def test_ignore_les_fichiers_etrangers(corpus):
    trouves = build_kb.collecter_documents()

    assert all(chemin.suffix.lower() in build_kb.FORMATS for chemin in trouves)


def test_un_dossier_absent_n_empeche_pas_la_collecte(corpus, monkeypatch):
    premier, second = corpus
    monkeypatch.setattr(config, "CORPUS_DIRS", (premier, second / "inexistant"))

    assert [chemin.name for chemin in build_kb.collecter_documents()] == ["cours.pdf"]


def test_signale_les_noms_presents_dans_plusieurs_dossiers(corpus):
    """Deux homonymes se remplaceraient dans l'index, sans le moindre message."""
    trouves = build_kb.collecter_documents()

    assert build_kb.noms_en_double(trouves) == []

    homonyme = trouves[0].parent.parent / "shamela" / "cours.pdf"
    homonyme.write_bytes(b"%PDF-1.4 autre")

    assert build_kb.noms_en_double(build_kb.collecter_documents()) == ["cours.pdf"]


# --- Choix de l'extracteur ------------------------------------------------

def test_un_epub_est_extrait_avec_ses_metadonnees(tmp_path):
    chemin = ecrire_epub(tmp_path / "livre.epub", [("P1", "الصفحة: 4", "نص")])

    metadonnees, pages, note = build_kb.extraire(chemin)

    assert metadonnees["language"] == "ar"
    assert metadonnees["title"] == "كتاب الاختبار"
    assert pages == [(4, "نص")]
    assert note == ""


def test_l_extension_est_reconnue_sans_tenir_compte_de_la_casse(tmp_path):
    chemin = ecrire_epub(tmp_path / "LIVRE.EPUB", [("P1", "الصفحة: 1", "نص")])

    _, pages, _ = build_kb.extraire(chemin)

    assert pages == [(1, "نص")]


def test_les_pages_absentes_sont_signalees(tmp_path):
    chemin = ecrire_epub(
        tmp_path / "trous.epub",
        [("P1", "الصفحة: 1", "a"), ("P2", "الصفحة: 6", "b")],
    )

    _, _, note = build_kb.extraire(chemin)

    assert "4 page(s)" in note
    assert "2-5" in note


def test_un_format_inconnu_est_refuse(tmp_path):
    chemin = tmp_path / "document.docx"
    chemin.write_bytes(b"PK")

    with pytest.raises(ingest.IngestError, match="Format non pris en charge"):
        build_kb.extraire(chemin)


# --- Enrichissement des chunks --------------------------------------------

def test_les_metadonnees_sont_recopiees_sur_chaque_chunk():
    chunks = [{"text": "a", "page": 1}, {"text": "b", "page": 2}]

    enrichis = build_kb.enrichir(chunks, {"title": "Livre", "language": "ar"})

    assert [chunk["title"] for chunk in enrichis] == ["Livre", "Livre"]
    assert [chunk["page"] for chunk in enrichis] == [1, 2]
    assert [chunk["text"] for chunk in enrichis] == ["a", "b"]


def test_sans_metadonnees_les_chunks_sont_rendus_tels_quels():
    chunks = [{"text": "a", "page": 1}]

    assert build_kb.enrichir(chunks, {}) == chunks


def test_l_enrichissement_ne_modifie_pas_les_chunks_d_origine():
    """Le chunk d'origine est partagé : l'altérer fausserait tout appelant."""
    chunks = [{"text": "a", "page": 1}]

    build_kb.enrichir(chunks, {"title": "Livre"})

    assert chunks == [{"text": "a", "page": 1}]


# --- Chaîne complète : EPUB → index ---------------------------------------

def test_un_epub_va_jusqu_a_l_index_avec_ses_metadonnees(tmp_path, session):
    """Le raccordement de bout en bout : page imprimée ET métadonnées indexées."""
    chemin = ecrire_epub(
        tmp_path / "livre.epub",
        [("P1", "الجزء: 1 - الصفحة: 7", "أول سطر\nثاني سطر")],
    )
    collection = CollectionFactice(embedding_model=config.EMBEDDING_MODEL)

    resultat, note = build_kb.ingerer_un_document(session, collection, chemin, args())

    assert resultat.status == "added"
    assert resultat.chunks == 1
    assert note == ""

    meta = next(iter(collection.chunks.values()))[1]
    assert meta["source"] == "livre.epub"
    assert meta["page"] == 7
    assert meta["line_start"] == 1
    assert meta["line_end"] == 2
    assert meta["language"] == "ar"
    assert meta["title"] == "كتاب الاختبار"


def test_un_epub_inchange_n_ecrit_rien_la_seconde_fois(tmp_path, session):
    """L'incrémentalité vaut aussi pour les EPUB, sans réextraction."""
    chemin = ecrire_epub(tmp_path / "livre.epub", [("P1", "الصفحة: 1", "نص")])
    collection = CollectionFactice(embedding_model=config.EMBEDDING_MODEL)

    premier, _ = build_kb.ingerer_un_document(session, collection, chemin, args())
    ecrits = len(collection.tailles_des_lots)

    second, _ = build_kb.ingerer_un_document(session, collection, chemin, args())

    assert premier.status == "added"
    assert second.status == "unchanged"
    assert len(collection.tailles_des_lots) == ecrits


def test_un_epub_illisible_remonte_une_erreur_claire(tmp_path, session):
    """.epub n'est pas une garantie de contenu : le message doit le dire."""
    chemin = tmp_path / "menteur.epub"
    chemin.write_bytes(b"rien a voir")
    collection = CollectionFactice(embedding_model=config.EMBEDDING_MODEL)

    with pytest.raises(Exception, match="archive ZIP"):
        build_kb.ingerer_un_document(session, collection, chemin, args())


def test_un_epub_sans_texte_est_ingere_vide(tmp_path, session):
    """Un EPUB vide laisse le registre propre : il ne sera pas « repris » sans fin."""
    chemin = tmp_path / "vide.epub"
    with zipfile.ZipFile(chemin, "w") as archive:
        archive.writestr(
            "META-INF/container.xml",
            '<?xml version="1.0"?>\n'
            '<container version="1.0"><rootfiles><rootfile '
            'full-path="OEBPS/content.opf"/></rootfiles></container>',
        )
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
        archive.writestr(
            "OEBPS/xhtml/P1.xhtml",
            '<?xml version="1.0" encoding="utf-8"?>\n'
            '<html xmlns="http://www.w3.org/1999/xhtml"><body>'
            '<div id="book-container"></div>'
            '<div class="center">الصفحة: 1</div>'
            "</body></html>",
        )
    collection = CollectionFactice(embedding_model=config.EMBEDDING_MODEL)

    resultat, _ = build_kb.ingerer_un_document(session, collection, chemin, args())

    assert resultat.status == "empty"
    assert resultat.chunks == 0
    assert collection.chunks == {}
