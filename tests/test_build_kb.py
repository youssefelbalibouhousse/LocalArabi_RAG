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
import sys
import zipfile

import build_kb
import pytest
from sqlmodel import Session
from test_epub import ecrire_epub
from test_ingest import CollectionFactice

from app import config, epub, ingest


def args(
    force: bool = False, retries: int = 1, continue_on_error: bool = False
) -> argparse.Namespace:
    """Les options telles que `argparse` les fournit au script."""
    return argparse.Namespace(
        force=force,
        only=None,
        forget=None,
        status=False,
        retries=retries,
        continue_on_error=continue_on_error,
    )


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


def test_un_epub_au_modele_divergent_est_re_extrait(tmp_path, session):
    """Le court-circuit AVANT extraction ne doit pas sauter un document dont les
    vecteurs ne sont plus justifiés : sans cela, un changement de modèle
    laisserait des vecteurs inconciliables en place, sans le moindre signal."""
    chemin = ecrire_epub(tmp_path / "livre.epub", [("P1", "الصفحة: 1", "نص")])
    collection = CollectionFactice(embedding_model=config.EMBEDDING_MODEL)

    premier, _ = build_kb.ingerer_un_document(session, collection, chemin, args())
    assert premier.status == "added"

    ligne = ingest.registre_pour(session, "livre.epub")
    ligne.embedding_model = "qwen3-embedding:0.6b"
    session.add(ligne)
    session.commit()
    collection.tailles_des_lots.clear()

    second, _ = build_kb.ingerer_un_document(session, collection, chemin, args())

    assert second.status == "updated"
    assert collection.tailles_des_lots == [1]


# --- Retrait d'un document du corpus --------------------------------------

@pytest.fixture(name="index")
def index_fixture(monkeypatch, engine):
    """`build_kb` branché sur une base EN MÉMOIRE et une collection factice.

    Le script travaille sur l'`engine` qu'il importe de `app.database` : sans
    cette substitution, un test écrirait dans la VRAIE base `data/app.db`.
    """
    collection = CollectionFactice(embedding_model=config.EMBEDDING_MODEL)
    monkeypatch.setattr(build_kb, "engine", engine)
    monkeypatch.setattr(build_kb.rag, "get_collection", lambda: collection)
    return collection


def indexer(collection, source: str, morceaux: int = 1) -> None:
    """Inscrit un document au registre et l'écrit dans la collection factice."""
    chunks = [
        {"text": f"texte {i}", "page": 1, "line_start": i + 1, "line_end": i + 1}
        for i in range(morceaux)
    ]
    with Session(build_kb.engine) as session:
        ingest.ingest_document(
            session, collection, source=source, fingerprint="v1", chunks=chunks
        )


def signaler_modele_divergent(source: str, modele: str = "qwen3-embedding:0.6b") -> None:
    """Réécrit le modèle d'embedding inscrit au registre pour un document.

    Reproduit l'état que laisse un passage d'essai lancé avec un autre
    `EMBEDDING_MODEL` : la ligne du registre raconte un modèle que l'INDEX n'a
    jamais utilisé — alors que les vecteurs, eux, n'ont pas bougé.
    """
    with Session(build_kb.engine) as session:
        ligne = ingest.registre_pour(session, source)
        ligne.embedding_model = modele
        session.add(ligne)
        session.commit()


def test_forget_retire_le_document_de_l_index_et_du_registre(index, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CORPUS_DIRS", (tmp_path,))
    indexer(index, "cours.pdf", morceaux=2)

    code = build_kb.oublier_un_document("cours.pdf")

    assert code == 0
    assert index.chunks == {}
    with Session(build_kb.engine) as session:
        assert ingest.registre_pour(session, "cours.pdf") is None


def test_forget_ne_touche_pas_aux_autres_documents(index, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CORPUS_DIRS", (tmp_path,))
    indexer(index, "cours.pdf")
    indexer(index, "autre.pdf")

    build_kb.oublier_un_document("cours.pdf")

    with Session(build_kb.engine) as session:
        assert ingest.registre_pour(session, "autre.pdf") is not None
    assert index.chunks != {}


def test_forget_refuse_un_document_absent_du_registre(index, capsys):
    assert build_kb.oublier_un_document("jamais-vu.pdf") == 1
    assert "pas au registre" in capsys.readouterr().out


def test_forget_avertit_si_le_fichier_est_encore_dans_le_corpus(
    index, tmp_path, monkeypatch, capsys
):
    """Sans avertissement, le document reviendrait au passage suivant et le
    retrait semblerait n'avoir servi à rien."""
    (tmp_path / "cours.pdf").write_bytes(b"%PDF-1.4 faux")
    monkeypatch.setattr(config, "CORPUS_DIRS", (tmp_path,))
    indexer(index, "cours.pdf")

    build_kb.oublier_un_document("cours.pdf")

    assert "TOUJOURS dans le corpus" in capsys.readouterr().out


def test_forget_ne_dit_rien_quand_le_fichier_a_disparu(
    index, tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(config, "CORPUS_DIRS", (tmp_path,))
    indexer(index, "cours.pdf")

    build_kb.oublier_un_document("cours.pdf")

    assert "TOUJOURS" not in capsys.readouterr().out


def test_forget_fonctionne_quand_le_corpus_est_vide(index, tmp_path, monkeypatch):
    """Retirer le DERNIER document doit rester possible : c'est justement le
    moment où l'on veut nettoyer un corpus qu'on vient de vider."""
    monkeypatch.setattr(config, "CORPUS_DIRS", (tmp_path,))  # dossier vide
    monkeypatch.setattr(build_kb, "create_db_and_tables", lambda: None)
    monkeypatch.setattr(sys, "argv", ["build_kb.py", "--forget", "cours.pdf"])
    indexer(index, "cours.pdf")

    assert build_kb.main() == 0
    assert index.chunks == {}


def test_status_reste_possible_corpus_vide(index, tmp_path, monkeypatch, capsys):
    """C'est le moment où l'on a le plus besoin de voir les orphelins."""
    monkeypatch.setattr(config, "CORPUS_DIRS", (tmp_path,))
    monkeypatch.setattr(build_kb, "create_db_and_tables", lambda: None)
    monkeypatch.setattr(sys, "argv", ["build_kb.py", "--status"])
    indexer(index, "cours.pdf")

    assert build_kb.main() == 0
    sortie = capsys.readouterr().out
    assert "cours.pdf" in sortie
    assert "orphelin" in sortie


def test_status_ne_signale_pas_d_orphelin_quand_tout_est_la(
    index, tmp_path, monkeypatch, capsys
):
    (tmp_path / "cours.pdf").write_bytes(b"%PDF-1.4 faux")
    monkeypatch.setattr(config, "CORPUS_DIRS", (tmp_path,))
    monkeypatch.setattr(build_kb, "create_db_and_tables", lambda: None)
    monkeypatch.setattr(sys, "argv", ["build_kb.py", "--status"])
    indexer(index, "cours.pdf")

    assert build_kb.main() == 0
    assert "orphelin" not in capsys.readouterr().out


def test_status_signale_une_ligne_de_registre_incoherente(
    index, tmp_path, monkeypatch, capsys
):
    """Découvert sur le corpus réel : un essai avec un autre modèle avait laissé
    le registre annonçant `qwen3-embedding:0.6b` alors que l'index était en
    `bge-m3`. `--status` affichait « concordent » — il rassurait à tort."""
    (tmp_path / "cours.pdf").write_bytes(b"%PDF-1.4 faux")
    monkeypatch.setattr(config, "CORPUS_DIRS", (tmp_path,))
    monkeypatch.setattr(build_kb, "create_db_and_tables", lambda: None)
    monkeypatch.setattr(sys, "argv", ["build_kb.py", "--status"])
    indexer(index, "cours.pdf")
    signaler_modele_divergent("cours.pdf")

    assert build_kb.main() == 0

    sortie = capsys.readouterr().out
    # Le modèle annoncé en tête est celui de l'INDEX…
    assert f"modèle d'embedding : {config.EMBEDDING_MODEL}" in sortie
    # …et la ligne du registre dit le sien, pour que l'écart soit visible.
    assert "qwen3-embedding:0.6b" in sortie
    assert "modèle divergent" in sortie
    assert "concordent" not in sortie


def test_status_reste_muet_quand_tout_concorde(index, tmp_path, monkeypatch, capsys):
    """Le bruit tue le signal : un état sain ne doit rien signaler du tout."""
    (tmp_path / "cours.pdf").write_bytes(b"%PDF-1.4 faux")
    monkeypatch.setattr(config, "CORPUS_DIRS", (tmp_path,))
    monkeypatch.setattr(build_kb, "create_db_and_tables", lambda: None)
    monkeypatch.setattr(sys, "argv", ["build_kb.py", "--status"])
    indexer(index, "cours.pdf")

    assert build_kb.main() == 0

    sortie = capsys.readouterr().out
    assert "concordent" in sortie
    assert "⚠️" not in sortie


# --- Tenue d'un long passage ----------------------------------------------

def preparer_passage(monkeypatch, tmp_path, *options: str) -> None:
    """Branche `main()` sur un faux corpus, sans base réelle ni Ollama."""
    monkeypatch.setattr(config, "CORPUS_DIRS", (tmp_path,))
    monkeypatch.setattr(build_kb, "create_db_and_tables", lambda: None)
    monkeypatch.setattr(build_kb, "check_ollama", lambda: True)
    monkeypatch.setattr(build_kb, "PAUSE_REPRISE_S", 0)  # pas d'attente en test
    monkeypatch.setattr(sys, "argv", ["build_kb.py", *options])


def test_un_document_qui_passe_du_premier_coup_n_est_pas_reessaye(
    index, tmp_path, monkeypatch
):
    monkeypatch.setattr(build_kb, "PAUSE_REPRISE_S", 0)
    chemin = ecrire_epub(tmp_path / "livre.epub", [("P1", "الصفحة: 1", "نص")])

    with Session(build_kb.engine) as session:
        resultat, _ = build_kb.ingerer_avec_reprises(session, index, chemin, args())

    assert resultat.status == "added"


def test_le_passage_annonce_les_documents_au_modele_divergent(
    index, tmp_path, monkeypatch, capsys
):
    """Une ré-ingestion que rien ne laissait prévoir doit être DITE, et avant :
    sinon on cherche la panne là où il n'y en a pas."""
    chemin = ecrire_epub(tmp_path / "livre.epub", [("P1", "الصفحة: 1", "نص")])
    preparer_passage(monkeypatch, tmp_path)
    with Session(build_kb.engine) as session:
        build_kb.ingerer_un_document(session, index, chemin, args())
    signaler_modele_divergent("livre.epub")

    assert build_kb.main() == 0

    assert "seront ré-embarqués" in capsys.readouterr().out


def test_une_panne_passagere_est_reessayee(index, tmp_path, monkeypatch):
    """Une coupure réseau sur un endpoint distant ne doit pas coûter un document."""
    monkeypatch.setattr(build_kb, "PAUSE_REPRISE_S", 0)
    chemin = ecrire_epub(tmp_path / "livre.epub", [("P1", "الصفحة: 1", "نص")])
    index.echouer_au_prochain_add = True  # la PREMIÈRE tentative échoue

    with Session(build_kb.engine) as session:
        resultat, _ = build_kb.ingerer_avec_reprises(session, index, chemin, args(retries=1))

    assert resultat.status == "added"
    assert index.chunks != {}


def test_aucune_reprise_quand_retries_vaut_zero(index, tmp_path, monkeypatch):
    monkeypatch.setattr(build_kb, "PAUSE_REPRISE_S", 0)
    chemin = ecrire_epub(tmp_path / "livre.epub", [("P1", "الصفحة: 1", "نص")])
    index.echouer_au_prochain_add = True

    with (
        Session(build_kb.engine) as session,
        pytest.raises(RuntimeError, match="panne simulée"),
    ):
        build_kb.ingerer_avec_reprises(session, index, chemin, args(retries=0))


def test_une_erreur_permanente_n_est_pas_reessayee(index, tmp_path, monkeypatch):
    """Réessayer un fichier illisible ne ferait que perdre du temps."""
    monkeypatch.setattr(build_kb, "PAUSE_REPRISE_S", 0)
    chemin = tmp_path / "menteur.epub"
    chemin.write_bytes(b"ceci n'est pas une archive")

    appels = []
    vraie_extraction = build_kb.extraire

    def extraire_en_comptant(chemin_a_lire):
        appels.append(chemin_a_lire)
        return vraie_extraction(chemin_a_lire)

    monkeypatch.setattr(build_kb, "extraire", extraire_en_comptant)

    with (
        Session(build_kb.engine) as session,
        pytest.raises(epub.EpubError, match="archive ZIP"),
    ):
        build_kb.ingerer_avec_reprises(session, index, chemin, args(retries=2))

    assert len(appels) == 1, "une erreur permanente ne doit être tentée qu'une fois"


def test_afficher_echecs_liste_les_documents(capsys):
    build_kb.afficher_echecs([("a.epub", ValueError("cassé")), ("b.epub", RuntimeError("réseau"))])

    sortie = capsys.readouterr().out
    assert "a.epub" in sortie
    assert "b.epub" in sortie
    assert "2 document(s) en échec" in sortie


def test_continue_on_error_traite_les_documents_suivants(
    index, tmp_path, monkeypatch, capsys
):
    """Sur 1 000 documents, un seul fichier illisible ne doit pas condamner les autres."""
    (tmp_path / "aaa.epub").write_bytes(b"ceci n'est pas une archive")
    ecrire_epub(tmp_path / "zzz.epub", [("P1", "الصفحة: 1", "نص")])
    preparer_passage(monkeypatch, tmp_path, "--continue-on-error", "--retries", "0")

    code = build_kb.main()

    assert code == 1, "un échec reste un échec : il doit être signalé"
    sortie = capsys.readouterr().out
    assert "aaa.epub" in sortie
    assert "zzz.epub" in sortie
    assert index.chunks != {}, "le document suivant devait être traité"


def test_sans_continue_on_error_le_passage_s_arrete_au_premier_echec(
    index, tmp_path, monkeypatch
):
    """Le comportement par défaut reste prudent : on s'arrête et on le dit."""
    (tmp_path / "aaa.epub").write_bytes(b"ceci n'est pas une archive")
    ecrire_epub(tmp_path / "zzz.epub", [("P1", "الصفحة: 1", "نص")])
    preparer_passage(monkeypatch, tmp_path, "--retries", "0")

    assert build_kb.main() == 1
    assert index.chunks == {}, "sans l'option, le document suivant ne doit pas être traité"
