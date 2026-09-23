"""Tests UNITAIRES de la logique RAG : découpage des PDF et formatage.

Aucun appel à Ollama ni à ChromaDB ici : on teste des fonctions pures.
"""

import build_kb
from app import rag

PAGE_UNIQUE = [(1, "ligne un\nligne deux\nligne trois")]


# --- Découpage des pages en chunks ---------------------------------------

def test_chunk_pages_conserve_la_page_et_les_numeros_de_lignes():
    chunks = build_kb.chunk_pages(PAGE_UNIQUE, chunk_size=1000)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk["page"] == 1
    assert chunk["line_start"] == 1
    assert chunk["line_end"] == 3
    assert "ligne un" in chunk["text"]


def test_chunk_pages_decoupe_quand_la_taille_est_depassee():
    texte = "\n".join(f"phrase numero {i}" for i in range(50))

    chunks = build_kb.chunk_pages([(1, texte)], chunk_size=60)

    assert len(chunks) > 1, "un texte long doit être découpé en plusieurs chunks"


def test_chunk_pages_ne_franchit_jamais_les_limites_de_page():
    pages = [(1, "a\n" * 40), (2, "b\n" * 40)]

    chunks = build_kb.chunk_pages(pages, chunk_size=30)

    pages_vues = {c["page"] for c in chunks}
    assert pages_vues == {1, 2}, "chaque chunk appartient à une seule page"


def test_chunk_pages_produit_des_intervalles_de_lignes_contigus():
    """Les lignes des chunks successifs doivent s'enchaîner sans trou ni doublon."""
    texte = "\n".join(f"phrase numero {i}" for i in range(50))

    chunks = build_kb.chunk_pages([(1, texte)], chunk_size=60)

    assert chunks[0]["line_start"] == 1
    for precedent, suivant in zip(chunks, chunks[1:]):
        assert suivant["line_start"] == precedent["line_end"] + 1


# --- Formatage des sources -----------------------------------------------

SOURCE = [{"source": "a.pdf", "page": 5, "line_start": 1, "line_end": 9}]


def test_format_sources_en_francais():
    assert rag.format_sources(SOURCE, "fr") == "a.pdf — page 5 (lignes 1-9)"


def test_format_sources_en_arabe():
    resultat = rag.format_sources(SOURCE, "ar")

    assert "a.pdf" in resultat
    assert "صفحة 5" in resultat


def test_format_sources_accepte_une_source_sans_nom_de_fichier():
    sans_fichier = [{"source": None, "page": 2, "line_start": 1, "line_end": 3}]

    assert rag.format_sources(sans_fichier, "fr") == "page 2 (lignes 1-3)"


# --- Formatage des extraits ----------------------------------------------

def test_format_excerpts_contient_le_texte_et_la_reference():
    docs = ["Ceci est un passage du document."]

    resultat = rag.format_excerpts(docs, SOURCE, "fr")

    assert "Ceci est un passage du document." in resultat
    assert "a.pdf — page 5 (lignes 1-9)" in resultat


def test_format_excerpts_tronque_les_textes_trop_longs():
    docs = ["x" * 500]

    resultat = rag.format_excerpts(docs, SOURCE, "fr", max_chars=50)

    assert "…" in resultat
    assert "x" * 500 not in resultat


def test_format_excerpts_est_bilingue():
    docs = ["un passage"]

    assert "Extraits cités" in rag.format_excerpts(docs, SOURCE, "fr")
    assert "المقتطفات" in rag.format_excerpts(docs, SOURCE, "ar")
