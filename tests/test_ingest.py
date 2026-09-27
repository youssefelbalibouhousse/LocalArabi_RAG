"""Tests de app/ingest.py (ingestion incrémentale et bornée).

Trois propriétés sont critiques et donc verrouillées explicitement :

1. **Reprise** — une panne en cours d'écriture ne doit jamais laisser un
   registre affirmant qu'un document est ingéré. Sinon le document ne serait
   plus jamais repris, et l'index resterait silencieusement incomplet.
2. **Isolation** — ré-ingérer un document ne doit pas toucher aux chunks des
   autres. C'est toute la différence avec l'ancien « supprimer puis recréer ».
3. **Économie** — un document inchangé ne doit déclencher AUCUNE écriture, ni
   aucun appel au serveur d'embeddings.

Comme pour le reste de la suite, aucun test ne touche à la vraie base, à
`chroma_db/`, ni à Ollama : la collection est un substitut en mémoire.
"""

import pytest

from app import config, ingest


class CollectionFactice:
    """Substitut en mémoire d'une collection ChromaDB.

    Enregistre ce qui lui est demandé (taille des lots, clauses `where`) pour
    que les tests puissent vérifier le COMPORTEMENT et pas seulement le résult.
    """

    def __init__(self, embedding_model: str | None = None):
        self.metadata = {"embedding_model": embedding_model} if embedding_model else {}
        self.chunks: dict[str, tuple[str, dict]] = {}
        self.tailles_des_lots: list[int] = []
        self.suppressions: list[dict] = []
        self.echouer_au_prochain_add = False
        self.echouer_au_prochain_delete = False

    def modify(self, metadata):
        """Remplace les métadonnées de la collection."""
        self.metadata = dict(metadata)

    def add(self, ids, documents, metadatas):
        """Écrit un lot ; peut être forcé à échouer pour simuler une panne."""
        if self.echouer_au_prochain_add:
            self.echouer_au_prochain_add = False
            raise RuntimeError("panne simulée du serveur d'embeddings")
        self.tailles_des_lots.append(len(ids))
        for identifiant, texte, meta in zip(ids, documents, metadatas, strict=True):
            self.chunks[identifiant] = (texte, meta)

    def delete(self, where):
        """Supprime les chunks visés par la clause `where`."""
        self.suppressions.append(where)
        if not where:
            raise ValueError("un delete sans where effacerait toute la base")
        if self.echouer_au_prochain_delete:
            self.echouer_au_prochain_delete = False
            raise RuntimeError("panne simulée de l'index")
        source = where.get("source")
        victimes = [
            cle for cle, (_, meta) in self.chunks.items() if meta.get("source") == source
        ]
        for cle in victimes:
            del self.chunks[cle]

    def get(self, where=None, include=None, limit=None, offset=None):
        """Renvoie les identifiants correspondant au filtre."""
        if where is None:
            return {"ids": list(self.chunks)}
        source = where.get("source")
        return {
            "ids": [
                cle for cle, (_, meta) in self.chunks.items() if meta.get("source") == source
            ]
        }

    def count(self):
        """Nombre de chunks présents."""
        return len(self.chunks)


def morceau(texte: str = "texte", page: int = 1) -> dict:
    """Un chunk minimal, tel que `chunk_pages` en produit."""
    return {"text": texte, "page": page, "line_start": 1, "line_end": 2}


@pytest.fixture(name="collection")
def collection_fixture() -> CollectionFactice:
    """Une collection simulée, neuve pour chaque test."""
    return CollectionFactice(embedding_model=config.EMBEDDING_MODEL)


# --- Empreinte de fichier -------------------------------------------------

def test_l_empreinte_est_stable_pour_un_contenu_identique(tmp_path):
    fichier = tmp_path / "a.pdf"
    fichier.write_bytes(b"%PDF-1.4 contenu")

    assert ingest.fingerprint_file(fichier) == ingest.fingerprint_file(fichier)


def test_l_empreinte_change_avec_le_contenu(tmp_path):
    fichier = tmp_path / "a.pdf"
    fichier.write_bytes(b"%PDF-1.4 contenu")
    avant = ingest.fingerprint_file(fichier)

    fichier.write_bytes(b"%PDF-1.4 contenu modifie")

    assert ingest.fingerprint_file(fichier) != avant


def test_l_empreinte_ne_depend_pas_de_la_date_de_modification(tmp_path):
    """Justifie le choix de l'empreinte plutôt que du mtime."""
    premier = tmp_path / "a.pdf"
    second = tmp_path / "b.pdf"
    premier.write_bytes(b"identique")
    second.write_bytes(b"identique")

    assert ingest.fingerprint_file(premier) == ingest.fingerprint_file(second)


# --- Garde-fou sur le modèle d'embedding ----------------------------------

def test_le_modele_est_inscrit_au_premier_usage():
    collection = CollectionFactice()

    assert ingest.ensure_embedding_model(collection, "bge-m3") == "bge-m3"
    assert collection.metadata["embedding_model"] == "bge-m3"


def test_le_meme_modele_est_accepte():
    collection = CollectionFactice(embedding_model="bge-m3")

    assert ingest.ensure_embedding_model(collection, "bge-m3") == "bge-m3"


def test_un_modele_different_est_refuse():
    """Sans ce refus, mélanger deux modèles donnerait des résultats faux en silence."""
    collection = CollectionFactice(embedding_model="bge-m3")

    with pytest.raises(ingest.EmbeddingModelMismatch, match="bge-m3"):
        ingest.ensure_embedding_model(collection, "autre-modele")


def test_le_message_d_explication_indique_la_marche_a_suivre():
    collection = CollectionFactice(embedding_model="bge-m3")

    with pytest.raises(ingest.EmbeddingModelMismatch) as capture:
        ingest.ensure_embedding_model(collection, "autre")

    assert "chroma_db/" in str(capture.value)


# --- Préparation et écriture ---------------------------------------------

def test_les_identifiants_suivent_le_format_attendu():
    ids, textes, _ = ingest.preparer_chunks("a.pdf", [morceau("un"), morceau("deux")])

    assert ids == ["a.pdf::chunk_0", "a.pdf::chunk_1"]
    assert textes == ["un", "deux"]


def test_les_metadonnees_ne_contiennent_pas_le_texte():
    """Le texte est stocké une seule fois : le dupliquer gaspillerait la place."""
    _, _, metadatas = ingest.preparer_chunks("a.pdf", [morceau()])

    assert "text" not in metadatas[0]
    assert metadatas[0]["source"] == "a.pdf"
    assert metadatas[0]["chunk_index"] == 0
    assert metadatas[0]["page"] == 1


def test_les_champs_inconnus_sont_transmis():
    """Un champ ajouté plus tard doit être indexé sans modifier ce module."""
    enrichi = {**morceau(), "author": "ابن حجر", "category": "fikh"}

    _, _, metadatas = ingest.preparer_chunks("a.pdf", [enrichi])

    assert metadatas[0]["author"] == "ابن حجر"
    assert metadatas[0]["category"] == "fikh"


def test_l_ecriture_respecte_la_taille_des_lots():
    collection = CollectionFactice()
    morceaux = [morceau(f"t{i}") for i in range(10)]
    ids, textes, metadatas = ingest.preparer_chunks("a.pdf", morceaux)

    ecrits = ingest.ecrire_par_lots(collection, ids, textes, metadatas, batch_size=3)

    assert ecrits == 10
    assert collection.tailles_des_lots == [3, 3, 3, 1]


def test_un_lot_invalide_est_refuse():
    collection = CollectionFactice()

    with pytest.raises(ingest.IngestError, match="batch_size"):
        ingest.ecrire_par_lots(collection, ["a"], ["t"], [{}], batch_size=0)


def test_la_suppression_ne_vise_qu_un_document():
    collection = CollectionFactice()
    ingest.ecrire_par_lots(collection, ["a::0"], ["t"], [{"source": "a.pdf"}], 10)
    ingest.ecrire_par_lots(collection, ["b::0"], ["t"], [{"source": "b.pdf"}], 10)

    ingest.supprimer_chunks(collection, "a.pdf")

    assert ingest.compter_chunks(collection, "a.pdf") == 0
    assert ingest.compter_chunks(collection, "b.pdf") == 1


# --- Ingestion d'un document ---------------------------------------------

def test_un_premier_passage_ajoute_le_document(session, collection):
    resultat = ingest.ingest_document(
        session, collection, source="a.pdf", fingerprint="v1", chunks=[morceau(), morceau()]
    )

    assert resultat.status == "added"
    assert resultat.chunks == 2
    assert ingest.registre_pour(session, "a.pdf").fingerprint == "v1"


def test_un_document_inchange_n_ecrit_rien(session, collection):
    """Le gain principal du module : ne pas recalculer ce qui n'a pas bougé."""
    ingest.ingest_document(
        session, collection, source="a.pdf", fingerprint="v1", chunks=[morceau()]
    )
    collection.tailles_des_lots.clear()

    resultat = ingest.ingest_document(
        session, collection, source="a.pdf", fingerprint="v1", chunks=[morceau()]
    )

    assert resultat.status == "unchanged"
    assert collection.tailles_des_lots == []


def test_un_document_modifie_est_remplace(session, collection):
    ingest.ingest_document(
        session, collection, source="a.pdf", fingerprint="v1", chunks=[morceau("ancien")]
    )

    resultat = ingest.ingest_document(
        session, collection, source="a.pdf", fingerprint="v2", chunks=[morceau("nouveau")]
    )

    assert resultat.status == "updated"
    assert resultat.chunks == 1
    assert ingest.compter_chunks(collection, "a.pdf") == 1
    assert collection.chunks["a.pdf::chunk_0"][0] == "nouveau"


def test_force_re_ingere_meme_un_document_inchange(session, collection):
    ingest.ingest_document(
        session, collection, source="a.pdf", fingerprint="v1", chunks=[morceau()]
    )

    resultat = ingest.ingest_document(
        session, collection, source="a.pdf", fingerprint="v1", chunks=[morceau()], force=True
    )

    assert resultat.status == "updated"


def test_un_document_sans_texte_ne_laisse_aucune_trace(session, collection):
    """Sinon il serait « ingéré avec zéro chunk » et jamais repris."""
    ingest.ingest_document(
        session, collection, source="a.pdf", fingerprint="v1", chunks=[morceau()]
    )

    resultat = ingest.ingest_document(
        session, collection, source="a.pdf", fingerprint="v2", chunks=[]
    )

    assert resultat.status == "empty"
    assert ingest.registre_pour(session, "a.pdf") is None
    assert ingest.compter_chunks(collection, "a.pdf") == 0


def test_re_ingerer_un_document_ne_touche_pas_aux_autres(session, collection):
    ingest.ingest_document(
        session, collection, source="a.pdf", fingerprint="v1", chunks=[morceau()]
    )
    ingest.ingest_document(
        session, collection, source="b.pdf", fingerprint="v1", chunks=[morceau()]
    )

    ingest.ingest_document(
        session, collection, source="a.pdf", fingerprint="v2", chunks=[morceau()], force=True
    )

    assert ingest.compter_chunks(collection, "b.pdf") == 1
    # Trois suppressions ciblées : a.pdf, b.pdf, puis a.pdf de nouveau. Aucune
    # n'est globale — c'est exactement ce que l'ancien script faisait de travers.
    assert collection.suppressions == [
        {"source": "a.pdf"},
        {"source": "b.pdf"},
        {"source": "a.pdf"},
    ]


def test_une_panne_en_cours_d_ecriture_laisse_le_document_a_reprendre(session, collection):
    """La garantie de reprise : le registre ne doit jamais mentir.

    On fait échouer l'écriture APRÈS la suppression des anciens chunks. Le
    document n'est alors plus dans l'index — et le registre doit être VIDE,
    sans quoi il ne serait plus jamais repris et l'index resterait incomplet
    sans que rien ne le signale.
    """
    ingest.ingest_document(
        session, collection, source="a.pdf", fingerprint="v1", chunks=[morceau()]
    )
    collection.echouer_au_prochain_add = True

    with pytest.raises(RuntimeError, match="panne simulée"):
        ingest.ingest_document(
            session, collection, source="a.pdf", fingerprint="v2", chunks=[morceau()]
        )

    assert ingest.registre_pour(session, "a.pdf") is None
    assert ingest.compter_chunks(collection, "a.pdf") == 0


def test_la_reprise_apres_panne_reindexe_completement(session, collection):
    """Corollaire du test précédent : la relance doit réparer l'index."""
    ingest.ingest_document(
        session, collection, source="a.pdf", fingerprint="v1", chunks=[morceau("un")]
    )
    collection.echouer_au_prochain_add = True
    with pytest.raises(RuntimeError):
        ingest.ingest_document(
            session, collection, source="a.pdf", fingerprint="v2", chunks=[morceau("deux")]
        )

    resultat = ingest.ingest_document(
        session, collection, source="a.pdf", fingerprint="v2", chunks=[morceau("deux")]
    )

    assert resultat.status == "added"
    assert collection.chunks["a.pdf::chunk_0"][0] == "deux"


def test_l_ingestion_utilise_la_taille_de_lot_configuree(session, collection):
    morceaux = [morceau(f"t{i}") for i in range(5)]

    ingest.ingest_document(
        session, collection, source="a.pdf", fingerprint="v1", chunks=morceaux, batch_size=2
    )

    assert collection.tailles_des_lots == [2, 2, 1]


def test_le_registre_retient_le_modele_et_le_nombre_de_chunks(session, collection):
    ingest.ingest_document(
        session, collection, source="a.pdf", fingerprint="v1", chunks=[morceau(), morceau()]
    )

    ligne = ingest.registre_pour(session, "a.pdf")

    assert ligne.chunk_count == 2
    assert ligne.embedding_model == config.EMBEDDING_MODEL
    assert ligne.updated_at is not None


# --- Comptage et oubli ----------------------------------------------------

def test_le_comptage_sans_source_donne_le_total(session, collection):
    ingest.ingest_document(
        session, collection, source="a.pdf", fingerprint="v1", chunks=[morceau(), morceau()]
    )
    ingest.ingest_document(
        session, collection, source="b.pdf", fingerprint="v1", chunks=[morceau()]
    )

    assert ingest.compter_chunks(collection) == 3
    assert ingest.compter_chunks(collection, "a.pdf") == 2


def test_oublier_un_document_ne_touche_pas_a_l_index(session, collection):
    """Sert à forcer une ré-ingestion sans effacer les chunks tout de suite."""
    ingest.ingest_document(
        session, collection, source="a.pdf", fingerprint="v1", chunks=[morceau()]
    )

    ingest.oublier_document(session, "a.pdf")

    assert ingest.registre_pour(session, "a.pdf") is None
    assert ingest.compter_chunks(collection, "a.pdf") == 1


def test_oublier_un_document_absent_ne_provoque_pas_d_erreur(session):
    """Idempotent : oublier deux fois n'est pas une faute."""
    ingest.oublier_document(session, "jamais-vu.pdf")


# --- Retrait complet d'un document ----------------------------------------

def test_documents_du_registre_liste_tout_le_registre(session, collection):
    """Sert à repérer les documents inscrits mais absents du corpus."""
    ingest.ingest_document(
        session, collection, source="b.pdf", fingerprint="v1", chunks=[morceau()]
    )
    ingest.ingest_document(
        session, collection, source="a.pdf", fingerprint="v1", chunks=[morceau()]
    )

    sources = [ligne.source for ligne in ingest.documents_du_registre(session)]

    assert sources == ["a.pdf", "b.pdf"]


def test_purger_retire_le_document_du_registre_et_de_l_index(session, collection):
    ingest.ingest_document(
        session, collection, source="a.pdf", fingerprint="v1", chunks=[morceau(), morceau()]
    )

    supprimes = ingest.purger_document(session, collection, "a.pdf")

    assert supprimes == 2
    assert ingest.registre_pour(session, "a.pdf") is None
    assert ingest.compter_chunks(collection, "a.pdf") == 0


def test_purger_ne_touche_pas_aux_autres_documents(session, collection):
    """Toute la différence avec l'ancien « supprimer puis recréer »."""
    ingest.ingest_document(
        session, collection, source="a.pdf", fingerprint="v1", chunks=[morceau()]
    )
    ingest.ingest_document(
        session, collection, source="b.pdf", fingerprint="v1", chunks=[morceau(), morceau()]
    )

    ingest.purger_document(session, collection, "a.pdf")

    assert ingest.registre_pour(session, "b.pdf") is not None
    assert ingest.compter_chunks(collection, "b.pdf") == 2


def test_purger_un_document_absent_renvoie_zero(session, collection):
    """Idempotent : retirer deux fois n'est pas une faute."""
    assert ingest.purger_document(session, collection, "jamais-vu.pdf") == 0


def test_une_panne_pendant_la_purge_laisse_le_document_non_inscrit(session, collection):
    """Le registre est effacé AVANT l'index — l'ordre de l'ingestion, et pour la
    même raison : une panne doit laisser le document « à reprendre », jamais un
    registre affirmant la présence de chunks qui n'existent plus."""
    ingest.ingest_document(
        session, collection, source="a.pdf", fingerprint="v1", chunks=[morceau()]
    )
    collection.echouer_au_prochain_delete = True

    with pytest.raises(RuntimeError, match="panne simulée"):
        ingest.purger_document(session, collection, "a.pdf")

    assert ingest.registre_pour(session, "a.pdf") is None


def test_un_document_purge_peut_etre_re_ingere(session, collection):
    """Le retrait n'est pas une impasse : ré-ingérer doit repartir de zéro."""
    ingest.ingest_document(
        session, collection, source="a.pdf", fingerprint="v1", chunks=[morceau()]
    )
    ingest.purger_document(session, collection, "a.pdf")

    resultat = ingest.ingest_document(
        session, collection, source="a.pdf", fingerprint="v1", chunks=[morceau()]
    )

    assert resultat.status == "added"
    assert ingest.compter_chunks(collection, "a.pdf") == 1
