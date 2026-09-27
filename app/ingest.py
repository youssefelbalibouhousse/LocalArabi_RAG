"""Ingestion incrémentale et bornée des documents dans la base vectorielle.

Pourquoi ce module existe
-------------------------
Le script historique (`scripts/build_kb.py`) supprimait la collection ENTIÈRE
avant de la reconstruire, et accumulait tout le corpus en mémoire :

    client.delete_collection(...)   # destructif
    collection.add(documents=...)   # un seul appel, non borné

Deux conséquences, toutes deux graves dès qu'on dépasse quelques dizaines de
documents :

1. **Aucune reprise.** Une interruption au milieu laisse une base vide ou
   partielle, sans moyen de savoir où l'on s'était arrêté.
2. **Aucune économie.** Ajouter UN document obligeait à recalculer les
   embeddings de TOUS les autres. À 20 millions de chunks, c'est des jours de
   calcul refaits pour rien.

Ce module apporte les trois propriétés qui manquaient :

**Incrémental**
    Un document dont le contenu n'a pas changé est ignoré. Le contenu est
    identifié par une empreinte SHA-256, jamais par sa date : une date de
    modification change sans que le contenu change (copie, restauration,
    changement d'horloge), et une empreinte ne ment pas.

**Borné**
    Les écritures sont découpées en lots d'une taille maîtrisée, sous le
    plafond d'écriture de ChromaDB et sous le délai d'expiration du serveur
    d'embeddings.

**Reprenable**
    Un document n'est inscrit au registre qu'APRÈS l'écriture réussie de tous
    ses chunks. Le registre est effacé AVANT de toucher à l'index : ainsi une
    interruption laisse le document « non ingéré », et la relance le reprend.
    C'est l'ordre inverse qui produirait le pire des cas — un registre affirmant
    qu'un document est indexé alors qu'il ne l'est plus.

Le registre vit dans `data/app.db`, PAS dans `chroma_db/` : l'index vectoriel
est DÉRIVÉ (il se régénère), le registre ne l'est pas — c'est le souvenir de ce
qui a été fait.
"""

import hashlib
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlmodel import Session, select

from app import config
from app.models import IngestedDocument

# Taille de lecture pour l'empreinte : 1 Mio. Assez gros pour ne pas multiplier
# les appels système, assez petit pour ne jamais charger un fichier entier.
FINGERPRINT_BLOCK = 1 << 20


class IngestError(Exception):
    """Erreur d'ingestion (message destiné à l'humain)."""


class EmbeddingModelMismatch(IngestError):
    """La collection a été construite avec un autre modèle d'embedding.

    Sans ce refus, changer `EMBEDDING_MODEL` produirait des résultats FAUX EN
    SILENCE : les anciens vecteurs ne seraient plus comparables aux nouveaux,
    mais ChromaDB ne signalerait rien — il ne connaît que des nombres.
    """


@dataclass(frozen=True)
class IngestResult:
    """Ce qui est arrivé à un document lors d'un passage d'ingestion."""

    source: str
    status: str  # "added" | "updated" | "unchanged" | "empty"
    chunks: int
    duration_s: float

    @property
    def libelle(self) -> str:
        """Résumé lisible, pour le journal d'ingestion."""
        if self.status == "unchanged":
            return "inchangé, ignoré"
        if self.status == "empty":
            return "aucun texte extractible, ignoré"
        return f"{self.chunks} chunks ({self.duration_s:.1f} s)"


def fingerprint_file(path: Path, block_size: int = FINGERPRINT_BLOCK) -> str:
    """Empreinte SHA-256 du CONTENU d'un fichier, lue par blocs.

    Par blocs, et non d'un seul coup : un PDF de plusieurs centaines de Mo ne
    doit pas être chargé en mémoire juste pour être identifié.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while bloc := handle.read(block_size):
            digest.update(bloc)
    return digest.hexdigest()


# --- Garde-fou sur le modèle d'embedding ---------------------------------

def collection_embedding_model(collection) -> str | None:
    """Modèle d'embedding inscrit dans les métadonnées de la collection."""
    meta = collection.metadata or {}
    return meta.get("embedding_model")


def ensure_embedding_model(collection, model: str | None = None) -> str:
    """Vérifie (ou inscrit) le modèle d'embedding de la collection.

    Inscrit le modèle au premier usage, puis refuse tout écart. Le premier
    passage sur une collection existante mais sans métadonnée l'adopte
    silencieusement : on ne peut pas deviner avec quoi elle a été construite,
    et refuser bloquerait une base légitime.
    """
    attendu = model if model is not None else config.EMBEDDING_MODEL
    existant = collection_embedding_model(collection)

    if existant is None:
        collection.modify(metadata={"embedding_model": attendu})
        return attendu

    if existant != attendu:
        raise EmbeddingModelMismatch(
            f"La base vectorielle a été construite avec « {existant} », "
            f"mais la configuration demande « {attendu} ».\n"
            "  Mélanger deux modèles dans un même index donnerait des résultats "
            "faux, sans erreur visible.\n"
            "  Pour changer de modèle : reconstruire l'index de zéro "
            "(supprimer chroma_db/ et vider le registre), ou rétablir "
            "EMBEDDING_MODEL."
        )

    return existant


# --- Registre -------------------------------------------------------------

def registre_pour(session: Session, source: str) -> IngestedDocument | None:
    """Ligne du registre correspondant à un document, si elle existe."""
    return session.exec(
        select(IngestedDocument).where(IngestedDocument.source == source)
    ).first()


def oublier_document(session: Session, source: str) -> None:
    """Retire un document du registre, sans toucher à l'index."""
    ligne = registre_pour(session, source)
    if ligne is not None:
        session.delete(ligne)
        session.commit()


# --- Écriture dans l'index ------------------------------------------------

def supprimer_chunks(collection, source: str) -> None:
    """Supprime les chunks d'UN document, sans toucher aux autres.

    ChromaDB exige un `where` non vide sur `delete` : sans lui, il refuserait
    l'opération, ce qui est une protection heureuse contre l'effacement
    accidentel de toute la base.
    """
    collection.delete(where={"source": source})


def ecrire_par_lots(
    collection,
    ids: Sequence[str],
    documents: Sequence[str],
    metadatas: Sequence[Mapping[str, Any]],
    batch_size: int,
) -> int:
    """Écrit les chunks par lots bornés et renvoie le nombre écrit.

    Un seul appel avec 20 millions de chunks poserait deux problèmes : dépasser
    le plafond d'écriture de ChromaDB (quelques milliers), et garder le serveur
    d'embeddings occupé plus longtemps que son délai d'expiration. On découpe
    donc — et chaque lot réussit ou échoue indépendamment.
    """
    if batch_size < 1:
        raise IngestError(f"batch_size={batch_size} est invalide : la valeur doit être >= 1.")

    total = 0
    for debut in range(0, len(ids), batch_size):
        fin = debut + batch_size
        paquet = slice(debut, fin)
        collection.add(
            ids=list(ids[paquet]),
            documents=list(documents[paquet]),
            metadatas=[dict(m) for m in metadatas[paquet]],
        )
        total += len(ids[paquet])

    return total


def preparer_chunks(source: str, chunks: Sequence[Mapping[str, Any]]):
    """Construit (ids, textes, metadatas) à partir des chunks d'un document.

    Les métadonnées reprennent TOUT le chunk sauf son texte : ainsi un champ
    ajouté plus tard (titre, auteur, catégorie…) sera indexé sans qu'il faille
    modifier ce module.
    """
    ids, textes, metadatas = [], [], []

    for index, chunk in enumerate(chunks):
        ids.append(f"{source}::chunk_{index}")
        textes.append(chunk["text"])
        metadatas.append({
            **{cle: valeur for cle, valeur in chunk.items() if cle != "text"},
            "source": source,
            "chunk_index": index,
        })

    return ids, textes, metadatas


# --- Ingestion d'un document ---------------------------------------------

def ingest_document(
    session: Session,
    collection,
    *,
    source: str,
    fingerprint: str,
    chunks: Sequence[Mapping[str, Any]],
    batch_size: int | None = None,
    force: bool = False,
) -> IngestResult:
    """Ingère un document, en ne refaisant que ce qui doit l'être.

    L'ORDRE des opérations est la garantie de reprise, et il n'est pas
    interchangeable :

    1. effacer la ligne du registre, puis valider ;
    2. supprimer les chunks existants du document ;
    3. écrire les nouveaux chunks par lots ;
    4. inscrire la ligne du registre, puis valider.

    Une interruption entre 1 et 4 laisse le document ABSENT du registre : la
    prochaine exécution le reprend intégralement. L'ordre inverse — inscrire
    d'abord, écrire ensuite — laisserait un registre affirmant qu'un document
    est indexé alors qu'il ne l'est pas, et le document ne serait PLUS JAMAIS
    repris. C'est exactement le piège que ce module existe pour éviter.
    """
    taille_lot = batch_size if batch_size is not None else config.INGEST_BATCH_SIZE
    debut = time.perf_counter()

    ligne = registre_pour(session, source)
    if not force and ligne is not None and ligne.fingerprint == fingerprint:
        return IngestResult(source, "unchanged", ligne.chunk_count, 0.0)

    if not chunks:
        # Un document vide ne doit pas non plus laisser de trace : sinon il
        # serait « ingéré » avec zéro chunk, et jamais repris.
        oublier_document(session, source)
        supprimer_chunks(collection, source)
        return IngestResult(source, "empty", 0, time.perf_counter() - debut)

    statut = "updated" if ligne is not None else "added"

    # (1) Le registre d'abord : à partir d'ici, une panne rend le document
    # « à refaire », ce qui est le comportement sûr.
    oublier_document(session, source)

    # (2) Puis l'index.
    supprimer_chunks(collection, source)

    # (3) Écriture bornée.
    ids, textes, metadatas = preparer_chunks(source, chunks)
    ecrits = ecrire_par_lots(collection, ids, textes, metadatas, taille_lot)

    # (4) Le registre enfin, une fois l'écriture prouvée.
    session.add(IngestedDocument(
        source=source,
        fingerprint=fingerprint,
        chunk_count=ecrits,
        embedding_model=collection_embedding_model(collection) or config.EMBEDDING_MODEL,
    ))
    session.commit()

    return IngestResult(source, statut, ecrits, time.perf_counter() - debut)


def compter_chunks(collection, source: str | None = None) -> int:
    """Nombre de chunks indexés, éventuellement pour un seul document."""
    if source is None:
        return collection.count()
    resultat = collection.get(where={"source": source}, include=[])
    return len(resultat.get("ids", []))
