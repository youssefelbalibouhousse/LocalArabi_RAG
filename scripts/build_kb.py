"""Ingestion : met à jour la base vectorielle à partir des documents de data/.

Usage :
    python scripts/build_kb.py                    # ingère ce qui a changé
    python scripts/build_kb.py --force            # ré-ingère tout
    python scripts/build_kb.py --only doc.pdf     # un seul document
    python scripts/build_kb.py --forget doc.pdf   # en RETIRER un (index + registre)
    python scripts/build_kb.py --status           # état registre/index, sans écrire

Deux formats, deux dossiers (voir `config.CORPUS_DIRS`) :
    · PDF  dans `data/documents/` — versionné, donc public ;
    · EPUB dans `data/shamela/`   — ignoré par Git (éditions sous droits).
Le dossier n'est qu'un rangement : l'ingestion est identique.

⚠️ Ce script est INCRÉMENTAL et NON DESTRUCTIF : il ne supprime plus la
collection entière. Un document dont le contenu n'a pas changé est ignoré
(comparaison d'empreintes SHA-256, jamais de dates de modification).

⚠️ Retirer un document est EXPLICITE (`--forget`) et jamais automatique : un
fichier disparu du disque n'est pas retiré pour autant. Un dossier déplacé ou
un disque non monté effacerait sinon un corpus entier sans qu'on l'ait demandé.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

from pypdf import PdfReader
from sqlmodel import Session

# Permet d'exécuter le script directement (python scripts/build_kb.py)
# en rendant le package `app` importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, epub, ingest, rag
from app.database import create_db_and_tables, engine
from app.rag import get_ollama_client

# Formats de documents reconnus par l'ingestion.
FORMATS = (".pdf", ".epub")


def extract_arabic_pdf(pdf_path):
    """Retourne une liste de pages : [(numero_page, texte), ...] (page 1-indexée)."""
    print(f"Reading {pdf_path.name}...")
    reader = PdfReader(str(pdf_path))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text and text.strip():
            pages.append((page_number, text))

    return pages


def extraire_epub(chemin: Path) -> tuple[dict, list, str]:
    """Extrait un EPUB → (métadonnées indexables, pages, note de diagnostic).

    Le numéro de page vient de la page IMPRIMÉE, pas du nom du fichier, et les
    morceaux d'une même page sont refusionnés par `app.epub` : la citation
    désigne donc un endroit que le lecteur retrouve dans son exemplaire.
    """
    metadonnees, pages = epub.lire_epub(chemin)

    note = ""
    absentes = epub.pages_absentes(pages)
    if absentes:
        note = (
            f"⚠️  {len(absentes)} page(s) absente(s) de l'édition : "
            f"{epub.formater_plages(absentes)}"
        )

    return metadonnees.pour_index(), [(page.numero, page.texte) for page in pages], note


def extraire(chemin: Path) -> tuple[dict, list, str]:
    """Extrait un document → (métadonnées, pages, note éventuelle).

    `pages` est une liste de couples `(numéro_de_page, texte)` — la même forme
    pour un PDF et pour un EPUB. C'est ce qui permet à `chunk_pages` de les
    traiter sans savoir d'où ils viennent, et donc à la taille de chunk de
    rester réglée à UN seul endroit.
    """
    suffixe = chemin.suffix.lower()

    if suffixe == ".epub":
        return extraire_epub(chemin)
    if suffixe == ".pdf":
        return {}, extract_arabic_pdf(chemin), ""

    raise ingest.IngestError(f"Format non pris en charge : « {chemin.name} ».")


def chunk_pages(pages, chunk_size=None):
    """Découpe chaque page en chunks sans franchir les limites de page.

    Retourne une liste de dicts : {text, page, line_start, line_end}
    où line_start/line_end sont les numéros de ligne (1-indexés) dans la page.
    """
    if chunk_size is None:
        chunk_size = config.CHUNK_SIZE

    chunks = []

    for page_number, text in pages:
        lines = text.split("\n")
        current_lines = []
        current_len = 0
        start_line = 1  # numéro de la première ligne du chunk en cours

        for offset, line in enumerate(lines):
            line_no = offset + 1

            # Si ajouter cette ligne dépasse la taille cible, on ferme le chunk.
            if current_lines and current_len + len(line) >= chunk_size:
                chunks.append({
                    "text": "\n".join(current_lines).strip(),
                    "page": page_number,
                    "line_start": start_line,
                    "line_end": line_no - 1,
                })
                current_lines = []
                current_len = 0
                start_line = line_no

            current_lines.append(line)
            current_len += len(line) + 1  # +1 pour le saut de ligne

        if current_lines and "\n".join(current_lines).strip():
            chunks.append({
                "text": "\n".join(current_lines).strip(),
                "page": page_number,
                "line_start": start_line,
                "line_end": len(lines),
            })

    return chunks


def enrichir(chunks, metadonnees):
    """Recopie les métadonnées du document sur chacun de ses chunks.

    `ingest.preparer_chunks` indexe tout ce qui accompagne le texte : les poser
    sur chaque chunk les rend donc interrogeables sans toucher au module
    d'ingestion. Un PDF n'apporte rien ici — pypdf ne rend pas de métadonnées
    fiables — d'où le cas « pas de métadonnées », qui n'est pas une anomalie.
    """
    if not metadonnees:
        return chunks
    return [{**chunk, **metadonnees} for chunk in chunks]


def check_ollama():
    """Vérifie que le serveur Ollama est joignable (indispensable pour les embeddings).

    Retourne True si joignable, False sinon (sans rien modifier).
    """
    try:
        get_ollama_client().list()
        return True
    except Exception as exc:
        print(f"❌ Impossible de joindre Ollama sur {config.OLLAMA_URL}.")
        print(f"   Détail : {exc}")
        print("   Démarrez Ollama (ou la pile Docker : docker compose up -d) puis réessayez.")
        return False


# --- Ingestion -----------------------------------------------------------

def ingerer_un_document(
    session, collection, chemin: Path, args
) -> tuple[ingest.IngestResult, str]:
    """Extrait, découpe et ingère un document, en ne refaisant que le nécessaire."""
    empreinte = ingest.fingerprint_file(chemin)

    # Court-circuit AVANT l'extraction. La même comparaison existe dans
    # `ingest_document`, mais elle arrive trop tard : il aurait fallu relire
    # tout le document pour s'apercevoir qu'il n'avait pas changé. C'est
    # exactement ce que l'empreinte permet d'éviter.
    if not args.force:
        ligne = ingest.registre_pour(session, chemin.name)
        if ligne is not None and ligne.fingerprint == empreinte:
            resultat = ingest.IngestResult(chemin.name, "unchanged", ligne.chunk_count, 0.0)
            return resultat, ""

    metadonnees, pages, note = extraire(chemin)
    if not pages:
        print(f"⚠️  Aucun texte extractible dans {chemin.name}.")

    chunks = chunk_pages(pages) if pages else []

    # Les métadonnées du document (titre, auteur, langue…) sont recopiées sur
    # CHAQUE chunk : `preparer_chunks` indexe tout ce qui accompagne le texte,
    # donc elles deviennent interrogeables sans toucher au module d'ingestion.
    chunks = enrichir(chunks, metadonnees)

    resultat = ingest.ingest_document(
        session,
        collection,
        source=chemin.name,
        fingerprint=empreinte,
        chunks=chunks,
        force=args.force,
    )
    return resultat, note


def afficher_resume(resultats) -> None:
    """Récapitulatif d'un passage d'ingestion."""
    comptes = {statut: 0 for statut in ("added", "updated", "unchanged", "empty")}
    for resultat in resultats:
        comptes[resultat.status] += 1

    ecrits = sum(r.chunks for r in resultats if r.status in ("added", "updated"))

    print()
    print("-" * 64)
    print(
        f"  {comptes['added']} ajouté(s) · {comptes['updated']} mis à jour · "
        f"{comptes['unchanged']} inchangé(s) · {comptes['empty']} vide(s)"
    )
    print(f"  {ecrits} chunk(s) écrit(s) au total")
    print(f"  Index : {config.CHROMA_DB_PATH}  ·  collection « {config.COLLECTION_NAME} »")
    print("-" * 64)


def collecter_documents() -> list[Path]:
    """Documents à ingérer, tous dossiers du corpus confondus."""
    return [
        chemin
        for dossier in config.CORPUS_DIRS
        if dossier.is_dir()
        for chemin in sorted(dossier.iterdir())
        if chemin.is_file() and chemin.suffix.lower() in FORMATS
    ]


def noms_en_double(chemins) -> list[str]:
    """Noms de fichiers présents dans plusieurs dossiers du corpus.

    Un document est identifié par son NOM (`source` dans l'index), pas par son
    chemin : deux homonymes se remplaceraient mutuellement dans la base, en
    silence. Mieux vaut refuser que d'indexer un document à la place d'un autre.
    """
    comptes = Counter(chemin.name for chemin in chemins)
    return sorted(nom for nom, total in comptes.items() if total > 1)


def afficher_etat(chemins, args) -> int:
    """Montre le registre face à l'index et au corpus, sans rien modifier.

    Deux désaccords à surveiller, de sens opposés :

    · **écart** — le registre annonce des chunks que l'index ne contient plus :
      `chroma_db/` a été effacé ou remplacé, la base ne doit PAS être crue ;
    · **orphelin** — le registre connaît un document absent du corpus : ses
      chunks sont encore servis comme sources d'un fichier introuvable, et
      aucun passage d'ingestion ne les retirera, puisqu'il ne parcourt que le
      corpus. Seul `--forget` les enlève.
    """
    collection = rag.get_collection()
    sur_disque = {chemin.name for chemin in chemins}

    print(f"{'document':<32}{'registre':>9}{'index':>8}   modèle d'embedding")
    print("-" * 76)

    ecart = False
    with Session(engine) as session:
        for chemin in chemins:
            ligne = ingest.registre_pour(session, chemin.name)
            attendu = ligne.chunk_count if ligne else 0
            reel = ingest.compter_chunks(collection, chemin.name)
            modele = ligne.embedding_model if ligne else "—"
            marque = ""
            if attendu != reel:
                ecart = True
                marque = "  ⚠️ écart"
            print(f"{chemin.name:<32}{attendu:>9}{reel:>8}   {modele}{marque}")

        orphelins = [
            ligne
            for ligne in ingest.documents_du_registre(session)
            if ligne.source not in sur_disque
        ]

    for ligne in orphelins:
        reel = ingest.compter_chunks(collection, ligne.source)
        print(
            f"{ligne.source:<32}{ligne.chunk_count:>9}{reel:>8}   "
            f"{ligne.embedding_model}  ⚠️ orphelin"
        )

    print("-" * 76)
    if ecart:
        print("⚠️  Un écart signifie que `chroma_db/` ne correspond plus au registre.")
        print("   Relancez avec --force pour reconstruire l'index de ces documents.")
    if orphelins:
        print("⚠️  Orphelin = inscrit au registre mais absent du corpus : ses chunks")
        print("   sont encore servis comme sources. Retirez-les avec --forget.")
    if not ecart and not orphelins:
        print("✅ Registre, index et corpus concordent.")
    return 0


def oublier_un_document(nom: str) -> int:
    """Retire un document de l'index ET du registre, sans toucher aux autres.

    Ollama n'est pas requis et n'est donc pas vérifié : supprimer des chunks
    n'embarque rien. Cette opération reste ainsi possible serveur d'embeddings
    éteint — l'exact inverse de l'ingestion, qui ne peut rien faire sans lui.
    """
    collection = rag.get_collection()

    with Session(engine) as session:
        if ingest.registre_pour(session, nom) is None:
            print(f"❌ « {nom} » n'est pas au registre : il n'y a rien à retirer.")
            print("   `--status` liste ce que le registre connaît.")
            return 1
        supprimes = ingest.purger_document(session, collection, nom)

    print(f"🗑️  {nom} … {supprimes} chunk(s) retiré(s) de l'index, registre effacé")

    # Le fichier est-il encore là ? Si oui, le prochain passage le ré-ingérera,
    # et l'utilisateur croira que le retrait n'a pas fonctionné.
    restants = [chemin for chemin in collecter_documents() if chemin.name == nom]
    if restants:
        print(f"   ⚠️  Le fichier est TOUJOURS dans le corpus ({restants[0].parent}).")
        print("      Il sera ré-ingéré au prochain passage : déplacez-le ou supprimez-le.")
    return 0


def analyser_arguments() -> argparse.Namespace:
    """Décrit la ligne de commande."""
    analyseur = argparse.ArgumentParser(
        description=(
            "Met à jour la base vectorielle à partir des documents "
            "(PDF, EPUB) des dossiers du corpus."
        ),
    )
    analyseur.add_argument(
        "--force",
        action="store_true",
        help="ré-ingère tout, même les documents inchangés",
    )
    analyseur.add_argument(
        "--status",
        action="store_true",
        help="affiche le registre et l'index, sans rien écrire",
    )

    # Ajouter et retirer s'excluent : demander les deux n'a aucun sens, et
    # argparse le refuse plutôt que d'en ignorer un en silence.
    actions = analyseur.add_mutually_exclusive_group()
    actions.add_argument(
        "--only",
        metavar="FICHIER",
        help="ne traiter qu'un seul document (nom de fichier, ex. cours.pdf)",
    )
    actions.add_argument(
        "--forget",
        metavar="FICHIER",
        help="retirer un document de l'index et du registre, sans toucher aux autres",
    )
    return analyseur.parse_args()


def main() -> int:
    args = analyser_arguments()

    # `--forget` passe AVANT tout le reste, pour deux raisons :
    #   · il ne dépend pas du corpus — retirer le dernier document d'un corpus
    #     vidé doit rester possible, alors que le contrôle « aucun document »
    #     refuserait de continuer ;
    #   · il ne dépend pas d'Ollama — supprimer des chunks n'embarque rien.
    if args.forget:
        create_db_and_tables()
        return oublier_un_document(args.forget)

    chemins = collecter_documents()
    doublons = noms_en_double(chemins)
    if doublons:
        print(f"❌ Nom(s) de fichier présent(s) dans plusieurs dossiers : {', '.join(doublons)}")
        print(
            "   Un document est identifié par son nom : ces fichiers se "
            "remplaceraient mutuellement dans l'index."
        )
        print("   Renommez-en un, puis relancez.")
        return 1

    if args.only:
        chemins = [chemin for chemin in chemins if chemin.name == args.only]
        if not chemins:
            print(f"❌ Aucun document nommé « {args.only} » dans les dossiers du corpus.")
            return 1

    if not chemins and not args.status:
        dossiers = " ou ".join(str(dossier) for dossier in config.CORPUS_DIRS)
        print(f"❌ Aucun document ({', '.join(FORMATS)}) trouvé dans {dossiers}")
        return 1

    create_db_and_tables()

    if args.status:
        # `--status` fonctionne même corpus vide : c'est précisément le moment
        # où l'on a besoin de voir les orphelins restés au registre.
        return afficher_etat(chemins, args)

    # Ollama AVANT de toucher à la base : sans lui, aucun embedding n'est
    # calculable, et tout travail commencé serait perdu.
    if not check_ollama():
        print("❌ Opération annulée : la base existante est intacte.")
        return 1

    collection = rag.get_collection()
    try:
        ingest.ensure_embedding_model(collection)
    except ingest.EmbeddingModelMismatch as erreur:
        print(f"❌ {erreur}")
        return 1

    resultats = []
    with Session(engine) as session:
        for chemin in chemins:
            print(f"📄 {chemin.name} …", end=" ", flush=True)
            try:
                resultat, note = ingerer_un_document(session, collection, chemin, args)
            except Exception as erreur:
                print(f"❌ échec : {erreur}")
                print(
                    "   Le document n'est PAS inscrit au registre : il sera repris "
                    "au prochain passage."
                )
                return 1
            resultats.append(resultat)
            print(resultat.libelle)
            if note:
                print(f"   {note}")

    afficher_resume(resultats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
