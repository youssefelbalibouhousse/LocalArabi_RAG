"""Ingestion : met à jour la base vectorielle à partir des PDF de data/documents/.

Usage :
    python scripts/build_kb.py                  # ingère ce qui a changé
    python scripts/build_kb.py --force          # ré-ingère tout
    python scripts/build_kb.py --only doc.pdf   # un seul document
    python scripts/build_kb.py --status         # état registre/index, sans écrire

⚠️ Ce script est INCRÉMENTAL et NON DESTRUCTIF : il ne supprime plus la
collection entière. Un document dont le contenu n'a pas changé est ignoré
(comparaison d'empreintes SHA-256, jamais de dates de modification).
"""

import argparse
import sys
from pathlib import Path

from pypdf import PdfReader
from sqlmodel import Session

# Permet d'exécuter le script directement (python scripts/build_kb.py)
# en rendant le package `app` importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, ingest, rag
from app.database import create_db_and_tables, engine
from app.rag import get_ollama_client


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

def ingerer_un_pdf(session, collection, pdf_path: Path, args) -> ingest.IngestResult:
    """Extrait, découpe et ingère un PDF, en ne refaisant que le nécessaire."""
    empreinte = ingest.fingerprint_file(pdf_path)

    # Court-circuit AVANT l'extraction. La même comparaison existe dans
    # `ingest_document`, mais elle arrive trop tard : il aurait fallu relire
    # tout le PDF pour s'apercevoir qu'il n'avait pas changé. C'est exactement
    # ce que l'empreinte permet d'éviter.
    if not args.force:
        ligne = ingest.registre_pour(session, pdf_path.name)
        if ligne is not None and ligne.fingerprint == empreinte:
            return ingest.IngestResult(pdf_path.name, "unchanged", ligne.chunk_count, 0.0)

    pages = extract_arabic_pdf(pdf_path)
    if not pages:
        print(f"⚠️  Aucun texte extractible dans {pdf_path.name}.")

    return ingest.ingest_document(
        session,
        collection,
        source=pdf_path.name,
        fingerprint=empreinte,
        chunks=chunk_pages(pages) if pages else [],
        force=args.force,
    )


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


def afficher_etat(pdf_paths, args) -> int:
    """Montre le registre face à l'index, sans rien modifier.

    L'écart entre les deux colonnes est LA chose à surveiller : un registre qui
    annonce des chunks que l'index ne contient plus signale un `chroma_db/`
    effacé ou remplacé — la base ne doit alors PAS être crue.
    """
    collection = rag.get_collection()

    print(f"{'document':<32}{'registre':>9}{'index':>8}   modèle d'embedding")
    print("-" * 76)

    ecart = False
    with Session(engine) as session:
        for pdf_path in pdf_paths:
            ligne = ingest.registre_pour(session, pdf_path.name)
            attendu = ligne.chunk_count if ligne else 0
            reel = ingest.compter_chunks(collection, pdf_path.name)
            modele = ligne.embedding_model if ligne else "—"
            marque = ""
            if attendu != reel:
                ecart = True
                marque = "  ⚠️ écart"
            print(f"{pdf_path.name:<32}{attendu:>9}{reel:>8}   {modele}{marque}")

    print("-" * 76)
    if ecart:
        print("⚠️  Un écart signifie que `chroma_db/` ne correspond plus au registre.")
        print("   Relancez avec --force pour reconstruire l'index de ces documents.")
    else:
        print("✅ Registre et index concordent.")
    return 0


def analyser_arguments() -> argparse.Namespace:
    """Décrit la ligne de commande."""
    analyseur = argparse.ArgumentParser(
        description="Met à jour la base vectorielle à partir des PDF de data/documents/.",
    )
    analyseur.add_argument(
        "--force",
        action="store_true",
        help="ré-ingère tout, même les documents inchangés",
    )
    analyseur.add_argument(
        "--only",
        metavar="FICHIER",
        help="ne traiter qu'un seul document (nom de fichier, ex. cours.pdf)",
    )
    analyseur.add_argument(
        "--status",
        action="store_true",
        help="affiche le registre et l'index, sans rien écrire",
    )
    return analyseur.parse_args()


def main() -> int:
    args = analyser_arguments()

    pdf_paths = sorted(config.DOCUMENTS_DIR.glob("*.pdf"))
    if args.only:
        pdf_paths = [chemin for chemin in pdf_paths if chemin.name == args.only]
        if not pdf_paths:
            print(f"❌ Aucun PDF nommé « {args.only} » dans {config.DOCUMENTS_DIR}")
            return 1

    if not pdf_paths:
        print(f"❌ Aucun PDF trouvé dans {config.DOCUMENTS_DIR}")
        return 1

    create_db_and_tables()

    if args.status:
        return afficher_etat(pdf_paths, args)

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
        for pdf_path in pdf_paths:
            print(f"📄 {pdf_path.name} …", end=" ", flush=True)
            try:
                resultat = ingerer_un_pdf(session, collection, pdf_path, args)
            except Exception as erreur:
                print(f"❌ échec : {erreur}")
                print(
                    "   Le document n'est PAS inscrit au registre : il sera repris "
                    "au prochain passage."
                )
                return 1
            resultats.append(resultat)
            print(resultat.libelle)

    afficher_resume(resultats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
