"""Ingestion : lit tous les PDF de data/documents/ et (re)construit la base vectorielle.

Usage :
    python scripts/build_kb.py
"""

import sys
from pathlib import Path

from pypdf import PdfReader

# Permet d'exécuter le script directement (python scripts/build_kb.py)
# en rendant le package `app` importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402
from app.rag import get_embedding_function  # noqa: E402
import chromadb  # noqa: E402


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


def main():
    pdf_paths = sorted(config.DOCUMENTS_DIR.glob("*.pdf"))
    if not pdf_paths:
        print(f"❌ Aucun PDF trouvé dans {config.DOCUMENTS_DIR}")
        return

    # On reconstruit la collection de zéro pour garantir des métadonnées cohérentes.
    client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
    try:
        client.delete_collection(name=config.COLLECTION_NAME)
    except Exception:
        pass  # la collection n'existait pas encore

    collection = client.create_collection(
        name=config.COLLECTION_NAME,
        embedding_function=get_embedding_function(),
    )

    documents, ids, metadatas = [], [], []

    for pdf_path in pdf_paths:
        pages = extract_arabic_pdf(pdf_path)
        if not pages:
            print(f"⚠️  Aucun texte extractible dans {pdf_path.name}, ignoré.")
            continue

        chunks = chunk_pages(pages)
        for i, chunk in enumerate(chunks):
            documents.append(chunk["text"])
            ids.append(f"{pdf_path.name}::chunk_{i}")
            metadatas.append({
                "source": pdf_path.name,
                "page": chunk["page"],
                "line_start": chunk["line_start"],
                "line_end": chunk["line_end"],
            })
        print(f"  → {len(chunks)} chunks depuis {pdf_path.name}")

    if not documents:
        print("❌ Aucun chunk créé (PDF vides ?).")
        return

    collection.add(documents=documents, ids=ids, metadatas=metadatas)

    print(f"\n🎉 Succès ! {len(documents)} chunks ajoutés dans ChromaDB.")
    print(f"📦 Collection : {config.COLLECTION_NAME}")
    print(f"📁 Base : {config.CHROMA_DB_PATH}")
    print(f"📚 Documents : {[p.name for p in pdf_paths]}")


if __name__ == "__main__":
    main()
