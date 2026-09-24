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

import chromadb

from app import config
from app.rag import get_embedding_function, get_ollama_client


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


def main():
    pdf_paths = sorted(config.DOCUMENTS_DIR.glob("*.pdf"))
    if not pdf_paths:
        print(f"❌ Aucun PDF trouvé dans {config.DOCUMENTS_DIR}")
        return

    # 1. Extraire et découper TOUS les PDF AVANT de toucher à la base.
    #    Ainsi, si l'extraction échoue, la collection existante est préservée.
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
        print("❌ Aucun chunk créé (PDF vides ?). La base existante n'a PAS été modifiée.")
        return

    # 2. Vérifier qu'Ollama est joignable AVANT de supprimer quoi que ce soit.
    if not check_ollama():
        print("❌ Opération annulée : la base existante a été conservée.")
        return

    # 3. Reconstruire la collection seulement maintenant (tout est prêt).
    client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)

    # On supprime l'ancienne collection si elle existe. Vérifier explicitement
    # évite un try/except aveugle : supprimer une collection absente lèverait
    # une exception, alors que c'est un cas parfaitement normal au 1er lancement.
    noms_existants = {c.name for c in client.list_collections()}
    if config.COLLECTION_NAME in noms_existants:
        client.delete_collection(name=config.COLLECTION_NAME)

    collection = client.create_collection(
        name=config.COLLECTION_NAME,
        embedding_function=get_embedding_function(),
    )
    collection.add(documents=documents, ids=ids, metadatas=metadatas)

    print(f"\n🎉 Succès ! {len(documents)} chunks ajoutés dans ChromaDB.")
    print(f"📦 Collection : {config.COLLECTION_NAME}")
    print(f"📁 Base : {config.CHROMA_DB_PATH}")
    print(f"📚 Documents : {[p.name for p in pdf_paths]}")


if __name__ == "__main__":
    main()
