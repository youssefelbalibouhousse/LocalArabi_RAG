import os
from pypdf import PdfReader
import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)


CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "arabic_documents"
PDF_FILENAME = "arabic_document.pdf"


def extract_arabic_pdf(pdf_path):
    """Retourne une liste de pages : [(numero_page, texte), ...] (page 1-indexée)."""
    print(f"Reading {pdf_path}...")
    reader = PdfReader(pdf_path)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text and text.strip():
            pages.append((page_number, text))

    return pages


def chunk_pages(pages, chunk_size=600):
    """Découpe chaque page en chunks sans franchir les limites de page.

    Retourne une liste de dicts : {text, page, line_start, line_end}
    où line_start/line_end sont les numéros de ligne (1-indexés) dans la page.
    """
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
    if not os.path.exists(PDF_FILENAME):
        print(f"❌ Error: Could not find {PDF_FILENAME} in your project directory!")
        return

    pages = extract_arabic_pdf(PDF_FILENAME)

    if not pages:
        print("❌ Error: No text could be extracted from the PDF.")
        return

    chunks = chunk_pages(pages)

    if not chunks:
        print("❌ Error: No chunks were created from the PDF text.")
        return

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    ef = OllamaEmbeddingFunction(
        model_name="bge-m3",
        url="http://localhost:11434",
    )

    # On reconstruit la collection de zéro pour que tous les chunks
    # portent bien les métadonnées (page / lignes).
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass  # la collection n'existait pas encore

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
    )

    documents = [c["text"] for c in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "page": c["page"],
            "line_start": c["line_start"],
            "line_end": c["line_end"],
        }
        for c in chunks
    ]

    collection.add(
        documents=documents,
        ids=ids,
        metadatas=metadatas,
    )

    print(f"🎉 Success! Added {len(documents)} Arabic chunks into ChromaDB.")
    print(f"📦 Collection name: {COLLECTION_NAME}")
    print(f"📁 ChromaDB path: {CHROMA_DB_PATH}")


if __name__ == "__main__":
    main()

import os
import chromadb

# 1. Connexion
client = chromadb.PersistentClient(path="./chroma_db")

print("\n=========================================")
print("📍 DIAGNOSTIC DE DOSSIER WINDOWS")
print("=========================================")
# Affiche le chemin complet sur ton disque dur
print(f"📁 Chemin absolu utilisé : {os.path.abspath('./chroma_db')}")

# Affiche toutes les collections présentes dans ce dossier spécifique
collections = client.list_collections()
print(f"🗂️ Collections présentes ici : {[col.name for col in collections]}")
print("=========================================\n")