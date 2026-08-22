"""Logique RAG réutilisable : connexion à la base, récupération, génération.

Partagée entre l'API (app.main) et le script d'ingestion (scripts.build_kb),
pour éviter toute duplication de configuration.
"""

import chromadb
import ollama
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)

from app import config


def get_embedding_function():
    """Fonction d'embedding bge-m3 servie par Ollama."""
    return OllamaEmbeddingFunction(
        model_name=config.EMBEDDING_MODEL,
        url=config.OLLAMA_URL,
    )


def get_collection():
    """Retourne la collection ChromaDB (créée si absente)."""
    client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
    return client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        embedding_function=get_embedding_function(),
    )


def retrieve(collection, question, n_results=None):
    """Récupère les chunks les plus proches de la question.

    Retourne (docs, sources) où chaque source contient le fichier d'origine,
    la page et l'intervalle de lignes.
    """
    if n_results is None:
        n_results = config.N_RESULTS

    results = collection.query(
        query_texts=[question],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    docs = results["documents"][0]
    metadatas = results["metadatas"][0]

    sources = []
    for meta in metadatas:
        if not meta:
            continue
        sources.append({
            "source": meta.get("source"),
            "page": meta.get("page"),
            "line_start": meta.get("line_start"),
            "line_end": meta.get("line_end"),
        })

    return docs, sources


def generate(question, context):
    """Envoie le contexte + la question au LLM et retourne la réponse en arabe."""
    augmented_prompt = f"""استخدم السياق التالي فقط للإجابة على السؤال بدقة باللغة العربية. إذا كان السياق لا يحتوي على الإجابة، قل "عذرًا، لا توجد معلومات كافية في الوثائق المرفقة".

السياق المستخرج:
{context}

السؤال: {question}
الإجابة:"""

    response = ollama.chat(
        model=config.LLM_MODEL,
        messages=[{"role": "user", "content": augmented_prompt}],
    )
    return response["message"]["content"]


def format_sources(sources):
    """Construit la mention arabe des sources : « fichier — صفحة X (الأسطر a-b) »."""
    parts = []
    for s in sources:
        prefix = f"{s['source']} — " if s.get("source") else ""
        parts.append(
            f"{prefix}صفحة {s['page']} (الأسطر {s['line_start']}-{s['line_end']})"
        )
    return "، ".join(parts)
