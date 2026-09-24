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

# Connexions mises en cache pour ne pas recréer un client/une collection à
# chaque requête, et pour ne pas charger Ollama au démarrage (chargement
# paresseux : la collection n'est créée qu'au premier appel).
_client = None
_collection = None
_ollama_client = None
_openai_client = None


def get_embedding_function():
    """Fonction d'embedding bge-m3 servie par Ollama."""
    return OllamaEmbeddingFunction(
        model_name=config.EMBEDDING_MODEL,
        url=config.OLLAMA_URL,
    )


def get_client():
    """Retourne le client ChromaDB persistant (créé une seule fois)."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
    return _client


def get_collection():
    """Retourne la collection ChromaDB (créée si absente, une seule fois)."""
    global _collection
    if _collection is None:
        _collection = get_client().get_or_create_collection(
            name=config.COLLECTION_NAME,
            embedding_function=get_embedding_function(),
        )
    return _collection


def get_ollama_client():
    """Retourne un client Ollama pointant vers l'URL configurée (une seule fois)."""
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = ollama.Client(host=config.OLLAMA_URL)
    return _ollama_client


def get_openai_client():
    """Retourne un client pour toute API compatible OpenAI (une seule fois).

    L'import est paresseux : le paquet `openai` n'est chargé que si ce
    fournisseur est réellement utilisé (inutile en mode Ollama).
    """
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI

        _openai_client = OpenAI(
            base_url=config.LLM_BASE_URL or None,
            # Certains serveurs locaux (vLLM, LM Studio) n'exigent pas de clé,
            # mais le SDK refuse une valeur vide.
            api_key=config.LLM_API_KEY or "not-needed",
        )
    return _openai_client


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
    distances = results["distances"][0]

    # Filtre de pertinence : on ignore les chunks dont la distance dépasse le
    # seuil configuré (désactivé si DISTANCE_THRESHOLD <= 0).
    threshold = config.DISTANCE_THRESHOLD
    filtered_docs, sources = [], []

    for doc, meta, distance in zip(docs, metadatas, distances, strict=True):
        if not meta:
            continue
        if threshold > 0 and distance is not None and distance > threshold:
            continue
        filtered_docs.append(doc)
        sources.append({
            "source": meta.get("source"),
            "page": meta.get("page"),
            "line_start": meta.get("line_start"),
            "line_end": meta.get("line_end"),
        })

    return filtered_docs, sources


def build_prompt(question, context, language="ar"):
    """Construit l'invite envoyée au modèle, dans la langue demandée."""
    if language == "fr":
        return f"""Utilise uniquement le contexte suivant pour répondre précisément à la question, en français. Si le contexte ne contient pas la réponse, dis : « Désolé, il n'y a pas assez d'informations dans les documents fournis. ».

Contexte extrait :
{context}

Question : {question}
Réponse :"""

    return f"""استخدم السياق التالي فقط للإجابة على السؤال بدقة باللغة العربية. إذا كان السياق لا يحتوي على الإجابة، قل "عذرًا، لا توجد معلومات كافية في الوثائق المرفقة".

السياق المستخرج:
{context}

السؤال: {question}
الإجابة:"""


def generate(question, context, language="ar"):
    """Envoie l'invite au fournisseur configuré et retourne la réponse.

    Le fournisseur est choisi par `config.LLM_PROVIDER` :
      - "ollama" : serveur Ollama auto-hébergé (défaut) ;
      - "openai" : toute API compatible OpenAI (Groq, Together, vLLM, ...).

    `language` : "ar" (arabe) ou "fr" (français). Le modèle lit le contexte
    (qui peut être en arabe) et rédige sa réponse dans la langue cible.
    """
    prompt = build_prompt(question, context, language)

    if config.LLM_PROVIDER == "openai":
        response = get_openai_client().chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""

    response = get_ollama_client().chat(
        model=config.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


def format_sources(sources, language="ar"):
    """Construit la mention des sources dans la langue demandée.

    Arabe : « fichier — صفحة X (الأسطر a-b) »
    Français : « fichier — page X (lignes a-b) »
    """
    parts = []
    for s in sources:
        prefix = f"{s['source']} — " if s.get("source") else ""
        if language == "fr":
            parts.append(
                f"{prefix}page {s['page']} (lignes {s['line_start']}-{s['line_end']})"
            )
        else:
            parts.append(
                f"{prefix}صفحة {s['page']} (الأسطر {s['line_start']}-{s['line_end']})"
            )
    separator = ", " if language == "fr" else "، "
    return separator.join(parts)


def format_excerpts(docs, sources, language="ar", max_chars=200):
    """Construit la liste des extraits (phrases) cités, chacun suivi de sa référence.

    Exemple :
    1. « ...texte du passage... » — fichier.pdf — page 5 (lignes 1-9)
    """
    header = "📄 Extraits cités :" if language == "fr" else "📄 المقتطفات:"
    items = []
    for i, (doc, src) in enumerate(zip(docs, sources, strict=True), start=1):
        text = (doc or "").strip().replace("\n", " ")
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "…"
        ref = format_sources([src], language)
        items.append(f"{i}. « {text} » — {ref}")
    return header + "\n" + "\n".join(items)
