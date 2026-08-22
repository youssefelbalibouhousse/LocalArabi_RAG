"""API FastAPI du chatbot RAG arabe."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import config, rag

app = FastAPI(title="Arabic RAG Chatbot")

# Autorise le navigateur (ou un fichier HTML local) à interroger l'API.
# Les origines sont configurables (voir config.CORS_ALLOW_ORIGINS).
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connexion unique à la collection au démarrage.
collection = rag.get_collection()


@app.get("/ask")
def ask_arabic(question: str):
    # 1. RETRIEVE — chunks les plus proches + leurs sources (fichier/page/lignes)
    docs, sources = rag.retrieve(collection, question)
    context = "\n\n".join(docs)

    # 2. GENERATE — réponse en arabe basée uniquement sur le contexte
    answer = rag.generate(question, context)

    # 3. Ajoute la mention des sources à la réponse
    if sources:
        answer = f"{answer}\n\n📄 المصادر: {rag.format_sources(sources)}"

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "context_used": docs,
    }
