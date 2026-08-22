from fastapi import FastAPI
import ollama
import chromadb
from fastapi.middleware.cors import CORSMiddleware
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)

app = FastAPI()

# Add the CORS middleware to allow requests from your browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all websites/local files to connect
    allow_credentials=True,
    allow_methods=["*"],  # Allows GET, POST, etc.
    allow_headers=["*"],  # Allows all headers
)


# Connect to ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")

# Set up the bge-m3 multilingual embedding function
ef = OllamaEmbeddingFunction(
    model_name="bge-m3",
    url="http://localhost:11434",
)

# Connect to the Arabic collection we built in Step 3
collection = client.get_or_create_collection(
    name="arabic_documents",
    embedding_function=ef,
)


@app.get("/ask")
def ask_arabic(question: str):
    # Step 1: RETRIEVE - Query ChromaDB using Arabic text
    results = collection.query(
        query_texts=[question],
        n_results=5,  # Get the top 5 matching context chunks
        include=["documents", "metadatas", "distances"],
    )

    docs = results["documents"][0]
    metadatas = results["metadatas"][0]

    # Combine retrieved context
    context = "\n\n".join(docs)

    # Build a list of sources (page + line range) from the chunk metadata
    sources = []
    for meta in metadatas:
        if not meta:
            continue
        sources.append({
            "page": meta.get("page"),
            "line_start": meta.get("line_start"),
            "line_end": meta.get("line_end"),
        })

    # AJOUTE CETTE LIGNE DE CODE POUR DEBUGGER :
    print("--- CONTEXTE RÉCUPÉRÉ DEPUIS CHROMADB ---")
    print(context)
    print("-----------------------------------------")

    # Step 2: AUGMENT - Customize the prompt instructions in Arabic!
    augmented_prompt = f"""استخدم السياق التالي فقط للإجابة على السؤال بدقة باللغة العربية. إذا كان السياق لا يحتوي على الإجابة، قل "عذرًا، لا توجد معلومات كافية في الوثائق المرفقة".

السياق المستخرج:
{context}

السؤال: {question}
الإجابة:"""

    # Step 3: GENERATE - Send context to Qwen2.5 (excellent Arabic capability)
    response = ollama.chat(
        model="llama3.1",
        messages=[{"role": "user", "content": augmented_prompt}],
    )

    # Format a readable Arabic mention of the sources used
    if sources:
        sources_text = "، ".join(
            f"صفحة {s['page']} (الأسطر {s['line_start']}-{s['line_end']})"
            for s in sources
        )
        answer_with_sources = (
            f"{response['message']['content']}\n\n📄 المصادر: {sources_text}"
        )
    else:
        answer_with_sources = response["message"]["content"]

    return {
        "question": question,
        "answer": answer_with_sources,
        "sources": sources,
        "context_used": docs,
    }
