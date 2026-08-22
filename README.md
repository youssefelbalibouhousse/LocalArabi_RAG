# Chatbot RAG arabe

Chatbot de questions/réponses en arabe sur des documents PDF, basé sur
**FastAPI**, **ChromaDB** et **Ollama** (embeddings `bge-m3`, génération `llama3.1`).
Chaque réponse cite ses sources : fichier, page et intervalle de lignes.

## Structure

```
app/
├── config.py        # Configuration centralisée (chemins, modèles, URL)
├── rag.py           # Logique RAG : récupération + génération
└── main.py          # API FastAPI (route /ask)
scripts/
└── build_kb.py      # Ingestion des PDF → base vectorielle
data/documents/      # Déposer ici les PDF à indexer
frontend/index.html  # Interface web simple
chroma_db/           # Base vectorielle générée (non versionnée)
```

## Prérequis

- Python 3.10+
- [Ollama](https://ollama.com/) installé et lancé, avec les modèles nécessaires :
  ```bash
  ollama pull bge-m3
  ollama pull llama3.1
  ```

## Installation

```bash
python -m venv venv
venv/Scripts/activate          # Windows
pip install -r requirements.txt
```

Optionnel : copier `.env.example` en `.env` pour surcharger la configuration.

## Utilisation

1. **Déposer les PDF** dans `data/documents/`.
2. **Construire la base vectorielle** :
   ```bash
   python scripts/build_kb.py
   ```
   (à relancer à chaque ajout/modification de document)
3. **Lancer l'API** :
   ```bash
   uvicorn app.main:app --reload
   ```
4. **Interroger** : `http://localhost:8000/ask?question=<votre question>`
   ou ouvrir `frontend/index.html`.

## Exemple de réponse

```json
{
  "question": "...",
  "answer": "... الإجابة ...\n\n📄 المصادر: arabic_document.pdf — صفحة 5 (الأسطر 3-12)",
  "sources": [
    {"source": "arabic_document.pdf", "page": 5, "line_start": 3, "line_end": 12}
  ],
  "context_used": ["..."]
}
```
