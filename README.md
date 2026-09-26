# Chatbot RAG arabe

[![CI](https://github.com/youssefelbalibouhousse/LocalArabi_RAG/actions/workflows/ci.yml/badge.svg)](https://github.com/youssefelbalibouhousse/LocalArabi_RAG/actions/workflows/ci.yml)

Chatbot de questions/réponses en arabe sur des documents PDF, basé sur
**FastAPI**, **ChromaDB** et **Ollama** (embeddings `bge-m3`, génération `llama3.1`).
Chaque réponse cite ses sources : fichier, page et intervalle de lignes.

## Structure

```
app/
├── config.py        # Configuration centralisée (chemins, modèles, URL)
├── rag.py           # Logique RAG : récupération + génération
├── language.py      # Détection arabe/français (respect de la langue)
├── auth.py          # JWT + hachage des mots de passe
├── ratelimit.py     # Limitation de débit (fenêtre glissante)
└── main.py          # API FastAPI (routes)
frontend/
├── index.html       # Balisage
├── app.js           # Logique du client
├── css/input.css    # SOURCE de la feuille de styles — à modifier
└── app.css          # Feuille COMPILÉE (non versionnée)
scripts/
├── build_kb.py          # Ingestion des PDF → base vectorielle
├── backup.py            # Sauvegarde + vérification + restauration
├── schedule_backup.py   # Sauvegarde quotidienne automatique
└── create_user.py       # Création d'un compte en ligne de commande
tests/               # Suite pytest
data/documents/      # Déposer ici les PDF à indexer
chroma_db/           # Base vectorielle générée (non versionnée)
```

## Prérequis

- Python 3.10+
- **Node.js 20+** — uniquement pour compiler la feuille de styles du frontend
  (l'application elle-même reste 100 % Python à l'exécution)
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

npm install                    # outils de compilation de la feuille de styles
npm run build:css              # produit frontend/app.css
```

Optionnel : copier `.env.example` en `.env` pour surcharger la configuration.

> 💡 `frontend/app.css` est un **artefact de build**, au même titre qu'un `.pyc` :
> il n'est **pas versionné**. La source est `frontend/css/input.css`. Après toute
> modification du balisage ou de la feuille, relancez `npm run build:css`
> (ou `npm run watch:css` pendant le développement).
>
> En Docker, rien à installer : l'image compile la feuille elle-même
> (build multi-étapes, Node n'existe que le temps de la compilation).

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

## Utilisation avec Docker

```bash
# Construire l'image et lancer tous les services (API + Ollama)
docker compose up --build

# Au premier lancement, le conteneur "ollama-pull" télécharge les modèles
# bge-m3 et llama3.1. C'est long (plusieurs Go) et fait UNE SEULE fois.

# Interface web :  http://localhost:8000
# Documentation :  http://localhost:8000/docs
# Santé :          http://localhost:8000/health

# Reconstruire la base vectorielle (à l'intérieur du conteneur) :
docker compose run --rm api python scripts/build_kb.py

# Arrêter les conteneurs :
docker compose down
```

Les volumes `./chroma_db`, `./data` et `./backups` sont montés depuis le projet :
les données sont donc partagées entre le lancement local (`uvicorn`) et Docker.

Déploiement en production (HTTPS, comptes des testeurs) : voir `docs/DEPLOIEMENT.md`.

## Sauvegardes

Une archive horodatée des données du projet, avec empreintes SHA-256 et
restauration vérifiée :

```bash
python scripts/backup.py                      # crée une archive dans backups/
python scripts/backup.py --no-vectors         # sans l'index (régénérable)
python scripts/backup.py --list               # liste les archives
python scripts/backup.py --restore <archive> --dry-run   # vérifie sans écrire
python scripts/backup.py --restore <archive>  # restaure
```

Automatiser (tâche quotidienne, **idempotente**) :

```bash
python scripts/schedule_backup.py              # montre le plan, ne modifie rien
python scripts/schedule_backup.py --install    # installe la tâche quotidienne
python scripts/schedule_backup.py --status     # est-elle active ?
```

Chaque exécution automatique utilise `--verify --log` : l'archive est vérifiée
dès sa création, et le déroulement est écrit dans `backups/backup.log`.

Sauvegardé : comptes (`data/app.db`), PDF sources, `.env` (clé JWT) et base
vectorielle. Les connexions SQLite vivantes sont copiées via l'**API de
sauvegarde de SQLite** (instantané cohérent) et non par simple copie de
fichier — voir `docs/DEPLOIEMENT.md` pour la stratégie complète (règle 3-2-1,
volume `caddy_data`, automatisation par cron).

> 🏆 **Une sauvegarde jamais restaurée n'existe pas.**

## Tests et qualité du code

```bash
pip install -r requirements-dev.txt   # dépendances de développement (une fois)

pytest                                # la suite de tests complète
pytest --cov=app --cov=scripts        # avec la couverture de code
ruff check .                          # analyse statique
```

Ces deux commandes tournent automatiquement à chaque `git push` (voir
`.github/workflows/ci.yml`) : une Pull Request ne peut pas être fusionnée si
les tests ou l'analyse statique échouent.

## Choix techniques

| Sujet | Décision |
|---|---|
| Configuration | `app/config.py` est la **source unique de vérité**, surchargeable par variables d'environnement |
| Fournisseur LLM | `LLM_PROVIDER=ollama` (auto-hébergé) ou `openai` (Groq, Together, vLLM…) |
| Inscription | Ouverte en développement, **fermée par défaut en production** |
| Clé JWT | Minimum 32 octets (RFC 7518) ; l'application refuse de démarrer en production avec la clé de développement |
| Langue de la réponse | Consigne placée **en fin d'invite**, puis langue **réellement produite** vérifiée (`app/language.py`, comptage d'alphabet) et réécrite si besoin — la langue n'est pas laissée au bon vouloir du modèle |
| Sauvegardes | Archives horodatées + empreintes SHA-256 ; copie SQLite via l'API de sauvegarde (pas de « torn copy ») |
