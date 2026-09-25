# Project Guidelines

Chatbot RAG arabe/français : questions-réponses sur des PDF, avec citation des
sources (fichier, page, lignes). Voir `README.md` (usage) et `docs/DEPLOIEMENT.md`
(Docker, dépannage).

## Architecture

- `app/config.py` — **source unique de vérité** pour toute la configuration
  (chemins, modèles, URL). Toujours surchargeable par variable d'environnement ;
  ne jamais coder une valeur en dur ailleurs.
- `app/rag.py` — logique RAG partagée (récupération, génération, formatage des
  sources). Les connexions (ChromaDB, client Ollama) sont **mises en cache** et
  chargées paresseusement : ne pas les recréer par requête.
- `app/main.py` — routes FastAPI. Le `StaticFiles` du frontend est monté **en
  dernier**, sinon il masquerait les routes de l'API.
- `app/auth.py` — bcrypt + JWT. `scripts/build_kb.py` — ingestion des PDF.
- Le paramètre `language` (`"ar"` ou `"fr"`) traverse /ask → rag → réponse :
  toute nouvelle chaîne affichée à l'utilisateur doit être déclinée dans les 2 langues.

## Commandes

```powershell
venv\Scripts\Activate.ps1              # environnement virtuel
pip install -r requirements-dev.txt    # deps + outils de test
uvicorn app.main:app --reload          # lancer en local
venv\Scripts\python.exe -m pytest      # suite de tests (doit rester verte)
docker compose up -d --build           # lancer en conteneurs
python scripts/backup.py               # sauvegarder les données
```

Reconstruire la base vectorielle : `python scripts/build_kb.py`
(ou `docker compose run --rm api python scripts/build_kb.py`).

## Conventions

- **Docstrings et commentaires en français**, identifiants en anglais.
- Annoter les types ; préférer des fonctions courtes à responsabilité unique.
- Commits au format Conventional Commits (`feat:`, `fix:`, `chore:`, `docs:`),
  un commit par changement logique.
- Tests : `pytest`, fixtures dans `tests/conftest.py`.

## Pièges à ne pas reproduire

- **Isolation des tests** : ne jamais toucher à la vraie base (`data/app.db`) ni
  à `chroma_db/`, et ne jamais appeler le vrai Ollama. Utiliser la base en
  mémoire et `monkeypatch` (voir `tests/conftest.py`).
- **Réseau Docker** : dans un conteneur, `localhost` désigne le conteneur
  lui-même. Pour joindre un autre service, utiliser son **nom de service**
  (`ollama`) et toujours passer par `config.OLLAMA_URL`. Ne jamais coder
  `localhost` en dur dans un appel réseau.
- **`build_kb.py` est destructif** : il supprime puis recrée la collection. La
  vérification `check_ollama()` et l'extraction des PDF se font **avant** la
  suppression — préserver cet ordre, sinon une base vide est laissée derrière.
- **Diagnostic « pas d'informations »** : une réponse sans la mention des sources
  (`📄 Sources` / `📄 المصادر`) signifie que la base vectorielle est **vide**, pas
  que le modèle a échoué. Avec la mention des sources, c'est le modèle qui juge
  les extraits insuffisants.
- **Secrets** : ne jamais committer `.env`, `chroma_db/`, `data/`, `backups/`. La clé
  `SECRET_KEY` doit faire **au moins 32 octets** (RFC 7518) ; ne jamais exposer
  `hashed_password` dans une réponse d'API.
- **Sauvegardes** : `scripts/backup.py` copie `data/app.db` via l'**API de sauvegarde
  de SQLite** (`Connection.backup()`), jamais par copie de fichier : une copie brute
  d'une base vivante peut être incohérente (« torn copy »). Toute restauration passe
  par une extraction en dossier temporaire, une vérification SHA-256, puis
  seulement l'écriture — ne pas court-circuiter cet ordre.
