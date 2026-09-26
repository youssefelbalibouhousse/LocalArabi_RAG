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
- `frontend/` — balisage (`index.html`), logique (`app.js`) et **source** des styles
  (`css/input.css`). `app.css` est **compilé** par Tailwind : ne jamais l'éditer.
  Le frontend n'a aucune classe utilitaire écrite dans le HTML : tout passe par des
  classes sémantiques définies dans `input.css`.

## Commandes

```powershell
venv\Scripts\Activate.ps1              # environnement virtuel
pip install -r requirements-dev.txt    # deps + outils de test
uvicorn app.main:app --reload          # lancer en local
venv\Scripts\python.exe -m pytest      # suite de tests (doit rester verte)
docker compose up -d --build           # lancer en conteneurs
python scripts/backup.py               # sauvegarder les données
python scripts/schedule_backup.py --status   # la sauvegarde auto est-elle planifiée ?
npm run build:css                      # compiler la feuille de styles du frontend
npm run watch:css                      # la recompiler en continu
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
- **Planification des sauvegardes** : `scripts/schedule_backup.py` est **idempotent**
  (ligne repérée par un marqueur dans le crontab, option `/F` sous Windows). Ne jamais
  inscrire une seconde tâche non marquée : deux sauvegardes concurrentes liraient la
  même base SQLite en parallèle. La tâche planifiée utilise toujours `--verify --log`.
  Dans `backup.py`, ne pas retirer `--log` d'un contexte planifié : le planificateur ne
  conserve pas la sortie standard, donc un échec deviendrait invisible.
- **Langue de la réponse** : toute route interrogeant le modèle doit passer par
  `rag.answer_question()` et **jamais** par `rag.generate()` : c'est là que la langue
  réellement produite est vérifiée (`app/language.py`) puis corrigée. Dans
  `build_prompt`, la consigne de langue doit rester **en fin d'invite** — c'est
  volontaire (les derniers tokens pèsent le plus), ne pas la remonter au début.
- **Frontend** : le style vit dans `frontend/css/input.css` ; `frontend/app.css` est un
  artefact **compilé** (non versionné, écrasé à chaque build) — ne jamais l'éditer à la
  main. `@import "tailwindcss" source(none)` est indispensable : sans lui, Tailwind
  scanne tout le dépôt et le build local diverge du build Docker. Enfin, **jamais de
  `letter-spacing` sur de l'arabe** : l'espacement casse la liaison des lettres
  (الحروف المتصلة). La typographie arabe se règle par la hauteur de ligne (1.85 mini)
  et le choix de police, jamais par l'espacement des lettres.
