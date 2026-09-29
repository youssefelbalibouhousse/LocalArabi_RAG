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
- `app/lexical.py` — recherche lexicale (BM25) et fusion de classements RRF.
  Module **pur** (ni base ni modèle), comme `app/epub.py` : testable sans rien
  lancer. La liste de mots vides et la longueur minimale font partie de la
  configuration **mesurée** — les modifier invalide le tableau du README, et la
  mesure doit être refaite avant de croire à un gain.
- `app/main.py` — routes FastAPI. Le `StaticFiles` du frontend est monté **en
  dernier**, sinon il masquerait les routes de l'API.
- `app/auth.py` — bcrypt + JWT. `scripts/build_kb.py` — ingestion des documents.
- `app/epub.py` — extraction des EPUB (corpus Shamela). Module **pur** (ni base ni
  modèle) : lisible et testable sans rien lancer. Voir la note « EPUB » ci-dessous.
- Le corpus est réparti en deux dossiers (`config.CORPUS_DIRS`) :
  `data/documents/` (versionné) et `data/shamela/` (ignoré par Git, éditions sous
  droits). Le dossier n'est qu'un rangement — l'ingestion est identique.
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
python scripts/benchmark_embeddings.py  # débit de l'endpoint d'embeddings
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
- **`build_kb.py` est incrémental et NON destructif** : il ne supprime jamais la
  collection. Chaque document est remplacé individuellement
  (`where={"source": ...}`). L'ordre des opérations dans `app/ingest.py` est une
  **garantie de reprise** et n'est pas interchangeable : *effacer le registre →
  supprimer les chunks → écrire → inscrire le registre*. Inscrire **avant**
  d'écrire laisserait un registre affirmant qu'un document est indexé alors qu'il
  ne l'est plus — et il ne serait plus jamais repris. `check_ollama()` reste
  **avant** toute écriture.
- **Retirer un document est EXPLICITE** (`--forget`) et jamais automatique. Un
  fichier disparu du disque n'est pas retiré : un dossier déplacé ou un disque
  non monté viderait sinon l'index sans qu'on l'ait demandé. Un document inscrit
  mais absent du corpus est un **orphelin** (`--status` le signale) ; ses chunks
  restent servis comme sources tant que `--forget` n'a pas été lancé.
  `ingest.purger_document()` efface le registre **avant** l'index — même ordre
  que l'ingestion, même raison. `--forget` ne vérifie pas Ollama : supprimer
  n'embarque rien.
- **`build_kb.py` doit TENIR un long passage, pas le subir.** Sans
  `--continue-on-error`, `main()` s'arrête au PREMIER échec : sur 1 000
  documents, un seul fichier illisible laisse les suivants non traités. Le
  compteur « i/N » et le récapitulatif d'échecs ne sont pas décoratifs — sur un
  passage de plusieurs heures, ce sont eux qui rendent l'avancement lisible.
  `ingerer_avec_reprises()` réessaie les pannes passagères (réseau, redémarrage)
  mais **jamais** `PERMANENTES` (`epub.EpubError`, `ingest.IngestError`) :
  réessayer un fichier illisible ne fait que perdre du temps.
- **Mesurer un débit AVANT de louer une machine.** `scripts/benchmark_embeddings.py`
  donne les chunks/s, la dimension des vecteurs, et vérifie la marge entre
  `INGEST_BATCH_SIZE × ms/chunk` et `EMBEDDING_TIMEOUT` (il sort en code 1 si
  elle est insuffisante). Il **préchauffe** avant de mesurer : le premier appel
  charge le modèle — 3,9 s pour un seul texte à froid, contre 0,4 s ensuite —
  et la mesure décrirait le chargement au lieu du débit.
- **Un seul modèle d'embedding par index** : `ensure_embedding_model()` inscrit le
  modèle dans les métadonnées de la collection et **refuse** tout écart. Sans ce
  refus, changer `EMBEDDING_MODEL` produirait des résultats faux en silence.
- **Diagnostic « pas d'informations »** : une réponse sans la mention des sources
  (`📄 Sources` / `📄 المصادر`) signifie que la base vectorielle est **vide**, pas
  que le modèle a échoué. Avec la mention des sources, c'est le modèle qui juge
  les extraits insuffisants.
- **Secrets** : ne jamais committer `.env`, `chroma_db/`, `data/`, `backups/`. La clé
  `SECRET_KEY` doit faire **au moins 32 octets** (RFC 7518) ; ne jamais exposer
  `hashed_password` dans une réponse d'API.
- **Corpus sous droits** : `data/shamela/` est ignoré par Git pour une raison
  juridique, pas technique — le texte des EPUB est numérique, mais les éditions
  modernes restent la propriété de leurs éditeurs. Ne jamais retirer cette ligne
  du `.gitignore`, ni pousser un EPUB de ce dossier. Le motif d'URL de
  téléchargement (`/epubs/{id//100}/{id}.epub`) est déterministe : il sert à
  retrouver un ouvrage **choisi**, pas à parcourir le catalogue — ce que les
  conditions d'utilisation de Shamela interdisent explicitement.
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
- **EPUB : trois invariants, tous mesurés sur `12445.epub`** — ne pas les
  « simplifier » :
  1. **l'ordre de lecture vient du `spine` de l'OPF**, jamais de l'ordre des
     entrées de l'archive ZIP (dans l'archive réelle, `P117.xhtml` précède
     `P11.xhtml`) ;
  2. **le numéro de page est l'étiquette imprimée** (`الصفحة: 81`), pas le nom du
     fichier (`P80.xhtml` porte la page 81, `P161.xhtml` la page 154) ; citer un
     numéro absent de l'exemplaire du lecteur ne servirait à rien ;
  3. **les morceaux consécutifs d'une même page imprimée sont refusionnés**
     (17 pages sur 154). Sans cela `(source, page)` ne désigne plus un endroit
     unique et la numérotation des lignes repart de 1 au milieu d'une page.
  Le repli, quand l'étiquette manque, est la POSITION dans l'ordre de lecture :
  `PageEpub.etiquetee` vaut alors `False`, et ce n'est pas une page imprimée.
- **XHTML : les entités nommées doivent être résolues avant l'analyse.**
  `ElementTree` ne connaît que `&amp; &lt; &gt; &quot; &apos;` et ne charge pas
  les DTD externes : un `&nbsp;` fait échouer la lecture. La substitution produit
  `&#160;` (ASCII, donc valable quel que soit l'encodage déclaré). Les espaces
  INSÉCABLES (U+00A0) qui en résultent sont ensuite normalisés : ils ressemblent
  à des espaces sans en être, et cassent silencieusement toute comparaison.
- **`INGEST_BATCH_SIZE` et `EMBEDDING_TIMEOUT` se lisent ENSEMBLE.** Un lot part
  en **une seule** requête d'embedding : le produit « taille du lot × temps par
  chunk » doit rester très en dessous du délai. Mesuré ici : ~0,55 s par chunk,
  donc 256 chunks ≈ 140 s alors que le défaut de la bibliothèque ChromaDB est de
  60 s — l'écriture échouait sur un « timed out in add » sans rapport apparent
  avec sa cause. Le PDF ne passait que parce qu'il est petit (52 chunks).
  Toujours passer `timeout=` explicitement dans `rag.get_embedding_function()`.
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
- **Recherche hybride** : `rag.retrieve()` mène une recherche vectorielle **et** une
  recherche lexicale (BM25), puis les **fusionne par rangs réciproques** (`app/lexical.py`).
  Mesuré sur 59 questions : `hit@10` 76,3 % → **96,6 %**, 25 questions mieux classées,
  3 moins bien, **latence inchangée**. Pourquoi : dans `تفسير ابن المنذر`, 85 % des
  chunks sont des chaînes de transmetteurs — le vecteur les confond, et quatre questions
  citant un verset mot pour mot n'étaient **jamais** retrouvées alors que l'index
  contenait ce verset **à la page attendue** (BM25 le trouve au rang 1).
  Ne **jamais** fusionner des scores — un score BM25 et une distance L2 n'ont aucune
  unité commune : uniquement des **rangs**. L'index lexical est un **cache dérivé** de
  ChromaDB, jamais une seconde source de vérité ; il se reconstruit quand le **nombre**
  de chunks change, donc **redémarrer l'API après une ré-ingestion** (modifier le contenu
  sans changer le nombre ne le déclenche pas).
- **La constante et le nombre de candidats de la fusion ne sont PAS indépendants.**
  Avec `HYBRID_RRF_K=60` et un poids de 3 pour le vecteur, le rang 1 lexical
  (1/61 = 0,0164) était battu par le 50e candidat vectoriel (3/110 = 0,0273) : le
  lexical ne pouvait que reclasser ce que le vecteur avait déjà vu, et quatre pages
  classées 1res par BM25 restaient introuvables. Poids **égaux** (1:1) et constante
  **10** : mesuré, et stable sur toute la troncature (50, 100, 200).
- **Tokenisation arabe : ne jamais écrire la plage `\u0600-\u06FF` dans une expression
  régulière.** Elle contient la ponctuation arabe — le « ؟ » final de chaque question se
  collait au dernier mot, qui ne correspondait alors plus à rien, **sans aucun message**.
  La liste de mots vides et la longueur minimale font partie de la configuration
  **mesurée** : les modifier invalide le tableau du README, et la mesure doit être refaite
  avant de croire à un gain.
