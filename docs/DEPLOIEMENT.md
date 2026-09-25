# Déploiement local avec Docker

Ce document récapitule comment lancer le projet en **conteneurs Docker**
(API FastAPI + ChromaDB + Ollama), et explique un bug classique rencontré.

## Prérequis

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installé et démarré
  (icône verte « Engine running » dans la barre des tâches).
- Les modèles `bge-m3` et `llama3.1` disponibles dans le volume Docker `ollama`
  (voir « Première installation des modèles » ci-dessous).

## Lancement rapide

```bash
# 1. Construire l'image API et démarrer la pile (détaché)
docker compose up -d --build

# 2. Vérifier l'état des conteneurs
docker compose ps
#    → api doit afficher "(healthy)", ollama "Up"

# 3. Ouvrir l'application dans le navigateur
#    http://localhost:8000
#    Documentation API : http://localhost:8000/docs
#    Santé :            http://localhost:8000/health
```

## Tester le chatbot (flux complet)

1. Ouvrir `http://localhost:8000`.
2. Cliquer sur **« إنشاء حساب جديد »** (créer un compte) : nom d'utilisateur + mot de passe.
3. Se connecter.
4. Poser une question **précise** sur le contenu des PDF de `data/documents/`.

Ou en ligne de commande (PowerShell) :

```powershell
# Connexion (OAuth2 : format form-urlencoded)
$login = Invoke-RestMethod -Method Post -Uri "http://localhost:8000/login" `
  -Body "username=alice&password=motdepasse123" `
  -ContentType "application/x-www-form-urlencoded"

# Question (jeton JWT dans l'en-tête)
$headers = @{ Authorization = "Bearer $($login.access_token)" }
Invoke-RestMethod -Method Get `
  -Uri "http://localhost:8000/ask?question=VOTRE_QUESTION_EN_ARABE" `
  -Headers $headers
```

## Commandes utiles

```bash
docker compose up -d --build   # construire + démarrer
docker compose down            # arrêter + supprimer les conteneurs (les volumes restent)
docker compose logs -f api     # suivre les logs de l'API
docker compose ps              # état des conteneurs
docker compose exec -T ollama ollama list   # lister les modèles disponibles
```

## Reconstruire la base vectorielle (après ajout de PDF)

```bash
# Déposer les PDF dans data/documents/ puis :
docker compose run --rm api python scripts/build_kb.py
```

## Déploiement en PRODUCTION (pilote avec testeurs externes)

En production, on ajoute **Caddy** en façade : il obtient et renouvelle
automatiquement le certificat **HTTPS**, et l'API n'est plus exposée directement.

### Prérequis

1. Un **serveur** (VPS) avec Docker installé, accessible sur Internet.
2. Un **nom de domaine** dont l'enregistrement DNS **A** pointe vers l'IP du serveur.
3. Les ports **80** et **443** ouverts dans le pare-feu.

> ℹ️ Let's Encrypt ne délivre pas de certificat pour une IP ou pour `localhost` :
> le nom de domaine est indispensable.

### Étapes

```bash
# 1. Créer le fichier .env à partir du modèle
cp .env.example .env

# 2. Renseigner les valeurs de PRODUCTION (voir ci-dessous)

# 3. Démarrer la pile complète (base + surcharge de production)
docker compose -f compose.yaml -f compose.prod.yaml up -d --build

# 4. Créer les comptes des testeurs (l'inscription publique est fermée)
docker compose run --rm api python scripts/create_user.py testeur1
```

Contenu minimal du `.env` de production :

```bash
DOMAIN=chatbot.mondomaine.com
ENVIRONMENT=production          # ferme l'inscription + refuse la clé de dev
ALLOW_REGISTRATION=false
SECRET_KEY=<python -c "import secrets; print(secrets.token_hex(32))">
```

### Vérifications

```bash
docker compose ps                       # api (healthy), caddy, ollama
docker compose logs -f caddy            # obtention du certificat + requêtes
curl -I https://chatbot.mondomaine.com  # doit répondre 200 en HTTPS
```

### Ce qui change par rapport au développement

| | Développement | Production |
|---|---|---|
| Point d'entrée | `http://localhost:8000` | `https://<DOMAIN>` (port 443) |
| Accès à l'API | direct sur le port 8000 | **uniquement via Caddy** |
| Certificat TLS | aucun | Let's Encrypt, renouvelé automatiquement |
| Inscription publique | ouverte | **fermée** |

### ⚠️ À sauvegarder

Toutes les données ne se valent pas : ce qui compte, c'est de savoir si elles
sont **régénérables**.

| Donnée | Régénérable ? | Priorité |
|---|---|---|
| `data/app.db` (comptes) | ❌ non — hachages bcrypt irréversibles | 🔴 critique |
| `data/documents/` (PDF sources) | ❌ non — sans eux, plus rien à indexer | 🔴 critique |
| `.env` (`SECRET_KEY`) | ❌ non — la perdre invalide tous les jetons | 🟠 importante |
| `chroma_db/` (base vectorielle) | ✅ oui — **dérivée** des PDF | 🟡 confort |
| volume `caddy_data` (certificats TLS) | ✅ oui — Let's Encrypt (quota limité) | 🟢 faible |

> 💡 **Dérivée ≠ inutile** : sauvegarder `chroma_db/` évite une réindexation de
> plusieurs minutes. Mais si vous devez choisir, gardez les PDF.

#### Sauvegarder

```bash
# Archive complète (comptes + PDF + index + .env)
docker compose exec api python scripts/backup.py

# Archive légère : sans la base vectorielle (régénérable)
docker compose exec api python scripts/backup.py --no-vectors

# Lister les archives
docker compose exec api python scripts/backup.py --list
```

Les archives sont écrites dans `./backups/` **sur l'hôte** (volume monté) :
elles survivent donc à un `docker compose down`, et même à un `down -v`.

Chaque archive contient un `MANIFEST.json` : date, liste des fichiers, tailles
et empreintes **SHA-256**. C'est ce qui permet de prouver, des mois plus tard,
que l'archive est intacte.

#### Vérifier et restaurer

```bash
# Vérifier l'intégrité SANS rien écrire (à faire périodiquement !)
docker compose exec api python scripts/backup.py --restore <archive> --dry-run

# Restaurer réellement (arrêter l'API d'abord)
docker compose stop api
docker compose exec api python scripts/backup.py --restore <archive>
docker compose start api
```

La restauration **refuse de commencer** si une seule vérification échoue
(chemins dangereux, empreinte invalide, fichier manquant) : une restauration
partielle serait pire que pas de restauration du tout.

#### Les certificats TLS (volume Docker)

Le script ne peut pas les lire : il s'exécute *dans* le conteneur `api`, qui
n'a pas accès au volume d'un autre conteneur. On passe donc par un conteneur
jetable :

```bash
docker run --rm -v fastapiproject_caddy_data:/data -v "$PWD/backups":/backup \
  alpine tar czf /backup/caddy_data.tar.gz -C /data .
```

#### Rétention et automatisation

`BACKUP_RETENTION` (défaut : `7`) fixe le nombre d'archives conservées ; les
plus anciennes sont supprimées automatiquement après chaque sauvegarde.

Pour automatiser la sauvegarde, une seule commande suffit — et elle est
**idempotente** : la relancer remplace la tâche au lieu d'en créer une seconde.

```bash
# SUR LE SERVEUR : l'application tourne dans Docker, donc --mode docker
python3 scripts/schedule_backup.py --mode docker           # plan, ne modifie RIEN
python3 scripts/schedule_backup.py --mode docker --install
python3 scripts/schedule_backup.py --status                # est-ce actif ?

# Sur un poste de développement (venv local)
python scripts/schedule_backup.py --install
```

Le planificateur utilisé dépend du système :

| Système | Outil | Idempotence |
|---|---|---|
| Linux / macOS | `crontab` | ligne repérée par un **marqueur** (`# arabic-rag-backup`) |
| Windows | Planificateur de tâches | option `/F` (écrase l'existante) |

Chaque exécution lance `scripts/backup.py --verify --log` :

- **`--verify`** : l'archive est vérifiée (empreintes SHA-256) aussitôt créée ;
- **`--log`** : le déroulement et le **code de sortie** sont écrits dans
  `backups/backup.log`. Le planificateur ne conserve pas la sortie standard :
  sans journal, une sauvegarde qui échoue échoue **en silence**.

> 🔒 **Pourquoi l'idempotence est vitale** : deux tâches concurrentes
> sauvegarderaient en parallèle, donc liraient la même base SQLite au même
> moment. Le marqueur (cron) et l'option `/F` (Windows) l'empêchent.

**Vérifier que la sauvegarde tourne réellement** — une tâche *planifiée* n'est
pas une tâche *exécutée* :

```bash
python scripts/schedule_backup.py --status
tail -n 20 backups/backup.log
python scripts/backup.py --list        # une archive par jour est attendue
```

> 🏆 **La règle d'or : une sauvegarde jamais restaurée n'existe pas.**
> Testez une restauration sur un dossier vide au moins une fois avant
> d'ouvrir l'application à des testeurs.

#### La règle 3-2-1

Trois copies, sur deux supports différents, dont **une hors site**. Un disque
externe, un NAS distant ou un stockage objet (`rclone`, `scp`, Backblaze B2…)
couvrent le dernier « 1 » — le seul qui sauve lors d'un vrai sinistre.

## Modifier l'apparence du frontend

Le frontend est stylé par une feuille **compilée** (`frontend/app.css`), qui
n'est pas versionnée : on modifie la source, puis on compile.

```bash
npm install            # une seule fois
npm run build:css      # compile frontend/css/input.css -> frontend/app.css
npm run watch:css      # recompile automatiquement à chaque modification
```

| Fichier | Rôle |
|---|---|
| `frontend/css/input.css` | **La source** : jetons de design, typographie, composants |
| `frontend/index.html` | Le balisage (classes sémantiques) |
| `frontend/app.js` | La logique du client |
| `frontend/app.css` | **Compilé** — ne jamais le modifier à la main |

> ⚠️ **Ne jamais éditer `frontend/app.css`** : il est écrasé à chaque
> compilation. Toute modification manuelle est perdue au build suivant.

En Docker, la compilation est intégrée : l'image utilise un **build
multi-étapes** (Node n'existe que dans la première étape, jamais dans l'image
finale). `docker compose up --build` suffit donc — rien à installer sur le
serveur.

### Deux invariants à ne pas casser

1. **Jamais de `letter-spacing` sur de l'arabe.** L'espacement casse la liaison
   des lettres (الحروف المتصلة) : « محمد » devient « م ح م د ». Une seule ligne
   de CSS peut rendre le texte illisible.
2. **Jamais d'`@import "tailwindcss"` sans `source(none)`.** Sinon Tailwind
   scanne tout le dossier du projet et le build local ne produit plus le même
   fichier que le build Docker (20,7 Ko contre 13,3 Ko mesurés) — une classe
   pouvait fonctionner en développement puis manquer en production.

## Première installation des modèles (une seule fois)

Les modèles se trouvent dans le volume Docker `ollama`. Deux options :

1. **Téléchargement direct** (nécessite une bonne connexion) :
   le service `ollama-pull` s'en charge automatiquement au premier `docker compose up`.

2. **Copie locale depuis un Ollama natif** (recommandé si la connexion est
   instable) :
   ```powershell
   # Liste des blobs du modèle voulu dans ~/.ollama/models/blobs
   # Copier chaque blob + le manifeste dans le conteneur :
   docker cp "C:\Users\<user>\.ollama\models\blobs\sha256-<digest>" `
     "fastapiproject-ollama-1:/root/.ollama/models/blobs/sha256-<digest>"
   docker cp "C:\Users\<user>\.ollama\models\manifests\registry.ollama.ai\library\llama3.1\latest" `
     "fastapiproject-ollama-1:/root/.ollama/models/manifests/registry.ollama.ai/library/llama3.1/latest"
   ```

## ⚠️ Bug classique : `localhost` dans un conteneur

**Symptôme** : `GET /ask` renvoie `500` avec
`ConnectionError: Failed to connect to Ollama`, alors que tout marche en local.

**Cause** : `rag.py` appelait `ollama.chat(...)` sans préciser d'URL. La
bibliothèque `ollama` utilise alors `localhost:11434` par défaut. Or, dans un
conteneur, `localhost` désigne **le conteneur lui-même**, pas la machine ni les
autres conteneurs. Ollama est joignable via son **nom de service** `ollama`.

**Solution** : utiliser un client configuré avec `config.OLLAMA_URL` :

```python
# app/rag.py
def get_ollama_client():
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = ollama.Client(host=config.OLLAMA_URL)
    return _ollama_client

# puis : get_ollama_client().chat(model=..., messages=...)
```

**Règle à retenir** :
- `localhost` → le conteneur courant.
- `ollama` (nom de service) → un autre conteneur du même réseau Compose.
- `host.docker.internal` → la machine hôte.

## Architecture

```
Navigateur ──► api (FastAPI, port 8000) ──► ollama (port interne 11434)
                     │                            │
                     ▼                            ▼
              volumes ./chroma_db, ./data    volume ollama (modèles)
```
