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

- Le volume **`caddy_data`** (contient les certificats TLS — les perdre force
  une réémission, ce qui est limité en fréquence par Let's Encrypt).
- Le dossier **`data/`** (comptes utilisateurs) et **`chroma_db/`** (base vectorielle).

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
