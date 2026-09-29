# Chatbot RAG arabe

[![CI](https://github.com/youssefelbalibouhousse/LocalArabi_RAG/actions/workflows/ci.yml/badge.svg)](https://github.com/youssefelbalibouhousse/LocalArabi_RAG/actions/workflows/ci.yml)

Chatbot de questions/réponses en arabe sur des documents **PDF et EPUB**, basé sur
**FastAPI**, **ChromaDB** et **Ollama** (embeddings `bge-m3`, génération `llama3.1`).
Chaque réponse cite ses sources : fichier, page et intervalle de lignes.

## Structure

```
app/
├── config.py        # Configuration centralisée (chemins, modèles, URL)
├── rag.py           # Logique RAG : récupération + génération
├── lexical.py       # Recherche lexicale (BM25) et fusion de classements
├── ingest.py        # Ingestion incrémentale et bornée (registre, lots)
├── epub.py          # Extraction des EPUB (corpus Shamela)
├── evaluation.py    # Mesure de la qualité de la récupération (hit@k, MRR)
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
├── build_kb.py              # Ingestion des documents (PDF, EPUB) → base vectorielle
├── benchmark_embeddings.py  # Débit de l'endpoint d'embeddings, avant de payer
├── eval_rag.py              # Évaluation de la récupération (jeu d'or, mesures)
├── backup.py                # Sauvegarde + vérification + restauration
├── schedule_backup.py       # Sauvegarde quotidienne automatique
└── create_user.py           # Création d'un compte en ligne de commande
eval/
├── golden.jsonl         # Jeu d'or : questions de référence (versionné)
└── results/             # Rapports de mesure (non versionnés)
tests/               # Suite pytest
data/documents/      # Corpus versionné : les PDF et EPUB de démonstration
data/shamela/        # Corpus sous droits, IGNORÉ par Git (voir « Corpus »)
chroma_db/           # Base vectorielle générée (non versionnée)
```

## Corpus

Deux formats sont lus, dans deux dossiers (`config.CORPUS_DIRS`) :

| Dossier | Format | Versionné | Pourquoi |
|---|---|---|---|
| `data/documents/` | PDF, EPUB | oui | Corpus de démonstration, public |
| `data/shamela/` | EPUB | **non** | Téléchargé depuis [shamela.ws](https://shamela.ws) : le texte est numérique, mais les éditions modernes restent sous droits d'éditeurs tiers |

Le dossier n'est qu'un rangement : les deux sont indexés de la même façon.

**Pourquoi l'EPUB plutôt qu'un PDF ?** Un PDF océrisé transporte les erreurs de
l'OCR dans sa couche texte, et aucune extraction ne peut les réparer — il n'y a
plus d'image à ré-océriser. Sur le PDF de démonstration, 77 % des chunks
contiennent une forme abîmée du mot central. Un EPUB de Shamela, lui, donne le
texte **numérique** : lettres liées, hamzas correctes, notes de bas de page en
clair — et, détail décisif, **la pagination imprimée**, ce qui permet à la
citation `(fichier, page, lignes)` de désigner une page que le lecteur retrouve
dans son exemplaire.

> ⚠️ Un fichier identifié par son **nom**, deux homonymes dans deux dossiers se
> remplaceraient dans l'index. `build_kb.py` refuse de démarrer dans ce cas.

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

1. **Déposer les documents** (PDF, EPUB) dans `data/documents/` — ou dans
   `data/shamela/` pour les EPUB sous droits (ignoré par Git).
2. **Mettre à jour la base vectorielle** :
   ```bash
   python scripts/build_kb.py                 # n'ingère que ce qui a changé
   python scripts/build_kb.py --status        # registre face à l'index et au corpus
   python scripts/build_kb.py --force         # tout ré-ingérer
   python scripts/build_kb.py --only x.pdf    # un seul document
   python scripts/build_kb.py --forget x.pdf  # en RETIRER un (index + registre)
   ```
   L'ingestion est **incrémentale** : un document dont le contenu n'a pas changé
   est ignoré (comparaison d'empreintes SHA-256, jamais de dates). Elle est aussi
   **non destructive** : seule la part d'un document est remplacée, jamais toute
   la collection.

   **Retirer un document est une action explicite** (`--forget`). Effacer le
   fichier ne suffit pas, et c'est volontaire : un dossier déplacé ou un disque
   non monté effacerait sinon un corpus entier sans que personne ne l'ait
   demandé. Tant que le document reste au registre, `--status` le signale comme
   **orphelin** — ses chunks sont encore servis comme sources d'un fichier que
   plus personne ne peut ouvrir.

   **Sur un long passage** (des centaines de documents, ou un serveur
   d'embeddings distant), deux options évitent de tout recommencer :

   ```bash
   python scripts/build_kb.py --continue-on-error      # ne s'arrête pas au 1er échec
   python scripts/build_kb.py --retries 2              # réessaie les pannes passagères
   ```

   `--continue-on-error` traite les documents suivants malgré les échecs, puis
   les récapitule à la fin — un seul fichier illisible ne condamne pas les 999
   autres. `--retries` réessaie une panne **passagère** (coupure réseau, serveur
   qui redémarre) mais **jamais** une erreur permanente (fichier illisible,
   configuration incohérente) : réessayer n'y changerait rien. Un document en
   échec n'est pas inscrit au registre, donc le passage suivant le reprend.
3. **Lancer l'API** :
   ```bash
   uvicorn app.main:app --reload
   ```
4. **Interroger** : `http://localhost:8000/ask?question=<votre question>`
   ou ouvrir `frontend/index.html`.

## Mesurer le débit d'embeddings

Louer une machine GPU se paie à l'heure : mieux vaut savoir ce qu'elle rend
avant de signer. Ce script mesure le débit de l'endpoint configuré, la dimension
des vecteurs produits, et **vérifie la cohérence** de la configuration.

```bash
python scripts/benchmark_embeddings.py
python scripts/benchmark_embeddings.py --total 390000     # projeter 1 000 livres
python scripts/benchmark_embeddings.py --taille-lot 64    # éprouver un autre lot
```

Un lot part en **une seule** requête d'embedding : si son temps de calcul
dépasse `EMBEDDING_TIMEOUT`, l'écriture échoue sur un « timed out in add » qui
ne dit ni le lot, ni le délai, ni la cause. Le script refuse une marge faible et
sort en code 1 — utilisable dans un script d'automatisation.

Valeurs mesurées ici, sur un i7-1165G7 **sans GPU dédié** (`bge-m3`, chunks de
~520 caractères, ~2,2 chunk/s) :

| Corpus | Chunks | Durée mesurée |
|---|---|---|
| 1 livre | 390 | 3 min |
| 100 livres | 39 000 | 4,8 h |
| 1 000 livres | 390 000 | 2 jours |
| 8 000 livres | 3,1 M | 16 jours |

C'est ce chiffre — pas une intuition — qui décide d'une location de GPU.

## Exemple de réponse

```json
{
  "question": "...",
  "answer": "... الإجابة ...\n\n📄 المصادر: الإجماع لابن المنذر — صفحة 81 (الأسطر 1-5)",
  "sources": [
    {
      "source": "12445.epub",
      "title": "الإجماع لابن المنذر ت فؤاد ط المسلم",
      "page": 81,
      "line_start": 1,
      "line_end": 5
    }
  ],
  "context_used": ["..."]
}
```

> 💡 `source` est le **nom de fichier** : la clé d'identification, celle que
> connaissent le registre, l'index et le jeu d'or. `title` est ce qui est
> **montré** — la seule mention qui permette au lecteur de retrouver la page
> dans son exemplaire. La citation nomme aussi l'**édition** : la pagination
> appartient à cette impression-là, pas à l'ouvrage en général.

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

## Évaluation de la qualité des réponses

Mesurer si la réponse est *bonne* demande un juge. Mesurer si la
**récupération** est bonne, non : il suffit de questions dont on connaît la
source. C'est ce que fait ce harnais, et sans lui tout réglage (taille de
chunk, chevauchement, modèle d'embedding, reranker, seuil de distance) se
décide à l'intuition.

```bash
# 1. Afficher des extraits au hasard pour écrire des questions
python scripts/eval_rag.py --sample 10

# 2. Écrire les questions dans eval/golden.jsonl, puis mesurer
python scripts/eval_rag.py --run --label baseline

# 3. Comparer deux mesures (ex. avant / après ajout d'un reranker)
python scripts/eval_rag.py --compare baseline avec-reranker
```

Deux métriques complémentaires :

| Métrique | Question à laquelle elle répond |
|---|---|
| `hit@k` | A-t-on trouvé le bon passage dans les *k* premiers ? |
| `MRR@k` | L'a-t-on bien **classé** ? (le rang 1 ne vaut pas le rang 5) |

`hit@k` seule ne suffit pas : un bon extrait au rang 5 est compté comme celui du
rang 1, alors qu'il est plus souvent ignoré par le modèle. Un **reranker**
améliore rarement `hit@k` — il ne peut pas trouver ce que la recherche a raté —
mais presque toujours le MRR.

Chaque rapport enregistre le modèle d'embedding, la taille de chunk, la taille
du corpus et une **empreinte du corpus**. La comparaison vous avertit si les deux
mesures ne portent pas sur le même corpus : sans cet avertissement, on
attribuerait au réglage testé l'effet d'un simple changement de documents.

Le jeu d'or (`eval/golden.jsonl`) est **versionné** — c'est un actif de test, et
il s'enrichit à chaque question que vos testeurs posent. Les rapports
(`eval/results/`) ne le sont pas : ils contiennent des extraits du corpus et
sont régénérables.

Le jeu d'or **grandit avec le corpus** : quand vous ajoutez des documents, vous
ajoutez des questions — mais vous ne supprimez pas les anciennes, qui restent
valables tant que leurs documents restent indexés. Si un document est retiré, les
questions qui le visent deviennent impossibles à satisfaire : elles sont alors
**exclues du calcul** et signalées dans le rapport, au lieu d'être comptées
« introuvables » — ce qui ferait chuter le score à cause d'un document retiré, et
non de la qualité de la récupération.

> 🎯 Une question **non retrouvée** est plus instructive qu'un score : regardez
> si le bon passage est absent du top-*k* (problème de découpage ou de modèle)
> ou seulement mal classé (problème que le reranker résout).

### Questions hors corpus : la bonne réponse est « il n'y a rien »

Un jeu d'or ne contient pas que des questions auxquelles le corpus répond. Les
plus révélatrices sont celles auxquelles il **ne peut pas** répondre — c'est
exactement ce qu'un testeur essaie en premier. Elles s'écrivent avec une liste
`expected` **vide** :

```json
{"id": "q073", "question": "Quelle est la recette de la tarte tatin ?",
 "lang": "fr", "status": "validated", "expected": [], "notes": "Hors corpus : cuisine."}
```

`expected: []` est un **choix** ; une clé `expected` **absente** reste une erreur.
Les confondre ferait passer une question mal saisie pour un exercice de refus, et
elle serait comptée réussie quel que soit son résultat.

Les deux populations ne se mesurent pas de la même façon, et le rapport les
sépare. Une question hors corpus **n'a pas de rang** : la récupération rend
toujours *k* chunks, même hors sujet. Ce qui se mesure, c'est la **distance du
chunk le plus proche** — et c'est elle qui dit si le seuil de distance de
l'application est réglable :

| | min | médiane | max |
|---|---|---|---|
| 59 questions répondables | 0,26 | 0,38 | **0,52** |
| 15 questions hors corpus | **0,54** | 0,61 | 0,67 |

**Les deux populations ne se recouvrent pas** sur cette mesure — le seuil
(`DISTANCE_THRESHOLD`, aujourd'hui désactivé) serait donc calibrable autour de
0,53. ⚠️ Mais la marge est de **0,02**, et elle repose sur deux extrêmes, les
statistiques les moins stables qui soient : 74 questions ne suffisent pas à
adopter ce seuil. À vérifier sur un jeu hors corpus plus large avant d'y toucher.

### Le refus ne se mesure pas par une phrase

Les 15 questions hors corpus ont été passées dans le chemin de production réel
(récupération → génération → assemblage). Résultat lu et classé à la main :

| | nombre |
|---|---|
| refus corrects | **12 / 15** |
| **fabrications** | **3 / 15** |

Les trois fabrications sont du pire type : à « en quelle année est tombé le mur
de Berlin ? » le système répond « **1989** », et à « qui a gagné la Coupe du monde
1998 ? » il répond « **la France** » — deux faits exacts, tirés de la mémoire du
modèle et non du corpus, puis décorés d'une citation vers une page qui parle d'un
sultan ottoman.

> ⚠️ **Un détecteur automatique de refus s'est trompé, et c'est le résultat le
> plus utile de cette mesure.** En cherchant la phrase exacte du prompt, il
> comptait **5 refus sur 15** là où il y en a 12 : il manquait les reformulations
> arabes (« لا توجد الإجابة في السياق المستخرج ») et butait sur une apostrophe
> en français. **Un taux de refus mesuré par correspondance de phrase est faux**,
> et il l'est dans le sens rassurant — il fait croire au pire. Cette mesure
> demande un juge, ou une lecture.

⚠️ **Structurel, et indépendant du modèle** : les sources sont ajoutées à la
réponse **dès que la récupération a rendu quelque chose** (`app/main.py`). Comme
`DISTANCE_THRESHOLD` est désactivé, elle rend toujours quelque chose : **15
réponses sur 15 portaient une citation**, refus compris. L'utilisateur ne peut
donc pas distinguer « ceci vient de la page citée » de « le système a refusé et on
a collé des sources sans rapport dessous ». C'est le point à corriger en priorité
pour un pilote — et la séparation des distances ci-dessus montre qu'on a de quoi
le faire.

## Recherche hybride (lexicale + vectorielle)

Chercher par le **sens** ne suffit pas sur ce corpus. Dans `تفسير ابن المنذر`,
**85 % des chunks** sont des chaînes de transmetteurs (« حدّثنا فلان عن فلان ») :
ils se ressemblent tous, et un vecteur les confond. Le symptôme est net — quatre
questions du jeu d'or **citent un verset mot pour mot** ; l'index contenait ce
verset, **à la page attendue**, et la recherche vectorielle ne l'a jamais
renvoyé. La recherche lexicale, elle, le trouve au **rang 1**.

Les deux recherches échouent différemment — BM25 ne comprend pas les
paraphrases, le vecteur noie les termes rares — et c'est exactement pourquoi il
faut les **fusionner** plutôt que d'en choisir une :

| recherche | `hit@10` | `MRR@10` | introuvables |
|---|---|---|---|
| vectorielle seule | 78,0 % | 0,52 | 13 |
| **fusion (RRF)** | **98,3 %** | **0,69** | **1** |

59 questions, 3 355 chunks, 3 ouvrages, jeu d'or **relu et validé** — **26
questions mieux classées, 3 moins bien, 30 inchangées**.

> ⏱️ **La latence n'est pas un critère ici, et mieux vaut le dire.** Les deux
> configurations mesurent entre 314 et 384 ms selon l'exécution. La MÊME
> configuration mesurée deux fois de suite a varié de **60 ms** (384 → 324 ms)
> alors que ses scores étaient identiques au point près. À cette échelle, comparer
> la latence de deux réglages, c'est mesurer du bruit. Ce qui est solide : la
> moitié lexicale ne parcourt pas le corpus — l'index inversé ne note que les
> documents contenant les mots de la question.

```bash
python scripts/eval_rag.py --run --label hybride      # fusion (défaut)
HYBRID_ENABLED=false python scripts/eval_rag.py --run --label vectoriel
python scripts/eval_rag.py --compare vectoriel hybride
```

Réglages (`.env`) : `HYBRID_ENABLED`, `HYBRID_CANDIDATES` (candidats demandés à
chaque recherche avant fusion), `HYBRID_RRF_K` (constante d'aplatissement),
`HYBRID_VECTOR_WEIGHT` / `HYBRID_LEXICAL_WEIGHT`.

> ⚠️ **Pourquoi fusionner des RANGS et non des scores ?** Un score BM25 et une
> distance vectorielle n'ont aucune unité commune : les additionner serait une
> faute de dimension. Le rang, lui, est toujours un entier de 1 à N.

### Le réglage n'est pas un détail : une erreur de conception, trouvée et corrigée

La première version fusionnait 50 candidats par recherche avec un poids de 3 pour
le vecteur. Le rang 1 lexical ne valait alors que $1/(60+1) = 0{,}0164$, quand le
**50e** candidat vectoriel valait déjà $3/(60+50) = 0{,}0273$ : la recherche
lexicale **ne pouvait que reclasser ce que le vecteur avait déjà vu**, jamais
introduire ce qu'il avait manqué. Quatre questions dont la page attendue était
classée **1re sur 3 355** par BM25 restaient introuvables.

Le symptôme était invisible dans le score global : la fusion gagnait bien
+13 points. C'est en cherchant *pourquoi* quatre questions précises échouaient
qu'on a vu que l'explication annoncée était fausse.

Grille mesurée ensuite, à poids égaux :

| constante | `hit@5` | `hit@10` | MRR |
|---|---|---|---|
| 5 | 89,8 % | 96,6 % | 0,686 |
| **10** | **89,8 %** | **98,3 %** | **0,687** |
| 20 | 86,4 % | 98,3 % | 0,667 |
| 60 | 84,7 % | 94,9 % | 0,648 |

`hit@5` est ce que le **modèle** voit (`N_RESULTS=5`) ; `hit@10` ce que le harnais
évalue. La constante 10 est le seul point qui satisfasse les deux — et le résultat
ne dépend **pas** de la troncature (50, 100 et 200 candidats donnent la même
chose), ce qui est la meilleure garantie contre un réglage qui ne vaudrait que
pour ces 59 questions.

> 💡 **Les poids sont égaux (1:1), et ce n'est pas un hasard.** Toutes les
> configurations pondérées mesurées sont erratiques, là où toutes les
> configurations à poids égaux se tiennent entre 94,9 % et 98,3 %. La mise en
> forme standard de RRF est symétrique : introduire un poids revient à décider à
> l'avance quelle recherche a raison.

Et il y avait une raison cachée à cette erreur de poids. Le premier calibrage
concluait « 3 pour le vecteur » sur un lexical **handicapé par le bug de
ponctuation** : mesuré sur le même jeu d'or, le lexical seul donne **74,6 %** de
`hit@10` avec la plage `\u0600-\u06FF`, et **88,1 %** avec `\w`. Le bug coûtait
13,5 points à la recherche lexicale — et c'est une recherche amoindrie que les
poids arbitraient. **Un défaut de mesure dans un composant fausse le réglage d'un
autre**, sans que rien ne le signale.

Deux enseignements, tous deux mesurés :

| tokenisation | lexical seul `hit@10` | MRR |
|---|---|---|
| plage `\u0600-\u06FF` (le `؟` collé au mot) | 74,6 % | 0,562 |
| `\w` | **88,1 %** | **0,686** |

Sur ce corpus, la recherche **lexicale seule bat la recherche vectorielle seule**
(88,1 % contre 78,0 %) — ce qui ne se voyait pas tant que la tokenisation était
fausse.

⚠️ **L'index lexical est un cache dérivé de ChromaDB**, pas une seconde source de
vérité — il est donc impossible qu'il « mente » comme a pu le faire le registre.
Il est reconstruit quand le **nombre** de chunks change. Deux conséquences :

- **Après une ré-ingestion, redémarrez l'API** (ou appelez
  `rag.reinitialiser_index_lexical()`) : modifier le contenu d'un document sans
  changer le nombre de chunks ne déclenche pas la reconstruction.
- La mémoire croît avec le corpus (négligeable à 3 355 chunks — quelques Mo ;
  à revoir vers plusieurs millions, où un index persistant prendrait le relais :
  `SQLite FTS5` fournit un BM25 classé sans limite de mémoire).

## Choix techniques

| Sujet | Décision |
|---|---|
| Configuration | `app/config.py` est la **source unique de vérité**, surchargeable par variables d'environnement |
| Recherche | **Hybride** : recherche vectorielle ET lexicale (BM25), fusionnées par rangs réciproques. Mesuré sur un jeu d'or validé : `hit@10` 78,0 % → **98,3 %**. Voir « Recherche hybride » ci-dessus |
| Ingestion | **Incrémentale et non destructive** : empreinte SHA-256 du contenu, remplacement par document, écriture par lots bornés. Registre dans `data/app.db`. Un index refuse de mélanger deux modèles d'embedding. |
| Retrait | **Explicite** (`--forget`) et jamais automatique : un fichier disparu du disque n'est pas retiré pour autant. La purge efface le registre **avant** l'index, comme l'ingestion, pour qu'une panne laisse le document repris plutôt qu'inscrit à tort. |
| Passages longs | `--continue-on-error` poursuit malgré les échecs et les récapitule ; `--retries` réessaie les pannes passagères (réseau, redémarrage), jamais les erreurs permanentes |
| Formats | PDF (`pypdf`) et EPUB (`app/epub.py`), ramenés à la **même forme** en sortie : `chunk_pages` et toute la chaîne traitent les deux sans distinction |
| Taille des lots | `INGEST_BATCH_SIZE` (32 par défaut) et `EMBEDDING_TIMEOUT` (120 s) se lisent **ensemble** : un lot est embarqué en une seule requête. Mesuré ici : ~0,55 s par chunk, donc 256 chunks dépassaient le délai de 60 s de la bibliothèque — l'écriture échouait en « timed out in add » |
| Réglage de la fusion | `HYBRID_RRF_K` et les poids **ne sont pas indépendants** de `HYBRID_CANDIDATES` : une constante trop grande devant le bassin empêche la recherche lexicale d'introduire un extrait que le vecteur a manqué. Mesuré, et documenté dans `config.py` |
| Fournisseur LLM | `LLM_PROVIDER=ollama` (auto-hébergé) ou `openai` (Groq, Together, vLLM…) |
| Inscription | Ouverte en développement, **fermée par défaut en production** |
| Clé JWT | Minimum 32 octets (RFC 7518) ; l'application refuse de démarrer en production avec la clé de développement |
| Langue de la réponse | Consigne placée **en fin d'invite**, puis langue **réellement produite** vérifiée (`app/language.py`, comptage d'alphabet) et réécrite si besoin — la langue n'est pas laissée au bon vouloir du modèle |
| Citations | L'ouvrage est nommé par son **titre** (`rag.etiquette_source`), jamais par son nom de fichier : « 12445.epub » ne désigne rien pour un lecteur. Un même ouvrage cité d'affilée n'est nommé qu'**une fois**. Le nom de fichier reste la clé d'identification (registre, index, jeu d'or) — ajouter ou renommer des ouvrages n'invalide donc aucune question du jeu d'or |
| Sauvegardes | Archives horodatées + empreintes SHA-256 ; copie SQLite via l'API de sauvegarde (pas de « torn copy ») |
