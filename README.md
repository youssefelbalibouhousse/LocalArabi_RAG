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
├── fidelite.py      # Contrôle qu'une réponse est ancrée dans le contexte
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
├── mesurer_reponses.py      # Mesure des RÉPONSES, avec répétitions
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
chunk le plus proche** :

| | min | médiane | max |
|---|---|---|---|
| 59 questions répondables | 0,26 | 0,38 | **0,52** |
| 24 questions hors corpus | **0,41** | 0,58 | 0,67 |

### ⚠️ La distance ne sépare PAS « hors sujet » de « réponse absente »

Une première série de 15 questions hors corpus donnait des distances **toutes
supérieures** à 0,53, et une population répondable plafonnant à 0,52 : les deux
semblaient séparées, et le seuil `DISTANCE_THRESHOLD` calibrable. **C'était une
illusion due à des questions trop faciles** — astronomie, cuisine, football, très
loin du sujet.

Les 9 questions ajoutées ensuite sont du **même domaine** que le corpus : elles
portent sur des réalités modernes (avion, assurance, monnaie électronique, don
d'organes) que ces ouvrages ne peuvent pas trancher, mais leur vocabulaire est
celui du corpus. Leur distance tombe **en plein milieu** des questions
répondables :

| question (réponse absente du corpus) | distance |
|---|---|
| الصلاة في الطائرة | 0,414 |
| التبرع بالأعضاء | 0,422 |
| حقنة في الوريد | 0,425 |
| بيع الأسهم في البورصة | 0,481 |
| العمل في البنوك | 0,491 |
| التأمين على السيارات | 0,513 |

**Conclusion : la distance mesure la proximité de SUJET, pas la présence d'une
RÉPONSE.** Aucun seuil ne les distingue — le rapport le dit désormais lui-même
(« les deux populations SE RECOUVRENT »). Un seuil assez haut pour laisser passer
ces questions ne protège de rien ; un seuil assez bas pour les écarter
supprimerait 10 vraies réponses sur 59.

C'est la population qui compte pour un pilote : la question a l'air normale, et
c'est précisément là que le système invente le plus.

### Ce que le système fait vraiment : c'est une DISTRIBUTION, pas un verdict

⚠️ **La même question reçoit des réponses contradictoires.** Mesuré sur les 9
questions proches du domaine, deux exécutions chacune
(`scripts/mesurer_reponses.py --repetitions 2`) :

| question | exécution #1 | exécution #2 |
|---|---|---|
| ما حكم استخدام مكبر الصوت في الأذان؟ | « لا بأس به. » | « **يحرم** استخدام مكبر الصوت في الأذان. » |
| هل يفطر الصائم بأخذ حقنة في الوريد؟ | « لا. » | « لا. » |
| ما حكم التبرع بالأعضاء بعد الوفاة؟ | *refus* | « حكمًا شرعيًا يعتمد على الاختصاصات القانونية في الدولة » |
| ما حكم صلاة الجمعة عن بعد في زمن الوباء؟ | « ليس على المسافر… » | « الجمعة جائزة خلف كل إمام… » |
| q075, q076, q077, q078, q083 | *refus* | *refus* |

**Deux fatwas opposées à une minute d'intervalle, sur la même question** — et aucune
des deux n'est dans le corpus. Deux réponses contradictoires ne peuvent pas venir
du même texte : c'est la démonstration la plus directe que quelque chose est
inventé, et elle ne demande aucun juge.

⚠️ **Conséquence de méthode** : les comptes publiés plus haut (12 refus sur 15, puis
4 sur 9) sont **un échantillon chacun**, pas une propriété. Une comparaison
d'invites faite sur une seule exécution de chaque côté n'est pas interprétable —
c'est pourquoi `scripts/mesurer_reponses.py` répète chaque question.

### Trois vérifications déterministes essayées, trois échecs

Toutes visaient le même but : empêcher qu'une réponse soit inventée, sans juge et
sans rappeler le modèle.

**1. Le seuil de distance** — échec, mesuré plus haut : les questions proches du
domaine ont une distance dans la plage des questions répondables.

**2. « La question emploie un mot que le corpus n'a jamais »** — attrape **20/20**
des questions hors corpus, mais refuse à tort **11 questions répondables sur 57**.
Les onze sont de la **morphologie** : `بماذا`, `الراجل`, `افترق`, `كرهها`,
`بنجومه` (bـ + نجوم + ـه), `زواج` là où le corpus écrit `نكاح`, `جواز`, `مخلوقه`.
Le mot est absent comme *chaîne*, sa racine est partout. Le critère est faux, et
il l'est systématiquement.

**3. Le contrôle de fidélité après génération** (`app/fidelite.py`, module pur) —
cherche dans le contexte les éléments de la réponse. Appliqué aux 7 réponses
réelles qui ne sont pas des refus : **0 attrapée par les nombres**, 3 par les mots,
et **3 inchécables** parce qu'elles font un à trois mots (« لا. », « لا بأس به. »).

| réponse | nombres absents | mots absents du contexte |
|---|---|---|
| « لا. » | — | aucun (rien à vérifier) |
| « لا بأس به. » | — | aucun (rien à vérifier) |
| « الجمعة جائزة خلف كل إمام… » | — | aucun |
| « حكمًا شرعيًا يعتمد على الاختصاصات القانونية » | — | 14 |

**La raison de ces trois échecs est la même** : les inventions du modèle sont faites
du **vocabulaire du corpus**. Il en connaît la langue, le style et les tournures.
Ce qui distingue une citation d'une invention n'est pas lexical, c'est **sémantique**
— et aucune vérification de mots ne peut le voir.

### La piste qui reste : rendre la réponse vérifiable par construction

Puisqu'aucun contrôle *a posteriori* ne fonctionne sur du texte libre, il reste à
changer ce qu'on demande au modèle : **une citation verbatim du contexte à l'appui
de chaque affirmation**. Le contrôle devient alors exact — la citation est dans le
contexte, ou elle n'y est pas.

C'est implémenté et mesuré : `CITATIONS_OBLIGATOIRES` (désactivé par défaut)
exige la citation entre `[[ ]]` — un délimiteur absent du corpus, contrairement
aux guillemets arabes « » qui ponctuent les textes édités — et
`fidelite.citations_non_verifiees()` vérifie chaque citation **littéralement**,
après normalisation arabe. Mesuré sur deux populations, deux exécutions chacune :

| | 18 réponses hors corpus | 12 réponses à des questions répondables |
|---|---|---|
| refus | 4 | 0 |
| **citation vérifiée** | 4 | 4 |
| aucune citation | 9 | 5 |
| **citation fabriquée** | 1 | **3** |
| → deviendraient des refus si l'on exigeait la citation | 10 / 18 | **8 / 12** |

**Le mécanisme marche comme détecteur, échoue comme empêchement.**

- Il **démontre** une invention quand le modèle cite : `q083` a cité « إن القاضي
  فإنه يحكم بشيء يجده في ديوانه بخطه » pour une question sur les banques — cette
  phrase n'est pas dans le contexte. Le régime normal ne laissait aucune prise.
- Mais le modèle **n'obéit qu'un tiers du temps** (9 réponses sans citation sur 18,
  5 sur 12), et **fabrique la citation 3 fois sur 12 sur des questions où la
  réponse existe pourtant**. L'invention n'est donc pas empêchée : elle reste
  seulement *détectable quand le modèle choisit de citer*.
- **Exiger la citation n'est pas livrable** : cela refuserait deux tiers des
  questions auxquelles le système sait répondre. Le produit deviendrait inutilisable.

⚠️ Et même une citation **vérifiée** n'est pas une réponse **juste** : les quatre
citations validées sur les questions hors corpus étaient **toutes hors sujet**
(`q075` répond sur les actions en citant une règle sur la dette). La citation rend
une réponse *vérifiable*, pas *correcte*.

### La voie structurelle : ne plus laisser le modèle écrire de texte libre

Les quatre tentatives échouent toutes sur le même mur, et il faut le nommer :
**une génération libre est invérifiable**, parce que le modèle écrit dans la langue
du corpus et que rien de lexical ne distingue ce qu'il a lu de ce qu'il sait.

La voie structurelle était donc la seule qui restait : faire **sélectionner** au
modèle un passage parmi ceux qui ont été récupérés — ou déclarer qu'aucun ne
convient — au lieu de le laisser rédiger. Le texte montré à l'utilisateur est
alors **toujours un passage du corpus**, avec sa référence : l'invention devient
**impossible par construction** plutôt que détectée après coup.

**C'est construit et mesuré** (`ANSWER_MODE`, dont la valeur par défaut reste
`texte` : le régime de production). Trois régimes sont nommés plutôt que décrits
par des booléens, parce que deux drapeaux auraient une combinaison qui ne veut
rien dire — `texte`, `selection`, `refus_puis_selection`.

#### Ce que la construction a coûté à découvrir

Le premier protocole demandait au modèle d'écrire un **numéro**. Il a répondu
`370`, `369`, `1062` — des numéros de **ligne** : le corpus en porte un en tête de
*chaque* ligne (« 369 - وأجمعوا… »). L'invite contenait donc deux numérotations
concurrentes de même forme. Ma conclusion suivante — « il faut des lettres » — était
**fausse**, et la mesure l'a démenti : avec des lettres, le modèle a rendu `أ` et
`د`, c'est-à-dire « حرف المقطع » traduit dans *son* alphabet (أ، ب، ج، د), et
`1062` est revenu. **Changer d'alphabet déplace l'ambiguïté, elle ne disparaît
pas** : la cause est que la sortie est du texte libre.

La correction qui agit sur la cause est de **contraindre la sortie** : le schéma
JSON passé au client (`{"passage": entier 0..n}`) est appliqué *pendant la
génération des tokens*. Un numéro de ligne recopié ne **peut plus** sortir.

| protocole (12 questions × 2) | choix tombant sur la page attendue | illisibles |
|---|---|---|
| numéro, texte libre | 12 puis 11 / 24 | 3 puis 5 |
| lettre, texte libre | 8 / 24 | 6 |
| **JSON contraint** | **15 / 24** | **0** |
| *base « prendre le premier extrait », sans modèle* | *10 / 24* | — |
| *plafond de la récupération (le bon extrait était visible)* | *20 / 24* | — |

⚠️ La base « rang 1 » est indispensable à la lecture : montrer le premier extrait
**sans appeler le modèle** réussit déjà **30 / 59** (50,8 %) sur les 59 répondables.
Un taux de choix justes qui ne la dépasse pas signifie que le modèle coûte 70 s par
question pour faire moins bien que rien.

#### ⚠️ Et ce qu'il NE fait pas : le mécanisme n'empêche pas de désigner un mauvais passage

Mesuré sur les **24 questions sans réponse dans le corpus** (48 essais) :

| | |
|---|---|
| abstentions du modèle (« aucun passage ne répond ») | **4 / 48** |
| passages désignés malgré tout | **44 / 48** |
| choix illisibles | **0 / 48** |

**Sur 20 des 24 questions hors corpus, le modèle n'a jamais renoncé.** L'invention
de *texte* est supprimée ; la *mauvaise réponse* ne l'est pas. Le mécanisme change
donc la nature de l'échec, il ne le supprime pas.

⚠️ **La comparaison avec le texte libre a été faite, et elle est sévère pour la
sélection.** Les deux régimes ont été mesurés sur les **mêmes 24 questions × 2
essais (48)** :

| | texte libre | sélection de passage |
|---|---|---|
| **renonce** | **34 / 48 — 71 %** | **4 / 48 — 8 %** |
| ne renonce pas | 14 / 48 — 29 % | 44 / 48 — 92 % |

⚠️ Le détecteur par phrase n'en comptait que **27** : il a **manqué 7 refus sur 34
(21 %)**, tous des reformulations parfaitement légitimes (« لا أجد الإجابة في
السياق السابق », « لا يوجد صلة للسؤال في السياق », « لا تجد الإجابة في السياق
المذكور »). Le chiffre définitif vient d'une **lecture** des 21 réponses non
comptées — c'est la seule méthode que ce projet ait trouvée fiable, et le détecteur
se trompe encore ici, dans le sens qui fait passer le système pour pire qu'il n'est.

⚠️ **Et le découpage explique l'ancien « 12 / 15 », qui m'avait induit en erreur :**

| sous-population | renonce | invente |
|---|---|---|
| 15 questions éloignées (astronomie, cuisine, football) | 24 / 30 — 80 % | 6 / 30 |
| 9 questions proches du domaine (banque, bourse, organes…) | 10 / 18 — 56 % | **8 / 18 — 44 %** |

Le « 12 / 15 » ne portait que sur les **éloignées**, c'est-à-dire les faciles. Sur
les mêmes questions, le texte libre renonce **71 %**, et **56 % seulement** là où
c'est difficile. Comparer ce chiffre à la sélection mesurée sur les 24 questions
revenait à avantager le texte libre.

Les 14 non-refus sont de la pire espèce, et deux se **contredisent** : à « ما حكم
استخدام مكبر الصوت في الأذان » le modèle a répondu « لا يباح » puis « لا بأس به »,
à une minute d'intervalle. Ailleurs : « عام 1989 » (chute du mur de Berlin), « Au »
(symbole chimique de l'or), « لا. » puis « لا. » à une question sur la piqûre
intraveineuse, et une recette de couscous.

**Conclusion : le texte libre renonce 71 % du temps, la sélection 8 %.** Le premier
sait dire « je ne sais pas », le second ne le dit pas — mais le premier invente du
texte, et le second le rend impossible. C'est ce qui justifie la piste suivante :
**combiner les deux**, la question ouverte décidant s'il faut renoncer, la sélection
servant ensuite à *ancrer* la réponse dans un passage réel. Le coût tombe du bon
côté — un seul appel sur une question sans réponse, deux sur une question où
l'utilisateur attend vraiment quelque chose.

#### Une affirmation retirée du gabarit

Le titre du passage affiché disait « **المقطع الذي يجيب عن السؤال** — le passage
qui répond à la question ». C'est une affirmation que le système n'est pas en
mesure de tenir : dans 44 des 48 cas ci-dessus, elle aurait été **fausse**, sous
une forme que l'utilisateur n'a aucun moyen de contester puisqu'elle vient du
système et non du texte. Il dit maintenant que le passage a été **sélectionné**, ce
qui est vérifiable, et invite à le contrôler : « المقطع المختار من الكتاب (يُرجى
التحقّق منه) ».

#### Un chiffre faux, publié, et corrigé

Le premier rapport annonçait « **0 abstention sur 48** ». C'était mon code qui
mentait : `if not choix:` — `0` étant falsy en Python, l'abstention était rendue
comme un échec de lecture. Il y en avait **4**. L'erreur allait dans le sens le
plus défavorable au mécanisme, et elle était invisible à la lecture du chiffre :
`0` et `None` produisent le **même texte vide**, et ne se distinguent que par le
second élément du triplet retourné. Même famille de piège que `all([])` dans
`est_orpheline`. C'est pourquoi la sortie **brute** du modèle est conservée dans
les rapports — c'est elle qui a permis de recompter sans refaire les 30 minutes de
mesure.

> Le projet le disait depuis le début, pour la qualité des réponses : « mesurer si
> la réponse est bonne demande un juge ». Les trois échecs ci-dessus ne font que
> le confirmer sur ce corpus — et ajoutent une raison de plus de rendre la réponse
> **vérifiable** plutôt que de chercher à la juger automatiquement.

### Ce qui a été corrigé : plus de citation sous un refus

Les sources étaient ajoutées **dès que la récupération rendait quelque chose**
(`app/main.py`) — et sans seuil de distance, elle rend toujours quelque chose :
**24 réponses sur 24 portaient une citation, refus compris**. Le lecteur ne pouvait
donc pas distinguer « ceci vient de la page citée » de « on a collé une source sans
rapport sous un refus ».

`rag.est_un_refus()` reconnaît les formules de refus **réellement observées** et
supprime alors la citation — de la réponse comme du champ `sources` de l'API.

⚠️ Ce garde-fou **n'empêche pas d'inventer** : il enlève une citation trompeuse.

### Une invite durcie a été essayée, puis retirée

Quatre règles numérotées ont été ajoutées, interdisant explicitement d'inférer, de
faire des analogies et d'attribuer des opinions absentes. Elle donnait 1 refus sur
9 là où l'invite précédente en donnait 4 — mais **ces deux mesures étaient des
échantillons uniques** d'un processus très variable (voir les fatwas opposées
ci-dessus). L'expérience est donc consignée dans `app/rag.py` **sans conclusion
tranchée** : elle devra être refaite avec des répétitions.

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
