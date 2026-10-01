"""Logique RAG réutilisable : connexion à la base, récupération, génération.

Partagée entre l'API (app.main) et le script d'ingestion (scripts.build_kb),
pour éviter toute duplication de configuration.
"""

import json
import logging
import re
from collections.abc import Mapping

import chromadb
import ollama
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)

from app import config
from app.language import detect_language
from app.lexical import IndexLexical, fusionner_rrf, normaliser

logger = logging.getLogger(__name__)

# Connexions mises en cache pour ne pas recréer un client/une collection à
# chaque requête, et pour ne pas charger Ollama au démarrage (chargement
# paresseux : la collection n'est créée qu'au premier appel).
_client = None
_collection = None
_ollama_client = None
_openai_client = None

# --- Recherche hybride : index lexical -----------------------------------
#
# L'index lexical est DÉRIVÉ de l'index vectoriel : les mêmes chunks, lus dans
# ChromaDB. Ce n'est pas une seconde source de vérité, seulement un cache —
# c'est ce qui le rend sûr. Une source de vérité peut mentir (ce projet en a
# fait l'expérience avec le registre) ; un cache, lui, se reconstruit.
#
# ⚠️ LIMITE CONNUE — la reconstruction est déclenchée par un changement du
# NOMBRE de chunks. Modifier le CONTENU d'un document sans changer ce nombre
# (ce que `build_kb.py --force` peut produire) laisserait l'index lexical sur
# l'ancien texte : les deux moitiés de la recherche ne décriraient plus le même
# corpus. Remède : redémarrer l'API après une ré-ingestion, ou appeler
# `reinitialiser_index_lexical()`. Documenté dans `README.md`.
_index_lexical: IndexLexical | None = None
_etat_index_lexical: tuple[str | None, int] | None = None
_extraits: dict[str, tuple[str, Mapping]] = {}


def reinitialiser_index_lexical() -> None:
    """Oublie l'index lexical : il sera reconstruit au prochain appel.

    Utile après une ré-ingestion, ou entre deux tests qui se succèdent sur des
    corpus différents.
    """
    global _index_lexical, _etat_index_lexical
    _index_lexical = None
    _etat_index_lexical = None
    _extraits.clear()


def _index_lexical_a_jour(collection) -> IndexLexical:
    """Retourne l'index lexical des chunks de la collection, en le construisant au besoin.

    Coût assumé : la première requête qui suit un démarrage (ou un changement
    de corpus) lit toute la collection en mémoire. Les suivantes ne paient
    rien. Pour 3 355 chunks c'est quelques dixièmes de seconde ; pour un corpus
    de plusieurs millions de chunks, il faudra un index persistant à la place
    (voir la note de passage à l'échelle dans `README.md`).

    La garde est le COUPLE (nom de collection, nombre de chunks) : une
    ingestion ajoute ou retire des chunks, le compte change, l'index est refait.
    """
    global _index_lexical, _etat_index_lexical

    etat = (getattr(collection, "name", None), collection.count())
    if _index_lexical is not None and _etat_index_lexical == etat:
        return _index_lexical

    resultat = collection.get(include=["documents", "metadatas"])
    _extraits.clear()
    for identifiant, document, meta in zip(
        resultat["ids"], resultat["documents"], resultat["metadatas"], strict=True
    ):
        _extraits[identifiant] = (document or "", meta or {})

    _index_lexical = IndexLexical(
        (identifiant, extrait[0]) for identifiant, extrait in _extraits.items()
    )
    _etat_index_lexical = etat
    logger.info("Index lexical construit : %d chunks.", _index_lexical.taille)
    return _index_lexical


def get_embedding_function():
    """Fonction d'embedding bge-m3 servie par Ollama.

    Le délai d'expiration est passé EXPLICITEMENT : celui de la bibliothèque
    (60 s) est trop court pour un lot d'ingestion complet, et l'écriture
    échouait alors sur un « timed out in add » difficile à relier à sa cause.
    Voir `config.EMBEDDING_TIMEOUT`.
    """
    return OllamaEmbeddingFunction(
        model_name=config.EMBEDDING_MODEL,
        url=config.OLLAMA_URL,
        timeout=config.EMBEDDING_TIMEOUT,
    )


def get_client():
    """Retourne le client ChromaDB persistant (créé une seule fois)."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
    return _client


def get_collection():
    """Retourne la collection ChromaDB (créée si absente, une seule fois)."""
    global _collection
    if _collection is None:
        _collection = get_client().get_or_create_collection(
            name=config.COLLECTION_NAME,
            embedding_function=get_embedding_function(),
        )
    return _collection


def get_ollama_client():
    """Retourne un client Ollama pointant vers l'URL configurée (une seule fois)."""
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = ollama.Client(host=config.OLLAMA_URL)
    return _ollama_client


def get_openai_client():
    """Retourne un client pour toute API compatible OpenAI (une seule fois).

    L'import est paresseux : le paquet `openai` n'est chargé que si ce
    fournisseur est réellement utilisé (inutile en mode Ollama).
    """
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI

        _openai_client = OpenAI(
            base_url=config.LLM_BASE_URL or None,
            # Certains serveurs locaux (vLLM, LM Studio) n'exigent pas de clé,
            # mais le SDK refuse une valeur vide.
            api_key=config.LLM_API_KEY or "not-needed",
        )
    return _openai_client


def _source_de(meta: Mapping) -> dict:
    """La description d'un extrait, telle qu'elle remonte jusqu'à la citation."""
    return {
        # `source` (le nom de fichier) reste la CLÉ : c'est lui que le registre,
        # l'index et le jeu d'or connaissent. `title` n'est là que pour
        # l'AFFICHAGE, et il peut manquer — un PDF n'en apporte pas.
        "source": meta.get("source"),
        "title": meta.get("title"),
        "page": meta.get("page"),
        "line_start": meta.get("line_start"),
        "line_end": meta.get("line_end"),
    }


def retrieve(collection, question, n_results=None):
    """Récupère les chunks les plus proches de la question.

    Retourne (docs, sources) où chaque source contient le fichier d'origine,
    son titre, la page et l'intervalle de lignes. Le titre sert à NOMMER
    l'ouvrage dans la citation (voir `etiquette_source`) ; le nom de fichier
    reste la clé d'identification.

    Deux recherches sont menées puis FUSIONNÉES quand `config.HYBRID_ENABLED`
    est vrai (voir `app/lexical.py` pour les mesures qui le justifient).
    """
    if n_results is None:
        n_results = config.N_RESULTS

    if not config.HYBRID_ENABLED:
        return _retrieve_vectoriel(collection, question, n_results)
    return _retrieve_hybride(collection, question, n_results)


def _retrieve_vectoriel(collection, question, n_results):
    """Recherche vectorielle seule — le comportement d'avant la fusion."""
    results = collection.query(
        query_texts=[question],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    docs = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # Filtre de pertinence : on ignore les chunks dont la distance dépasse le
    # seuil configuré (désactivé si DISTANCE_THRESHOLD <= 0).
    threshold = config.DISTANCE_THRESHOLD
    filtered_docs, sources = [], []

    for doc, meta, distance in zip(docs, metadatas, distances, strict=True):
        if not meta:
            continue
        if threshold > 0 and distance is not None and distance > threshold:
            continue
        filtered_docs.append(doc)
        sources.append(_source_de(meta))

    return filtered_docs, sources


def _retrieve_hybride(collection, question, n_results):
    """Recherche vectorielle ET lexicale, fusionnées par rangs réciproques.

    Le bassin de candidats est plus large que ce qu'on rendra : la fusion ne
    peut pas classer ce qu'on ne lui a pas donné. Rendre les `n_results`
    derniers après avoir demandé 50 candidats à chaque recherche, c'est laisser
    chacune des deux retrouver le passage qu'elle seule connaît.

    ⚠️ `DISTANCE_THRESHOLD` ne s'applique qu'aux candidats VECTORIELS, avant la
    fusion : un extrait écarté pour distance peut revenir si la recherche
    lexicale le place bien. C'est cohérent (le seuil juge une distance, et BM25
    n'en produit pas), mais cela mérite d'être su avant de l'activer.
    """
    bassin = max(config.HYBRID_CANDIDATES, n_results)

    results = collection.query(
        query_texts=[question],
        n_results=bassin,
        include=["documents", "metadatas", "distances"],
    )

    threshold = config.DISTANCE_THRESHOLD
    identifiants_vectoriels = []
    for doc, meta, identifiant, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["ids"][0],
        results["distances"][0],
        strict=True,
    ):
        if not meta:
            continue
        if threshold > 0 and distance is not None and distance > threshold:
            continue
        identifiants_vectoriels.append(identifiant)
        # Les extraits vus par la recherche vectorielle sont mémorisés : si
        # l'index lexical date d'avant une ingestion, la fusion peut désigner un
        # identifiant qu'il ne connaît pas encore.
        _extraits.setdefault(identifiant, (doc or "", meta))

    index = _index_lexical_a_jour(collection)
    identifiants_lexicaux = index.classer(question, bassin)

    ordre = fusionner_rrf(
        [identifiants_vectoriels, identifiants_lexicaux],
        [config.HYBRID_VECTOR_WEIGHT, config.HYBRID_LEXICAL_WEIGHT],
        config.HYBRID_RRF_K,
        n_results,
    )

    docs, sources = [], []
    for identifiant in ordre:
        extrait = _extraits.get(identifiant)
        if extrait is None or not extrait[1]:
            continue
        docs.append(extrait[0])
        sources.append(_source_de(extrait[1]))

    return docs, sources


def _invite_citations(question, context, language):
    """Invite qui EXIGE une citation verbatim du contexte.

    Variante expérimentale (voir `config.CITATIONS_OBLIGATOIRES`). Le délimiteur
    `[[ ]]` est choisi parce qu'il est facile à taper et ABSENT du corpus : les
    guillemets arabes « » y ponctuent les textes édités, et ne permettraient pas
    de distinguer la citation du commentaire.

    ⚠️ La consigne de langue reste EN DERNIER, comme dans l'invite normale : c'est
    l'invariant du projet, et il vaut pour toutes les variantes.
    """
    if language == "fr":
        return f"""Utilise uniquement le contexte ci-dessous pour répondre.

Contexte extrait :
{context}

Question : {question}

Règle : chaque affirmation doit s'appuyer sur un passage RECOPIÉ LITTÉRALEMENT du contexte, placé entre [[ et ]] — par exemple [[le passage recopié]]. Ne reformule pas ce passage.

Si le contexte ne contient aucun passage qui appuie la réponse, écris uniquement cette phrase : « Désolé, il n'y a pas assez d'informations dans les documents fournis. »

Consigne de langue, impérative : les extraits ci-dessus sont en arabe, mais tu dois rédiger ta réponse UNIQUEMENT EN FRANÇAIS. N'écris aucune phrase en arabe.

Réponse en français :"""

    return f"""استخدم السياق أدناه فقط للإجابة على السؤال.

السياق المستخرج:
{context}

السؤال: {question}

قاعدة إلزامية: كل حكم تذكره يجب أن يستند إلى نص منقول حرفيًا من السياق، موضوع بين [[ و ]]، هكذا: [[النص المنقول]]. انقل النص كما هو دون إعادة صياغة.

إذا لم تجد في السياق أي نص يدعم الجواب، فاكتب هذه الجملة وحدها: "عذرًا، لا توجد معلومات كافية في الوثائق المرفقة"

تنبيه إلزامي: يجب أن تكتب إجابتك باللغة العربية فقط، ولا تكتب أي جملة بلغة أخرى.

الإجابة بالعربية:"""


# Étiquettes des passages soumis au modèle. Des CHIFFRES par défaut.
#
# ⚠️ MESURÉ, et l'enseignement est l'inverse de l'intuition. Le corpus porte un
# numéro de ligne en tête de CHAQUE ligne (« 369 - وأجمعوا… ») et des marqueurs de
# note (« (1) ») : avec des étiquettes chiffrées, le modèle a répondu « 370 »,
# « 369 », « 1062 » — des numéros de LIGNE — ou « 6 » pour cinq passages. Passer à
# des LETTRES (zéro lettre latine sur 60 extraits examinés) n'a PAS suffi : le
# modèle a rendu « أ » et « د », c'est-à-dire « حرف المقطع » traduit dans SON
# alphabet (أ، ب، ج، د), et « 1062 » est resté. Sur 24 essais : 12 puis 11 choix
# justes avec des chiffres, 8 avec des lettres (base « premier extrait » = 10).
#
# **Changer d'alphabet déplace l'ambiguïté, elle ne la supprime pas** : la cause
# est que la sortie est du TEXTE LIBRE. On garde donc les chiffres et on contraint
# la SORTIE (`config.SELECTION_CHOIX`).
ETIQUETTES = "ABCDEFGH"


def etiquettes_selection(nombre: int, lettres=None) -> list[str]:
    """Les étiquettes des ``nombre`` passages.

    ``lettres`` force le jeu d'étiquettes ; ``None`` suit
    ``config.SELECTION_ETIQUETTES``. ⚠️ Ce paramètre doit être HONORÉ partout où
    l'on construit une correspondance étiquette → rang : sinon `lire_choix`
    chercherait « C » dans un jeu d'étiquettes chiffrées et ne lirait plus rien.
    """
    if lettres is None:
        lettres = config.SELECTION_ETIQUETTES == "lettres"
    if lettres:
        return list(ETIQUETTES[:nombre])
    return [str(rang) for rang in range(1, nombre + 1)]


def schema_choix(nombre_de_passages: int) -> dict:
    """Schéma JSON de la réponse : un entier entre 0 et le nombre de passages.

    C'est ce qui rend un choix ILLISIBLE impossible plutôt que détecté. Un
    numéro de ligne recopié (« 1062 ») ne peut pas être produit : la contrainte
    porte sur la génération des tokens, pas sur une vérification après coup.
    ``0`` reste permis, et c'est l'abstention.
    """
    return {
        "type": "object",
        "properties": {
            "passage": {
                "type": "integer",
                "minimum": 0,
                "maximum": nombre_de_passages,
            }
        },
        "required": ["passage"],
    }


def numeroter_passages(documents, etiquettes=None, lettres=None):
    """Étiquette les passages soumis au modèle : ``[A] …``, ``[B] …``

    L'étiquetage est ce qui permet au modèle de RÉPONDRE PAR UNE ÉTIQUETTE au lieu
    de rédiger : c'est le seul format dont la sortie se vérifie sans rien
    comprendre au sens.

    ``etiquettes`` impose la liste exacte ; ``lettres`` choisit le jeu d'étiquettes
    sans avoir à le construire. Les deux sont transmis à
    ``etiquettes_selection``, qui reste le seul endroit qui décide.
    """
    if etiquettes is None:
        etiquettes = etiquettes_selection(len(documents), lettres)
    return "\n\n".join(
        f"[{etiquette}] {texte}"
        for etiquette, texte in zip(etiquettes, documents, strict=True)
    )


def construire_invite_selection(question, contexte, language="ar", lettres=None, json_format=None):
    """Invite qui demande de CHOISIR un passage, et non de rédiger.

    La tâche est fermée — rendre une étiquette — ce qui laisse beaucoup moins de
    liberté qu'une réponse libre. Un modèle de 8 milliards de paramètres sait mal
    s'abstenir quand on lui demande de rédiger ; il s'abstient mieux quand on lui
    demande de choisir entre des étiquettes.

    ⚠️ La consigne « ne recopie rien du texte » n'est PAS décorative : sans elle,
    le modèle rend un numéro de LIGNE du corpus (« 1062 ») au lieu d'un numéro de
    passage, et la réponse est illisible.
    """
    if lettres is None:
        lettres = config.SELECTION_ETIQUETTES == "lettres"
    if json_format is None:
        json_format = config.SELECTION_CHOIX == "json"
    quoi = ("حرف المقطع" if lettres else "رقم المقطع") if language == "ar" else (
        "la LETTRE" if lettres else "le numéro"
    )

    if json_format:
        # La contrainte de sortie fait le travail ; l'invite ne fait que nommer
        # la forme attendue. Sans elle, le modèle commente ou recopie.
        if language == "fr":
            return f"""Lis les passages étiquetés ci-dessous.

Passages :
{contexte}

Question : {question}

Réponds UNIQUEMENT par un objet JSON : {{"passage": N}}, où N est {quoi} du passage qui répond DIRECTEMENT à la question, ou 0 si aucun passage ne répond. Ne recopie rien du texte et n'ajoute aucun commentaire.

JSON :"""
        return f"""اقرأ المقاطع الموسومة أدناه.

المقاطع:
{contexte}

السؤال: {question}

أجب بصيغة JSON فقط: {{"passage": N}}
حيث N هو {quoi} الذي يجيب عن السؤال إجابة مباشرة، أو 0 إذا لم يجب أي مقطع. لا تنسخ شيئًا من النص ولا تضف شرحًا.

JSON:"""

    if language == "fr":
        return f"""Lis les passages étiquetés ci-dessous, puis réponds par une seule étiquette.

Passages :
{contexte}

Question : {question}

Écris {quoi} du passage qui répond DIRECTEMENT à la question. Écris cette étiquette seule, sans aucun autre mot, et ne recopie rien du texte. Si aucun passage ne répond à la question, écris : 0

Étiquette :"""

    return f"""اقرأ المقاطع الموسومة أدناه، ثم أجب بوسم واحد فقط.

المقاطع:
{contexte}

السؤال: {question}

اكتب {quoi} الذي يجيب عن السؤال إجابة مباشرة. اكتب الوسم وحده، دون أي كلمة أخرى، ولا تنسخ شيئًا من النص. وإذا لم يجب أي مقطع عن السؤال، اكتب: 0

الوسم:"""


def lire_choix(
    reponse: str,
    nombre_de_passages: int,
    lettres=None,
    json_format=None,
) -> int | None:
    """Le passage choisi par le modèle, ou ``None`` si la réponse est inutilisable.

    Renvoie ``0`` quand le modèle déclare qu'aucun passage ne répond — ce qui
    n'est PAS un échec de lecture, mais une réponse : c'est la bonne conduite sur
    une question que le corpus ne peut pas trancher.

    Renvoie ``None`` quand la réponse ne contient aucune étiquette exploitable, ou
    quand elle sort des bornes. On ne DEVINE pas : un choix illisible vaut un
    refus, et il doit être mesurable comme tel. Mesuré : « 6 » pour cinq passages
    reste illisible — il n'y a pas de sixième passage, donc il n'y a pas de choix.

    On lit le DERNIER jeton utile : un modèle qui commente avant de conclure écrit
    sa conclusion à la fin. Un jeton PRÉCÉDÉ D'UN SIGNE n'en est pas un : « -2 »
    ne vaut pas 2, il vaut illisible.
    """
    if lettres is None:
        lettres = config.SELECTION_ETIQUETTES == "lettres"
    if json_format is None:
        json_format = config.SELECTION_CHOIX == "json"
    texte = reponse or ""

    if json_format:
        # La contrainte de sortie garantit un objet bien formé, mais pas le
        # contenu : un champ absent, un type inattendu ou un texte quelconque
        # restent possibles avec un autre fournisseur. On vérifie donc quand
        # même — « la contrainte devrait l'empêcher » n'est pas une preuve.
        try:
            donnees = json.loads(texte.strip())
        except ValueError:
            return None
        if not isinstance(donnees, dict):
            return None
        valeur = donnees.get("passage")
        if isinstance(valeur, bool) or not isinstance(valeur, int):
            return None
        return valeur if 0 <= valeur <= nombre_de_passages else None

    if lettres:
        # Avec des étiquettes-lettres, on ne lit QUE des lettres ISOLÉES (et « 0 »
        # pour l'abstention) :
        #  - un chiffre dans la réponse est un numéro de ligne du corpus recopié,
        #    donc précisément ce qu'on ne veut pas interpréter ;
        #  - une lettre COLLÉE à d'autres lettres est un mot (« aucun » contient
        #    un « a »), pas une étiquette. Sans cette borne, « aucun » serait lu
        #    comme le passage A — c'est-à-dire deviné.
        jetons = re.findall(r"(?<![-\u2212])\d+|(?<![A-Za-z])[A-Za-z](?![A-Za-z])", texte)
        correspondance = {
            etiquette: rang
            for rang, etiquette in enumerate(
                etiquettes_selection(nombre_de_passages, lettres), start=1
            )
        }
        for jeton in reversed(jetons):
            if jeton == "0":
                return 0
            rang = correspondance.get(jeton.upper())
            if rang is not None:
                return rang
        return None

    nombres = re.findall(r"(?<![-\u2212])\d+", texte)
    if not nombres:
        return None
    choix = int(nombres[-1])
    return choix if 0 <= choix <= nombre_de_passages else None


def format_passage_choisi(passage: str, source: Mapping, language="ar") -> str:
    """Le texte montré à l'utilisateur : le passage, et sa référence.

    Aucune phrase n'est rédigée par le modèle : c'est ce qui rend l'invention
    impossible. L'introduction et la référence sont des gabarits fixes.

    ⚠️ L'introduction dit que le passage a été SÉLECTIONNÉ, et rien de plus. Elle
    disait « المقطع الذي يجيب عن السؤال » — « le passage qui répond à la
    question » — c'est-à-dire une affirmation que le système n'est PAS en mesure
    de tenir : mesuré, sur 48 essais de questions SANS réponse dans le corpus, le
    modèle n'a renoncé que 4 fois. Dans les 44 autres cas, le titre aurait
    affirmé une chose fausse. Le système sait quel passage il a choisi ; il ne
    sait pas si ce passage répond. Il ne doit donc l'affirmer dans aucun cas.
    """
    intro = (
        "Passage sélectionné dans l'ouvrage (à vérifier) :"
        if language == "fr"
        else "المقطع المختار من الكتاب (يُرجى التحقّق منه):"
    )
    return (
        f"{intro}\n\n{passage.strip()}\n\n"
        f"{format_sources([source], language)}"
    )


def repondre_par_selection(question, documents, sources, language="ar"):
    """Répond en SÉLECTIONNANT un passage parmi ceux qui ont été récupérés.

    Retourne ``(texte, rang_choisi, brute)`` :

    - ``rang_choisi`` vaut ``0`` quand le modèle déclare qu'aucun passage ne
      répond. C'est une RÉPONSE, pas un échec : c'est la bonne conduite sur une
      question que le corpus ne peut pas trancher. Le texte rendu est vide.
    - ``rang_choisi`` vaut ``None`` quand la réponse est illisible (hors
      protocole, ou rang hors bornes). Le texte rendu est vide AUSSI — mais les
      deux causes ne se confondent pas, et l'appelant doit pouvoir les compter
      séparément.

    ⚠️ **``0`` et ``None`` ne doivent JAMAIS être confondus.** Ce code a écrit
    ``if not choix:`` : ``0`` étant falsy, l'abstention était rendue comme un
    échec de lecture. Le rapport annonçait donc « 0 abstention sur 48 » là où il y
    en avait 4 — l'erreur allait dans le sens le plus défavorable au mécanisme,
    et elle était invisible à la lecture du chiffre. C'est le même piège que
    ``all([])`` dans ``evaluation.est_orpheline`` : une valeur falsy qui fait
    disparaître un cas.

    ⚠️ ``brute`` — la sortie NON interprétée du modèle — est retournée parce que
    sans elle un choix illisible est INDÉMÉLABLE : le texte rendu est vide dans
    tous les cas, qu'on ait affaire à un modèle muet, à un numéro hors bornes,
    ou à trois paragraphes de prose. C'est ce qui a permis de découvrir que les
    ``{"passage": 0}`` comptés comme illisibles étaient des abstentions.
    """
    if not documents:
        return "", None, ""

    contexte = numeroter_passages(documents)
    brute = generate_selection(question, contexte, language, len(documents))
    choix = lire_choix(brute, len(documents))

    # ⚠️ ``is None``, et non ``not choix`` : ``0`` est une abstention, c'est-à-dire
    # une réponse. Le confondre avec un échec de lecture a fait publier
    # « 0 abstention sur 48 » là où il y en avait 4.
    if choix is None:
        return "", None, brute

    if choix == 0:
        return "", 0, brute

    return (
        format_passage_choisi(documents[choix - 1], sources[choix - 1], language),
        choix,
        brute,
    )


def repondre_par_refus_puis_selection(question, documents, sources, language="ar"):
    """Renonce d'abord (question ouverte), ancre ensuite (sélection).

    Retourne ``(texte, rang_choisi, brute)`` avec la convention de
    `repondre_par_selection` : ``0`` pour un renoncement, ``None`` pour une
    lecture impossible.

    ⚠️ POURQUOI DEUX RÉGIMES PLUTÔT QU'UN. Mesuré sur les 24 questions SANS
    réponse dans le corpus (2 répétitions, 48 essais) :

        texte libre seul ....... renonce 34 / 48  (71 %)
        sélection seule ........ renonce  4 / 48  ( 8 %)

    Le modèle SAIT dire « je ne sais pas » en question ouverte, et ne le dit
    presque jamais quand on lui demande de choisir parmi des passages : une
    question à choix multiple appelle une réponse, une question ouverte admet
    l'ignorance. On prend donc chaque régime là où il est le meilleur — le
    premier décide s'il faut renoncer, le second empêche l'invention de texte.

    ⚠️ Le premier appel sert à DÉCIDER, pas à rédiger : son texte n'est JAMAIS
    montré. C'est ce qui permet d'utiliser `generate` (un seul appel) au lieu
    d'`answer_question`, dont la reprise de langue coûterait un second appel pour
    un texte qu'on jette de toute façon.

    ⚠️ Ce régime n'est PAS parfait, et sa faiblesse est connue : la décision
    repose sur `est_un_refus`, un détecteur par formules qui manquait 7 refus sur
    34 avant d'être étendu aux cas observés. Un refus manqué fait passer une
    question sans réponse jusqu'à la sélection, qui montrera alors un passage hors
    sujet. Mesuré : le texte libre ne renonce pas 14 fois sur 48, et dans ces cas
    il produit des fatwas brèves et deux réponses CONTRADICTOIRES à la même
    question. Ce régime les transforme en passages hors sujet — trois fois mieux
    que 92 %, pas parfait.
    """
    if not documents:
        return "", None, ""

    # 1. Question OUVERTE : elle décide s'il y a de quoi répondre.
    brouillon = generate(question, "\n\n".join(documents), language)
    if est_un_refus(brouillon, language):
        # Le brouillon est conservé comme sortie brute : c'est lui qui permettra
        # de juger un renoncement, ou de constater un refus manqué.
        return "", 0, brouillon

    # 2. Le modèle n'a pas renoncé : on ANCRE la réponse dans un passage réel.
    return repondre_par_selection(question, documents, sources, language)


def generate_selection(question, contexte, language="ar", nombre_de_passages=0):
    """Envoie l'invite de sélection au fournisseur configuré.

    Séparée de `generate` parce que la tâche n'est pas la même : `generate`
    rédige, celle-ci choisit une étiquette. Les garder distinctes permet de
    mesurer l'une sans toucher à l'autre — et c'est ce qu'on veut savoir.

    `nombre_de_passages` sert à construire le schéma de sortie : c'est lui qui
    borne l'entier au nombre réel de passages, donc qui rend un choix hors bornes
    impossible à produire.
    """
    prompt = construire_invite_selection(question, contexte, language)
    schema = (
        schema_choix(nombre_de_passages)
        if config.SELECTION_CHOIX == "json"
        else None
    )
    return _appeler_modele([{"role": "user", "content": prompt}], format=schema)


def build_prompt(question, context, language="ar"):
    """Construit l'invite envoyée au modèle, dans la langue demandée.

    ⚠️ La consigne de langue est volontairement placée **en fin d'invite**.

    Les derniers tokens d'une invite pèsent davantage sur la génération : la
    consigne la plus proche du point de départ de la réponse est donc la mieux
    suivie. Placée au début, avant un long contexte arabe, elle était souvent
    noyée — et le modèle « continuait » en arabe.

    On précise aussi explicitement que les extraits sont en arabe : sans cela,
    le modèle imite la langue du contexte, ce qui paraît naturel.

    ⚠️ MESURÉ, PUIS REVENU EN ARRIÈRE (29/09). Une version DURCIE de cette invite a
    été essayée : quatre règles numérotées interdisant explicitement d'inférer, de
    faire des analogies et d'attribuer des opinions absentes du texte. Résultat
    mesuré sur les 9 questions proches du domaine et sans réponse dans le corpus :
    **1 refus sur 9, contre 4 sur 9 avant**. Et la FORME des réponses a changé :
    elles sont devenues des fatwas courtes et assertives (« لا يجوز بيع الأسهم في
    البورصة. », « لا بأس بأن يعمل في البنوك. »), là où l'ancienne invite faisait
    au moins citer le corpus. **Ajouter des interdictions a rendu le modèle plus
    assertif, pas plus fidèle.**

    Réserve honnête : une exécution de chaque côté, et le modèle est stochastique.
    Le changement de forme, lui, est net sur 6 questions sur 9. Avant de retenter un
    durcissement, mesurer PLUSIEURS exécutions par question — c'est le seul
    protocole qui puisse trancher.
    """
    if config.CITATIONS_OBLIGATOIRES:
        return _invite_citations(question, context, language)

    if language == "fr":
        return f"""Utilise uniquement le contexte ci-dessous pour répondre précisément à la question. Si le contexte ne contient pas la réponse, dis : « Désolé, il n'y a pas assez d'informations dans les documents fournis. ».

Contexte extrait :
{context}

Question : {question}

Consigne de langue, impérative : les extraits ci-dessus sont en arabe, mais tu dois rédiger ta réponse UNIQUEMENT EN FRANÇAIS. N'écris aucune phrase en arabe.

Réponse en français :"""

    return f"""استخدم السياق أدناه فقط للإجابة على السؤال بدقة. إذا كان السياق لا يحتوي على الإجابة، قل "عذرًا، لا توجد معلومات كافية في الوثائق المرفقة".

السياق المستخرج:
{context}

السؤال: {question}

تنبيه إلزامي: يجب أن تكتب إجابتك باللغة العربية فقط، ولا تكتب أي جملة بلغة أخرى.

الإجابة بالعربية:"""


# Les formules de refus, par langue.
#
# ⚠️ Sert à deux choses, et la seconde est plus exigeante que la première :
#   1. NE PAS CITER DE SOURCE sous un refus (voir `est_un_refus`) ;
#   2. DÉCIDER de renoncer dans le régime `refus_puis_selection`.
# Pour (2), un refus manqué fait passer une question sans réponse jusqu'à la
# sélection, qui montrera un passage hors sujet. L'erreur est donc coûteuse dans
# les deux sens, et la liste a été étendue aux formules RÉELLEMENT observées.
#
# ⚠️ Ce n'est toujours PAS une mesure du taux de refus. Mesuré le 29/09, un
# détecteur de ce genre comptait 5 refus sur 15 là où une lecture en trouve 12.
# Et le 01/10, sur 48 essais hors corpus, il en manquait **7 sur 34 (21 %)**, tous
# des reformulations légitimes. Les sept formes ci-dessous ont été relevées en
# lisant les réponses RÉELLES (le fichier UTF-8, pas le terminal) :
#
#   لا توجد المعلومات اللازمة لرد السؤال.        (et non « لا توجد معلومات »)
#   لا يوجد صلة للسؤال في السياق.
#   لا أوجد الإجابة في السياق السابق.            (أوجد, pas أجد)
#   لا يوجد سؤال صحيح يمكن الإجابة عليه…
#   لا تجد الإجابة في السياق المذكور.
#   لا يوجد معلومات في السياق المرفق عن…         (يوجد مع معلومات !)
#   لا يوجد mention لذلك في النص.               (mélange arabe/latin)
#
# ⚠️ Deux de ces formes ont d'abord été écrites FAUX, parce que je les avais lues
# sur la sortie d'un terminal Windows déformée par cp1252 : « أجد » au lieu de
# « أوجد », et « توجد » au lieu de « يوجد ». **Ne jamais lire de l'arabe dans un
# terminal cp1252** — passer par un fichier UTF-8.
#
# Et il en manquera encore d'autres. **Un taux de refus se lit, il ne se compte pas.**
FORMULES_DE_REFUS = {
    "ar": (
        "لا توجد معلومات",
        "لا توجد المعلومات",
        "لا يوجد معلومات",
        "لا توجد الاجابه",
        "لا اوجد معلومات",
        "لا يوجد سياق",
        "لا يوجد جواب",
        "لا يوجد صله",
        "لا يوجد سوال صحيح",
        "لا يوجد mention",
        "لا توجد في الوثايق",
        "لا اوجد الاجابه",
        "لا تجد الاجابه",
    ),
    "fr": (
        "pas assez d informations",
        "aucune information",
        "ne contiennent pas",
        "ne mentionnent pas",
        "ne permet pas de repondre",
    ),
}


def est_un_refus(reponse: str, language: str = "ar") -> bool:
    """La réponse déclare-t-elle qu'elle n'a pas de quoi répondre ?

    Sert à NE PAS ajouter de citation sous un refus. Les sources sont ajoutées
    dès que la récupération rend quelque chose, et sans seuil de distance elle
    rend toujours quelque chose : sans ce test, une réponse « je n'ai pas
    l'information » se retrouve décorée d'une page sans rapport, et rien ne
    distingue plus « ceci vient de la page citée » de « on a collé une source
    sous un refus ». C'est la promesse du produit qui est en jeu : toute
    affirmation doit être vérifiable dans le texte cité.

    ⚠️ Reconnaissance par formules, donc imparfaite par construction : elle peut
    manquer un refus formulé autrement. Elle ne doit JAMAIS servir à mesurer un
    taux de refus (voir `FORMULES_DE_REFUS`), seulement à éviter une citation
    trompeuse.
    """
    normalisee = _normaliser_pour_refus(reponse)
    return any(
        _normaliser_pour_refus(formule) in normalisee
        for formule in FORMULES_DE_REFUS.get(language, FORMULES_DE_REFUS["ar"])
    )


def _normaliser_pour_refus(texte: str) -> str:
    """Forme comparable d'un texte, pour chercher une formule de refus.

    ⚠️ La normalisation ARABE est indispensable, et son absence a fait échouer deux
    tests dès l'écriture : « الإجابة » et « اجابه » sont le même mot, mais aucune
    des deux chaînes ne contient l'autre. C'est la même faute que la plage de
    caractères qui avalait la ponctuation : une comparaison de texte brut sur de
    l'arabe est fausse par accident, et silencieusement.

    Les apostrophes françaises sont retirées pour la même raison : « d'informations »
    ne contient pas la sous-chaîne « d informations ».
    """
    texte = normaliser(texte)
    for signe in ("'", "’", "`", "،", "؛"):
        texte = texte.replace(signe, " ")
    return " ".join(texte.split())


def build_repair_instruction(language):
    """Consigne de reprise : demander une réécriture dans la bonne langue.

    La réponse fautive est renvoyée au modèle comme message `assistant` (voir
    `build_messages`) : il voit ainsi ce qu'il a produit et le corrige, au lieu
    de repartir de zéro. C'est le principe d'un tour de conversation.
    """
    if language == "fr":
        return (
            "TA RÉPONSE PRÉCÉDENTE N'ÉTAIT PAS EN FRANÇAIS. "
            "Réécris-la intégralement EN FRANÇAIS, sans une seule phrase en arabe. "
            "Conserve exactement les mêmes informations et n'ajoute rien."
        )

    return (
        "إجابتك السابقة لم تكن باللغة العربية. "
        "أعد كتابتها كاملة باللغة العربية فقط، دون أي جملة بلغة أخرى. "
        "احتفظ بنفس المعلومات ولا تضف شيئًا."
    )


def build_messages(question, context, language="ar", previous_answer=None):
    """Construit la liste de messages envoyée au modèle.

    Sans `previous_answer` : un seul message. Avec : on ajoute la réponse fautive
    puis la consigne de reprise, ce qui forme un tour de correction.
    """
    messages = [{"role": "user", "content": build_prompt(question, context, language)}]

    if previous_answer:
        messages.append({"role": "assistant", "content": previous_answer})
        messages.append({"role": "user", "content": build_repair_instruction(language)})

    return messages


def _appeler_modele(messages, format=None):
    """Envoie une liste de messages au fournisseur configuré.

    Point de passage UNIQUE : `generate` (rédaction), la réécriture de langue et
    la sélection de passage (choix d'un numéro) partagent le même appel, donc le
    même fournisseur et le même modèle. Une seconde copie de cet appel
    finirait par diverger — et `LLM_PROVIDER` cesserait d'être respecté quelque
    part sans que rien ne le signale.

    `format` : schéma JSON qui CONTRAINT la sortie, ou ``None`` pour du texte
    libre. La contrainte est appliquée pendant la génération des tokens : elle
    rend une forme interdite impossible à produire, là où une consigne dans
    l'invite ne fait que la déconseiller. Avec un fournisseur OpenAI, seul
    ``json_object`` est demandé — le schéma n'est pas transporté, car tous les
    serveurs compatibles ne le comprennent pas.
    """
    if config.LLM_PROVIDER == "openai":
        kwargs = {}
        if format is not None:
            kwargs["response_format"] = {"type": "json_object"}
        response = get_openai_client().chat.completions.create(
            model=config.LLM_MODEL,
            messages=messages,
            **kwargs,
        )
        return response.choices[0].message.content or ""

    response = get_ollama_client().chat(
        model=config.LLM_MODEL,
        messages=messages,
        format=format,
    )
    return response["message"]["content"]


def generate(question, context, language="ar", previous_answer=None):
    """Envoie l'invite au fournisseur configuré et retourne la réponse.

    Le fournisseur est choisi par `config.LLM_PROVIDER` :
      - "ollama" : serveur Ollama auto-hébergé (défaut) ;
      - "openai" : toute API compatible OpenAI (Groq, Together, vLLM, ...).

    `language` : "ar" (arabe) ou "fr" (français). Le modèle lit le contexte
    (qui peut être en arabe) et rédige sa réponse dans la langue cible.

    `previous_answer` : réponse à faire réécrire (voir `build_repair_instruction`).
    """
    return _appeler_modele(build_messages(question, context, language, previous_answer))


def answer_question(question, context, language="ar"):
    """Génère la réponse et VÉRIFIE qu'elle est bien dans la langue demandée.

    Stratégie de **défense en profondeur** — chaque couche rattrape les limites
    de la précédente :

    1. l'invite place déjà la consigne de langue en fin de texte (efficacité
       probabiliste : on améliore les chances, sans garantie) ;
    2. la langue RÉELLEMENT produite est détectée (efficacité déterministe :
       aucune place au hasard) ;
    3. si elle est fausse, une réécriture est demandée, dans la limite de
       `config.LANGUAGE_MAX_RETRIES` (une reprise coûte un appel LLM).

    Une langue indétectable (réponse vide, chiffres seuls, texte réellement
    mélangé) est considérée comme **acceptable** : on ne peut rien prouver, et
    relancer pour rien coûterait du temps GPU sans raison.
    """
    answer = generate(question, context, language)

    if not config.LANGUAGE_ENFORCEMENT_ENABLED:
        return answer

    detected = detect_language(answer)
    if detected is None or detected == language:
        return answer

    logger.info(
        "Langue non respectée (%s au lieu de %s) : demande de réécriture.",
        detected,
        language,
    )

    for _ in range(config.LANGUAGE_MAX_RETRIES):
        corrected = generate(question, context, language, previous_answer=answer)

        if not corrected.strip():
            # Réponse vide : rien d'exploitable, on garde la tentative précédente.
            continue

        detected = detect_language(corrected)
        if detected is None or detected == language:
            return corrected

        answer = corrected

    logger.warning(
        "Langue toujours non respectée après %d reprise(s) : attendu %s, obtenu %s.",
        config.LANGUAGE_MAX_RETRIES,
        language,
        detect_language(answer),
    )
    return answer


def etiquette_source(source: Mapping) -> str:
    """Comment NOMMER un ouvrage dans une citation.

    Le **titre**, quand on le connaît : c'est la seule mention qu'un lecteur peut
    retrouver dans son exemplaire. Le nom du fichier n'est qu'un repli — un PDF
    n'apporte pas de titre fiable (`pypdf` n'en rend pas), et « 12445.epub » ne
    désigne rien pour personne.

    Le nom de fichier reste la CLÉ d'identification partout ailleurs (registre,
    index, jeu d'or) : cette fonction ne change que ce qui est MONTRÉ. C'est
    exactement ce qui permet d'ajouter vingt ouvrages, ou d'en renommer un, sans
    invalider les questions de référence.
    """
    titre = (source.get("title") or "").strip()
    return titre or source.get("source") or ""


def format_sources(sources, language="ar"):
    """Construit la mention des sources dans la langue demandée.

    Arabe : « ouvrage — صفحة X (الأسطر a-b) »
    Français : « ouvrage — page X (lignes a-b) »

    L'ouvrage est nommé par son titre quand on le connaît (voir
    `etiquette_source`).
    """
    parts = []
    precedent = None
    for s in sources:
        nom = etiquette_source(s)
        # Un même ouvrage cité d'affilée ne se nomme qu'UNE fois : sans cela,
        # cinq passages du même livre répètent cinq fois un titre qui peut faire
        # quarante caractères, et la référence devient illisible. Dès que
        # l'ouvrage change, le nom revient — sinon on ne saurait plus de quel
        # livre parle le passage.
        prefix = f"{nom} — " if nom and nom != precedent else ""
        precedent = nom
        if language == "fr":
            parts.append(
                f"{prefix}page {s['page']} (lignes {s['line_start']}-{s['line_end']})"
            )
        else:
            parts.append(
                f"{prefix}صفحة {s['page']} (الأسطر {s['line_start']}-{s['line_end']})"
            )
    separator = ", " if language == "fr" else "، "
    return separator.join(parts)


def format_excerpts(docs, sources, language="ar", max_chars=200):
    """Construit la liste des extraits (phrases) cités, chacun suivi de sa référence.

    Exemple :
    1. « ...texte du passage... » — fichier.pdf — page 5 (lignes 1-9)
    """
    header = "📄 Extraits cités :" if language == "fr" else "📄 المقتطفات:"
    items = []
    for i, (doc, src) in enumerate(zip(docs, sources, strict=True), start=1):
        text = (doc or "").strip().replace("\n", " ")
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "…"
        ref = format_sources([src], language)
        items.append(f"{i}. « {text} » — {ref}")
    return header + "\n" + "\n".join(items)
