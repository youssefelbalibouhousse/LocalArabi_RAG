"""Logique RAG réutilisable : connexion à la base, récupération, génération.

Partagée entre l'API (app.main) et le script d'ingestion (scripts.build_kb),
pour éviter toute duplication de configuration.
"""

import logging
from collections.abc import Mapping

import chromadb
import ollama
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)

from app import config
from app.language import detect_language
from app.lexical import IndexLexical, fusionner_rrf

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


def build_prompt(question, context, language="ar"):
    """Construit l'invite envoyée au modèle, dans la langue demandée.

    ⚠️ La consigne de langue est volontairement placée **en fin d'invite**.

    Les derniers tokens d'une invite pèsent davantage sur la génération : la
    consigne la plus proche du point de départ de la réponse est donc la mieux
    suivie. Placée au début, avant un long contexte arabe, elle était souvent
    noyée — et le modèle « continuait » en arabe.

    On précise aussi explicitement que les extraits sont en arabe : sans cela,
    le modèle imite la langue du contexte, ce qui paraît naturel.
    """
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


def generate(question, context, language="ar", previous_answer=None):
    """Envoie l'invite au fournisseur configuré et retourne la réponse.

    Le fournisseur est choisi par `config.LLM_PROVIDER` :
      - "ollama" : serveur Ollama auto-hébergé (défaut) ;
      - "openai" : toute API compatible OpenAI (Groq, Together, vLLM, ...).

    `language` : "ar" (arabe) ou "fr" (français). Le modèle lit le contexte
    (qui peut être en arabe) et rédige sa réponse dans la langue cible.

    `previous_answer` : réponse à faire réécrire (voir `build_repair_instruction`).
    """
    messages = build_messages(question, context, language, previous_answer)

    if config.LLM_PROVIDER == "openai":
        response = get_openai_client().chat.completions.create(
            model=config.LLM_MODEL,
            messages=messages,
        )
        return response.choices[0].message.content or ""

    response = get_ollama_client().chat(
        model=config.LLM_MODEL,
        messages=messages,
    )
    return response["message"]["content"]


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
