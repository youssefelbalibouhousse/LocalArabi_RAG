"""Tests UNITAIRES de la logique RAG : découpage des PDF et formatage.

Aucun appel à Ollama ni à ChromaDB ici : on teste des fonctions pures.
"""

from itertools import pairwise
from types import SimpleNamespace

import build_kb
import pytest

from app import config, rag

PAGE_UNIQUE = [(1, "ligne un\nligne deux\nligne trois")]


# --- Découpage des pages en chunks ---------------------------------------

def test_chunk_pages_conserve_la_page_et_les_numeros_de_lignes():
    chunks = build_kb.chunk_pages(PAGE_UNIQUE, chunk_size=1000)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk["page"] == 1
    assert chunk["line_start"] == 1
    assert chunk["line_end"] == 3
    assert "ligne un" in chunk["text"]


def test_chunk_pages_decoupe_quand_la_taille_est_depassee():
    texte = "\n".join(f"phrase numero {i}" for i in range(50))

    chunks = build_kb.chunk_pages([(1, texte)], chunk_size=60)

    assert len(chunks) > 1, "un texte long doit être découpé en plusieurs chunks"


def test_chunk_pages_ne_franchit_jamais_les_limites_de_page():
    pages = [(1, "a\n" * 40), (2, "b\n" * 40)]

    chunks = build_kb.chunk_pages(pages, chunk_size=30)

    pages_vues = {c["page"] for c in chunks}
    assert pages_vues == {1, 2}, "chaque chunk appartient à une seule page"


def test_chunk_pages_produit_des_intervalles_de_lignes_contigus():
    """Les lignes des chunks successifs doivent s'enchaîner sans trou ni doublon."""
    texte = "\n".join(f"phrase numero {i}" for i in range(50))

    chunks = build_kb.chunk_pages([(1, texte)], chunk_size=60)

    assert chunks[0]["line_start"] == 1
    for precedent, suivant in pairwise(chunks):
        assert suivant["line_start"] == precedent["line_end"] + 1


def test_chunk_pages_ne_depasse_jamais_la_taille_cible():
    """`chunk_size` est un MAXIMUM, pas une préférence.

    Mesuré sur le corpus réel : « الأوسط » contient une ligne de 15 870
    caractères, qui produisait un chunk de 15 870 caractères — 26 fois la
    cible. Un tel chunk dépasse la fenêtre du modèle d'embedding, qui tronque
    en silence : le vecteur ne représente alors que le début du texte.
    """
    enorme = " ".join(f"mot{i}" for i in range(600))

    chunks = build_kb.chunk_pages([(1, f"courte\n{enorme}\nautre")], chunk_size=100)

    assert max(len(c["text"]) for c in chunks) <= 100


def test_une_ligne_decoupee_garde_son_numero_d_origine():
    """Découper ne doit pas DÉCALER la numérotation : sinon la citation ne
    désigne plus le bon endroit du livre."""
    enorme = " ".join(f"mot{i}" for i in range(200))

    chunks = build_kb.chunk_pages([(1, f"premiere\n{enorme}\nderniere")], chunk_size=80)

    morceaux = [c for c in chunks if "mot" in c["text"]]
    assert len(morceaux) > 1, "la ligne énorme doit être découpée"
    # Aucun morceau ne peut prétendre venir d'ailleurs que de la ligne 2. Le
    # dernier peut déborder sur la ligne 3 s'il reste de la place — il doit
    # alors l'annoncer, et c'est justement ce que cette assertion vérifie.
    assert all(c["line_start"] <= 2 <= c["line_end"] for c in morceaux)
    assert morceaux[0]["line_start"] == 2
    # La première et la dernière ligne gardent leurs numéros à elles.
    assert chunks[0]["line_start"] == 1
    assert chunks[-1]["line_end"] == 3


def test_decouper_ligne_ne_coupe_pas_au_milieu_d_un_mot():
    ligne = " ".join(f"mot{i}" for i in range(50))

    morceaux = build_kb.decouper_ligne(ligne, 40)

    assert all(len(morceau) <= 40 for morceau in morceaux)
    assert all(morceau == morceau.strip() for morceau in morceaux)
    assert " ".join(morceaux) == ligne


def test_decouper_ligne_coupe_quand_meme_sans_espace():
    """Un « mot » plus long que la cible doit être coupé, faute de mieux : ne
    pas le couper laisserait le chunk dépasser la fenêtre du modèle."""
    morceaux = build_kb.decouper_ligne("a" * 250, 100)

    assert [len(morceau) for morceau in morceaux] == [100, 100, 50]


def test_decouper_ligne_laisse_une_ligne_courte_intacte():
    assert build_kb.decouper_ligne("texte court", 100) == ["texte court"]


# --- Formatage des sources -----------------------------------------------

SOURCE = [{"source": "a.pdf", "page": 5, "line_start": 1, "line_end": 9}]

# Un ouvrage qui PORTE un titre : la seule mention qu'un lecteur peut retrouver
# dans son exemplaire. « 12445.epub » ne désigne rien pour personne.
SOURCE_TITREE = [
    {
        "source": "12445.epub",
        "title": "الاعتكاف",
        "page": 81,
        "line_start": 1,
        "line_end": 5,
    }
]


def test_format_sources_en_francais():
    assert rag.format_sources(SOURCE, "fr") == "a.pdf — page 5 (lignes 1-9)"


def test_format_sources_en_arabe():
    resultat = rag.format_sources(SOURCE, "ar")

    assert "a.pdf" in resultat
    assert "صفحة 5" in resultat


def test_format_sources_accepte_une_source_sans_nom_de_fichier():
    sans_fichier = [{"source": None, "page": 2, "line_start": 1, "line_end": 3}]

    assert rag.format_sources(sans_fichier, "fr") == "page 2 (lignes 1-3)"


def test_format_sources_prefere_le_titre_au_nom_de_fichier():
    """La citation doit nommer l'OUVRAGE, pas le fichier."""
    resultat = rag.format_sources(SOURCE_TITREE, "fr")

    assert resultat == "الاعتكاف — page 81 (lignes 1-5)"
    assert "12445.epub" not in resultat


def test_un_titre_blanc_ne_masque_pas_le_nom_de_fichier():
    """Un titre vide n'est pas un titre : sans ce repli, la citation perdrait
    toute référence utilisable."""
    sans_titre = [
        {"source": "a.pdf", "title": "   ", "page": 5, "line_start": 1, "line_end": 9}
    ]

    assert rag.format_sources(sans_titre, "fr") == "a.pdf — page 5 (lignes 1-9)"


def test_un_meme_ouvrage_cite_d_affilee_ne_se_nomme_qu_une_fois():
    """Sinon cinq passages du même livre répètent cinq fois un titre qui peut
    faire quarante caractères, et la référence devient illisible."""
    memes = [
        {"source": "a.epub", "title": "Livre", "page": 15, "line_start": 1, "line_end": 3},
        {"source": "a.epub", "title": "Livre", "page": 15, "line_start": 4, "line_end": 8},
    ]

    resultat = rag.format_sources(memes, "fr")

    assert resultat == "Livre — page 15 (lignes 1-3), page 15 (lignes 4-8)"
    assert resultat.count("Livre") == 1


def test_le_nom_revient_quand_l_ouvrage_change():
    """Sans ce retour, on ne saurait plus de quel livre parle le passage."""
    melange = [
        {"source": "a.epub", "title": "Livre A", "page": 1, "line_start": 1, "line_end": 2},
        {"source": "b.epub", "title": "Livre B", "page": 2, "line_start": 1, "line_end": 2},
        {"source": "a.epub", "title": "Livre A", "page": 3, "line_start": 1, "line_end": 2},
    ]

    resultat = rag.format_sources(melange, "fr")

    assert resultat == (
        "Livre A — page 1 (lignes 1-2), Livre B — page 2 (lignes 1-2), "
        "Livre A — page 3 (lignes 1-2)"
    )


class _FakeCollection:
    """Collection minimale, fidèle à ce que `rag.retrieve` utilise réellement.

    `retrieve` ne se contente plus d'appeler `query` : la recherche hybride lit
    aussi `name`, `count` et `get` (pour construire l'index lexical). Un double
    qui n'exposerait que `query` ne décrirait plus la vraie interface.
    """

    def __init__(self, documents, metadatas, distances=None, identifiants=None,
                 name="fictive"):
        self.name = name
        self._documents = documents
        self._metadatas = metadatas
        self._distances = distances or [0.1] * len(documents)
        self._identifiants = identifiants or [f"id-{i}" for i in range(len(documents))]

    def count(self):
        return len(self._documents)

    def ajouter(self, document, meta, identifiant, distance=0.1):
        """Ajoute un chunk au faux corpus, en tenant les quatre listes ensemble.

        Les listes parallèles du double sont ce qui a déjà menti une fois : ajouter
        un document sans sa distance faisait lever `zip(strict=True)` dans le code
        de production. Une méthode unique évite d'avoir à y penser.
        """
        self._documents.append(document)
        self._metadatas.append(meta)
        self._identifiants.append(identifiant)
        self._distances.append(distance)

    def get(self, **kwargs):
        return {
            "ids": list(self._identifiants),
            "documents": list(self._documents),
            "metadatas": list(self._metadatas),
        }

    def query(self, **kwargs):
        return {
            "ids": [list(self._identifiants)],
            "documents": [list(self._documents)],
            "metadatas": [list(self._metadatas)],
            "distances": [list(self._distances)],
        }


def _meta(source, page, title=None):
    """Un métadonnée de chunk, réduite à ce qui compte pour ces tests."""
    return {
        "source": source,
        "title": title or source,
        "page": page,
        "line_start": 1,
        "line_end": 5,
    }


def test_retrieve_transmet_le_titre_de_l_ouvrage():
    """`format_sources` ne peut pas inventer un titre qu'on ne lui a pas donné :
    si `retrieve` l'oublie, la citation retombe sur le nom de fichier sans que
    rien ne le signale."""
    collection = _FakeCollection(
        documents=["texte"],
        metadatas=[
            {
                "source": "12445.epub",
                "title": "الاعتكاف",
                "page": 81,
                "line_start": 1,
                "line_end": 5,
            }
        ],
    )

    docs, sources = rag.retrieve(collection, "سؤال")

    assert docs == ["texte"]
    assert sources[0]["title"] == "الاعتكاف"
    assert sources[0]["source"] == "12445.epub"


# --- Recherche hybride ----------------------------------------------------
#
# Le cas reproduit ici est celui qui a motivé la fusion, mesuré sur le
# `تفسير ابن المنذر` : la recherche vectorielle ne renvoie jamais le chunk qui
# porte le verset, parce que celui-ci est noyé dans une chaîne de transmetteurs.
# Le chunk est pourtant dans l'index, à la bonne page.

CHAINE = "حدثنا علي بن المبارك قال حدثنا زيد بن ثور عن ابن جريج"


def _corpus_hybride():
    """Un faux corpus où le verset n'arrive qu'en bas du classement vectoriel."""
    documents = [
        f"{CHAINE} في تفسير اية الدين والمعاملات والبيوع",
        f"{CHAINE} في تفسير اية الوضوء والصلوات والخيول",
        f"{CHAINE} ثم صرفكم عنهم ليبتليكم قال يعني بذلك يوم احد",
    ]
    metadatas = [
        _meta("22549.epub", 11, "تفسير ابن المنذر"),
        _meta("22549.epub", 12, "تفسير ابن المنذر"),
        _meta("22549.epub", 446, "تفسير ابن المنذر"),
    ]
    return _FakeCollection(documents, metadatas, name="hybride")


def test_la_recherche_hybride_sauve_le_passage_que_le_vecteur_ignore():
    """Le cœur de la fonctionnalité, et sa raison d'être.

    La recherche vectorielle classe le verset en DERNIER (c'est le classement que
    le faux lui prête). Sans fusion, l'extrait serait le 3e des 5 rendus… ou pas
    rendu du tout si le bassin était plus étroit. La fusion doit le remonter.
    """
    collection = _corpus_hybride()

    docs, sources = rag.retrieve(collection, "ما معنى ثم صرفكم عنهم ليبتليكم", n_results=1)

    assert len(sources) == 1
    assert sources[0]["page"] == 446
    assert "ليبتليكم" in docs[0]


def test_la_recherche_hybride_ne_rend_que_le_nombre_demande():
    """La fusion élargit le bassin, elle ne change pas ce qui est montré au modèle."""
    collection = _corpus_hybride()

    docs, sources = rag.retrieve(collection, "تفسير اية الدين", n_results=2)

    assert len(docs) == len(sources) == 2


def test_la_recherche_vectorielle_seule_reste_disponible(monkeypatch):
    """`HYBRID_ENABLED=false` doit rendre exactement l'ancien comportement.

    C'est ce qui permet de comparer les deux mesures, et de revenir en arrière
    sans redéployer — le drapeau existe pour être utilisé.
    """
    monkeypatch.setattr(config, "HYBRID_ENABLED", False)
    collection = _corpus_hybride()

    docs, sources = rag.retrieve(collection, "ما معنى ثم صرفكم عنهم ليبتليكم", n_results=1)

    # Classement vectoriel seul : le verset est le dernier du faux corpus.
    assert sources[0]["page"] == 11
    assert "ليبتليكم" not in docs[0]


def test_l_index_lexical_est_reconstruit_quand_le_corpus_change():
    """Un cache qui ne se rafraîchit pas servirait un corpus qui n'existe plus."""
    rag.reinitialiser_index_lexical()
    collection = _corpus_hybride()
    rag.retrieve(collection, "تفسير اية الدين")
    avant = rag._index_lexical.taille

    collection.ajouter("نص جديد عن الزكاة والصدقات", _meta("22549.epub", 900), "id-3")

    rag.retrieve(collection, "تفسير اية الدين")

    assert avant == 3
    assert rag._index_lexical.taille == 4


def test_le_seuil_de_distance_ecarte_les_candidats_vectoriels(monkeypatch):
    """`DISTANCE_THRESHOLD` continue de filtrer la moitié vectorielle.

    Le seuil juge une distance ; BM25 n'en produit pas. Il ne peut donc
    s'appliquer qu'avant la fusion, aux candidats vectoriels.
    """
    monkeypatch.setattr(config, "HYBRID_ENABLED", False)
    monkeypatch.setattr(config, "DISTANCE_THRESHOLD", 0.5)
    collection = _FakeCollection(
        documents=["proche", "lointain"],
        metadatas=[_meta("12445.epub", 1), _meta("12445.epub", 2)],
        distances=[0.2, 0.9],
    )

    docs, sources = rag.retrieve(collection, "سؤال")

    assert docs == ["proche"]
    assert len(sources) == 1


def test_l_index_lexical_est_reconstruit_apres_reinitialisation():
    """`reinitialiser_index_lexical` doit vraiment oublier, puis se reconstruire.

    C'est le geste à faire après une ré-ingestion : sans lui, la moitié lexicale
    de la recherche décrirait l'ancien corpus pendant que la moitié vectorielle
    décrit le nouveau.
    """
    collection = _corpus_hybride()
    rag.retrieve(collection, "تفسير اية الدين")
    assert rag._index_lexical is not None

    rag.reinitialiser_index_lexical()

    assert rag._index_lexical is None
    assert rag._extraits == {}

    rag.retrieve(collection, "تفسير اية الدين")

    assert rag._index_lexical.taille == 3
    assert len(rag._extraits) == 3


# --- Formatage des extraits ----------------------------------------------

def test_format_excerpts_contient_le_texte_et_la_reference():
    docs = ["Ceci est un passage du document."]

    resultat = rag.format_excerpts(docs, SOURCE, "fr")

    assert "Ceci est un passage du document." in resultat
    assert "a.pdf — page 5 (lignes 1-9)" in resultat


def test_format_excerpts_tronque_les_textes_trop_longs():
    docs = ["x" * 500]

    resultat = rag.format_excerpts(docs, SOURCE, "fr", max_chars=50)

    assert "…" in resultat
    assert "x" * 500 not in resultat


def test_format_excerpts_est_bilingue():
    docs = ["un passage"]

    assert "Extraits cités" in rag.format_excerpts(docs, SOURCE, "fr")
    assert "المقتطفات" in rag.format_excerpts(docs, SOURCE, "ar")


# --- Invite envoyée au modèle --------------------------------------------

def test_build_prompt_francais_contient_contexte_et_question():
    prompt = rag.build_prompt("Quelle est la règle ?", "un contexte", "fr")

    assert "un contexte" in prompt
    assert "Quelle est la règle ?" in prompt
    assert "français" in prompt


def test_build_prompt_arabe_contient_contexte_et_question():
    prompt = rag.build_prompt("ما الحكم؟", "سياق", "ar")

    assert "سياق" in prompt
    assert "ما الحكم؟" in prompt
    assert "باللغة العربية" in prompt


# --- Choix du fournisseur de génération ----------------------------------

class _FakeOpenAI:
    """Doublure du client OpenAI : enregistre les appels et renvoie un texte fixe.

    `self.chat = self` et `self.completions = self` reproduisent la chaîne
    d'appel réelle `client.chat.completions.create(...)` sans réseau.
    """

    def __init__(self, texte):
        self.texte = texte
        self.appels = []
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.appels.append(kwargs)
        message = SimpleNamespace(content=self.texte)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _FakeOllama:
    """Doublure du client Ollama."""

    def __init__(self, texte):
        self.texte = texte
        self.appels = []

    def chat(self, **kwargs):
        self.appels.append(kwargs)
        return {"message": {"content": self.texte}}


def test_generate_utilise_ollama_par_defaut(monkeypatch):
    faux = _FakeOllama("Réponse locale.")
    monkeypatch.setattr(config, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(rag, "get_ollama_client", lambda: faux)

    resultat = rag.generate("question", "contexte", "ar")

    assert resultat == "Réponse locale."
    assert faux.appels[0]["model"] == config.LLM_MODEL
    assert "contexte" in faux.appels[0]["messages"][0]["content"]


def test_generate_utilise_api_openai_si_configuree(monkeypatch):
    faux = _FakeOpenAI("Réponse cloud.")
    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(rag, "get_openai_client", lambda: faux)

    resultat = rag.generate("question", "contexte", "fr")

    assert resultat == "Réponse cloud."
    assert faux.appels[0]["model"] == config.LLM_MODEL
    assert "contexte" in faux.appels[0]["messages"][0]["content"]


def test_generate_retombe_sur_ollama_si_fournisseur_inconnu(monkeypatch):
    """Un nom de fournisseur mal orthographié ne doit pas casser le service."""
    faux = _FakeOllama("Repli local.")
    monkeypatch.setattr(config, "LLM_PROVIDER", "opnai")  # faute de frappe
    monkeypatch.setattr(rag, "get_ollama_client", lambda: faux)

    assert rag.generate("question", "contexte", "ar") == "Repli local."


def test_generate_renvoie_une_chaine_meme_si_le_cloud_repond_vide(monkeypatch):
    """Une réponse vide du fournisseur ne doit jamais produire None."""
    faux = _FakeOpenAI(None)
    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(rag, "get_openai_client", lambda: faux)

    assert rag.generate("question", "contexte", "fr") == ""


# --- Placement de la consigne de langue dans l'invite ---------------------

def test_build_prompt_place_la_consigne_de_langue_apres_le_contexte():
    """Les derniers tokens pèsent le plus : la consigne doit venir EN DERNIER.

    C'est la correction de la cause n°1 du bug : placée au début, avant un long
    contexte arabe, la consigne « en français » était noyée.
    """
    prompt = rag.build_prompt("Quelle règle ?", "المحتوى المستخرج", "fr")

    assert prompt.index("Consigne de langue") > prompt.index("Contexte extrait")
    assert prompt.index("Consigne de langue") > prompt.index("Quelle règle ?")


def test_build_prompt_francais_previent_que_le_contexte_est_en_arabe():
    """Sans cet avertissement, le modèle « continue » dans la langue du contexte."""
    prompt = rag.build_prompt("Question ?", "المحتوى", "fr")

    assert "en arabe" in prompt
    assert "UNIQUEMENT EN FRANÇAIS" in prompt


def test_build_prompt_arabe_place_la_consigne_a_la_fin():
    prompt = rag.build_prompt("ما الحكم؟", "contexte latin", "ar")

    assert prompt.index("تنبيه إلزامي") > prompt.index("السياق المستخرج")
    assert prompt.index("تنبيه إلزامي") > prompt.index("ما الحكم؟")


def test_build_repair_instruction_est_bilingue():
    assert "FRANÇAIS" in rag.build_repair_instruction("fr")
    assert "باللغة العربية" in rag.build_repair_instruction("ar")


def test_build_messages_initial_ne_contient_qu_un_message():
    messages = rag.build_messages("Question ?", "contexte", "fr")

    assert len(messages) == 1
    assert messages[0]["role"] == "user"


def test_build_messages_de_reprise_forme_un_tour_de_correction():
    """La réponse fautive est renvoyée au modèle : il voit quoi corriger."""
    messages = rag.build_messages("Question ?", "contexte", "fr", previous_answer="إجابة")

    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[1]["content"] == "إجابة"
    assert "PAS EN FRANÇAIS" in messages[2]["content"]


# --- Vérification et reprise (answer_question) ---------------------------

REPONSE_FR = "Voici la réponse en français, d'après les documents."
REPONSE_AR = "هذه هي الإجابة بالعربية حسب الوثائق المرفقة."
# Concaténation volontaire : ruff refuse (RUF001) les littéraux qui mélangent
# les alphabets — or ici, le mélange est précisément le cas à couvrir.
REPONSE_MELANGEE = "abcde" + "المدرسة"


def _fausse_generation(*reponses):
    """Doublure de `rag.generate`.

    Renvoie les réponses dans l'ordre, puis la dernière indéfiniment (ce qui
    simule un modèle qui « persiste » dans son erreur), et journalise chaque
    appel pour pouvoir les compter.
    """
    attente = list(reponses)
    appels = []

    def faux_generate(question, context, language="ar", previous_answer=None):
        appels.append({
            "question": question,
            "context": context,
            "language": language,
            "previous_answer": previous_answer,
        })
        return attente.pop(0) if len(attente) > 1 else attente[0]

    return faux_generate, appels


def test_answer_question_ne_relance_pas_si_la_langue_est_bonne(monkeypatch):
    """Le cas normal ne doit coûter qu'UN appel au modèle."""
    faux, appels = _fausse_generation(REPONSE_FR)
    monkeypatch.setattr(rag, "generate", faux)

    resultat = rag.answer_question("Question ?", "contexte", "fr")

    assert resultat == REPONSE_FR
    assert len(appels) == 1
    assert appels[0]["previous_answer"] is None


def test_answer_question_relance_quand_la_langue_est_fausse(monkeypatch):
    """C'est LE bug corrigé : le modèle répond en arabe malgré language="fr"."""
    faux, appels = _fausse_generation(REPONSE_AR, REPONSE_FR)
    monkeypatch.setattr(rag, "generate", faux)

    resultat = rag.answer_question("Question ?", "contexte", "fr")

    assert resultat == REPONSE_FR
    assert len(appels) == 2
    # La reprise doit transmettre la réponse fautive (tour de correction).
    assert appels[1]["previous_answer"] == REPONSE_AR
    assert appels[1]["language"] == "fr"


def test_answer_question_fonctionne_aussi_dans_l_autre_sens(monkeypatch):
    """Symétrie : un modèle qui répond en français quand on demande l'arabe."""
    faux, appels = _fausse_generation(REPONSE_FR, REPONSE_AR)
    monkeypatch.setattr(rag, "generate", faux)

    assert rag.answer_question("Question ?", "contexte", "ar") == REPONSE_AR
    assert len(appels) == 2


def test_answer_question_renvoie_la_derniere_tentative_si_rien_ne_marche(monkeypatch):
    """Le modèle persiste dans l'erreur : on rend la dernière réponse obtenue.

    Mieux vaut une réponse imparfaite qu'aucune réponse du tout — et le
    problème est signalé dans les journaux (voir le `logger.warning`).
    """
    faux, appels = _fausse_generation(REPONSE_AR)
    monkeypatch.setattr(rag, "generate", faux)

    resultat = rag.answer_question("Question ?", "contexte", "fr")

    assert resultat == REPONSE_AR
    assert len(appels) == 2  # 1 appel initial + 1 reprise (défaut)


def test_answer_question_respecte_le_nombre_maximal_de_reprises(monkeypatch):
    """Chaque reprise coûte un appel LLM : le plafond doit être respecté."""
    faux, appels = _fausse_generation(REPONSE_AR)
    monkeypatch.setattr(rag, "generate", faux)
    monkeypatch.setattr(config, "LANGUAGE_MAX_RETRIES", 3)

    rag.answer_question("Question ?", "contexte", "fr")

    assert len(appels) == 4  # 1 + 3


def test_answer_question_sans_reprise_si_le_plafond_est_zero(monkeypatch):
    faux, appels = _fausse_generation(REPONSE_AR)
    monkeypatch.setattr(rag, "generate", faux)
    monkeypatch.setattr(config, "LANGUAGE_MAX_RETRIES", 0)

    assert rag.answer_question("Question ?", "contexte", "fr") == REPONSE_AR
    assert len(appels) == 1


def test_answer_question_accepte_une_langue_indeterminee(monkeypatch):
    """« Je ne sais pas quelle langue c'est » n'est pas « c'est faux ».

    Relancer sur une réponse ambiguë gaspillerait un appel GPU sans preuve
    qu'il y ait quoi que ce soit à corriger.
    """
    faux, appels = _fausse_generation(REPONSE_MELANGEE)
    monkeypatch.setattr(rag, "generate", faux)

    resultat = rag.answer_question("Question ?", "contexte", "fr")

    assert resultat == REPONSE_MELANGEE
    assert len(appels) == 1


def test_answer_question_accepte_une_reprise_devenue_ambigue(monkeypatch):
    """Après reprise, une réponse mélangée est acceptée (plus rien n'est prouvable)."""
    faux, appels = _fausse_generation(REPONSE_AR, REPONSE_MELANGEE)
    monkeypatch.setattr(rag, "generate", faux)

    assert rag.answer_question("Question ?", "contexte", "fr") == REPONSE_MELANGEE
    assert len(appels) == 2


def test_answer_question_garde_la_premiere_reponse_si_la_reprise_est_vide(monkeypatch):
    """Une reprise vide ne doit pas effacer une réponse, même imparfaite."""
    faux, appels = _fausse_generation(REPONSE_AR, "   ")
    monkeypatch.setattr(rag, "generate", faux)

    assert rag.answer_question("Question ?", "contexte", "fr") == REPONSE_AR
    assert len(appels) == 2


def test_answer_question_sans_verification_si_desactivee(monkeypatch):
    """Interrupteur de secours : utile pour diagnostiquer ou maîtriser la facture."""
    faux, appels = _fausse_generation(REPONSE_AR)
    monkeypatch.setattr(rag, "generate", faux)
    monkeypatch.setattr(config, "LANGUAGE_ENFORCEMENT_ENABLED", False)

    assert rag.answer_question("Question ?", "contexte", "fr") == REPONSE_AR
    assert len(appels) == 1


# --- Reconnaissance du refus ---------------------------------------------
#
# Ce test sert à NE PAS citer de source sous un refus — jamais à mesurer un taux
# de refus. Il s'est déjà trompé dans ce rôle-là : 5 refus comptés sur 15 là où
# une lecture en trouve 12. Les formules sont donc larges, et l'erreur restante
# est assumée : manquer un refus formulé autrement laisse une citation de trop,
# ce qui est moins grave que l'inverse.


@pytest.mark.parametrize(
    "reponse",
    [
        # Les formules que l'invite impose
        "عذرًا، لا توجد معلومات كافية في الوثائق المرفقة.",
        "Désolé, il n'y a pas assez d'informations dans les documents fournis.",
        # Les reformulations RÉELLEMENT observées, qui avaient échappé au premier détecteur
        "لا توجد الإجابة في السياق المستخرج.",
        "لا أوجد معلومات كافية في الوثائق المرفقة.",
        "لا يوجد سياق يتعلق بمحرك الاحتراق الداخلي.",
        "لا توجد في الوثائق المرفقة المعلومات الكافية حول حكم الصلاة في الطائرة.",
        "حسب الوثائق المرفقة لا يوجد جواب مناسب للاستفسار.",
        "لا توجد معلومات واضحة في السياق المرفق حول حكم العمل في البنوك.",
        "Les extraits ne mentionnent pas du tout la cuisine.",
    ],
)
def test_les_refus_reellement_observes_sont_reconnus(reponse):
    langue = "fr" if any(c.isascii() and c.isalpha() for c in reponse) else "ar"

    assert rag.est_un_refus(reponse, langue)


@pytest.mark.parametrize(
    "reponse",
    [
        "وأجمعوا على أن في المأمومة ثلث الدية.",
        "La réponse se trouve à la page 81 de l'ouvrage.",
        "",
    ],
)
def test_une_reponse_normale_n_est_pas_prise_pour_un_refus(reponse):
    """Un faux positif supprimerait les citations d'une vraie réponse."""
    langue = "fr" if any(c.isascii() and c.isalpha() for c in reponse) else "ar"

    assert not rag.est_un_refus(reponse, langue)


def test_l_apostrophe_francaise_ne_fait_pas_echouer_la_reconnaissance():
    """Elle a déjà fait échouer un détecteur : « d'informations » ne contient
    pas la sous-chaîne « d informations »."""
    assert rag.est_un_refus(
        "Désolé, il n'y a pas assez d'informations dans les documents fournis.", "fr"
    )
