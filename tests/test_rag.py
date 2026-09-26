"""Tests UNITAIRES de la logique RAG : découpage des PDF et formatage.

Aucun appel à Ollama ni à ChromaDB ici : on teste des fonctions pures.
"""

from itertools import pairwise
from types import SimpleNamespace

import build_kb

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


# --- Formatage des sources -----------------------------------------------

SOURCE = [{"source": "a.pdf", "page": 5, "line_start": 1, "line_end": 9}]


def test_format_sources_en_francais():
    assert rag.format_sources(SOURCE, "fr") == "a.pdf — page 5 (lignes 1-9)"


def test_format_sources_en_arabe():
    resultat = rag.format_sources(SOURCE, "ar")

    assert "a.pdf" in resultat
    assert "صفحة 5" in resultat


def test_format_sources_accepte_une_source_sans_nom_de_fichier():
    sans_fichier = [{"source": None, "page": 2, "line_start": 1, "line_end": 3}]

    assert rag.format_sources(sans_fichier, "fr") == "page 2 (lignes 1-3)"


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
