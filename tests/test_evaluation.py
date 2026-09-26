"""Tests de app/evaluation.py (harnais d'évaluation de la récupération).

Deux propriétés sont critiques et donc verrouillées explicitement :

1. une ligne invalide du jeu d'or doit FAIRE ÉCHOUER la lecture, avec le numéro
   de ligne — un jeu d'or silencieusement amputé de ses questions difficiles
   afficherait un excellent score et donnerait une confiance injustifiée ;
2. la correspondance ne doit JAMAIS dépendre de ``line_start`` / ``line_end`` :
   ces bornes varient avec le découpage, et les utiliser ferait mesurer le
   chunker au lieu de la récupération.

Comme pour le reste de la suite, tout est pur : aucune base vectorielle, aucun
appel à Ollama, aucun fichier réel touché (tmp_path uniquement).
"""

import json

import pytest

from app import evaluation
from app.evaluation import (
    EvaluationReport,
    ExpectedSource,
    GoldenError,
    GoldenQuestion,
    QuestionResult,
)

# --- Utilitaires de construction -----------------------------------------

def ligne(**surcharges) -> str:
    """Construit une ligne JSON valide du jeu d'or, surchargeable champ par champ."""
    champs = {
        "id": "q001",
        "question": "ما حكم صلاة الجماعة؟",
        "expected": [{"source": "arabic_document.pdf", "page": 3}],
    }
    champs.update(surcharges)
    return json.dumps(champs, ensure_ascii=False)


def resultat(identifiant: str, rang: int | None, latence: float = 10.0) -> QuestionResult:
    """Construit un résultat de question minimal."""
    return QuestionResult(
        question_id=identifiant,
        question=f"question {identifiant}",
        rank=rang,
        latency_ms=latence,
    )


def rapport(
    label: str,
    rangs: dict[str, int | None],
    k: int = 5,
    empreinte: str | None = None,
) -> EvaluationReport:
    """Construit un rapport à partir d'un dictionnaire {identifiant: rang}."""
    contexte = {"empreinte_corpus": empreinte} if empreinte else {}
    return evaluation.build_report(
        label=label,
        k=k,
        results=[resultat(identifiant, rang) for identifiant, rang in rangs.items()],
        context=contexte,
    )


# --- Analyse d'une ligne du jeu d'or -------------------------------------

def test_une_ligne_complete_est_analysee():
    question = evaluation.parse_golden_line(ligne(), 1)

    assert question is not None
    assert question.id == "q001"
    assert question.question == "ما حكم صلاة الجماعة؟"
    assert question.lang == "ar"
    assert question.status == evaluation.STATUS_DRAFT
    assert question.expected == (ExpectedSource("arabic_document.pdf", 3),)


def test_les_champs_facultatifs_ont_des_defauts_utilisables():
    brut = json.dumps({"question": "سؤال", "expected": [{"source": "a.pdf"}]})

    question = evaluation.parse_golden_line(brut, 7)

    assert question is not None
    assert question.id == "q007"  # déduit du numéro de ligne
    assert question.expected == (ExpectedSource("a.pdf", None),)
    assert question.notes == ""


def test_une_ligne_vide_est_ignoree():
    assert evaluation.parse_golden_line("   ", 1) is None


def test_un_commentaire_est_ignore():
    assert evaluation.parse_golden_line("# une note de lecture", 1) is None


def test_plusieurs_sources_attendues_sont_acceptees():
    brut = ligne(expected=[{"source": "a.pdf", "page": 1}, {"source": "b.pdf"}])

    question = evaluation.parse_golden_line(brut, 1)

    assert question is not None
    assert len(question.expected) == 2


def test_un_json_invalide_echoue_en_citant_la_ligne():
    with pytest.raises(GoldenError, match="ligne 12"):
        evaluation.parse_golden_line("{ceci n'est pas du json", 12)


def test_un_json_qui_n_est_pas_un_objet_echoue():
    with pytest.raises(GoldenError, match="objet JSON"):
        evaluation.parse_golden_line("[1, 2, 3]", 4)


def test_une_question_absente_echoue():
    with pytest.raises(GoldenError, match="« question »"):
        evaluation.parse_golden_line(json.dumps({"expected": [{"source": "a.pdf"}]}), 1)


def test_une_question_vide_echoue():
    with pytest.raises(GoldenError, match="« question »"):
        evaluation.parse_golden_line(ligne(question="   "), 1)


def test_une_liste_attendue_vide_echoue():
    """Sans source attendue, la question serait comptée introuvable à jamais."""
    with pytest.raises(GoldenError, match="« expected »"):
        evaluation.parse_golden_line(ligne(expected=[]), 1)


def test_une_source_attendue_sans_nom_de_fichier_echoue():
    with pytest.raises(GoldenError, match="« source »"):
        evaluation.parse_golden_line(ligne(expected=[{"page": 3}]), 1)


@pytest.mark.parametrize("page", [0, -1, "3", 2.5, True])
def test_une_page_invalide_echoue(page):
    """True est écarté bien qu'il soit un entier en Python : c'est un piège classique."""
    with pytest.raises(GoldenError, match="« page »"):
        evaluation.parse_golden_line(ligne(expected=[{"source": "a.pdf", "page": page}]), 1)


def test_un_statut_inconnu_echoue():
    with pytest.raises(GoldenError, match="« status »"):
        evaluation.parse_golden_line(ligne(status="peut-etre"), 1)


# --- Lecture et écriture du fichier --------------------------------------

def test_la_lecture_ignore_commentaires_et_lignes_vides(tmp_path):
    chemin = tmp_path / "golden.jsonl"
    chemin.write_text(
        "# un commentaire\n" + ligne(id="q001") + "\n\n" + ligne(id="q002") + "\n",
        encoding="utf-8",
    )

    questions = evaluation.load_golden(chemin)

    assert [q.id for q in questions] == ["q001", "q002"]


def test_un_identifiant_en_double_echoue(tmp_path):
    """Deux fois le même identifiant rendrait l'analyse des écarts incohérente."""
    chemin = tmp_path / "golden.jsonl"
    chemin.write_text(ligne(id="q001") + "\n" + ligne(id="q001") + "\n", encoding="utf-8")

    with pytest.raises(GoldenError, match="déjà utilisé"):
        evaluation.load_golden(chemin)


def test_un_fichier_absent_donne_un_message_actionnable(tmp_path):
    with pytest.raises(GoldenError, match="introuvable"):
        evaluation.load_golden(tmp_path / "absent.jsonl")


def test_l_ecriture_preserve_le_texte_arabe(tmp_path):
    """Sans ensure_ascii=False, le fichier serait illisible donc incorrigible."""
    chemin = tmp_path / "golden.jsonl"
    question = GoldenQuestion(
        id="q001",
        question="ما حكم صلاة الجماعة؟",
        expected=(ExpectedSource("arabic_document.pdf", 3),),
    )

    evaluation.append_golden(chemin, [question])
    contenu = chemin.read_text(encoding="utf-8")

    assert "ما حكم صلاة الجماعة؟" in contenu
    assert "\\u" not in contenu


def test_ecriture_puis_lecture_redonne_la_meme_question(tmp_path):
    chemin = tmp_path / "golden.jsonl"
    origine = GoldenQuestion(
        id="q003",
        question="Question ?",
        expected=(ExpectedSource("a.pdf", 2), ExpectedSource("b.pdf", None)),
        lang="fr",
        status=evaluation.STATUS_VALIDATED,
        notes="lignes 10-20",
    )

    evaluation.append_golden(chemin, [origine])

    assert evaluation.load_golden(chemin) == [origine]


# --- Correspondance -------------------------------------------------------

def test_le_meme_fichier_sans_page_attendue_correspond():
    assert evaluation.source_matches(
        {"source": "a.pdf", "page": 42}, ExpectedSource("a.pdf")
    )


def test_la_page_attendue_doit_correspondre():
    assert evaluation.source_matches(
        {"source": "a.pdf", "page": 3}, ExpectedSource("a.pdf", 3)
    )
    assert not evaluation.source_matches(
        {"source": "a.pdf", "page": 4}, ExpectedSource("a.pdf", 3)
    )


def test_un_autre_fichier_ne_correspond_pas():
    assert not evaluation.source_matches(
        {"source": "b.pdf", "page": 3}, ExpectedSource("a.pdf", 3)
    )


def test_une_page_absente_ne_correspond_pas_a_une_page_attendue():
    """Un chunk sans métadonnée de page ne peut pas « satisfaire » une page précise."""
    assert not evaluation.source_matches({"source": "a.pdf"}, ExpectedSource("a.pdf", 3))


def test_les_bornes_de_lignes_ne_sont_pas_utilisees():
    """Verrou explicite : le harnais mesure la récupération, pas le découpage."""
    attendu = ExpectedSource("a.pdf", 3)
    resultat_differents = {
        "source": "a.pdf",
        "page": 3,
        "line_start": 1,
        "line_end": 999,
    }

    assert evaluation.source_matches(resultat_differents, attendu)


# --- Rang du premier résultat correct -------------------------------------

def test_le_rang_du_premier_resultat_correct_est_trouve():
    sources = [{"source": "x.pdf", "page": 1}, {"source": "a.pdf", "page": 3}]

    assert evaluation.first_match_rank(sources, [ExpectedSource("a.pdf", 3)]) == 2


def test_le_rang_est_none_quand_rien_ne_correspond():
    sources = [{"source": "x.pdf", "page": 1}, {"source": "y.pdf", "page": 2}]

    assert evaluation.first_match_rank(sources, [ExpectedSource("a.pdf", 3)]) is None


def test_plusieurs_sources_attendues_equivalent_a_un_ou_logique():
    """Une réponse utile peut venir de plusieurs passages : le premier suffit."""
    sources = [{"source": "z.pdf"}, {"source": "b.pdf", "page": 9}]

    rang = evaluation.first_match_rank(
        sources, [ExpectedSource("a.pdf"), ExpectedSource("b.pdf", 9)]
    )

    assert rang == 2


def test_le_premier_rang_gagne_meme_si_une_autre_source_attendue_est_plus_loin():
    sources = [{"source": "b.pdf"}, {"source": "a.pdf"}]

    rang = evaluation.first_match_rank(
        sources, [ExpectedSource("a.pdf"), ExpectedSource("b.pdf")]
    )

    assert rang == 1


# --- Métriques élémentaires ----------------------------------------------

@pytest.mark.parametrize(
    ("rang", "k", "attendu"),
    [(1, 5, True), (5, 5, True), (6, 5, False), (None, 5, False)],
)
def test_hit_at_k(rang, k, attendu):
    assert evaluation.hit_at_k(rang, k) is attendu


@pytest.mark.parametrize(
    ("rang", "k", "attendu"),
    [(1, 5, 1.0), (2, 5, 0.5), (5, 5, 0.2), (6, 5, 0.0), (None, 5, 0.0)],
)
def test_reciprocal_rank(rang, k, attendu):
    assert evaluation.reciprocal_rank(rang, k) == pytest.approx(attendu)


# --- Empreinte du corpus --------------------------------------------------

def test_l_empreinte_ne_depend_pas_de_l_ordre():
    """L'ordre de retour de ChromaDB n'est pas garanti : il ne doit pas compter."""
    assert evaluation.corpus_fingerprint(["b", "a", "c"]) == evaluation.corpus_fingerprint(
        ["c", "a", "b"]
    )


def test_l_empreinte_change_avec_le_contenu():
    assert evaluation.corpus_fingerprint(["a", "b"]) != evaluation.corpus_fingerprint(
        ["a", "c"]
    )


def test_l_empreinte_distingue_les_decoupages_des_identifiants():
    """Sans séparateur, ["ab"] et ["a", "b"] produiraient la même empreinte."""
    assert evaluation.corpus_fingerprint(["ab"]) != evaluation.corpus_fingerprint(["a", "b"])


# --- Agrégation d'un rapport ---------------------------------------------

def test_les_metriques_agregees_sont_correctes():
    rapport_test = rapport("t", {"q1": 1, "q2": 2, "q3": None, "q4": 9}, k=5)

    assert rapport_test.count == 4
    assert rapport_test.hits == 2
    assert rapport_test.hit_at_k == pytest.approx(0.5)
    assert rapport_test.mrr == pytest.approx((1.0 + 0.5 + 0.0 + 0.0) / 4)


def test_les_questions_introuvees_sont_listees():
    rapport_test = rapport("t", {"q1": 1, "q2": None})

    assert [resultat.question_id for resultat in rapport_test.misses] == ["q2"]


def test_un_rapport_vide_ne_divise_pas_par_zero():
    vide = EvaluationReport(label="vide", k=5, results=())

    assert vide.hit_at_k == 0.0
    assert vide.mrr == 0.0
    assert vide.mean_latency_ms == 0.0


def test_le_resume_est_recalcule_a_la_relecture():
    """Le bloc « summary » du fichier est décoratif : « results » fait foi.

    Un résumé figé pourrait mentir après une correction manuelle du rapport.
    """
    donnees = rapport("t", {"q1": 1, "q2": None}, k=5).to_dict()
    donnees["summary"]["hit_at_k"] = 0.99  # résumé faussé volontairement

    relu = EvaluationReport.from_dict(donnees)

    assert relu.hit_at_k == pytest.approx(0.5)


def test_un_rapport_survit_a_un_aller_retour_json():
    origine = evaluation.build_report(
        label="baseline",
        k=5,
        results=[resultat("q1", 1), resultat("q2", None)],
        context={"modele_embedding": "bge-m3"},
    )

    relu = EvaluationReport.from_dict(json.loads(json.dumps(origine.to_dict())))

    assert relu.label == origine.label
    assert relu.k == origine.k
    assert [r.rank for r in relu.results] == [1, None]
    assert relu.context == {"modele_embedding": "bge-m3"}


# --- Accord en nombre -----------------------------------------------------

@pytest.mark.parametrize(
    ("nombre", "attendu"),
    [(0, "0 question"), (1, "1 question"), (2, "2 questions"), (11, "11 questions")],
)
def test_l_accord_suit_la_regle_francaise_du_zero_singulier(nombre, attendu):
    """En français, zéro prend le singulier : « 0 question », pas « 0 questions »."""
    assert evaluation.accorder(nombre, "question") == attendu


def test_un_pluriel_irregulier_peut_etre_fourni():
    assert evaluation.accorder(3, "cheval", "chevaux") == "3 chevaux"


# --- Affichage ------------------------------------------------------------

def test_le_rapport_affiche_les_metriques_en_francais():
    texte = evaluation.format_report(rapport("baseline", {"q1": 1, "q2": 2}, k=5))

    assert "baseline" in texte
    assert "hit@5" in texte
    assert "100,0 %" in texte  # virgule décimale et espace avant le %
    assert "MRR@5" in texte


def test_le_rapport_liste_les_questions_non_retrouvees():
    texte = evaluation.format_report(rapport("t", {"q1": 1, "q2": None}, k=5))

    assert "Questions non retrouvées" in texte
    assert "q2" in texte


def test_la_comparaison_compte_progressions_et_regressions():
    avant = rapport("avant", {"q1": 1, "q2": 4, "q3": None}, k=5)
    apres = rapport("apres", {"q1": 1, "q2": 2, "q3": 3}, k=5)

    texte = evaluation.format_comparison(avant, apres)

    assert "Questions mieux classées : 2" in texte
    assert "Questions moins bien classées : 0" in texte
    assert "Inchangées : 1" in texte


def test_la_comparaison_detecte_une_regression():
    avant = rapport("avant", {"q1": 1}, k=5)
    apres = rapport("apres", {"q1": 3}, k=5)

    texte = evaluation.format_comparison(avant, apres)

    assert "Questions moins bien classées : 1" in texte
    assert "Questions mieux classées : 0" in texte


def test_la_comparaison_alerte_si_le_corpus_a_change():
    """Sans cet avertissement, on attribuerait au réglage l'effet des documents."""
    avant = rapport("avant", {"q1": 1}, k=5, empreinte="aaaaaaaaaaaa")
    apres = rapport("apres", {"q1": 1}, k=5, empreinte="bbbbbbbbbbbb")

    texte = evaluation.format_comparison(avant, apres)

    assert "corpus DIFFÉRENT" in texte


def test_la_comparaison_alerte_si_le_top_k_differe():
    avant = rapport("avant", {"q1": 1}, k=5)
    apres = rapport("apres", {"q1": 1}, k=10)

    texte = evaluation.format_comparison(avant, apres)

    assert "top-k différent" in texte


def test_une_comparaison_sans_avertissement_quand_tout_concorde():
    avant = rapport("avant", {"q1": 1}, k=5, empreinte="identique")
    apres = rapport("apres", {"q1": 1}, k=5, empreinte="identique")

    texte = evaluation.format_comparison(avant, apres)

    assert "⚠️" not in texte
