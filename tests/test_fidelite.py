"""Tests de app/fidelite.py — la vérification qu'une réponse est ancrée.

Ce module porte la promesse du produit : rien ne doit être inventé, tout doit
pouvoir être vérifié dans le texte cité. Il est donc PUR — aucun modèle, aucune
base — ce qui permet de le mesurer hors ligne sur des réponses déjà enregistrées.
"""

import pytest

from app import fidelite

# --- Extraction des éléments vérifiables ---------------------------------

def test_les_chiffres_latins_et_arabes_sont_comparables():
    """« ١٩٨٩ » et « 1989 » sont le même nombre : les ouvrages emploient les deux."""
    assert fidelite.nombres("سنة ١٩٨٩") == ["1989"]


def test_les_nombres_sont_extraits_dans_l_ordre():
    assert fidelite.nombres("ثلاثة 40 ثم 60") == ["40", "60"]


def test_les_mots_trop_courts_sont_ignores():
    """« من », « في », « و » ne portent aucun contenu vérifiable."""
    assert "من" not in fidelite.mots_verifiables("من في و ما")


# --- Le contrôle de fidélité ---------------------------------------------

CONTEXTE = "125 - وأجمعوا على أنه لا شيء على الصائم إذا ذرعه القيء"


def test_un_nombre_absent_du_contexte_est_signale():
    """Le cas mesuré : à « en quelle année est tombé le mur de Berlin ? » le
    système a répondu « سنة 1989 » — un fait exact, absent de tout le corpus,
    tiré de la mémoire du modèle."""
    nombres_absents, _ = fidelite.elements_absents("سنة 1989.", CONTEXTE)

    assert nombres_absents == ("1989",)


def test_un_nombre_present_dans_le_contexte_n_est_pas_signale():
    nombres_absents, _ = fidelite.elements_absents("وأجمعوا على 125 حكمًا.", CONTEXTE)

    assert nombres_absents == ()


def test_un_mot_absent_est_signale_separement_des_nombres():
    """Les deux n'ont pas la même force : un nombre absent est une preuve, un mot
    absent n'est qu'un soupçon (morphologie arabe)."""
    nombres_absents, mots_absents = fidelite.elements_absents(
        "يجب التأمين على السيارات.", CONTEXTE
    )

    assert nombres_absents == ()
    assert "التامين" in mots_absents


def test_une_reponse_recopiee_du_contexte_est_ancree():
    """Le cas normal : le modèle cite le passage, donc ses mots y sont."""
    reponse = "قال: لا شيء على الصائم إذا ذرعه القيء."

    assert fidelite.est_ancree(reponse, CONTEXTE)


def test_une_reponse_avec_un_nombre_invente_n_est_pas_ancree():
    assert not fidelite.est_ancree("سنة 1989.", CONTEXTE)


def test_un_verdict_vide_ne_signale_rien():
    """Sans réponse, il n'y a rien à vérifier — et surtout pas un refus automatique."""
    nombres_absents, mots_absents = fidelite.elements_absents("", CONTEXTE)

    assert nombres_absents == ()
    assert mots_absents == ()


def test_le_verdict_ignore_les_mots_qui_ne_sont_que_suspects():
    """`est_ancree` ne porte que sur les nombres, volontairement.

    Un verdict large inclurait les mots absents — et refuser une réponse juste
    coûte plus cher que laisser passer une réponse douteuse tant que le taux de
    faux positifs n'est pas mesuré.
    """
    reponse = "يجب التأمين على السيارات."  # deux mots absents, aucun nombre

    assert fidelite.est_ancree(reponse, CONTEXTE)


@pytest.mark.parametrize(
    ("elements", "attendu"),
    [
        ((), ""),
        (("a",), "a"),
        (("a", "b", "c", "d", "e"), "a, b, c, d, e"),
        (("a", "b", "c", "d", "e", "f"), "a, b, c, d, e (+1)"),
    ],
)
def test_exemples_borne_l_affichage(elements, attendu):
    assert fidelite.exemples(elements) == attendu

# --- La citation verbatim ------------------------------------------------
#
# Seconde voie, après l'échec de la première : chercher les ÉLÉMENTS d'une
# réponse libre ne marche pas, parce que les inventions du modèle sont faites du
# vocabulaire du corpus. On ne vérifie donc plus la réponse, on vérifie la
# CITATION que le modèle doit produire.

CONTEXTE_CITATION = (
    "قال جابر: بايعناه وعمر آخذ بيده تحت الشجرة وهي سمرة، بايعناه على أن لا نفر."
)


def test_les_citations_sont_extraites_entre_les_delimiteurs():
    reponse = "الجواب: [[قال جابر: بايعناه تحت الشجرة]]. وهذا يدل على الجواز."

    assert fidelite.citations(reponse) == ("قال جابر: بايعناه تحت الشجرة",)


def test_plusieurs_citations_sont_extraites_dans_l_ordre():
    reponse = "أولا [[النص الأول]] ثم [[النص الثاني]]."

    assert fidelite.citations(reponse) == ("النص الأول", "النص الثاني")


def test_une_reponse_sans_citation_n_en_contient_aucune():
    """C'est un FAIT, pas un jugement : une réponse sans citation n'est pas vérifiable."""
    assert fidelite.citations("لا بأس به.") == ()


def test_un_delimiteur_ouvert_sans_fermeture_est_ignore():
    """Ne PAS deviner où la citation s'arrête : mieux vaut ne rien extraire."""
    assert fidelite.citations("الجواب [[نص مقطوع") == ()


def test_une_citation_recopiee_est_verifiee():
    reponse = f"الجواب: [[{CONTEXTE_CITATION}]]"

    assert fidelite.citations_non_verifiees(reponse, CONTEXTE_CITATION) == ()


def test_une_citation_absente_du_contexte_est_signalee():
    """Le cas qui compte : une citation que le texte ne contient pas est FABRIQUÉE.

    C'est le seul cas où l'invention est mécaniquement démontrable — et c'est
    exactement ce que la première voie ne savait pas faire.
    """
    reponse = "الجواب: [[يحرم استخدام مكبر الصوت في الأذان]]"

    non_verifiees = fidelite.citations_non_verifiees(reponse, CONTEXTE_CITATION)

    assert non_verifiees == ("يحرم استخدام مكبر الصوت في الأذان",)


def test_la_verification_tolere_diacritiques_et_espaces():
    """Le modèle recopie avec ou sans voyelles, et change parfois les espaces.

    La comparaison reste littérale — c'est la PHRASE qui doit s'y trouver — mais
    elle ne doit pas échouer sur ce qui ne change pas le texte.
    """
    contexte = "قَالَ جَابِرٌ: بَايَعْنَاهُ وَعُمَرُ آخِذٌ بِيَدِهِ تَحْتَ الشَّجَرَةِ"
    reponse = "الجواب: [[قال جابر: بايعناه وعمر آخذ بيده تحت الشجرة]]"

    assert fidelite.citations_non_verifiees(reponse, contexte) == ()


def test_une_citation_partielle_ne_passe_pas():
    """Vérifier la PHRASE et non ses mots : c'est tout l'intérêt du mécanisme.

    Un contrôle par mots validerait cette citation, parce que chacun de ses mots
    existe quelque part dans le contexte — et c'est précisément l'échec mesuré de
    la première voie.
    """
    reponse = "الجواب: [[قول آخر قال به بعض الناس]]"

    assert len(fidelite.citations_non_verifiees(reponse, CONTEXTE_CITATION)) == 1


def test_les_citations_valides_et_fabriquees_sont_separees():
    reponse = (
        f"الجواب: [[{CONTEXTE_CITATION}]] "
        "ثم [[نص لا وجود له في السياق]]"
    )

    non_verifiees = fidelite.citations_non_verifiees(reponse, CONTEXTE_CITATION)

    assert non_verifiees == ("نص لا وجود له في السياق",)
