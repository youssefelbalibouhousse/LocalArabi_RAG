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
