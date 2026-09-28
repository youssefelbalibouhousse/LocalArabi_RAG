"""Tests de scripts/benchmark_embeddings.py.

Le script interroge un serveur réel, ce qu'aucun test ne peut faire. Mais ce qui
DÉCIDE est du calcul pur : projeter une durée, et juger si la taille du lot tient
dans le délai d'expiration. C'est ce calcul qui est verrouillé ici — parce que
c'est lui qui a coûté une panne réelle (« timed out in add »), et parce qu'il
servira à engager de l'argent sur une machine louée.
"""

import benchmark_embeddings as bench
import pytest

from app import config

# --- Mise en forme --------------------------------------------------------

@pytest.mark.parametrize(
    ("valeur", "attendu"),
    [(1.5, "1,5"), (2.0, "2,0"), (12.34, "12,3")],
)
def test_formater_nombre_utilise_la_virgule(valeur, attendu):
    """Comme les rapports d'évaluation : les chiffres se lisent à la française."""
    assert bench.formater_nombre(valeur) == attendu


@pytest.mark.parametrize(
    ("secondes", "attendu"),
    [
        (1.5, "1,5 s"),
        (45, "45,0 s"),
        (120, "2 min"),
        (7200, "2,0 h"),
        (172800, "2,0 jours"),
    ],
)
def test_formater_duree_change_d_unite_a_la_bonne_echelle(secondes, attendu):
    """« 172800 s » ne se lit pas ; « 2,0 jours » dit ce qu'il faut décider."""
    assert bench.formater_duree(secondes) == attendu


# --- Projection -----------------------------------------------------------

def test_projeter_multiplie_le_temps_par_chunk():
    assert bench.projeter(500, 1000) == 500


def test_projeter_sur_zero_chunk_ne_dure_pas():
    assert bench.projeter(500, 0) == 0


# --- Cohérence entre la taille du lot et le délai d'expiration ------------

def test_un_lot_qui_tient_largement_est_accepte():
    coherent, message = bench.verdict_coherence(taille_lot=32, ms_par_chunk=550, timeout_s=120)

    assert coherent is True
    assert "✅" in message


def test_un_lot_qui_depasse_le_delai_est_refuse():
    """Le cas réel : 256 chunks × 0,55 s = 141 s, pour un délai de 60 s."""
    coherent, message = bench.verdict_coherence(taille_lot=256, ms_par_chunk=550, timeout_s=60)

    assert coherent is False
    assert "DÉPASSE" in message


def test_une_marge_faible_est_signalee_sans_etre_acceptee():
    """2× de marge passe ici et casse ailleurs : ce n'est pas assez."""
    coherent, message = bench.verdict_coherence(taille_lot=64, ms_par_chunk=550, timeout_s=120)

    assert coherent is False
    assert "marge faible" in message


def test_la_regression_du_25_juin_reste_detectee():
    """Le réglage exact qui a échoué en production doit être refusé, toujours.

    256 chunks par lot et le délai par défaut de la bibliothèque ChromaDB (60 s)
    : l'écriture échouait sur un « timed out in add » sans rapport apparent avec
    sa cause. Si ce test passe au vert, c'est que le verdict s'est ramolli.
    """
    coherent, _ = bench.verdict_coherence(taille_lot=256, ms_par_chunk=550, timeout_s=60)

    assert coherent is False


# --- Échantillon ----------------------------------------------------------

def test_sans_corpus_l_echantillon_est_fabrique_et_signale_comme_tel():
    textes, reels = bench.echantillon(None, 4)

    assert len(textes) == 4
    assert reels is False
    assert all(len(texte) == config.CHUNK_SIZE for texte in textes)
