"""Tests de la recherche lexicale et de la fusion de classements.

Ce module est PUR : ni ChromaDB, ni Ollama, ni fichier. Les tests portent donc
sur ce qui décide du résultat — la normalisation des mots, le classement BM25,
et la façon dont deux classements se combinent.
"""

import pytest

from app.lexical import IndexLexical, decouper, fusionner_rrf, normaliser

# --- Normalisation : deux graphies d'un même mot doivent se rencontrer ------


def test_les_diacritiques_sont_ignores():
    """Un texte vocalisé et le même sans voyelles sont le même texte.

    Sans cela, « الصَّلَاة » (tel qu'un EPUB le fournit) et « الصلاة » (tel
    qu'un lecteur l'écrit) seraient deux mots étrangers l'un à l'autre.
    """
    assert normaliser("الصَّلَاة") == normaliser("الصلاة")


def test_le_tatweel_est_ignore():
    """Le tatweel n'allonge que le tracé : il ne change pas le mot."""
    assert decouper("مـــال") == decouper("مال")


@pytest.mark.parametrize(
    ("graphie", "autre"),
    [
        ("إجماع", "اجماع"),
        ("أبي", "ابي"),
        ("آية", "ايه"),
        ("موسى", "موسي"),
    ],
)
def test_les_variantes_de_lettres_sont_unifiees(graphie, autre):
    """Aléf, tâ marbûta et hamza s'écrivent de plusieurs façons selon l'éditeur."""
    assert decouper(graphie) == decouper(autre)


def test_les_mots_vides_sont_ecartes():
    """« في », « عن », « ابن » sont partout : ils ne désignent aucun sujet."""
    assert decouper("الإجماع في الصلاة عن ابن المنذر") == ["الاجماع", "الصلاه", "المنذر"]


def test_la_ponctuation_arabe_ne_colle_pas_au_dernier_mot():
    """Toute question arabe se termine par « ؟ ».

    Une plage de caractères trop large l'aurait avalé, collant la ponctuation au
    dernier mot — le terme le plus porteur d'une question — et ce mot n'aurait
    plus correspondu à rien. Aucun message n'aurait signalé la perte.
    """
    assert decouper("كيف يرث الخنثى؟") == ["يرث", "الخنثي"]
    assert decouper("البيع، والشراء؛ والربا") == ["البيع", "والشراء", "والربا"]


def test_les_mots_trop_courts_sont_ecartes():
    """Les particules grammaticales arabes (و، ب، ل) n'ont pas de longueur utile."""
    assert decouper("و ب ل ما لا") == []


def test_les_chiffres_arabes_indiens_sont_ramenes_aux_chiffres_latins():
    """Les éditions de Shamela numérotent en chiffres arabes-indiens."""
    assert "81" in normaliser("الصفحة: ٨١")


# --- Classement lexical ----------------------------------------------------


@pytest.fixture(name="index")
def index_fixture() -> IndexLexical:
    """Un petit index où un seul chunk parle de الخنثى (l'hermaphrodite).

    Le corpus imite le vrai : plusieurs chunks de chaînes de transmetteurs, très
    longs et presque identiques, et UN chunk qui porte le mot distinctif noyé au
    milieu.
    """
    chaine = "حدثنا علي بن المبارك قال حدثنا زيد عن ابن ثور عن ابن جريج"
    return IndexLexical([
        ("m1", f"{chaine} في ميراث الرجل والمراة والولد والاب والام والزوجة"),
        ("m2", f"{chaine} في ميراث الرجل والمراة والشريك والاخ والاخت"),
        ("m3", f"{chaine} في ميراث الرجل والمراة والولد والاب والام والزوجة"),
        ("m4", f"{chaine} مسالة في ميراث الخنثى الذي لا يعرف اذكر هو ام انثى"),
    ])


def test_le_chunk_portant_le_mot_rare_est_classe_premier(index):
    """Le fait décisif : le mot rare pèse plus que la formule répétée.

    Les quatre chunks contiennent « حدثنا… عن… » ; un seul contient الخنثى.
    Le vecteur confond les quatre, BM25 non — c'est toute la raison d'être de la
    recherche lexicale.
    """
    assert index.classer("كيف يرث الخنثى؟", limite=4)[0] == "m4"


def test_un_mot_absent_de_l_index_n_empeche_pas_de_classer(index):
    """Une question peut contenir un mot inconnu du corpus : les autres comptent."""
    resultats = index.classer("الخنثى في بلاد بعيدة جدا", limite=4)

    assert resultats[0] == "m4"


def test_une_question_sans_aucun_mot_connu_ne_renvoie_rien(index):
    """Faute de mot commun, il n'y a pas de classement — et surtout pas un ordre arbitraire."""
    assert index.classer("كلمات لا توجد في هذا النص ابدا", limite=4) == []


def test_le_nombre_de_resultats_demande_est_respecte(index):
    assert len(index.classer("ميراث الخنثى", limite=2)) == 2


def test_un_document_court_est_favorise_a_egalite_de_frequence():
    """La normalisation par la longueur, sans laquelle le plus long gagne toujours.

    Le chunk long répète le mot deux fois, le court une seule : le court doit
    tout de même l'emporter, sinon les chunks qui empilent les chaînes de
    transmetteurs seraient systématiquement avantagés.
    """
    index = IndexLexical([
        ("court", "الخنثى ميراث"),
        ("long", "الخنثى الخنثى " + " ".join(["وقال"] * 60)),
    ])

    assert index.classer("الخنثى", limite=2)[0] == "court"


def test_un_index_vide_ne_provoque_pas_d_erreur():
    """Le cas se produit réellement : base vectorielle vide, avant toute ingestion."""
    index = IndexLexical([])

    assert index.taille == 0
    assert index.classer("سؤال", limite=5) == []


def test_la_taille_est_le_nombre_de_chunks_indexes(index):
    assert index.taille == 4


def test_le_classement_authentique_reproduit_le_cas_du_tafsir():
    """Le cas mesuré : le verset existe, mais le vecteur ne le renvoie jamais.

    Le chunk qui contient le verset est entouré de longues chaînes de
    transmetteurs, exactement comme dans `تفسير ابن المنذر`. La question cite le
    verset : le classement lexical doit le mettre en tête.
    """
    chaine = " ".join(["حدثنا احمد بن شبيب قال حدثنا يزيد عن سعيد عن قتادة"] * 3)
    index = IndexLexical([
        ("v1", f"{chaine} في تفسير اية الدين والمعاملات"),
        ("v2", f"{chaine} ثم صرفكم عنهم ليبتليكم قال يعني بذلك يوم احد"),
        ("v3", f"{chaine} في تفسير اية الوضوء والصلاة"),
    ])

    assert index.classer("ما معنى ثم صرفكم عنهم ليبتليكم", limite=3)[0] == "v2"


# --- Fusion de classements -------------------------------------------------


def test_un_document_bien_classe_par_les_deux_recherches_l_emporte():
    """Le principe de la fusion : deux avis concordants valent mieux qu'un seul.

    « b » est deuxième pour l'une et premier pour l'autre ; « a » n'est vu que
    par la première, « c » que par la seconde. C'est « b » qui doit sortir en
    tête — sans quoi fusionner ne servirait à rien (on aurait juste choisi une
    des deux recherches).
    """
    fusion = fusionner_rrf(
        [["a", "b"], ["b", "c"]], poids=[1.0, 1.0], constante=60, limite=2
    )

    assert fusion == ["b", "a"]


def test_un_document_que_la_seule_recherche_lexicale_connait_est_conserve():
    """Le cas réel : le passage que le vecteur ignore est sauvé par le lexique."""
    fusion = fusionner_rrf(
        [["a", "b", "c"], ["z", "a", "b"]], poids=[3.0, 1.0], constante=60, limite=4
    )

    assert "z" in fusion


def test_le_poids_relatif_change_l_ordre():
    """Un poids plus fort emporte le désaccord : c'est le levier de réglage.

    Les deux recherches ne sont pas d'accord — « a » premier pour l'une,
    « b » premier pour l'autre. Le poids décide lequel est cru.
    """
    avec_vecteur_fort = fusionner_rrf(
        [["a"], ["b"]], poids=[3.0, 1.0], constante=60, limite=2
    )
    avec_lexical_fort = fusionner_rrf(
        [["a"], ["b"]], poids=[1.0, 3.0], constante=60, limite=2
    )

    assert avec_vecteur_fort == ["a", "b"]
    assert avec_lexical_fort == ["b", "a"]


def test_la_constante_decide_entre_excellence_et_consensus():
    """La constante arbitre entre « excellent une fois » et « correct deux fois ».

    « solo » est premier d'une seule recherche, et absent de l'autre.
    « consensus » est 50e des deux. C'est le réglage qui départage :
      · constante élevée (60) : consensus gagne — être vu par les deux compte plus ;
      · constante faible (1)  : solo gagne — le rang 1 vaut alors 1/2, écrasant.
    """
    # Les deux remplissages doivent être DISJOINTS : un même identifiant présent
    # dans les deux classements gagnerait par consensus, et le test mesurerait
    # autre chose que ce qu'il annonce.
    premier = ["solo"] + [f"p{i}" for i in range(1, 49)] + ["consensus"]
    second = [f"s{i}" for i in range(1, 49)] + ["consensus"]

    doux = fusionner_rrf([premier, second], [1.0, 1.0], constante=60, limite=5)
    dur = fusionner_rrf([premier, second], [1.0, 1.0], constante=1, limite=5)

    assert doux[0] == "consensus"
    assert dur[0] == "solo"


def test_la_fusion_respecte_la_limite_demandee():
    fusion = fusionner_rrf(
        [["a", "b", "c", "d"], ["e", "f", "g"]], poids=[1.0, 1.0], constante=60, limite=3
    )

    assert len(fusion) == 3


def test_la_fusion_refuse_des_poids_en_nombre_different():
    """Une faute de comptage silencieuse fausserait les poids sans rien dire."""
    with pytest.raises(ValueError):
        fusionner_rrf([["a"], ["b"]], poids=[1.0], constante=60, limite=2)


def test_la_fusion_accepte_un_classement_vide():
    """La recherche lexicale rend une liste vide quand aucun mot ne correspond."""
    fusion = fusionner_rrf([["a", "b"], []], poids=[3.0, 1.0], constante=60, limite=2)

    assert fusion == ["a", "b"]
