"""Tests de app/language.py — détection arabe / français.

Ce module est le FONDEMENT du respect de la langue : s'il se trompe, le chatbot
soit relance le modèle pour rien (faux positif, coût GPU inutile), soit laisse
passer une réponse dans la mauvaise langue (faux négatif, le bug d'origine).

Les deux directions sont donc testées, ainsi que les cas où la fonction doit
**refuser de conclure** — la valeur `None` n'est pas une erreur, c'est une
information : « je ne sais pas ».
"""

from app import config, language

# --- Cas nominaux ---------------------------------------------------------

def test_detecte_le_francais():
    assert language.detect_language("La règle est claire et précise.") == "fr"


def test_detecte_le_francais_accentue():
    """Les lettres accentuées doivent compter comme du latin (contrôle Unicode)."""
    assert language.detect_language("Où est l'élève ? À côté.") == "fr"


def test_detecte_l_arabe():
    assert language.detect_language("هذه إجابة بالعربية عن السؤال") == "ar"


def test_detecte_l_arabe_sans_lettres_diacritiques():
    assert language.detect_language("الصلاة واجبة على كل مسلم") == "ar"


def test_detecte_l_arabe_malgre_un_nom_de_fichier_latin():
    """Un « a.pdf » cité dans une réponse arabe ne doit pas tromper le comptage."""
    reponse = "الصلاة واجبة كما ورد في الملف a.pdf على وجه التحديد"

    assert language.detect_language(reponse) == "ar"


def test_detecte_le_francais_citant_un_mot_arabe():
    """Une réponse française qui cite un mot arabe entre guillemets reste française."""
    reponse = "Le document indique « الصلاة » puis en précise les conditions."

    assert language.detect_language(reponse) == "fr"


# --- Cas où la fonction refuse de conclure --------------------------------

def test_texte_vide_est_indetermine():
    assert language.detect_language("") is None


def test_texte_absent_est_indetermine():
    assert language.detect_language(None) is None


def test_texte_sans_lettres_est_indetermine():
    """Chiffres et ponctuation : aucune écriture, donc aucune conclusion possible."""
    assert language.detect_language("1234 - 56,78 !! (90 %)") is None


def test_chiffres_arabes_indiens_ne_comptent_pas_comme_arabe():
    """Les chiffres arabo-indiens (١٢٣) ne sont PAS des lettres arabes.

    Piège réel : leur nom Unicode commence par « ARABIC », donc un contrôle
    qui se fierait au seul préfixe du nom les compterait à tort. C'est
    `isalpha()` qui les élimine — ce test verrouille ce comportement.
    """
    assert language.detect_language("١٢٣٤٥٦٧٨٩٠") is None


def test_texte_reellement_melange_est_indetermine():
    """5 lettres latines contre 7 arabes : aucune écriture ne domine."""
    # Concaténation volontaire : ruff refuse (RUF001) les littéraux qui mélangent
    # les alphabets — or ici, le mélange est précisément le sujet du test.
    assert language.detect_language("abcde" + "المدرسة") is None


def test_une_ecriture_dominante_suffit_a_conclure():
    """11 lettres latines contre 3 arabes : le latin domine nettement."""
    assert language.detect_language("abcdefg hijk" + "نعم") == "fr"


# --- Seuil de dominance ---------------------------------------------------

def test_le_seuil_est_respecte():
    """7 latines contre 3 arabes = 70 % : juste à la limite, doit conclure."""
    texte = "abcdefg" + "نعم"

    assert language.detect_language(texte) == "fr"


def test_un_seuil_plus_exigeant_rend_le_texte_indetermine():
    texte = "abcdefg" + "نعم"

    assert language.detect_language(texte, threshold=0.9) is None


# --- Configuration --------------------------------------------------------

def test_la_verification_de_langue_est_activee_par_defaut():
    """« Secure by default » : la protection doit être active sans configuration."""
    assert config.LANGUAGE_ENFORCEMENT_ENABLED is True


def test_le_nombre_de_reprises_est_borné():
    assert config.LANGUAGE_MAX_RETRIES >= 0
