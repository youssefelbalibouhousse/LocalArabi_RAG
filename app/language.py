"""Détection de la langue d'un texte : arabe ou français.

Pourquoi un simple COMPTAGE DE CARACTÈRES plutôt qu'un modèle de détection ?

Parce que l'arabe et le français utilisent des **alphabets différents**. Il
suffit donc de compter les lettres de chaque écriture — et le résultat est :

- **déterministe** : aucune part de hasard, contrairement à un modèle ;
- **instantané** : quelques microsecondes, aucun téléchargement ;
- **fiable même sur un texte court** : un détecteur probabiliste hésite sur
  « نعم » (3 lettres), alors que l'alphabet, lui, ne laisse aucun doute.

Cette fiabilité vient du choix des langues du projet. Détecter l'espagnol et le
français demanderait, à l'inverse, un vrai modèle statistique.
"""

import unicodedata

# Part à partir de laquelle une écriture est considérée comme dominante.
# En dessous, le texte est jugé MÉLANGÉ et la fonction refuse de se prononcer.
LANGUAGE_DOMINANCE_THRESHOLD = 0.7


def _count_scripts(text: str) -> tuple[int, int]:
    """Compte les lettres arabes et latines (chiffres et ponctuation ignorés)."""
    arabic = latin = 0

    for char in text:
        if not char.isalpha():
            continue

        # `unicodedata.name` donne le nom Unicode officiel du caractère :
        # « ARABIC LETTER BEH », « LATIN SMALL LETTER E WITH ACUTE »...
        #
        # Se fier à ce nom évite d'oublier un bloc de code — arabe étendu
        # (0x0750), formes de présentation (0xFE70)… — comme le ferait une
        # liste d'intervalles écrite à la main.
        name = unicodedata.name(char, "")

        if name.startswith("ARABIC"):
            arabic += 1
        elif name.startswith("LATIN"):
            latin += 1

    return arabic, latin


def detect_language(text: str, threshold: float = LANGUAGE_DOMINANCE_THRESHOLD):
    """Retourne "fr", "ar", ou None quand la langue ne peut pas être établie.

    None couvre trois cas qu'il faut savoir distinguer d'une erreur :

    1. le texte est vide ;
    2. le texte ne contient aucune lettre (chiffres, ponctuation, emoji) ;
    3. le texte est réellement **mélangé** (aucune écriture n'atteint le seuil).

    Dans ces trois cas, l'appelant doit S'ABSTENIR de conclure. C'est le sens de
    cette valeur : « je ne sais pas » n'est pas « c'est faux ».
    """
    if not text:
        return None

    arabic, latin = _count_scripts(text)
    total = arabic + latin
    if total == 0:
        return None

    if arabic / total >= threshold:
        return "ar"
    if latin / total >= threshold:
        return "fr"
    return None
