"""Vérifie qu'une réponse est ancrée dans le contexte qui lui a été fourni.

Pourquoi ce module existe
-------------------------
La promesse du produit est que **chaque affirmation provienne des textes cités,
et puisse y être vérifiée**. Cette promesse n'est pas tenue aujourd'hui : sur
24 questions dont la réponse n'est pas dans le corpus, le système a produit
8 fabrications — dont « 1989 » pour la chute du mur de Berlin, et une fatwa sur
les injections appuyée sur un passage traitant du vomi. Le tout correctement cité.

Ce module fournit le seul contrôle qu'on puisse faire SANS juge et SANS rappeler
le modèle : chercher dans le contexte les éléments de la réponse.

Ce qu'il attrape, et ce qu'il n'attrape pas
-------------------------------------------
Mesuré sur les fabrications observées :

- **attrapé** — un élément que le contexte ne contient nulle part. « 1989 » ne
  peut pas venir d'un texte qui ne l'écrit pas ; « فرنسا » non plus ; « البنوك »
  vient de la question, pas des extraits. C'est la classe des faits importés de la
  mémoire du modèle, la plus dangereuse parce qu'ils sont exacts.
- **NON attrapé** — la citation DÉTOURNÉE. À « quel est le jugement sur les
  injections ? » le modèle a répondu « لا شيء عليه », qui est bien dans le
  contexte… mais dans un passage sur le vomi. Tout est dans le texte, et pourtant
  la réponse est fausse. Ce contrôle ne juge pas la PERTINENCE, seulement la
  PRÉSENCE. Le déclarer résolu serait mentir.

⚠️ Le module ne refuse rien de lui-même : il rend une liste. Décider quoi faire
d'une réponse non ancrée est un arbitrage de produit (la refuser ? la signaler ?),
et il dépend du taux de faux positifs — qui doit être mesuré, pas supposé.

Pureté
------
Aucune base vectorielle, aucun appel au modèle, aucun fichier : on lui donne deux
textes, il rend une liste. Testable partout, y compris hors ligne sur des
réponses déjà enregistrées — c'est ce qui permet de mesurer son taux d'erreur sur
des données réelles sans repayer 25 minutes de génération.
"""

import re
from collections.abc import Sequence

from app.lexical import normaliser

# Chiffres arabes-indiens : les ouvrages et les modèles les emploient tous les
# deux, et « ١٩٨٩ » doit se comparer à « 1989 ».
CHIFFRES_ARABES = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

_NOMBRE = re.compile(r"\d+")

# Longueur minimale d'un mot vérifiable. En dessous, ce sont les particules
# (من، في، و) qui n'apportent aucune information vérifiable.
LONGUEUR_MINIMALE = 4


def normaliser_chiffres(texte: str) -> str:
    """Ramène les chiffres arabes-indiens aux chiffres latins."""
    return texte.translate(CHIFFRES_ARABES)


def nombres(texte: str) -> list[str]:
    """Les nombres écrits en chiffres, dans l'ordre d'apparition."""
    return _NOMBRE.findall(normaliser_chiffres(texte))


def mots_verifiables(texte: str) -> list[str]:
    """Les mots d'un texte qui portent un contenu vérifiable.

    On garde les mots assez longs pour ne pas être des particules. ⚠️ Aucune
    analyse morphologique : « كرهها » et « كره » restent deux chaînes
    différentes. C'est acceptable ici et PAS ailleurs — dans une RÉPONSE le
    modèle recopie le contexte, alors qu'une QUESTION est reformulée par
    l'utilisateur, avec ses propres flexions. Mesuré : ce critère appliqué aux
    questions refusait à tort 11 questions répondables sur 57, toutes pour des
    raisons de morphologie.
    """
    return [mot for mot in normaliser(texte).split() if len(mot) >= LONGUEUR_MINIMALE]


def elements_absents(reponse: str, contexte: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Éléments de la réponse introuvables dans le contexte.

    Retourne ``(nombres_absents, mots_absents)``, séparés parce qu'ils n'ont pas
    la même force :

    - un **nombre** absent est une preuve : il ne peut pas avoir été lu dans un
      texte qui ne l'écrit pas ;
    - un **mot** absent est un soupçon : la morphologie arabe fait qu'un mot
      présent au fond peut manquer à la forme (« كرهها » vs « كره »).
    """
    nombres_du_contexte = set(nombres(contexte))
    mots_du_contexte = set(normaliser(contexte).split())

    nombres_absents = tuple(
        nombre for nombre in nombres(reponse) if nombre not in nombres_du_contexte
    )
    mots_absents = tuple(
        dict.fromkeys(
            mot for mot in mots_verifiables(reponse) if mot not in mots_du_contexte
        )
    )
    return nombres_absents, mots_absents


def est_ancree(reponse: str, contexte: str) -> bool:
    """La réponse ne contient-elle AUCUN nombre absent du contexte ?

    C'est le seul verdict que ce module rende, et il est volontairement étroit :
    il porte sur les nombres, où l'absence est une preuve. Un verdict large
    inclurait les mots, dont l'absence n'est qu'un soupçon — et refuser une
    réponse juste coûte plus cher que laisser passer une réponse douteuse, tant
    que le taux de faux positifs n'a pas été mesuré sur un jeu suffisant.
    """
    nombres_absents, _ = elements_absents(reponse, contexte)
    return not nombres_absents


def exemples(elements: Sequence[str], maximum: int = 5) -> str:
    """Rend une liste d'éléments lisible dans un message (bornée)."""
    montres = list(elements[:maximum])
    reste = len(elements) - len(montres)
    texte = ", ".join(montres)
    return f"{texte} (+{reste})" if reste > 0 else texte
