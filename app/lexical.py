"""Recherche lexicale (BM25) et fusion de classements.

Pourquoi ce module
------------------
Un vecteur résume un texte entier en un point. Sur un corpus où des centaines de
passages ne diffèrent que par leur chaîne de transmetteurs (« حدّثنا فلان عن
فلان… »), deux passages sans rapport se ressemblent énormément : le vecteur a
perdu le petit nombre de mots qui les distingue. Une question qui **cite** un
verset est alors noyée parmi cent mots de formule.

La recherche lexicale fait l'inverse : elle ne compte que les mots, et un mot
rare pèse très lourd (c'est l'IDF de BM25). Elle retrouve donc ce que le vecteur
confond. Symétriquement, elle échoue là où le vecteur réussit — une question
formulée avec d'autres mots que le texte. Les deux se trompent différemment :
c'est précisément pour cela que les **fusionner** vaut mieux que choisir.

Mesuré sur les 59 questions du jeu d'or (3 355 chunks, 3 ouvrages) :

| recherche            | hit@10 | MRR@10 |
|----------------------|--------|--------|
| vectorielle seule    | 76,3 % | 0,53   |
| lexicale seule       | 72,9 % | 0,55   |
| **fusion (RRF)**     | **88,1 %** | **0,62** |

⚠️ **La liste de mots vides et la longueur minimale font partie de la
configuration MESURÉE.** Elles sont donc dans le code, pas dans une constante
qu'on ajuste au hasard : les modifier invalide le tableau ci-dessus, et il faut
refaire la mesure avant de croire à un changement.

Pureté
------
Ce module n'importe ni ChromaDB ni Ollama, ne lit aucun fichier et n'a aucun
effet de bord : on lui donne des textes, il rend des classements. Il est donc
testable sans base vectorielle ni modèle — même doctrine que `app/epub.py` et
`app/evaluation.py`.
"""

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

# Paramètres de BM25. Les valeurs 1,5 et 0,75 sont les valeurs usuellement
# retenues ; elles ne sont pas ajustées ici, faute de quoi l'ajustement serait
# indistinguable d'un surapprentissage sur 59 questions.
K1 = 1.5
B = 0.75

# Longueur minimale d'un mot retenu. En dessous, ce sont des particules
# grammaticales arabes (و، ب، ل، ما، لا) qui n'apportent aucun sujet.
LONGUEUR_MINIMALE = 3

# Mots-outils : présents partout, donc incapables de désigner un sujet.
#
# L'IDF les rend déjà presque nuls, mais pas tout à fait — et un mot comme
# « الذي » apparaît dans un fragment de texte sur deux. Les retirer rend aussi
# le classement plus lisible au diagnostic.
#
# Les mots interrogatifs (كيف، هل، متى) sont inclus : dans une question, ils
# portent la FORME de la demande, jamais son sujet. Toutes les questions les
# partageant, les garder reviendrait à ajouter un mot identique partout.
MOTS_VIDES = frozenset({
    # Prépositions, conjonctions, pronoms
    "في", "من", "على", "عن", "الى", "ان", "ما", "لا", "هذا", "هذه", "ذلك",
    "التي", "الذي", "الذين", "به", "بها", "له", "لها", "ثم", "قد", "كان",
    "هو", "هي", "كل", "بين", "عند", "غير", "اذا", "اما", "اي", "ولا", "ولم",
    "لم", "لما", "كما", "حتى", "بعد", "قبل", "او", "و", "ف", "ب", "ل",
    # Termes de transmission : le squelette de toute chaîne de transmetteurs.
    # Ce sont les mots les plus fréquents du corpus (`تفسير ابن المنذر` en
    # compte 1 562 chunks) : ils ne distinguent rien, ils noient.
    "قال", "قوله", "حدثنا", "اخبرنا", "الك", "الل", "وسلم", "عنه", "علي",
    "بن", "ابن", "ابي", "ابيه", "صلى", "عليه", "رسول", "الله", "النبي",
    # Mots interrogatifs
    "كيف", "هل", "متى", "اين", "ماذا", "لماذا",
})

# Aléf et yâ portent des variantes qui ne changent pas le mot : « إجماع » et
# « اجااع » doivent se rencontrer. La tâ marbûta s'écrit souvent comme un hâ en
# fin de mot. Sans ces substitutions, deux graphies d'un même mot deviennent
# deux mots étrangers, et le recouvrement lexical tombe à zéro par accident.
EQUIVALENCES = (
    ("أإآٱ", "ا"),
    ("ى", "ي"),
    ("ة", "ه"),
    ("ؤ", "و"),
    ("ئ", "ي"),
)

CHIFFRES_ARABES = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

# ⚠️ Mesuré : ne JAMAIS écrire ici la plage `\u0600-\u06FF`. Elle contient la
# ponctuation arabe — le point d'interrogation « ؟ » (U+061F), la virgule
# « ، » (U+060C), le point-virgule « ؛ » (U+061B). Le « ؟ » final de chaque
# question se collait alors au dernier mot (« الخنثي؟ »), et ce mot ne
# correspondait plus à rien : une question entière perdait son terme le plus
# porteur, sans le moindre message. `\w` couvre déjà les lettres arabes et
# laisse la ponctuation de côté.
_MOT = re.compile(r"\w+")


def normaliser(texte: str) -> str:
    """Ramène un texte arabe à une forme comparable.

    Retirés : les diacritiques (harakât) et le tatweel, purement ornementaux à
    l'écrit ; les variantes de aléf, yâ, tâ marbûta, hamza (voir ``EQUIVALENCES``).
    Les chiffres arabes-indiens sont ramenés aux chiffres latins.
    """
    # Les diacritiques sont des caractères combinants : on les écarte par
    # catégorie Unicode plutôt que par une liste, qui serait toujours incomplète.
    texte = "".join(c for c in texte if not unicodedata.combining(c))
    texte = texte.replace("\u0640", "")
    texte = unicodedata.normalize("NFKC", texte)
    for source, cible in EQUIVALENCES:
        for lettre in source:
            texte = texte.replace(lettre, cible)
    return texte.translate(CHIFFRES_ARABES).lower()


def decouper(texte: str) -> list[str]:
    """Les mots comparables d'un texte : normalisés, hors mots-outils."""
    return [
        mot
        for mot in _MOT.findall(normaliser(texte))
        if len(mot) >= LONGUEUR_MINIMALE and mot not in MOTS_VIDES
    ]


@dataclass(frozen=True)
class Correspondance:
    """Un mot et le rang du document qui le contient, avec sa fréquence.

    C'est une entrée de « liste de correspondances » (postings list) : à chaque
    mot, les documents où il apparaît. Parcourir cette liste est bien plus
    rapide que relire tous les documents pour chaque mot de la question.
    """

    rang: int
    frequence: int


class IndexLexical:
    """Index BM25 en mémoire, construit à partir des chunks de la collection.

    Mémoire : proportionnelle au corpus (les textes et un index inversé). C'est
    le prix assumé de cette approche — voir `config.HYBRID_CANDIDATES` et la
    note de passage à l'échelle dans `README.md`.
    """

    def __init__(self, chunks: Iterable[tuple[str, str]]) -> None:
        """Construit l'index.

        `chunks` : des couples (identifiant, texte). L'identifiant est celui de
        ChromaDB — c'est lui qui relie un classement lexical à un extrait.
        """
        self._identifiants: list[str] = []
        self._mots: list[list[str]] = []
        self._index: dict[str, list[Correspondance]] = {}

        for identifiant, texte in chunks:
            rang = len(self._identifiants)
            self._identifiants.append(identifiant)
            mots = decouper(texte)
            self._mots.append(mots)
            # `Counter` sur les mots DÉJÀ dédoublonnés par le comptage : chaque
            # mot n'entre qu'une fois dans la liste de correspondances, avec sa
            # fréquence. C'est l'index inversé.
            for mot, frequence in Counter(mots).items():
                self._index.setdefault(mot, []).append(Correspondance(rang, frequence))

        self._nombre = len(self._identifiants)
        self._longueurs = [len(mots) for mots in self._mots]
        self._longueur_moyenne = (
            sum(self._longueurs) / self._nombre if self._nombre else 0.0
        )

    @property
    def taille(self) -> int:
        """Nombre de chunks indexés."""
        return self._nombre

    def _idf(self, mot: str) -> float:
        """Rareté d'un mot : l'essentiel de BM25.

        Un mot présent partout a un IDF proche de zéro, un mot rare un IDF
        élevé — c'est ce qui permet à un terme distinctif de dominer des
        centaines de mots de formule.
        """
        documents_contenant = len(self._index.get(mot, ()))
        return math.log(1 + (self._nombre - documents_contenant + 0.5) / (documents_contenant + 0.5))

    def classer(self, requete: str, limite: int) -> list[str]:
        """Les identifiants des `limite` chunks les mieux classés, du meilleur au moins bon.

        Un mot de la question absent de l'index n'est simplement pas compté :
        c'est ce qui rend la recherche robuste à une faute de frappe ou à une
        variante de graphie non prévue.
        """
        scores: dict[int, float] = {}

        for mot in decouper(requete):
            correspondances = self._index.get(mot)
            if not correspondances:
                continue
            idf = self._idf(mot)
            for correspondance in correspondances:
                rang = correspondance.rang
                # Normalisation par la longueur : sans elle, un chunk long
                # gagnerait toujours, simplement parce qu'il contient plus de
                # mots — et les chunks les plus longs sont ici ceux qui
                # empilent les chaînes de transmetteurs.
                denominateur = correspondance.frequence + K1 * (
                    1 - B + B * self._longueurs[rang] / self._longueur_moyenne
                )
                gain = idf * correspondance.frequence * (K1 + 1) / denominateur
                scores[rang] = scores.get(rang, 0.0) + gain

        meilleurs = sorted(scores, key=lambda rang: -scores[rang])[:limite]
        return [self._identifiants[rang] for rang in meilleurs]


def fusionner_rrf(
    classements: Sequence[Sequence[str]],
    poids: Sequence[float],
    constante: int,
    limite: int,
) -> list[str]:
    """Fusionne plusieurs classements par rangs réciproques (RRF).

    Chaque classement vote pour les documents qu'il remonte : le *i*-ème reçoit
    ``poids / (constante + i)``. On additionne, on retrie.

    Pourquoi les RANGS et non les scores : un score BM25 et une distance
    vectorielle ne sont pas dans la même unité et n'ont aucune échelle commune.
    Additionner l'un à l'autre serait une faute de dimension. Le rang, lui, est
    toujours un entier de 1 à N — comparable par construction.

    `constante` aplatit l'influence des tout premiers rangs : plus elle est
    grande, plus un document bien classé par plusieurs recherches l'emporte sur
    un document excellents dans une seule.
    """
    if len(classements) != len(poids):
        raise ValueError("Chaque classement doit avoir son poids.")

    votes: dict[str, float] = {}
    for classement, poids_du_classement in zip(classements, poids, strict=True):
        for position, identifiant in enumerate(classement, start=1):
            votes[identifiant] = (
                votes.get(identifiant, 0.0) + poids_du_classement / (constante + position)
            )

    return sorted(votes, key=lambda identifiant: -votes[identifiant])[:limite]
