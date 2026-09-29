"""Mesure de la qualité de la récupération (le « harnais d'évaluation »).

Pourquoi ce module existe
-------------------------
Tout réglage du RAG — taille de chunk, chevauchement, modèle d'embedding, seuil
de distance, ajout d'un reranker — reste un pari tant qu'il n'est pas mesuré.
Ce module fournit l'instrument : un « jeu d'or » de questions dont on connaît la
source attendue, et les métriques qui disent si la récupération la retrouve, et
à quel rang.

Les deux métriques
------------------
``hit@k``
    Part des questions dont AU MOINS une source attendue figure dans les *k*
    premiers résultats. Répond à : « a-t-on trouvé le bon passage ? »

``MRR@k``
    Moyenne de ``1 / rang`` du PREMIER résultat correct (0 si absent). Répond à :
    « l'a-t-on bien classé ? » Un extrait correct au rang 5 compte moins qu'au
    rang 1 — ce qui est réaliste, car un extrait noyé au milieu du contexte est
    plus souvent ignoré par le modèle.

``hit@k`` seule ne suffit pas : elle ne distingue pas le rang 1 du rang *k*.
Un reranker améliore rarement ``hit@k`` (il ne peut pas trouver ce que la
récupération a raté) mais améliore presque toujours le MRR.

Les deux populations
--------------------
Un jeu d'or ne contient pas que des questions auxquelles le corpus répond. Il en
contient aussi dont la bonne réponse est : « il n'y a rien ». Elles s'écrivent
``"expected": []``, et elles ne se mesurent pas de la même façon :

* les questions **répondables** se notent par ``hit@k`` et ``MRR`` — le passage
  attendu est-il retrouvé, et à quel rang ;
* les questions **hors corpus** ne peuvent PAS se noter par un rang : il n'y a
  aucun passage à retrouver, et la récupération rend toujours *k* chunks, même
  pour une question qui ne concerne pas le corpus. Ce qui se mesure, c'est la
  **distance du chunk le plus proche** : si elle reste basse alors que rien ne
  répond, aucune distance ne distingue « le corpus répond » de « le corpus ne
  répond pas » — et le seuil de distance de l'application est un réglage
  introuvable. C'est précisément la question à laquelle ce harnais doit répondre
  avant qu'on ose activer ce seuil.

Mélanger les deux populations dans un même score n'aurait aucun sens : compter
« hors corpus » comme « introuvable » punirait la bonne réponse du système, et
l'ignorer ferait disparaître ce qu'on cherche justement à savoir.

Pureté
------
Ce module n'appelle ni ChromaDB ni Ollama, ne lit que les fichiers qu'on lui
passe explicitement, et n'a aucun effet de bord. Il est donc testable sans base
vectorielle ni modèle de langue.
"""

import hashlib
import json
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Statuts d'une question du jeu d'or.
# - "draft"     : écrite (souvent générée) mais PAS encore relue par un humain ;
# - "validated" : relue, la source attendue est confirmée.
# Une question « draft » reste utilisable : on compare des mesures entre elles,
# et le biais est identique avant/après. Seul le score ABSOLU en dépend.
STATUS_DRAFT = "draft"
STATUS_VALIDATED = "validated"
VALID_STATUSES = (STATUS_DRAFT, STATUS_VALIDATED)

DEFAULT_LANG = "ar"


class GoldenError(ValueError):
    """Le jeu d'or est illisible ou mal formé (message destiné à l'humain)."""


# --- Le jeu d'or ----------------------------------------------------------

@dataclass(frozen=True)
class ExpectedSource:
    """Passage attendu pour une question.

    ``page`` peut rester ``None`` : on accepte alors n'importe quelle page du
    fichier. C'est volontaire — au moment d'écrire une question à la main on n'a
    pas toujours la page sous les yeux, et un critère trop strict rendrait le jeu
    d'or pénible à remplir (donc il ne serait pas rempli).
    """

    source: str
    page: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Représentation JSON compacte (la page absente est omise)."""
        data: dict[str, Any] = {"source": self.source}
        if self.page is not None:
            data["page"] = self.page
        return data


@dataclass(frozen=True)
class GoldenQuestion:
    """Une question et ce qu'une récupération correcte doit ramener."""

    id: str
    question: str
    expected: tuple[ExpectedSource, ...]
    lang: str = DEFAULT_LANG
    status: str = STATUS_DRAFT
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Représentation JSON de la question."""
        return {
            "id": self.id,
            "question": self.question,
            "lang": self.lang,
            "status": self.status,
            "expected": [source.to_dict() for source in self.expected],
            "notes": self.notes,
        }

    @property
    def est_hors_corpus(self) -> bool:
        """La bonne réponse à cette question est-elle « rien » ?

        Écrite ``"expected": []``, elle vérifie que le système REFUSE de répondre
        au lieu d'inventer. C'est la question qu'un utilisateur pose le plus
        souvent sans le savoir : celle dont la réponse n'est pas dans les
        documents.
        """
        return not self.expected


def parse_expected(raw: Any, context: str) -> ExpectedSource:
    """Construit une source attendue à partir d'une entrée JSON."""
    if not isinstance(raw, dict):
        raise GoldenError(f"{context} : chaque entrée de « expected » doit être un objet.")

    source = raw.get("source")
    if not isinstance(source, str) or not source.strip():
        raise GoldenError(f"{context} : « source » est obligatoire et ne peut être vide.")

    page = raw.get("page")
    # bool est une sous-classe de int : on l'écarte explicitement, sinon
    # « "page": true » passerait pour la page 1.
    if page is not None and (isinstance(page, bool) or not isinstance(page, int) or page < 1):
        raise GoldenError(f"{context} : « page » doit être un entier >= 1 (ou absent).")

    return ExpectedSource(source=source.strip(), page=page)


def parse_golden_line(line: str, number: int) -> GoldenQuestion | None:
    """Analyse une ligne du jeu d'or.

    Renvoie ``None`` pour une ligne vide ou un commentaire (``#``) : JSON n'a pas
    de commentaires, mais un jeu d'or se relit à la main, et un fichier qu'on ne
    peut pas annoter devient vite incompréhensible.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    try:
        raw = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise GoldenError(f"ligne {number} : JSON invalide ({exc.msg}).") from exc

    if not isinstance(raw, dict):
        raise GoldenError(f"ligne {number} : un objet JSON est attendu.")

    context = f"ligne {number}"

    question = raw.get("question")
    if not isinstance(question, str) or not question.strip():
        raise GoldenError(f"{context} : « question » est obligatoire et ne peut être vide.")

    identifier = raw.get("id") or f"q{number:03d}"
    if not isinstance(identifier, str):
        raise GoldenError(f"{context} : « id » doit être une chaîne.")

    expected_raw = raw.get("expected")
    # Une liste VIDE est un choix délibéré (« le corpus ne doit pas répondre »),
    # une clé ABSENTE est une erreur de rédaction : ne pas confondre les deux,
    # sinon une question mal saisie passerait pour une question hors corpus et
    # serait comptée comme un exercice de refus.
    if expected_raw is None:
        raise GoldenError(
            f"{context} : « expected » est obligatoire. Utilisez ``[]`` pour une "
            "question dont la réponse n'est PAS dans le corpus."
        )
    if not isinstance(expected_raw, list):
        raise GoldenError(f"{context} : « expected » doit être une liste.")

    expected = tuple(
        parse_expected(item, f"{context}, expected[{index}]")
        for index, item in enumerate(expected_raw)
    )

    status = raw.get("status", STATUS_DRAFT)
    if status not in VALID_STATUSES:
        raise GoldenError(
            f"{context} : « status » doit valoir « {' » ou « '.join(VALID_STATUSES)} »."
        )

    lang = raw.get("lang", DEFAULT_LANG)
    if not isinstance(lang, str):
        raise GoldenError(f"{context} : « lang » doit être une chaîne.")

    notes = raw.get("notes", "")
    if not isinstance(notes, str):
        raise GoldenError(f"{context} : « notes » doit être une chaîne.")

    return GoldenQuestion(
        id=identifier,
        question=question.strip(),
        expected=expected,
        lang=lang,
        status=status,
        notes=notes,
    )


def to_json_line(question: GoldenQuestion) -> str:
    """Sérialise une question en une ligne JSON.

    ``ensure_ascii=False`` est indispensable : sans lui, le texte arabe serait
    écrit en séquences d'échappement (\\u0627...) et le fichier deviendrait
    illisible — donc impossible à relire et corriger à la main.
    """
    return json.dumps(question.to_dict(), ensure_ascii=False)


def load_golden(path: Path) -> list[GoldenQuestion]:
    """Charge le jeu d'or depuis un fichier JSONL.

    Une ligne invalide fait échouer la lecture, avec le numéro de ligne. C'est
    volontaire : un jeu d'or silencieusement amputé de ses questions difficiles
    afficherait un excellent score et donnerait une confiance injustifiée.
    """
    if not path.exists():
        raise GoldenError(
            f"Jeu d'or introuvable : {path}\n"
            "  Créez-le avec « python scripts/eval_rag.py --sample 10 », qui affiche "
            "des extraits\n  et leurs références, puis transcrivez vos questions dans "
            "ce fichier."
        )

    questions: list[GoldenQuestion] = []
    seen: set[str] = set()

    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        question = parse_golden_line(line, number)
        if question is None:
            continue
        if question.id in seen:
            raise GoldenError(f"ligne {number} : identifiant « {question.id} » déjà utilisé.")
        seen.add(question.id)
        questions.append(question)

    return questions


def append_golden(path: Path, questions: Sequence[GoldenQuestion]) -> None:
    """Ajoute des questions à la fin du jeu d'or (le fichier est créé si absent)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for question in questions:
            handle.write(to_json_line(question))
            handle.write("\n")


# --- Correspondance et métriques -----------------------------------------

def source_matches(retrieved: Mapping[str, Any], expected: ExpectedSource) -> bool:
    """Un résultat récupéré correspond-il à la source attendue ?

    Le fichier doit correspondre. Si la page est renseignée des DEUX côtés, elle
    doit correspondre aussi. On ne compare donc jamais ``line_start`` /
    ``line_end`` : ces bornes dépendent du découpage, et les utiliser rendrait le
    jeu d'or sensible au moindre changement de ``CHUNK_SIZE`` — le harnais
    mesurerait alors le chunker au lieu de la récupération.
    """
    if retrieved.get("source") != expected.source:
        return False
    if expected.page is None:
        return True
    return retrieved.get("page") == expected.page


def est_orpheline(question: GoldenQuestion, sources_indexees: Collection[str]) -> bool:
    """La question ne peut-elle plus être satisfaite, faute de source indexée ?

    Elle l'est seulement si AUCUNE de ses sources attendues n'est indexée : une
    question dont une source subsiste reste parfaitement mesurable.

    Sert à EXCLURE du calcul les questions devenues impossibles. Les compter
    « introuvables » ferait chuter le score à cause d'un document retiré du
    corpus, et non de la qualité de la récupération — un faux signal, qui
    masquerait un vrai problème le jour où il apparaîtrait pour de bon.

    ⚠️ Une question HORS CORPUS n'est jamais orpheline : elle n'attend aucune
    source. Sans ce cas particulier, ``all([])`` vaut ``True`` et toutes les
    questions de refus seraient silencieusement écartées de la mesure —
    c'est-à-dire exactement celles qu'on vient mesurer.
    """
    if question.est_hors_corpus:
        return False
    return all(source.source not in sources_indexees for source in question.expected)


def first_match_rank(
    retrieved_sources: Iterable[Mapping[str, Any]],
    expected: Sequence[ExpectedSource],
) -> int | None:
    """Rang (1-indexé) du premier résultat correspondant à une source attendue.

    Renvoie ``None`` si aucun des résultats ne correspond.
    """
    for rank, retrieved in enumerate(retrieved_sources, start=1):
        if any(source_matches(retrieved, candidate) for candidate in expected):
            return rank
    return None


def hit_at_k(rank: int | None, k: int) -> bool:
    """Le bon passage est-il dans les *k* premiers résultats ?"""
    return rank is not None and rank <= k


def reciprocal_rank(rank: int | None, k: int) -> float:
    """``1 / rang`` si le résultat tombe dans les *k* premiers, sinon 0."""
    if rank is None or rank > k:
        return 0.0
    return 1.0 / rank


class CorpusFingerprint:
    """Accumulateur d'empreinte de corpus, en mémoire constante.

    Pourquoi un accumulateur plutôt qu'un simple tri : une fois le corpus à
    l'échelle de la bibliothèque Shamela (~23 millions de chunks), constituer la
    liste complète des identifiants pour la trier demanderait plusieurs
    gigaoctets. Ici, chaque identifiant est incorporé puis oublié.

    L'empreinte est un OU-exclusif (XOR) des empreintes individuelles. XOR étant
    commutatif et associatif, le résultat ne dépend PAS de l'ordre de parcours —
    indispensable, car l'ordre de retour de la base vectorielle n'a aucune raison
    d'être stable d'une exécution à l'autre.
    """

    __slots__ = ("_accumulateur", "_nombre")

    def __init__(self) -> None:
        self._accumulateur = 0
        self._nombre = 0

    def add(self, identifier: str) -> None:
        """Incorpore un identifiant."""
        self._accumulateur ^= int.from_bytes(
            hashlib.sha256(identifier.encode("utf-8")).digest(), "big"
        )
        self._nombre += 1

    def update(self, identifiers: Iterable[str]) -> None:
        """Incorpore une série d'identifiants."""
        for identifier in identifiers:
            self.add(identifier)

    def hexdigest(self) -> str:
        """Empreinte finale : 12 caractères hexadécimaux.

        Le NOMBRE d'identifiants est incorporé séparément : un XOR seul ne
        distinguerait pas un corpus vide d'un corpus dont les identifiants
        s'annulent deux à deux.
        """
        contenu = f"{self._nombre}:{self._accumulateur}"
        return hashlib.sha256(contenu.encode("utf-8")).hexdigest()[:12]


def corpus_fingerprint(identifiers: Iterable[str]) -> str:
    """Empreinte stable d'un corpus, pour vérifier que deux mesures sont comparables.

    Deux rapports ne se comparent que si le corpus est le MÊME. Sans cette
    empreinte, on peut croire à un gain de qualité alors qu'on a seulement changé
    les documents indexés.

    Accepte un simple générateur : rien n'est conservé en mémoire.
    """
    empreinte = CorpusFingerprint()
    empreinte.update(identifiers)
    return empreinte.hexdigest()


# --- Rapports -------------------------------------------------------------

@dataclass(frozen=True)
class QuestionResult:
    """Résultat d'une question : à quel rang le bon passage est ressorti.

    ``min_distance`` est la distance L2 du chunk le plus proche de la question,
    relevée indépendamment de tout filtrage et de toute fusion. C'est la seule
    grandeur qui ait un sens pour une question hors corpus, où il n'y a aucun
    rang à trouver.

    ``hors_corpus`` dit à quelle population la question appartient. Sans lui, un
    rapport relu six mois plus tard serait impossible à interpréter : un ``rank``
    à ``None`` signifierait tour à tour « échec » et « réussite ».
    """

    question_id: str
    question: str
    rank: int | None
    latency_ms: float
    retrieved: tuple[dict[str, Any], ...] = ()
    min_distance: float | None = None
    hors_corpus: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Représentation JSON du résultat."""
        return {
            "id": self.question_id,
            "question": self.question,
            "rank": self.rank,
            "latency_ms": round(self.latency_ms, 1),
            "min_distance": (
                None if self.min_distance is None else round(self.min_distance, 4)
            ),
            "hors_corpus": self.hors_corpus,
            "retrieved": list(self.retrieved),
        }


@dataclass
class EvaluationReport:
    """Résultat complet d'une campagne de mesure, sur un corpus donné."""

    label: str
    k: int
    results: tuple[QuestionResult, ...]
    context: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    # Questions écartées de la mesure parce que leur source n'est plus indexée.
    # Conservées dans le rapport : sans elles, un score qui chute serait
    # inexplicable six mois plus tard.
    ignored_ids: tuple[str, ...] = ()

    @property
    def count(self) -> int:
        """Nombre de questions évaluées (les deux populations confondues)."""
        return len(self.results)

    @property
    def answerable(self) -> tuple[QuestionResult, ...]:
        """Questions auxquelles le corpus DOIT répondre."""
        return tuple(result for result in self.results if not result.hors_corpus)

    @property
    def hors_corpus(self) -> tuple[QuestionResult, ...]:
        """Questions auxquelles le corpus ne peut PAS répondre."""
        return tuple(result for result in self.results if result.hors_corpus)

    @property
    def hits(self) -> int:
        """Nombre de questions répondables dont le bon passage est dans le top-k."""
        return sum(1 for result in self.answerable if hit_at_k(result.rank, self.k))

    @property
    def hit_at_k(self) -> float:
        """Part des questions RÉPONDABLES retrouvées dans le top-k (entre 0 et 1).

        Le dénominateur exclut les questions hors corpus : il n'y a rien à
        retrouver pour elles, et les compter comme des échecs ferait dépendre le
        score de la proportion de questions de refus qu'on a pris la peine
        d'écrire.
        """
        total = len(self.answerable)
        return self.hits / total if total else 0.0

    @property
    def mrr(self) -> float:
        """MRR@k : qualité du classement du premier résultat correct."""
        total = len(self.answerable)
        if not total:
            return 0.0
        somme = sum(reciprocal_rank(result.rank, self.k) for result in self.answerable)
        return somme / total

    @property
    def mean_latency_ms(self) -> float:
        """Latence moyenne de récupération, en millisecondes."""
        if not self.count:
            return 0.0
        return sum(result.latency_ms for result in self.results) / self.count

    @property
    def misses(self) -> tuple[QuestionResult, ...]:
        """Questions RÉPONDABLES dont le bon passage n'a pas été retrouvé."""
        return tuple(result for result in self.answerable if result.rank is None)

    def to_dict(self) -> dict[str, Any]:
        """Représentation JSON complète, prête à écrire sur disque.

        Le bloc ``summary`` est fourni pour la LECTURE humaine (un fichier qu'on
        ouvre doit se comprendre d'un coup d'œil) mais n'est jamais relu : à la
        relecture, tout est recalculé depuis ``results``, qui reste la seule
        source de vérité. Un résumé figé pourrait mentir après une correction.
        """
        return {
            "label": self.label,
            "k": self.k,
            "created_at": self.created_at,
            "context": self.context,
            "summary": {
                "count": self.count,
                "answerable": len(self.answerable),
                "hors_corpus": len(self.hors_corpus),
                "ignored": len(self.ignored_ids),
                "hit_at_k": round(self.hit_at_k, 4),
                "mrr": round(self.mrr, 4),
                "mean_latency_ms": round(self.mean_latency_ms, 1),
            },
            "ignored_ids": list(self.ignored_ids),
            "results": [result.to_dict() for result in self.results],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvaluationReport":
        """Relit un rapport écrit par :meth:`to_dict`."""
        results = tuple(
            QuestionResult(
                question_id=str(item.get("id", "")),
                question=str(item.get("question", "")),
                rank=item.get("rank"),
                latency_ms=float(item.get("latency_ms") or 0.0),
                retrieved=tuple(item.get("retrieved") or ()),
                min_distance=(
                    None
                    if item.get("min_distance") is None
                    else float(item["min_distance"])
                ),
                hors_corpus=bool(item.get("hors_corpus", False)),
            )
            for item in data.get("results", [])
        )
        return cls(
            label=str(data.get("label", "sans-nom")),
            k=int(data.get("k", 5)),
            results=results,
            context=dict(data.get("context") or {}),
            created_at=str(data.get("created_at", "")),
            ignored_ids=tuple(str(item) for item in (data.get("ignored_ids") or ())),
        )


def build_report(
    label: str,
    k: int,
    results: Sequence[QuestionResult],
    context: Mapping[str, Any] | None = None,
    ignored_ids: Sequence[str] = (),
) -> EvaluationReport:
    """Assemble un rapport horodaté (UTC, format ISO 8601)."""
    return EvaluationReport(
        label=label,
        k=k,
        results=tuple(results),
        context=dict(context or {}),
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ignored_ids=tuple(ignored_ids),
    )


# --- Affichage ------------------------------------------------------------

def accorder(nombre: int, singulier: str, pluriel: str | None = None) -> str:
    """Accorde un nom en nombre, à la française : « 1 question », « 3 questions ».

    En français, zéro prend le singulier (« 0 question ») : la comparaison porte
    donc sur ``<= 1``, et non sur ``== 1``.
    """
    if nombre <= 1:
        return f"{nombre} {singulier}"
    return f"{nombre} {pluriel if pluriel is not None else singulier + 's'}"


def accorder_question(nombre: int, adjectif: str) -> str:
    """Accorde le nom « question » ET son adjectif : « 1 question ignorée ».

    Évite les « ignorée(s) » : l'adjectif est donné au féminin singulier, le
    pluriel étant formé en ajoutant « s ».
    """
    suffixe = "" if nombre <= 1 else "s"
    return f"{accorder(nombre, 'question')} {adjectif}{suffixe}"


def _pourcent(value: float) -> str:
    """Formate un ratio en pourcentage, à la française : « 85,0 % »."""
    return f"{value:.1%}".replace(".", ",").replace("%", " %")


def _nombre(value: float, decimals: int = 2) -> str:
    """Formate un nombre à la française (virgule décimale)."""
    return f"{value:.{decimals}f}".replace(".", ",")


def _rang_comparable(rank: int | None) -> float:
    """Rend les rangs comparables : « introuvable » devient l'infini."""
    return float("inf") if rank is None else float(rank)


@dataclass(frozen=True)
class DistancesPlusProches:
    """Distribution de la distance L2 au chunk le plus proche.

    C'est l'instrument qui décide si le seuil de distance de l'application est
    réglable. Si les deux populations se recouvrent — une question hors corpus
    peut avoir un chunk PLUS proche qu'une question répondable — alors aucun
    seuil ne les sépare, et en activer un écarterait de vraies réponses.
    """

    nombre: int
    minimum: float
    mediane: float
    maximum: float

    def __str__(self) -> str:
        return (
            f"min {_nombre(self.minimum)}  médiane {_nombre(self.mediane)}  "
            f"max {_nombre(self.maximum)}"
        )


def distances_plus_proches(
    results: Sequence[QuestionResult],
) -> DistancesPlusProches | None:
    """Décrit la distance au plus proche sur une population, ou ``None``.

    ``None`` signifie qu'aucune question de la population ne porte de distance :
    c'est le cas d'un rapport écrit avant que cette mesure n'existe.
    """
    valeurs = sorted(
        result.min_distance for result in results if result.min_distance is not None
    )
    if not valeurs:
        return None
    return DistancesPlusProches(
        nombre=len(valeurs),
        minimum=valeurs[0],
        mediane=valeurs[len(valeurs) // 2],
        maximum=valeurs[-1],
    )


def format_report(report: EvaluationReport) -> str:
    """Rend un rapport lisible dans un terminal."""
    # Les rangs ne concernent que les questions répondables : une question hors
    # corpus n'a pas de rang, et la compter « introuvable » serait un contresens.
    par_rang = [result.rank for result in report.answerable if result.rank is not None]
    premier = sum(1 for rank in par_rang if rank == 1)
    ensuite = len(par_rang) - premier
    introuvables = len(report.misses)

    lignes = [
        f"Évaluation « {report.label} » — {accorder(report.count, 'question')}"
        f" ({len(report.answerable)} répondable(s)"
        + (
            f", {len(report.hors_corpus)} hors corpus)"
            if report.hors_corpus
            else ")"
        )
        + f", top-{report.k}"
    ]

    if report.created_at:
        lignes.append(f"Mesuré le {report.created_at}")
    if report.context:
        details = " · ".join(f"{cle} = {valeur}" for cle, valeur in report.context.items())
        lignes.append(f"Contexte : {details}")

    lignes += [
        "-" * 64,
        f"{'hit@' + str(report.k):<17}: {_pourcent(report.hit_at_k):>9}"
        f"   ({report.hits}/{len(report.answerable)})",
        f"{'MRR@' + str(report.k):<17}: {_nombre(report.mrr):>9}",
        f"{'Latence moyenne':<17}: {report.mean_latency_ms:>6.0f} ms",
        "-" * 64,
        f"{'Rang 1':<17}: {premier:>9}",
        f"{'Rang 2 à ' + str(report.k):<17}: {ensuite:>9}",
        f"{'Introuvable':<17}: {introuvables:>9}",
    ]

    # Le seuil de distance ne peut se régler que si les deux populations ne se
    # recouvrent pas. C'est ce que ce bloc permet de voir d'un coup d'œil.
    repondables = distances_plus_proches(report.answerable)
    hors_corpus = distances_plus_proches(report.hors_corpus)
    if repondables or hors_corpus:
        lignes += ["", f"Distance L2 au chunk le plus proche (top-{report.k}) :"]
        if repondables:
            lignes.append(f"  répondables :  {repondables}  ({repondables.nombre})")
        if hors_corpus:
            lignes.append(f"  hors corpus :  {hors_corpus}  ({hors_corpus.nombre})")
        if repondables and hors_corpus:
            if hors_corpus.minimum <= repondables.maximum:
                lignes.append(
                    "  → Les deux populations SE RECOUVRENT : aucun seuil de distance\n"
                    "    ne peut les séparer. En activer un écarterait de vraies réponses."
                )
            else:
                lignes.append(
                    "  → Populations séparées : un seuil autour de "
                    f"{_nombre(hors_corpus.minimum)} les distinguerait."
                )

    if report.ignored_ids:
        lignes += [
            "",
            f"⚠️  {accorder_question(len(report.ignored_ids), 'ignorée')} : "
            "source non indexée",
            f"    {', '.join(report.ignored_ids)}",
        ]

    if report.misses:
        lignes += ["", "Questions non retrouvées :"]
        lignes += [
            f"  · {result.question_id} — {result.question}" for result in report.misses
        ]

    return "\n".join(lignes)


def format_comparison(before: EvaluationReport, after: EvaluationReport) -> str:
    """Compare deux rapports (avant / après) et alerte si la comparaison est pipée."""
    avertissements: list[str] = []

    if before.k != after.k:
        avertissements.append(
            f"⚠️  top-k différent ({before.k} vs {after.k}) : les scores ne sont pas comparables."
        )

    empreinte_avant = before.context.get("empreinte_corpus")
    empreinte_apres = after.context.get("empreinte_corpus")
    if empreinte_avant and empreinte_apres and empreinte_avant != empreinte_apres:
        chunks_avant = before.context.get("chunks_indexes", "?")
        chunks_apres = after.context.get("chunks_indexes", "?")
        avertissements.append(
            f"⚠️  corpus DIFFÉRENT entre les deux mesures "
            f"({chunks_avant} → {chunks_apres} chunks) : l'écart peut venir des\n"
            "    documents et non du réglage testé.\n"
            "    · Normal si vous testez l'AJOUT de documents (Shamela, par exemple).\n"
            "    · Suspect si vous testez un réglage (reranker, chunker…) : mesurez\n"
            "      alors les deux rapports sur le MÊME corpus."
        )

    rangs_avant = {
        result.question_id: result.rank for result in before.answerable
    }
    rangs_apres = {
        result.question_id: result.rank for result in after.answerable
    }
    communes = sorted(set(rangs_avant) & set(rangs_apres))

    if set(rangs_avant) != set(rangs_apres):
        seulement_avant = len(set(rangs_avant) - set(rangs_apres))
        seulement_apres = len(set(rangs_apres) - set(rangs_avant))
        avertissements.append(
            f"⚠️  les deux mesures ne portent pas sur le même jeu de questions "
            f"({len(rangs_avant)} avant, {len(rangs_apres)} après ; "
            f"{seulement_avant} seulement avant, {seulement_apres} seulement après).\n"
            "    Le score global est alors indicatif : seule la comparaison question\n"
            "    par question reste fiable."
        )

    progres, regressions, stables = 0, 0, 0
    for identifiant in communes:
        avant = _rang_comparable(rangs_avant[identifiant])
        apres = _rang_comparable(rangs_apres[identifiant])
        if apres < avant:
            progres += 1
        elif apres > avant:
            regressions += 1
        else:
            stables += 1

    ecart_hit = after.hit_at_k - before.hit_at_k
    ecart_mrr = after.mrr - before.mrr
    ecart_latence = after.mean_latency_ms - before.mean_latency_ms

    lignes = [f"Comparaison « {before.label} » → « {after.label} »"]
    lignes.append(
        f"Questions communes : {len(communes)}"
        f"  (avant : {len(before.answerable)}, après : {len(after.answerable)})"
    )
    lignes.append("-" * 64)
    lignes.append(f"{'':<17}{'avant':>10}{'après':>10}{'écart':>12}")
    lignes.append(
        f"{'hit@' + str(after.k):<17}{_pourcent(before.hit_at_k):>10}"
        f"{_pourcent(after.hit_at_k):>10}{'+' + _nombre(ecart_hit * 100, 1):>11} pts"
    )
    lignes.append(
        f"{'MRR@' + str(after.k):<17}{_nombre(before.mrr):>10}"
        f"{_nombre(after.mrr):>10}{('+' if ecart_mrr >= 0 else '') + _nombre(ecart_mrr):>12}"
    )
    lignes.append(
        f"{'Latence':<17}{_nombre(before.mean_latency_ms, 0) + ' ms':>10}"
        f"{_nombre(after.mean_latency_ms, 0) + ' ms':>10}"
        f"{('+' if ecart_latence >= 0 else '') + _nombre(ecart_latence, 0):>9} ms"
    )
    lignes.append("-" * 64)
    lignes.append(f"Questions mieux classées : {progres}")
    lignes.append(f"Questions moins bien classées : {regressions}")
    lignes.append(f"Inchangées : {stables}")

    if avertissements:
        lignes.append("")
        lignes.extend(avertissements)

    return "\n".join(lignes)
