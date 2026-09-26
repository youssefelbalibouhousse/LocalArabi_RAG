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

Pureté
------
Ce module n'appelle ni ChromaDB ni Ollama, ne lit que les fichiers qu'on lui
passe explicitement, et n'a aucun effet de bord. Il est donc testable sans base
vectorielle ni modèle de langue.
"""

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
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
    if not isinstance(expected_raw, list) or not expected_raw:
        raise GoldenError(f"{context} : « expected » doit être une liste non vide.")

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


def corpus_fingerprint(identifiers: Iterable[str]) -> str:
    """Empreinte stable d'un corpus, pour vérifier que deux mesures sont comparables.

    Deux rapports ne se comparent que si le corpus est le MÊME. Sans cette
    empreinte, on peut croire à un gain de qualité alors qu'on a simplement
    changé les documents indexés.

    Les identifiants sont TRIÉS avant hachage : l'ordre de retour de la base
    vectorielle n'a aucune raison d'être stable d'une exécution à l'autre et ne
    doit donc pas faire varier l'empreinte.

    Attention : cette fonction charge tous les identifiants en mémoire. Convenable
    jusqu'à quelques centaines de milliers de chunks, à revoir au-delà.
    """
    digest = hashlib.sha256()
    for identifier in sorted(identifiers):
        digest.update(identifier.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()[:12]


# --- Rapports -------------------------------------------------------------

@dataclass(frozen=True)
class QuestionResult:
    """Résultat d'une question : à quel rang le bon passage est ressorti."""

    question_id: str
    question: str
    rank: int | None
    latency_ms: float
    retrieved: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Représentation JSON du résultat."""
        return {
            "id": self.question_id,
            "question": self.question,
            "rank": self.rank,
            "latency_ms": round(self.latency_ms, 1),
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

    @property
    def count(self) -> int:
        """Nombre de questions évaluées."""
        return len(self.results)

    @property
    def hits(self) -> int:
        """Nombre de questions dont le bon passage est dans le top-k."""
        return sum(1 for result in self.results if hit_at_k(result.rank, self.k))

    @property
    def hit_at_k(self) -> float:
        """Part des questions retrouvées dans le top-k (entre 0 et 1)."""
        return self.hits / self.count if self.count else 0.0

    @property
    def mrr(self) -> float:
        """MRR@k : qualité du classement du premier résultat correct."""
        if not self.count:
            return 0.0
        total = sum(reciprocal_rank(result.rank, self.k) for result in self.results)
        return total / self.count

    @property
    def mean_latency_ms(self) -> float:
        """Latence moyenne de récupération, en millisecondes."""
        if not self.count:
            return 0.0
        return sum(result.latency_ms for result in self.results) / self.count

    @property
    def misses(self) -> tuple[QuestionResult, ...]:
        """Questions dont le bon passage n'a pas été retrouvé."""
        return tuple(result for result in self.results if result.rank is None)

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
                "hit_at_k": round(self.hit_at_k, 4),
                "mrr": round(self.mrr, 4),
                "mean_latency_ms": round(self.mean_latency_ms, 1),
            },
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
            )
            for item in data.get("results", [])
        )
        return cls(
            label=str(data.get("label", "sans-nom")),
            k=int(data.get("k", 5)),
            results=results,
            context=dict(data.get("context") or {}),
            created_at=str(data.get("created_at", "")),
        )


def build_report(
    label: str,
    k: int,
    results: Sequence[QuestionResult],
    context: Mapping[str, Any] | None = None,
) -> EvaluationReport:
    """Assemble un rapport horodaté (UTC, format ISO 8601)."""
    return EvaluationReport(
        label=label,
        k=k,
        results=tuple(results),
        context=dict(context or {}),
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
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


def _pourcent(value: float) -> str:
    """Formate un ratio en pourcentage, à la française : « 85,0 % »."""
    return f"{value:.1%}".replace(".", ",").replace("%", " %")


def _nombre(value: float, decimals: int = 2) -> str:
    """Formate un nombre à la française (virgule décimale)."""
    return f"{value:.{decimals}f}".replace(".", ",")


def _rang_comparable(rank: int | None) -> float:
    """Rend les rangs comparables : « introuvable » devient l'infini."""
    return float("inf") if rank is None else float(rank)


def format_report(report: EvaluationReport) -> str:
    """Rend un rapport lisible dans un terminal."""
    par_rang = [result.rank for result in report.results if result.rank is not None]
    premier = sum(1 for rank in par_rang if rank == 1)
    ensuite = len(par_rang) - premier
    introuvables = len(report.misses)

    lignes = [
        f"Évaluation « {report.label} » — {accorder(report.count, 'question')}, top-{report.k}"
    ]

    if report.created_at:
        lignes.append(f"Mesuré le {report.created_at}")
    if report.context:
        details = " · ".join(f"{cle} = {valeur}" for cle, valeur in report.context.items())
        lignes.append(f"Contexte : {details}")

    lignes += [
        "-" * 64,
        f"{'hit@' + str(report.k):<17}: {_pourcent(report.hit_at_k):>9}"
        f"   ({report.hits}/{report.count})",
        f"{'MRR@' + str(report.k):<17}: {_nombre(report.mrr):>9}",
        f"{'Latence moyenne':<17}: {report.mean_latency_ms:>6.0f} ms",
        "-" * 64,
        f"{'Rang 1':<17}: {premier:>9}",
        f"{'Rang 2 à ' + str(report.k):<17}: {ensuite:>9}",
        f"{'Introuvable':<17}: {introuvables:>9}",
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
        avertissements.append(
            "⚠️  corpus DIFFÉRENT entre les deux mesures : l'écart peut venir des "
            "documents,\n    pas du réglage testé."
        )

    rangs_avant = {result.question_id: result.rank for result in before.results}
    rangs_apres = {result.question_id: result.rank for result in after.results}
    communes = sorted(set(rangs_avant) & set(rangs_apres))

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
        f"  (avant : {before.count}, après : {after.count})"
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
