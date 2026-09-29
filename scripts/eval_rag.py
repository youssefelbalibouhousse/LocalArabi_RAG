"""Harnais d'évaluation de la récupération (qualité du RAG).

Cet outil répond à une seule question : **quand on pose une question dont on
connaît la réponse, la récupération ramène-t-elle le bon passage, et à quel
rang ?** Sans cette mesure, tout réglage du RAG — taille de chunk, chevauchement,
modèle d'embedding, reranker, seuil de distance — se décide à l'intuition.

Trois modes
-----------
``--sample N``
    Affiche N extraits tirés au hasard avec leurs références. Sert à ÉCRIRE des
    questions : on lit un extrait, on formule la question à laquelle il répond,
    et on note sa référence dans le jeu d'or.

``--run``
    Mesure le jeu d'or sur la base vectorielle ACTUELLE (nécessite Ollama pour
    calculer l'embedding de chaque question) et écrit un rapport horodaté.

``--compare A B``
    Compare deux rapports déjà écrits. Ne touche ni à la base ni au modèle :
    fonctionne hors ligne, sans Ollama.

Exemples
--------
    python scripts/eval_rag.py --sample 10
    python scripts/eval_rag.py --run --label baseline
    python scripts/eval_rag.py --run --label avec-reranker
    python scripts/eval_rag.py --compare baseline avec-reranker
"""

import argparse
import contextlib
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

# Permet d'exécuter le script directement (python scripts/eval_rag.py)
# en rendant le package `app` importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, evaluation, rag

# Emplacements par défaut, à la racine du projet.
BASE_DIR = Path(__file__).resolve().parent.parent
GOLDEN_PATH = BASE_DIR / "eval" / "golden.jsonl"
RESULTS_DIR = BASE_DIR / "eval" / "results"

# Nombre de résultats ramenés pendant la mesure.
# Volontairement PLUS GRAND que `N_RESULTS` (5, la valeur de l'application) :
# mesurer sur 10 résultats dit si le bon passage est au moins dans le voisinage
# du top-5, information que le seul hit@5 ne donnerait pas. Le MRR, lui, reste
# sensible au rang exact.
DEFAULT_K = 10

SEPARATEUR = "─" * 68

GABARIT = """\
Pour ajouter une question au jeu d'or, écrivez UNE ligne JSON par question
(le texte arabe reste lisible, le fichier est en UTF-8) :

  {"id": "q001", "question": "ما حكم ...؟", "lang": "ar", "status": "validated",
   "expected": [{"source": "arabic_document.pdf", "page": 3}],
   "notes": "lignes 12-24"}

  · "expected" accepte plusieurs entrées (une réponse utile peut venir de
    plusieurs passages).
  · "page" peut être omise : n'importe quelle page du fichier conviendra alors.
    C'est plus rapide à écrire, mais le score est moins exigeant.
  · "status" vaut "draft" (par défaut) tant que vous n'avez pas relu la question.
    Un brouillon reste mesurable : le biais est le même avant et après un
    réglage, donc les COMPARAISONS restent valides."""


def afficher(texte: str) -> None:
    """Écrit sur la sortie standard en UTF-8 (le texte arabe doit rester lisible).

    Sans cette précaution, une console Windows en cp1252 lèverait une
    UnicodeEncodeError en affichant du texte arabe. Un flux redirigé, ou une
    plateforme sans reconfiguration, n'est pas un échec : on écrit tel quel.
    """
    with contextlib.suppress(AttributeError, ValueError, OSError):
        sys.stdout.reconfigure(encoding="utf-8")
    print(texte)


# --- Accès à la base vectorielle -----------------------------------------

def inventorier_corpus(collection, taille_lot: int = 5000):
    """Parcourt la collection UNE fois : empreinte du corpus + sources indexées.

    Renvoie ``(accumulateur_d_empreinte, noms_de_fichiers)``.

    Un seul parcours pour deux besoins — l'empreinte du corpus (les identifiants)
    et la détection des questions orphelines (les noms de fichiers) : parcourir
    deux fois une base de plusieurs millions de chunks doublerait le temps de
    l'inventaire sans rien apporter.

    Rien n'est conservé en mémoire hormis les NOMS DE FICHIERS — quelques milliers
    de chaînes, même pour la bibliothèque Shamela entière. L'empreinte, elle, est
    calculée au fil de l'eau (voir ``evaluation.CorpusFingerprint``) : à 23 millions
    de chunks, constituer la liste complète des identifiants pour la trier
    demanderait plusieurs gigaoctets.

    On ne demande que les métadonnées, jamais les textes ni les vecteurs : c'est
    la charge la plus faible disponible, les identifiants étant de toute façon
    toujours renvoyés par l'API ChromaDB.
    """
    empreinte = evaluation.CorpusFingerprint()
    sources: set[str] = set()
    offset = 0

    while True:
        lot = collection.get(limit=taille_lot, offset=offset, include=["metadatas"])
        ids = list(lot.get("ids") or [])
        if not ids:
            break

        empreinte.update(ids)
        for meta in lot.get("metadatas") or []:
            source = (meta or {}).get("source")
            if source:
                sources.add(source)

        if len(ids) < taille_lot:
            break
        offset += taille_lot

    return empreinte, sources


def extraits_aleatoires(collection, nombre: int, seed: int) -> list[dict]:
    """Tire des extraits au hasard dans la collection.

    Chaque extrait est lu individuellement (``limit=1`` + ``offset``) plutôt que
    par un chargement global : le tirage reste instantané même quand la base
    contiendra des millions de chunks — ce qui est précisément l'objectif du
    projet.
    """
    total = collection.count()
    if total == 0:
        return []

    nombre = min(nombre, total)
    generateur = random.Random(seed)
    extraits: list[dict] = []

    for offset in sorted(generateur.sample(range(total), nombre)):
        lot = collection.get(limit=1, offset=offset, include=["documents", "metadatas"])
        documents = lot.get("documents") or []
        metadatas = lot.get("metadatas") or []
        if not documents:
            continue

        meta = metadatas[0] or {}
        extraits.append({
            "text": documents[0],
            "source": meta.get("source"),
            "page": meta.get("page"),
            "line_start": meta.get("line_start"),
            "line_end": meta.get("line_end"),
        })

    return extraits


def mise_en_contexte(collection, k: int, questions: list, empreinte, sources: set) -> dict:
    """Photographie de la configuration au moment de la mesure.

    Indispensable : un rapport dont on ignore le modèle, la taille de chunk ou la
    taille du corpus est inexploitable six mois plus tard — on ne saurait pas
    expliquer l'écart constaté.

    L'empreinte et les sources sont fournies par l'appelant : elles viennent du
    même parcours de la collection, qu'on ne refait donc pas ici.

    La recherche est décrite EN ENTIER, poids compris. Un rapport qui ne dit pas
    « fusion » mesure peut-être la fusion sans le savoir : deux mesures
    inconciliables porteraient alors le même nom, et on attribuerait au réglage
    suivant l'effet du précédent. C'est la même raison qui fait inscrire le modèle
    d'embedding dans les métadonnées de la collection.
    """
    pipeline = "vectoriel seul"
    hybride = {}
    if config.HYBRID_ENABLED:
        pipeline = "fusion lexicale + vectorielle (RRF)"
        hybride = {
            "hybride_candidats": config.HYBRID_CANDIDATES,
            "hybride_rrf_k": config.HYBRID_RRF_K,
            "hybride_poids_vecteur": config.HYBRID_VECTOR_WEIGHT,
            "hybride_poids_lexical": config.HYBRID_LEXICAL_WEIGHT,
        }

    return {
        "pipeline": pipeline,
        "modele_embedding": config.EMBEDDING_MODEL,
        "chunk_size": config.CHUNK_SIZE,
        "n_results_app": config.N_RESULTS,
        "k_evalue": k,
        "seuil_distance": config.DISTANCE_THRESHOLD,
        "chunks_indexes": collection.count(),
        "fichiers_indexes": len(sources),
        "empreinte_corpus": empreinte.hexdigest(),
        "questions_draft": sum(
            1 for question in questions if question.status == evaluation.STATUS_DRAFT
        ),
        **hybride,
    }


# --- Modes ----------------------------------------------------------------

def mode_sample(args: argparse.Namespace) -> int:
    """Affiche des extraits au hasard pour aider à écrire des questions."""
    collection = rag.get_collection()
    extraits = extraits_aleatoires(collection, args.sample, args.seed)

    if not extraits:
        afficher(
            "❌ La base vectorielle est vide.\n"
            "   Indexez d'abord des documents : python scripts/build_kb.py"
        )
        return 1

    blocs = [
        f"{evaluation.accorder(len(extraits), 'extrait')} au hasard "
        "— écrivez une question par extrait."
    ]

    for numero, extrait in enumerate(extraits, start=1):
        blocs += [
            "",
            SEPARATEUR,
            f"Extrait {numero}/{len(extraits)}",
            f"  Fichier : {extrait['source']}",
            f"  Page    : {extrait['page']}",
            f"  Lignes  : {extrait['line_start']}-{extrait['line_end']}",
            SEPARATEUR,
            extrait["text"],
        ]

    blocs += ["", SEPARATEUR, GABARIT]
    texte = "\n".join(blocs)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(texte + "\n", encoding="utf-8")
        afficher(f"✅ {len(extraits)} extraits écrits dans {args.out}")
    else:
        afficher(texte)

    return 0


def mode_run(args: argparse.Namespace) -> int:
    """Mesure le jeu d'or sur la base vectorielle actuelle."""
    try:
        questions = evaluation.load_golden(args.golden)
    except evaluation.GoldenError as exc:
        afficher(f"❌ {exc}")
        return 1

    if not questions:
        afficher(
            f"❌ Le jeu d'or est vide : {args.golden}\n"
            "   Écrivez des questions avec : python scripts/eval_rag.py --sample 10"
        )
        return 1

    label = args.label or datetime.now().strftime("%Y%m%d-%H%M%S")

    try:
        collection = rag.get_collection()
    except Exception as exc:
        afficher(f"❌ Base vectorielle injoignable : {exc}")
        return 1

    total = collection.count()
    if total == 0:
        afficher(
            "❌ La base vectorielle est vide.\n"
            "   Indexez d'abord des documents : python scripts/build_kb.py"
        )
        return 1

    empreinte, sources_indexees = inventorier_corpus(collection)

    a_mesurer = [
        question
        for question in questions
        if not evaluation.est_orpheline(question, sources_indexees)
    ]
    orphelines = [
        question
        for question in questions
        if evaluation.est_orpheline(question, sources_indexees)
    ]

    if not a_mesurer:
        afficher(
            "❌ Aucune question mesurable : la source d'AUCUNE d'entre elles n'est\n"
            "   indexée. Deux causes possibles, et deux remèdes :\n"
            "   · les documents visés ont été RETIRÉS du corpus — `build_kb.py\n"
            "     --status` les montre en « orphelin » du registre, et `--forget`\n"
            "     finit de les enlever ;\n"
            "   · le jeu d'or vise des documents qui n'ont jamais été indexés :\n"
            "     les questions doivent alors être réécrites pour le corpus actuel.\n"
            "   En attendant, le harnais refuse de mesurer : un score de 0 % dirait\n"
            "   le contraire de la vérité."
        )
        return 1

    afficher(
        f"Mesure de {evaluation.accorder(len(a_mesurer), 'question')} "
        f"(top-{args.k}) sur {evaluation.accorder(total, 'chunk')}…"
    )

    if orphelines:
        afficher("")
        afficher(
            f"⚠️  {evaluation.accorder_question(len(orphelines), 'ignorée')} : "
            "source non indexée."
        )
        for question in orphelines:
            attendues = ", ".join(source.source for source in question.expected)
            afficher(f"    · {question.id} — {attendues}")
        afficher(
            "    Exclues du calcul : comptées « introuvables », elles feraient chuter\n"
            "    le score à cause d'un document retiré du corpus, et non de la\n"
            "    qualité de la récupération.\n"
            "    Réindexez le document, ou retirez la question du jeu d'or."
        )
        afficher("")

    resultats = []
    for question in a_mesurer:
        debut = time.perf_counter()
        try:
            _, sources = rag.retrieve(collection, question.question, n_results=args.k)
        except Exception as exc:
            afficher(
                f"\n❌ Échec sur la question « {question.id} » : {exc}\n"
                f"   Ollama est-il démarré ({config.OLLAMA_URL}) et le modèle "
                f"« {config.EMBEDDING_MODEL} » disponible ?"
            )
            return 1
        latence_ms = (time.perf_counter() - debut) * 1000

        resultats.append(evaluation.QuestionResult(
            question_id=question.id,
            question=question.question,
            rank=evaluation.first_match_rank(sources, question.expected),
            latency_ms=latence_ms,
            retrieved=tuple(sources),
        ))

    rapport = evaluation.build_report(
        label=label,
        k=args.k,
        results=resultats,
        context=mise_en_contexte(
            collection, args.k, a_mesurer, empreinte, sources_indexees
        ),
        ignored_ids=[question.id for question in orphelines],
    )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    chemin = args.results_dir / f"{label}.json"
    chemin.write_text(
        json.dumps(rapport.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    afficher("")
    afficher(evaluation.format_report(rapport))
    afficher("")
    afficher(f"📄 Rapport enregistré : {chemin}")
    afficher(f"   Comparer plus tard : python scripts/eval_rag.py --compare {label} <autre-label>")

    return 0


def charger_rapport(results_dir: Path, label: str) -> evaluation.EvaluationReport:
    """Relit un rapport depuis le disque."""
    chemin = results_dir / f"{label}.json"
    if not chemin.exists():
        raise evaluation.GoldenError(
            f"Rapport introuvable : {chemin}\n"
            f"  Mesurez-le d'abord : python scripts/eval_rag.py --run --label {label}"
        )
    return evaluation.EvaluationReport.from_dict(
        json.loads(chemin.read_text(encoding="utf-8"))
    )


def mode_compare(args: argparse.Namespace) -> int:
    """Compare deux rapports (avant / après), sans base ni modèle."""
    avant, apres = args.compare

    try:
        rapport_avant = charger_rapport(args.results_dir, avant)
        rapport_apres = charger_rapport(args.results_dir, apres)
    except evaluation.GoldenError as exc:
        afficher(f"❌ {exc}")
        return 1

    afficher(evaluation.format_comparison(rapport_avant, rapport_apres))
    return 0


# --- Point d'entrée -------------------------------------------------------

def construire_analyseur() -> argparse.ArgumentParser:
    """Décrit la ligne de commande."""
    analyseur = argparse.ArgumentParser(
        description="Évalue la qualité de la récupération du RAG.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Exemples")[-1].strip(),
    )

    modes = analyseur.add_mutually_exclusive_group(required=True)
    modes.add_argument(
        "--sample",
        type=int,
        metavar="N",
        help="affiche N extraits au hasard (pour écrire des questions)",
    )
    modes.add_argument(
        "--run",
        action="store_true",
        help="mesure le jeu d'or et enregistre un rapport",
    )
    modes.add_argument(
        "--compare",
        nargs=2,
        metavar=("AVANT", "APRES"),
        help="compare deux rapports déjà enregistrés (hors ligne)",
    )

    analyseur.add_argument(
        "--golden",
        type=Path,
        default=GOLDEN_PATH,
        help=f"jeu d'or à utiliser (défaut : {GOLDEN_PATH})",
    )
    analyseur.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help=f"dossier des rapports (défaut : {RESULTS_DIR})",
    )
    analyseur.add_argument(
        "--label",
        help="nom du rapport à écrire (défaut : horodatage)",
    )
    analyseur.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=f"nombre de résultats examinés (défaut : {DEFAULT_K})",
    )
    analyseur.add_argument(
        "--seed",
        type=int,
        default=0,
        help="graine du tirage aléatoire de --sample (défaut : 0)",
    )
    analyseur.add_argument(
        "--out",
        type=Path,
        help="--sample : écrire dans un fichier plutôt que sur la console",
    )
    return analyseur


def main() -> int:
    """Lance le mode demandé et renvoie le code de sortie."""
    args = construire_analyseur().parse_args()

    if args.k < 1:
        afficher(f"❌ --k doit être >= 1 (reçu : {args.k}).")
        return 1

    if args.sample is not None:
        if args.sample < 1:
            afficher(f"❌ --sample doit être >= 1 (reçu : {args.sample}).")
            return 1
        return mode_sample(args)

    if args.run:
        return mode_run(args)

    return mode_compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
