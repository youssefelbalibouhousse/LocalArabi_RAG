"""Mesure les RÉPONSES, et non la récupération — avec répétitions.

Pourquoi ce script existe
-------------------------
`eval_rag.py` mesure la récupération : le bon passage est-il retrouvé, et à quel
rang. Il ne dit rien de la RÉPONSE. Or c'est là que se joue la promesse du
produit : chaque affirmation doit provenir des textes cités, et pouvoir y être
vérifiée.

Deux leçons ont rendu ce script nécessaire.

1. **Le modèle est stochastique.** Une comparaison d'invites faite sur UNE
   exécution de chaque côté n'est pas interprétable : elle ne distingue pas un
   effet d'une fluctuation. D'où `--repetitions`.
2. **Le refus ne se reconnaît pas à une phrase.** Un détecteur cherchant la
   formule de l'invite a compté 5 refus sur 15 là où une lecture en trouve 12, et
   2 sur 9 là où il y en a 4 — deux fois dans le sens rassurant. Le compte de
   refus est donc affiché comme un INDICE, et les réponses brutes sont conservées
   dans le rapport : c'est la lecture qui tranche.

Ce que le rapport contient, et pourquoi
---------------------------------------
Les **contextes** sont enregistrés avec les réponses. Sans eux, on ne peut pas
évaluer hors ligne un contrôle de fidélité (« cette réponse vient-elle vraiment
des extraits ? ») sans rappeler le modèle — et une mesure qui coûte 15 minutes
ne se refait pas assez souvent pour être honnête.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Permet d'exécuter le script directement (python scripts/mesurer_reponses.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, evaluation, fidelite, rag

BASE_DIR = Path(__file__).resolve().parent.parent
GOLDEN_PATH = BASE_DIR / "eval" / "golden.jsonl"
RESULTS_DIR = BASE_DIR / "eval" / "results"

SEPARATEUR = "─" * 68


def mesurer_question(collection, question, repetitions):
    """Répète la question et rassemble les réponses produites.

    Retourne un dictionnaire par répétition : la réponse, le contexte EXACT qui a
    été soumis au modèle, et les sources qui l'accompagneraient.
    """
    essais = []
    for numero in range(1, repetitions + 1):
        docs, sources = rag.retrieve(
            collection, question.question, n_results=config.N_RESULTS
        )
        contexte = "\n\n".join(docs)

        debut = time.perf_counter()
        # Aucun extrait : l'API ne consulte pas le modèle (voir app/main.py).
        reponse = (
            rag.answer_question(question.question, contexte, question.lang) if docs else ""
        )
        duree = (time.perf_counter() - debut) * 1000

        essais.append({
            "repetition": numero,
            "answer": reponse,
            "context": docs,
            "sources": sources,
            "refus": bool(rag.est_un_refus(reponse, question.lang)) if reponse else True,
            # Les citations sont extraites et vérifiées ICI, et conservées dans le
            # rapport : c'est le seul verdict mécaniquement démontrable. Une
            # citation absente du contexte est une invention PROUVÉE — là où une
            # réponse libre, faite du vocabulaire du corpus, ne se laisse pas
            # juger par des mots.
            "citations": list(fidelite.citations(reponse)),
            "citations_fabriquees": list(
                fidelite.citations_non_verifiees(reponse, contexte)
            ),
            "latency_ms": round(duree, 1),
        })

    return essais


def afficher_entete(question, essais):
    """Rend lisible ce qu'on vient d'obtenir, sans prétendre le juger."""
    lignes = [f"{question.id}  [{question.lang}]  {question.question}"]
    for essai in essais:
        marque = "refus ?" if essai["refus"] else "répond"
        lignes.append(
            f"   #{essai['repetition']} {marque:<7} {essai['latency_ms']:>7.0f} ms"
        )
        lignes.append(f"      {_apercu(essai['answer'])}")
    return "\n".join(lignes)


def _apercu(texte, longueur=220):
    """Un aperçu sur une ligne : le terminal n'affiche pas l'arabe sur plusieurs."""
    if not texte:
        return "(aucune réponse : la récupération n'a rien rendu)"
    return " ".join(texte.split())[:longueur]


def mode_rapport(args):
    """Mesure les questions demandées et écrit un rapport horodaté."""
    try:
        questions = evaluation.load_golden(args.golden)
    except evaluation.GoldenError as exc:
        print(f"❌ {exc}")
        return 1

    if args.hors_corpus:
        questions = [question for question in questions if question.est_hors_corpus]
    if args.ids:
        demandes = {identifiant.strip() for identifiant in args.ids.split(",")}
        questions = [question for question in questions if question.id in demandes]

    if not questions:
        print("❌ Aucune question ne correspond à la sélection.")
        return 1

    try:
        collection = rag.get_collection()
    except Exception as exc:
        print(f"❌ Base vectorielle injoignable : {exc}")
        return 1

    total = len(questions) * args.repetitions
    print(
        f"Mesure de {len(questions)} question(s) × {args.repetitions} répétition(s)"
        f" = {total} réponse(s) — modèle {config.LLM_MODEL}"
    )
    print("  (chaque réponse peut prendre une minute sur CPU)")

    label = args.label or datetime.now().strftime("%Y%m%d-%H%M%S")

    resultats = []
    for question in questions:
        essais = mesurer_question(collection, question, args.repetitions)
        resultats.append({
            "id": question.id,
            "question": question.question,
            "lang": question.lang,
            "hors_corpus": question.est_hors_corpus,
            "attempts": essais,
        })
        print("")
        print(afficher_entete(question, essais))

    refus = sum(
        1
        for resultat in resultats
        for essai in resultat["attempts"]
        if essai["refus"]
    )
    non_verifiables = [
        (resultat["id"], essai["repetition"])
        for resultat in resultats
        for essai in resultat["attempts"]
        if not essai["refus"]
        and (essai["citations_fabriquees"] or not essai["citations"])
    ]
    rapport = {
        "label": label,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "context": {
            "modele_generation": config.LLM_MODEL,
            "modele_embedding": config.EMBEDDING_MODEL,
            "repetitions": args.repetitions,
            "n_results_app": config.N_RESULTS,
            "hybride": config.HYBRID_ENABLED,
        },
        "summary": {
            "questions": len(resultats),
            "reponses": total,
            "refus_reconnus": refus,
            "non_verifiables": len(non_verifiables),
        },
        "results": resultats,
    }

    args.results_dir.mkdir(parents=True, exist_ok=True)
    chemin = args.results_dir / f"{label}.json"
    chemin.write_text(
        json.dumps(rapport, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("")
    print(SEPARATEUR)
    print(f"Réponses mesurées      : {total}")
    print(
        f"Refus RECONNUS         : {refus} / {total}"
        "   (indice seulement — la reconnaissance par phrase s'est déjà trompée"
        " dans le sens rassurant)"
    )
    print(
        f"NON VÉRIFIABLES        : {len(non_verifiables)} / {total - refus}"
        "   (affirmation sans citation, ou citation absente du contexte —"
        " invention démontrée)"
    )
    print(SEPARATEUR)
    print(f"📄 Rapport enregistré : {chemin}")
    print("   Les contextes y sont conservés : un contrôle de fidélité peut être")
    print("   évalué hors ligne, sans rappeler le modèle.")
    return 0


def main():
    analyseur = argparse.ArgumentParser(
        description="Mesure les réponses du RAG (et non la récupération)."
    )
    analyseur.add_argument("--golden", type=Path, default=GOLDEN_PATH)
    analyseur.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    analyseur.add_argument("--label", default=None)
    analyseur.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Combien de fois reposer chaque question (défaut 1).",
    )
    analyseur.add_argument(
        "--hors-corpus",
        action="store_true",
        help="Ne mesurer que les questions dont la réponse n'est pas dans le corpus.",
    )
    analyseur.add_argument(
        "--ids",
        default=None,
        help="Liste d'identifiants séparés par des virgules (ex. q075,q076).",
    )
    args = analyseur.parse_args()

    if args.repetitions < 1:
        print("❌ --repetitions doit valoir au moins 1.")
        return 1

    return mode_rapport(args)


if __name__ == "__main__":
    raise SystemExit(main())
