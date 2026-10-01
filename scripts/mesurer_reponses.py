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
3. **La latence ne se compare pas telle quelle.** La récupération étant
   déterministe, la répétition n d'une question renvoie une invite IDENTIQUE à la
   précédente : Ollama réutilise alors son cache de préfixe et répond en ~1 s au
   lieu de ~70 s. Mesuré : q019 à 1 216 ms après un passage antérieur, q046 à
   72 068 ms sur une invite neuve. Un écart de latence entre deux essais dit donc
   qui a payé le préchargement, pas qui a été plus rapide.

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

# Le terminal Windows par défaut (cp1252) ne sait pas écrire l'arabe : sans cela,
# l'affichage d'un extrait fait planter la mesure APRÈS avoir payé les appels au
# modèle — on perd la mesure entière pour un problème d'affichage. Le rapport est
# de toute façon écrit en UTF-8 explicitement plus bas.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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
        choix = None
        brute = ""
        if not docs:
            # Aucun extrait : l'API ne consulte pas le modèle (voir app/main.py).
            reponse = ""
        elif config.ANSWER_MODE == "selection":
            # Régime SÉLECTION : le modèle choisit une étiquette, il ne rédige pas.
            # Le texte rendu est un passage du corpus — l'invention est impossible
            # par construction, et le rang choisi dit si le choix est le bon.
            reponse, choix, brute = rag.repondre_par_selection(
                question.question, docs, sources, question.lang
            )
        elif config.ANSWER_MODE == "refus_puis_selection":
            # Régime REFUS PUIS SÉLECTION : la question ouverte décide s'il faut
            # renoncer, la sélection ancre ensuite la réponse. Le texte du premier
            # appel n'est jamais montré — d'où `generate` (un appel) et non
            # `answer_question` (qui pourrait en coûter deux).
            reponse, choix, brute = rag.repondre_par_refus_puis_selection(
                question.question, docs, sources, question.lang
            )
        else:
            reponse = rag.answer_question(question.question, contexte, question.lang)
        duree = (time.perf_counter() - debut) * 1000

        essais.append({
            "repetition": numero,
            "answer": reponse,
            "context": docs,
            "sources": sources,
            "selection": choix,
            # La sortie NON interprétée, conservée même quand elle est illisible :
            # sans elle, « le modèle n'a rien écrit », « il a écrit un numéro hors
            # bornes » et « il a écrit de la prose » sont indiscernables.
            "selection_brute": brute,
            "refus": (
                (choix is None or choix == 0)
                if config.ANSWER_MODE != "texte"
                else ((not reponse) or bool(rag.est_un_refus(reponse, question.lang)))
            ),
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


def _rang_attendu(sources, expected):
    """Rang (1-indexé) du premier extrait qui porte la page attendue, ou None."""
    for rang, source in enumerate(sources, start=1):
        if any(evaluation.source_matches(source, attendu) for attendu in expected):
            return rang
    return None


def _bilan_selection(questions, resultats):
    """Compte les choix du chemin de sélection — séparément par population.

    Sur une question **répondable**, « juste » veut dire : le passage choisi est
    celui que la question désigne (même source, même page). Sur une question
    **hors corpus**, il n'y a rien à choisir : la seule issue juste est
    l'abstention (``0``).

    Les deux populations ne se comptent donc pas ensemble : un taux global
    mélangerait « a bien choisi » et « a bien renoncé », qui ne sont pas la même
    compétence.

    ⚠️ Deux chiffres ne disent rien sans un TROISIÈME, et c'est le plus
    important : **prendre systématiquement le premier extrait**, sans aucun appel
    au modèle, réussit déjà 50,8 % des questions sur les 59 du jeu d'or. Un taux
    de choix justes inférieur à cette base signifierait que le modèle coûte 70 s
    par question pour faire moins bien que de ne rien lui demander. La base est
    donc calculée **sur les mêmes questions**, à côté du plafond de récupération
    (« le bon extrait était-il seulement visible parmi les 5 ? »).
    """
    par_id = {question.id: question for question in questions}
    bilan = {
        "repondables": {
            "essais": 0,
            "visibles": 0,
            "rang1_juste": 0,
            "justes": 0,
            "abstentions": 0,
        },
        "hors_corpus": {"essais": 0, "abstentions": 0, "choix": 0},
        "illisibles": 0,
        "questions": 0,
        "questions_choix_stable": 0,
    }
    for resultat in resultats:
        question = par_id[resultat["id"]]
        population = bilan["hors_corpus" if question.est_hors_corpus else "repondables"]
        bilan["questions"] += 1
        choix_vus = {
            essai["selection"] for essai in resultat["attempts"]
        }
        if len(choix_vus) == 1:
            # Un choix IDENTIQUE d'une répétition à l'autre : la tâche fermée est
            # reproductible, là où la rédaction libre ne l'était pas.
            bilan["questions_choix_stable"] += 1

        for essai in resultat["attempts"]:
            population["essais"] += 1
            choix = essai["selection"]
            if choix is None:
                # Ni un chiffre exploitable, ni « 0 » : le modèle n'a pas joué le
                # jeu. C'est un échec du protocole, pas un choix — compté à part.
                bilan["illisibles"] += 1

            if question.est_hors_corpus:
                if choix == 0:
                    population["abstentions"] += 1
                elif choix is not None:
                    population["choix"] += 1
                continue

            # Ce que la SÉLECTION ne peut pas corriger, et qu'il faut donc compter
            # à part : choisir juste exige que le bon extrait soit visible. Compté
            # pour TOUS les essais, choix illisible compris — c'est une propriété
            # de la récupération, pas du modèle.
            rang = _rang_attendu(essai["sources"], question.expected)
            if rang is not None:
                population["visibles"] += 1
                if rang == 1:
                    population["rang1_juste"] += 1

            if choix == 0:
                population["abstentions"] += 1
            elif choix is not None and choix <= len(essai["sources"]):
                choisi = essai["sources"][choix - 1]
                if any(
                    evaluation.source_matches(choisi, attendu)
                    for attendu in question.expected
                ):
                    population["justes"] += 1
    return bilan


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
    # En mode sélection le modèle ne RÉDIGE plus : il n'écrit aucune citation, et
    # le compteur ci-dessus mesurerait l'absence d'une chose qui n'est plus
    # demandée. On le neutralise au lieu d'afficher un chiffre trompeur.
    bilan_selection = (
        _bilan_selection(questions, resultats) if config.ANSWER_MODE != "texte" else None
    )
    rapport = {
        "label": label,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "context": {
            "modele_generation": config.LLM_MODEL,
            "modele_embedding": config.EMBEDDING_MODEL,
            "repetitions": args.repetitions,
            "n_results_app": config.N_RESULTS,
            "hybride": config.HYBRID_ENABLED,
            "selection_passage": config.ANSWER_MODE,
            "selection_etiquettes": config.SELECTION_ETIQUETTES,
            "selection_choix": config.SELECTION_CHOIX,
            "citations_obligatoires": config.CITATIONS_OBLIGATOIRES,
        },
        "summary": {
            "questions": len(resultats),
            "reponses": total,
            "refus_reconnus": refus,
            "non_verifiables": len(non_verifiables) if config.ANSWER_MODE == "texte" else None,
            "selection": bilan_selection,
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
    if bilan_selection:
        repondables = bilan_selection["repondables"]
        hors_corpus = bilan_selection["hors_corpus"]
        print(
            f"Sélection (répondables) : {repondables['justes']} / {repondables['essais']}"
            "   choix tombant sur la page attendue"
        )
        print(
            f"   plafond récupération : {repondables['visibles']} / {repondables['essais']}"
            "   (le bon extrait était dans les 5 montrés)"
        )
        print(
            f"   base « rang 1 »      : {repondables['rang1_juste']} / {repondables['essais']}"
            "   (prendre le premier extrait, SANS modèle : le modèle doit battre ceci)"
        )
        print(
            f"   dont abstentions     : {repondables['abstentions']}"
            "   (le modèle renonce sur une question répondable : c'est une perte)"
        )
        print(
            f"Sélection (hors corpus) : {hors_corpus['abstentions']} / {hors_corpus['essais']}"
            "   abstentions — attendu : toutes"
        )
        print(
            f"   dont choix d'un extrait : {hors_corpus['choix']}"
            "   (répondre là où il n'y a rien)"
        )
        print(
            f"Choix ILLISIBLE        : {bilan_selection['illisibles']} / {total}"
            "   (ni chiffre ni 0 : le protocole n'a pas été suivi)"
        )
        print(
            f"Choix REPRODUCTIBLE    : {bilan_selection['questions_choix_stable']}"
            f" / {bilan_selection['questions']} questions"
            "   (même numéro à toutes les répétitions — à comparer au texte libre,"
            " qui a produit deux fatwas opposées sur la même question)"
        )
    print(
        f"Refus RECONNUS         : {refus} / {total}"
        "   (indice seulement — la reconnaissance par phrase s'est déjà trompée"
        " dans le sens rassurant)"
    )
    if config.ANSWER_MODE != "texte":
        print(
            "NON VÉRIFIABLES        : sans objet"
            "   (le modèle ne rédige pas : le texte affiché est un extrait du corpus)"
        )
    else:
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
