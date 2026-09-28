"""Mesure le débit d'un serveur d'embeddings, AVANT de payer des heures de GPU.

Pourquoi ce script existe
-------------------------
Louer une machine GPU se paie à l'heure. Choisir un modèle d'embedding sans
savoir combien de chunks par seconde il produit, c'est signer un chèque en
blanc. Sur la machine de développement (i7-1165G7, sans GPU dédié), `bge-m3`
traite ~1,8 chunk/s — soit 4 minutes pour un livre, et **2,5 jours pour 1 000**.

Ce script répond à trois questions, dans cet ordre :

1. **Quel débit ?** chunks/s et millisecondes par chunk de l'endpoint configuré.
   C'est ce qui se compare entre deux fournisseurs, et ce qui se projette.
2. **Quelle dimension ?** la taille des vecteurs, dont dépend celle de l'index :
   un modèle à 4096 dimensions coûte 4× plus de disque qu'un modèle à 1024.
3. **La configuration tient-elle ?** la taille de lot multipliée par le temps par
   chunk doit rester TRÈS en dessous de `EMBEDDING_TIMEOUT`. Un lot part en
   **une seule** requête : trop gros, l'écriture échoue sur un
   « timed out in add » qui ne dit ni le lot, ni le délai, ni la cause. Cette
   panne a déjà été rencontrée ici, d'où ce contrôle automatique.

Usage :
    python scripts/benchmark_embeddings.py                  # endpoint configuré
    python scripts/benchmark_embeddings.py --total 390000   # projet sur 1 000 livres
    python scripts/benchmark_embeddings.py --taille-lot 64  # tester un autre lot

Code de sortie : 0 si la configuration est cohérente, 1 si la marge est
insuffisante — pour qu'un script d'automatisation puisse refuser un réglage
dangereux sans lire la sortie.
"""

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, rag

# Tailles mesurées : la plus petite donne le coût d'un appel isolé (latence
# réseau, surcoût par requête), la plus grande le débit réel. Comparer les deux
# dit si le serveur est limité par le calcul ou par les allers-retours réseau.
# Elles sont mesurées APRÈS le préchauffage — voir `main`.
PETITES_TAILLES = (1, 8)

# Nombre de chunks d'un ouvrage de ~150 pages, mesuré sur `12445.epub`
# (387 chunks). Sert à la projection : c'est la seule constante qui vienne du
# corpus, pas du matériel.
CHUNKS_PAR_LIVRE = 390

# Tailles de corpus projetées.
CORPUS_PROJECTION = (
    ("1 livre", 1),
    ("100 livres", 100),
    ("1 000 livres", 1_000),
    ("8 000 livres", 8_000),
)

# Marge minimale exigée entre le temps d'un lot et le délai d'expiration.
# 5× n'est pas du zèle : la machine qui exécutera le vrai passage peut être
# plus lente, et un lot qui frôle le délai échoue de façon intermittente.
MARGE_EXIGEE = 5.0


def formater_nombre(valeur: float, decimales: int = 1) -> str:
    """Nombre à la française (virgule décimale), comme les rapports d'évaluation."""
    return f"{valeur:.{decimales}f}".replace(".", ",")


def formater_duree(secondes: float) -> str:
    """Durée lisible : « 1,8 s », « 42 min », « 2,5 jours »."""
    if secondes < 90:
        return f"{formater_nombre(secondes)} s"

    minutes = secondes / 60
    if minutes < 90:
        return f"{minutes:.0f} min"

    heures = minutes / 60
    if heures < 36:
        return f"{formater_nombre(heures)} h"

    return f"{formater_nombre(heures / 24)} jours"


def projeter(ms_par_chunk: float, chunks: int) -> float:
    """Durée, en secondes, pour embarquer `chunks` chunks au débit donné."""
    return ms_par_chunk * chunks / 1000.0


def verdict_coherence(
    taille_lot: int, ms_par_chunk: float, timeout_s: int
) -> tuple[bool, str]:
    """La taille de lot tient-elle dans le délai d'expiration ?

    Renvoie (cohérent, explication). Volontairement sévère : une marge de 2×
    passe pour confortable sur la machine de mesure et casse sur une autre.
    """
    duree = projeter(ms_par_chunk, taille_lot)
    marge = timeout_s / duree if duree > 0 else float("inf")

    if marge >= MARGE_EXIGEE:
        return True, (
            f"{formater_nombre(duree)} s par lot → "
            f"{formater_nombre(marge)}× sous le délai  ✅"
        )
    if marge >= 2:
        return False, (
            f"{formater_nombre(duree)} s par lot → {formater_nombre(marge)}× sous "
            "le délai  ⚠️  marge faible : une machine plus lente échouera"
        )
    return False, (
        f"{formater_nombre(duree)} s par lot → DÉPASSE ou frôle le délai de "
        f"{timeout_s} s  ❌  baisser INGEST_BATCH_SIZE, ou monter EMBEDDING_TIMEOUT"
    )


def echantillon(collection, nombre: int) -> tuple[list[str], bool]:
    """Textes à embarquer, et si ce sont de vrais chunks du corpus.

    Le temps d'embedding dépend de la LONGUEUR du texte : mesurer sur des
    phrases courtes donne un débit flatteur et faux. On prend donc de vrais
    chunks si l'index en contient. À défaut, on fabrique des textes de la
    taille configurée — approximation assumée, signalée dans le rapport.
    """
    if collection is not None and collection.count() > 0:
        resultat = collection.get(limit=nombre, include=["documents"])
        textes = [texte for texte in (resultat.get("documents") or []) if texte]
        if textes:
            return textes, True

    gabarit = "وأجمعوا على أن العلم لا ينفك عن العمل، وأن العمل لا يقبل إلا بالإخلاص. "
    # Division ARRONDIE AU SUPÉRIEUR : une division entière donnerait un texte
    # plus COURT que la taille configurée, donc une latence sous-estimée —
    # exactement le biais que ce script doit éviter.
    repetition = max(1, math.ceil(config.CHUNK_SIZE / len(gabarit)))
    return [(gabarit * repetition)[: config.CHUNK_SIZE] for _ in range(nombre)], False


def chronometrer(fonction, textes: list[str]) -> tuple[float, int]:
    """Renvoie (durée en secondes, dimension des vecteurs produits)."""
    debut = time.perf_counter()
    vecteurs = fonction(textes)
    duree = time.perf_counter() - debut
    return duree, len(vecteurs[0])


def message_injoignable(erreur: Exception) -> None:
    """Explique pourquoi la mesure n'a pas pu se faire, et ce qu'il faut faire."""
    print()
    print(f"❌ Impossible de joindre le serveur d'embeddings ({config.OLLAMA_URL}).")
    print(f"   {type(erreur).__name__} : {erreur}")
    print("   Démarrez Ollama, ou vérifiez OLLAMA_URL, puis relancez.")


def analyser_arguments() -> argparse.Namespace:
    """Décrit la ligne de commande."""
    analyseur = argparse.ArgumentParser(
        description="Mesure le débit d'embeddings de l'endpoint configuré.",
    )
    analyseur.add_argument(
        "--taille-lot",
        type=int,
        default=None,
        metavar="N",
        help=f"taille de lot à éprouver (défaut : INGEST_BATCH_SIZE = "
        f"{config.INGEST_BATCH_SIZE})",
    )
    analyseur.add_argument(
        "--total",
        type=int,
        default=None,
        metavar="CHUNKS",
        help="projette aussi la durée pour ce nombre total de chunks",
    )
    return analyseur.parse_args()


def main() -> int:
    args = analyser_arguments()
    taille_lot = args.taille_lot if args.taille_lot is not None else config.INGEST_BATCH_SIZE

    if taille_lot < 1:
        print(f"❌ --taille-lot {taille_lot} est invalide : la valeur doit être >= 1.")
        return 2

    collection = rag.get_collection() if Path(config.CHROMA_DB_PATH).exists() else None
    besoin = max([*PETITES_TAILLES, taille_lot, 64])
    textes, reels = echantillon(collection, besoin)

    def fonction(entrees):
        return rag.get_embedding_function()(entrees)

    print(f"Serveur d'embeddings : {config.OLLAMA_URL}")
    print(f"Modèle              : {config.EMBEDDING_MODEL}")
    print(
        "Échantillon         : "
        + (
            f"{len(textes)} chunks réels du corpus"
            if reels
            else f"{len(textes)} textes fabriqués de {config.CHUNK_SIZE} caractères "
            "(approximatif : aucun corpus indexé)"
        )
    )
    print()

    # PRÉCHAUFFAGE. Le premier appel charge le modèle en mémoire : mesuré sur la
    # machine de développement, 3,9 s pour un SEUL texte à froid, contre 0,4 s
    # une fois chaud. Sans cette étape, la plus petite taille mesurée décrirait
    # le chargement du modèle et non le débit — et comparer deux fournisseurs
    # serait trompeur si l'un était déjà chaud et l'autre non.
    try:
        duree_chargement, _ = chronometrer(fonction, textes[:1])
    except Exception as erreur:
        message_injoignable(erreur)
        return 2

    print(f"Premier appel : {formater_duree(duree_chargement)} (préchauffage, hors mesure)")
    print()

    tailles = sorted(
        {taille for taille in (*PETITES_TAILLES, taille_lot) if taille <= len(textes)}
    )
    print(f"{'taille':>8}{'durée':>12}{'ms/chunk':>12}{'chunks/s':>12}")

    mesures: dict[int, float] = {}
    for taille in tailles:
        try:
            duree, dimension = chronometrer(fonction, textes[:taille])
        except Exception as erreur:
            message_injoignable(erreur)
            return 2

        ms_par_chunk = duree * 1000 / taille
        mesures[taille] = ms_par_chunk
        print(
            f"{taille:>8}{formater_duree(duree):>12}"
            f"{formater_nombre(ms_par_chunk, 0):>12}"
            f"{formater_nombre(1000 / ms_par_chunk):>12}"
        )

    if not mesures:
        print("❌ Aucune mesure exploitable.")
        return 2

    # Le débit retenu est celui du PLUS GRAND lot : le coût par chunk y est le
    # plus élevé, donc la projection la plus prudente.
    reference = max(mesures)
    ms_par_chunk = mesures[reference]

    print()
    print(f"Dimension des vecteurs : {dimension}")
    print(
        f"Débit retenu           : {formater_nombre(1000 / ms_par_chunk)} chunk/s "
        f"({formater_nombre(ms_par_chunk)} ms/chunk, mesuré sur un lot de {reference})"
    )

    coherent, message = verdict_coherence(
        taille_lot, ms_par_chunk, config.EMBEDDING_TIMEOUT
    )
    print()
    print("Cohérence de la configuration")
    print(f"  INGEST_BATCH_SIZE : {taille_lot}")
    print(f"  EMBEDDING_TIMEOUT : {config.EMBEDDING_TIMEOUT} s")
    print(f"  {message}")

    print()
    print("Projection")
    lignes = [(nom, livres * CHUNKS_PAR_LIVRE) for nom, livres in CORPUS_PROJECTION]
    if args.total is not None:
        lignes.append((f"{args.total} chunks (demandé)", args.total))
    for nom, chunks in lignes:
        print(
            f"  {nom:<26} {chunks:>10} chunks   {formater_duree(projeter(ms_par_chunk, chunks)):>12}"
        )

    return 0 if coherent else 1


if __name__ == "__main__":
    raise SystemExit(main())
