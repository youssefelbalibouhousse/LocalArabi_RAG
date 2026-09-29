"""Ingestion : met à jour la base vectorielle à partir des documents de data/.

Usage :
    python scripts/build_kb.py                    # ingère ce qui a changé
    python scripts/build_kb.py --force            # ré-ingère tout
    python scripts/build_kb.py --only doc.pdf     # un seul document
    python scripts/build_kb.py --forget doc.pdf   # en RETIRER un (index + registre)
    python scripts/build_kb.py --status           # état registre/index, sans écrire

Deux formats, deux dossiers (voir `config.CORPUS_DIRS`) :
    · PDF  dans `data/documents/` — versionné, donc public ;
    · EPUB dans `data/shamela/`   — ignoré par Git (éditions sous droits).
Le dossier n'est qu'un rangement : l'ingestion est identique.

⚠️ Ce script est INCRÉMENTAL et NON DESTRUCTIF : il ne supprime plus la
collection entière. Un document dont le contenu n'a pas changé est ignoré
(comparaison d'empreintes SHA-256, jamais de dates de modification).

⚠️ Retirer un document est EXPLICITE (`--forget`) et jamais automatique : un
fichier disparu du disque n'est pas retiré pour autant. Un dossier déplacé ou
un disque non monté effacerait sinon un corpus entier sans qu'on l'ait demandé.
"""

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

from pypdf import PdfReader
from sqlmodel import Session

# Permet d'exécuter le script directement (python scripts/build_kb.py)
# en rendant le package `app` importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, epub, ingest, rag
from app.database import create_db_and_tables, engine
from app.rag import get_ollama_client

# Formats de documents reconnus par l'ingestion.
FORMATS = (".pdf", ".epub")

# Erreurs qui ne GUÉRIRONT PAS en réessayant : le fichier est illisible, ou la
# configuration est incohérente. Réessayer ne ferait que perdre du temps.
PERMANENTES = (epub.EpubError, ingest.IngestError)

# Pause entre deux tentatives, en secondes, allongée à chaque reprise.
# Un serveur d'embeddings distant qui redémarre a besoin de quelques secondes ;
# insister immédiatement ne ferait qu'accumuler les échecs.
PAUSE_REPRISE_S = 3.0

# Version du DÉCOUPAGE. À incrémenter dès que `chunk_pages` ou `decouper_ligne`
# change de comportement : sans cela, un index bâti par la version précédente
# resterait cru à jour (voir `ensure_decoupage`).
PIPELINE_VERSION = 2


def extract_arabic_pdf(pdf_path):
    """Retourne une liste de pages : [(numero_page, texte), ...] (page 1-indexée)."""
    print(f"Reading {pdf_path.name}...")
    reader = PdfReader(str(pdf_path))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text and text.strip():
            pages.append((page_number, text))

    return pages


def extraire_epub(chemin: Path) -> tuple[dict, list, str]:
    """Extrait un EPUB → (métadonnées indexables, pages, note de diagnostic).

    Le numéro de page vient de la page IMPRIMÉE, pas du nom du fichier, et les
    morceaux d'une même page sont refusionnés par `app.epub` : la citation
    désigne donc un endroit que le lecteur retrouve dans son exemplaire.
    """
    metadonnees, pages = epub.lire_epub(chemin)

    note = ""
    absentes = epub.pages_absentes(pages)
    if absentes:
        note = (
            f"⚠️  {len(absentes)} page(s) absente(s) de l'édition : "
            f"{epub.formater_plages(absentes)}"
        )

    return metadonnees.pour_index(), [(page.numero, page.texte) for page in pages], note


def extraire(chemin: Path) -> tuple[dict, list, str]:
    """Extrait un document → (métadonnées, pages, note éventuelle).

    `pages` est une liste de couples `(numéro_de_page, texte)` — la même forme
    pour un PDF et pour un EPUB. C'est ce qui permet à `chunk_pages` de les
    traiter sans savoir d'où ils viennent, et donc à la taille de chunk de
    rester réglée à UN seul endroit.
    """
    suffixe = chemin.suffix.lower()

    if suffixe == ".epub":
        return extraire_epub(chemin)
    if suffixe == ".pdf":
        return {}, extract_arabic_pdf(chemin), ""

    raise ingest.IngestError(f"Format non pris en charge : « {chemin.name} ».")


def decouper_ligne(ligne: str, taille: int) -> list[str]:
    """Découpe une ligne en morceaux d'au plus `taille` caractères.

    Indispensable parce qu'une ligne peut être arbitrairement longue : « الأوسط »
    en contient une de **15 870 caractères**. Tant que rien ne la coupe,
    `CHUNK_SIZE` n'est pas un maximum mais une préférence, et un chunk peut
    dépasser la fenêtre du modèle d'embedding — qui **tronque alors en
    silence** : le vecteur ne représente plus que le début du texte, et rien ne
    le signale.

    La coupe se fait de préférence sur un ESPACE, et seulement à défaut au
    milieu d'un mot : un texte arabe coupé n'importe où reste lisible, mais
    autant l'éviter quand c'est possible.
    """
    if len(ligne) <= taille:
        return [ligne]

    morceaux = []
    reste = ligne
    while len(reste) > taille:
        coupe = reste.rfind(" ", 0, taille + 1)
        if coupe <= 0:
            # Aucun espace dans la fenêtre : un seul « mot » trop long, il faut
            # couper dedans. C'est le seul cas où la coupe abîme le texte.
            coupe = taille
        morceaux.append(reste[:coupe])
        reste = reste[coupe:].lstrip()

    if reste:
        morceaux.append(reste)
    return morceaux


def chunk_pages(pages, chunk_size=None):
    """Découpe chaque page en chunks sans franchir les limites de page.

    Retourne une liste de dicts : {text, page, line_start, line_end}
    où line_start/line_end sont les numéros de ligne (1-indexés) dans la page.

    ⚠️ `chunk_size` est un MAXIMUM, pas une préférence. Mesuré sur le corpus
    réel : « الأوسط » contient 773 chunks dépassant 1 000 caractères, dont un de
    15 870 — 26 fois la cible. Ces chunks dépassaient la fenêtre de bge-m3
    (8 192 tokens), qui tronque sans le dire, et faisaient à eux seuls échouer
    l'écriture en « timed out in add ». Les lignes trop longues sont donc
    découpées par `decouper_ligne`.
    """
    if chunk_size is None:
        chunk_size = config.CHUNK_SIZE

    chunks = []

    for page_number, text in pages:
        # Chaque morceau garde le numéro de la ligne D'ORIGINE dont il vient :
        # découper une ligne ne doit pas DÉCALER la numérotation, sinon la
        # citation ne désigne plus le bon endroit du livre. Deux morceaux d'une
        # même ligne partagent donc leur intervalle — ils sont bien tous les
        # deux à cette ligne-là.
        morceaux = [
            (numero, morceau)
            for numero, ligne in enumerate(text.split("\n"), start=1)
            for morceau in decouper_ligne(ligne, chunk_size)
        ]

        current_lines = []
        current_len = 0
        start_line = 1  # numéro de la première ligne du chunk en cours
        end_line = 1

        for numero, morceau in morceaux:
            # Si ajouter ce morceau dépasse la taille cible, on ferme le chunk.
            if current_lines and current_len + len(morceau) >= chunk_size:
                chunks.append({
                    "text": "\n".join(current_lines).strip(),
                    "page": page_number,
                    "line_start": start_line,
                    "line_end": end_line,
                })
                current_lines = []
                current_len = 0
                start_line = numero

            current_lines.append(morceau)
            current_len += len(morceau) + 1  # +1 pour le saut de ligne
            end_line = numero

        if current_lines and "\n".join(current_lines).strip():
            chunks.append({
                "text": "\n".join(current_lines).strip(),
                "page": page_number,
                "line_start": start_line,
                "line_end": end_line,
            })

    return chunks


def enrichir(chunks, metadonnees):
    """Recopie les métadonnées du document sur chacun de ses chunks.

    `ingest.preparer_chunks` indexe tout ce qui accompagne le texte : les poser
    sur chaque chunk les rend donc interrogeables sans toucher au module
    d'ingestion. Un PDF n'apporte rien ici — pypdf ne rend pas de métadonnées
    fiables — d'où le cas « pas de métadonnées », qui n'est pas une anomalie.
    """
    if not metadonnees:
        return chunks
    return [{**chunk, **metadonnees} for chunk in chunks]


def ensure_decoupage(collection, force: bool = False) -> None:
    """Signale si l'index n'a pas forcément été bâti avec le découpage actuel.

    Le registre identifie un document par l'empreinte du CONTENU de son fichier.
    Changer la façon de découper ne change pas cette empreinte : un passage
    ordinaire répondrait « inchangé » et garderait des chunks bâtis par un autre
    découpage. C'est le piège du modèle d'embedding, au même endroit — le
    registre prouve ce qui a été ingéré, jamais COMMENT.

    La recette (`chunk_size`, `pipeline_version`) est inscrite dans les
    métadonnées de la COLLECTION : le découpage est une propriété de tout
    l'index, pas d'un document. Elle n'est inscrite que là où elle est VRAIE —
    collection vide, ou `--force` qui reconstruit tout.

    ⚠️ Avertir plutôt que refuser, et c'est délibéré. Un refus bloque tout, y
    compris le travail légitime : il obligerait à reconstruire un index dont les
    chunks sont parfaitement utilisables. Mesuré : bge-m3 tronque au-delà de
    8 192 tokens, soit ~20 000 caractères d'arabe — un chunk de 1 961 caractères
    en est dix fois loin. Un garde-fou sans issue avait déjà rendu `--force`
    inopérant ; la visibilité vaut mieux qu'un blocage.
    """
    recette = {"chunk_size": config.CHUNK_SIZE, "pipeline_version": PIPELINE_VERSION}
    existante = dict(collection.metadata or {})

    if all(existante.get(cle) == valeur for cle, valeur in recette.items()):
        return

    if force or not collection.count():
        collection.modify(metadata={**existante, **recette})
        return

    # Index peuplé, recette absente ou différente : inscrire celle du code
    # ferait taire l'avertissement alors que les chunks, eux, n'ont pas bougé.
    # Le signal doit durer tant que le doute dure.
    indentifie = ", ".join(f"{cle}={valeur}" for cle, valeur in recette.items())
    print(
        f"⚠️  L'index ne confirme pas son découpage ({indentifie}).\n"
        "   Ses chunks n'ont jamais été comparés à celui du code : ils peuvent\n"
        "   provenir d'une autre version. `--force` les reconstruirait."
    )


def check_ollama():
    """Vérifie que le serveur Ollama est joignable (indispensable pour les embeddings).

    Retourne True si joignable, False sinon (sans rien modifier).
    """
    try:
        get_ollama_client().list()
        return True
    except Exception as exc:
        print(f"❌ Impossible de joindre Ollama sur {config.OLLAMA_URL}.")
        print(f"   Détail : {exc}")
        print("   Démarrez Ollama (ou la pile Docker : docker compose up -d) puis réessayez.")
        return False


# --- Ingestion -----------------------------------------------------------

def ingerer_un_document(
    session, collection, chemin: Path, args
) -> tuple[ingest.IngestResult, str]:
    """Extrait, découpe et ingère un document, en ne refaisant que le nécessaire."""
    empreinte = ingest.fingerprint_file(chemin)

    # Court-circuit AVANT l'extraction. La même décision existe dans
    # `ingest_document`, mais elle arrive trop tard : il aurait fallu relire
    # tout le document pour s'apercevoir qu'il n'avait pas changé. C'est
    # exactement ce que l'empreinte permet d'éviter.
    #
    # Les deux appellent le MÊME prédicat, et c'est volontaire : la comparaison
    # porte sur le contenu ET sur le modèle d'embedding de l'index. Dupliquée,
    # la condition aurait fini par ne plus dire la même chose aux deux endroits.
    if not args.force:
        ligne = ingest.registre_pour(session, chemin.name)
        a_jour = ingest.deja_a_jour(ligne, collection, empreinte)
        if a_jour is not None:
            resultat = ingest.IngestResult(
                chemin.name, "unchanged", a_jour.chunk_count, 0.0
            )
            return resultat, ""

    metadonnees, pages, note = extraire(chemin)
    if not pages:
        print(f"⚠️  Aucun texte extractible dans {chemin.name}.")

    chunks = chunk_pages(pages) if pages else []

    # Les métadonnées du document (titre, auteur, langue…) sont recopiées sur
    # CHAQUE chunk : `preparer_chunks` indexe tout ce qui accompagne le texte,
    # donc elles deviennent interrogeables sans toucher au module d'ingestion.
    chunks = enrichir(chunks, metadonnees)

    resultat = ingest.ingest_document(
        session,
        collection,
        source=chemin.name,
        fingerprint=empreinte,
        chunks=chunks,
        force=args.force,
    )
    return resultat, note


def ingerer_avec_reprises(
    session, collection, chemin: Path, args
) -> tuple[ingest.IngestResult, str]:
    """Ingère un document, en réessayant les pannes PASSAGÈRES.

    Une panne d'ingestion est souvent temporaire : un serveur d'embeddings
    distant qui redémarre, une coupure réseau, un délai dépassé. Réessayer
    coûte peu — le document est ré-ingéré de zéro, ce qui est sûr — alors que
    ne pas réessayer oblige à relancer tout le passage.

    Les erreurs PERMANENTES remontent immédiatement : le fichier est illisible
    ou la configuration est incohérente, et aucune reprise n'y changera rien.

    `args.retries` compte les REPRISES, pas les tentatives : 0 = une seule
    tentative, 1 = deux tentatives au total.
    """
    for tentative in range(args.retries + 1):
        try:
            return ingerer_un_document(session, collection, chemin, args)
        except PERMANENTES:
            raise
        except Exception:
            if tentative == args.retries:
                raise
            print(f"⏳ reprise {tentative + 2}/{args.retries + 1} …", end=" ", flush=True)
            time.sleep(PAUSE_REPRISE_S * (tentative + 1))

    # Inatteignable : la boucle retourne ou relance à chaque tour.
    raise AssertionError("boucle de reprise incohérente")


def afficher_resume(resultats) -> None:
    """Récapitulatif d'un passage d'ingestion."""
    comptes = {statut: 0 for statut in ("added", "updated", "unchanged", "empty")}
    for resultat in resultats:
        comptes[resultat.status] += 1

    ecrits = sum(r.chunks for r in resultats if r.status in ("added", "updated"))

    print()
    print("-" * 64)
    print(
        f"  {comptes['added']} ajouté(s) · {comptes['updated']} mis à jour · "
        f"{comptes['unchanged']} inchangé(s) · {comptes['empty']} vide(s)"
    )
    print(f"  {ecrits} chunk(s) écrit(s) au total")
    print(f"  Index : {config.CHROMA_DB_PATH}  ·  collection « {config.COLLECTION_NAME} »")
    print("-" * 64)


def afficher_echecs(echecs) -> None:
    """Récapitulatif des documents en échec, à la fin d'un passage.

    Rassembler les échecs en fin de course, plutôt que de s'arrêter au premier,
    est ce qui rend un long passage exploitable : sur 1 000 documents, un seul
    fichier illisible masquerait sinon l'état des 999 autres.
    """
    print()
    print("-" * 64)
    print(f"  ⚠️  {len(echecs)} document(s) en échec — NON inscrits au registre")
    for nom, erreur in echecs:
        print(f"     · {nom}")
        print(f"       {type(erreur).__name__} : {erreur}")
    print("  Ils seront repris au prochain passage : rien à rattraper à la main.")
    print("-" * 64)


def collecter_documents() -> list[Path]:
    """Documents à ingérer, tous dossiers du corpus confondus."""
    return [
        chemin
        for dossier in config.CORPUS_DIRS
        if dossier.is_dir()
        for chemin in sorted(dossier.iterdir())
        if chemin.is_file() and chemin.suffix.lower() in FORMATS
    ]


def noms_en_double(chemins) -> list[str]:
    """Noms de fichiers présents dans plusieurs dossiers du corpus.

    Un document est identifié par son NOM (`source` dans l'index), pas par son
    chemin : deux homonymes se remplaceraient mutuellement dans la base, en
    silence. Mieux vaut refuser que d'indexer un document à la place d'un autre.
    """
    comptes = Counter(chemin.name for chemin in chemins)
    return sorted(nom for nom, total in comptes.items() if total > 1)


def afficher_etat(chemins, args) -> int:
    """Montre le registre face à l'index et au corpus, sans rien modifier.

    Trois désaccords à surveiller :

    · **écart** — le registre annonce des chunks que l'index ne contient plus :
      `chroma_db/` a été effacé ou remplacé, la base ne doit PAS être crue ;
    · **orphelin** — le registre connaît un document absent du corpus : ses
      chunks sont encore servis comme sources d'un fichier introuvable, et
      aucun passage d'ingestion ne les retirera, puisqu'il ne parcourt que le
      corpus. Seul `--forget` les enlève ;
    · **modèle divergent** — le document a été inscrit sous un autre modèle
      d'embedding que celui de l'index : ses vecteurs ne sont pas prouvés, et
      le passage suivant le ré-embarquera de lui-même.

    Le modèle annoncé en tête est celui de l'INDEX, jamais celui du registre :
    c'est l'index qui décide de la comparabilité des vecteurs. La colonne dit ce
    que le registre raconte — les deux doivent coïncider, et c'est précisément
    ce que la marque « modèle divergent » vérifie. Afficher la valeur du
    registre sous un titre qui laisse croire à celle de l'index, c'est rassurer
    à tort.
    """
    collection = rag.get_collection()
    modele_index = ingest.modele_de_l_index(collection)
    sur_disque = {chemin.name for chemin in chemins}

    print(f"Index « {collection.name} » — modèle d'embedding : {modele_index}")
    print()
    print(f"{'document':<32}{'registre':>9}{'index':>8}   modèle inscrit au registre")
    print("-" * 76)

    ecart = False
    divergence = False
    absent = False
    with Session(engine) as session:
        for chemin in chemins:
            ligne = ingest.registre_pour(session, chemin.name)
            attendu = ligne.chunk_count if ligne else 0
            reel = ingest.compter_chunks(collection, chemin.name)
            modele = ligne.embedding_model if ligne else "—"
            marque = ""
            if ligne is None and reel == 0:
                # Présent dans le corpus, absent des DEUX côtés : il n'est pas
                # ingéré. Sans ce signal, deux zéros se ressemblent et l'état
                # paraît sain alors qu'un ouvrage manque ENTIÈREMENT.
                absent = True
                marque = "  ⚠️ non ingéré"
            elif attendu != reel:
                ecart = True
                marque = "  ⚠️ écart"
            if ligne is not None and ligne.embedding_model != modele_index:
                divergence = True
                marque += "  ⚠️ modèle divergent"
            print(f"{chemin.name:<32}{attendu:>9}{reel:>8}   {modele}{marque}")

        orphelins = [
            ligne
            for ligne in ingest.documents_du_registre(session)
            if ligne.source not in sur_disque
        ]

    for ligne in orphelins:
        reel = ingest.compter_chunks(collection, ligne.source)
        marque = "  ⚠️ orphelin"
        if ligne.embedding_model != modele_index:
            divergence = True
            marque += " · modèle divergent"
        print(
            f"{ligne.source:<32}{ligne.chunk_count:>9}{reel:>8}   "
            f"{ligne.embedding_model}{marque}"
        )

    print("-" * 76)
    if ecart:
        print("⚠️  Un écart signifie que `chroma_db/` ne correspond plus au registre.")
        print("   Relancez avec --force pour reconstruire l'index de ces documents.")
    if orphelins:
        print("⚠️  Orphelin = inscrit au registre mais absent du corpus : ses chunks")
        print("   sont encore servis comme sources. Retirez-les avec --forget.")
    if divergence:
        print(f"⚠️  « modèle divergent » : l'index est en « {modele_index} », mais ces")
        print("   documents ont été inscrits sous un autre modèle. On ne peut donc pas")
        print("   prouver que leurs vecteurs viennent du modèle courant.")
        print("   Le prochain passage les ré-embarquera de lui-même : rien à faire à la main.")
    if absent:
        print("⚠️  « non ingéré » : présent dans le corpus, absent de l'index ET du")
        print("   registre. Un passage l'ingérera — mais d'ici là, cet ouvrage est")
        print("   TOTALEMENT absent des réponses, sans autre signe.")
    if not ecart and not orphelins and not divergence and not absent:
        print("✅ Registre, index et corpus concordent.")
    return 0


def oublier_un_document(nom: str) -> int:
    """Retire un document de l'index ET du registre, sans toucher aux autres.

    Ollama n'est pas requis et n'est donc pas vérifié : supprimer des chunks
    n'embarque rien. Cette opération reste ainsi possible serveur d'embeddings
    éteint — l'exact inverse de l'ingestion, qui ne peut rien faire sans lui.

    ⚠️ Un document peut avoir des chunks dans l'index SANS ligne au registre :
    c'est l'état exact que laisse une ingestion interrompue, qui efface le
    registre AVANT d'écrire. Refuser dans ce cas serait le raisonnement à
    l'envers — il y a bel et bien quelque chose à retirer, et rien d'autre ne
    l'enlèvera : aucun passage d'ingestion ne parcourt un document qui échoue.
    """
    collection = rag.get_collection()

    with Session(engine) as session:
        ligne = ingest.registre_pour(session, nom)
        restes = ingest.compter_chunks(collection, nom)

        if ligne is None and not restes:
            print(f"❌ « {nom} » n'est ni au registre ni dans l'index : rien à retirer.")
            print("   `--status` liste ce que le registre connaît.")
            return 1

        # Sans ligne au registre mais avec des chunks : ingestion PARTIELLE.
        partiel = ligne is None
        supprimes = ingest.purger_document(session, collection, nom)

    if partiel:
        print(
            f"🗑️  {nom} … {supprimes} chunk(s) retiré(s) de l'index "
            "(ingestion partielle : aucune ligne au registre)"
        )
    else:
        print(f"🗑️  {nom} … {supprimes} chunk(s) retiré(s) de l'index, registre effacé")

    # Le fichier est-il encore là ? Si oui, le prochain passage le ré-ingérera,
    # et l'utilisateur croira que le retrait n'a pas fonctionné.
    restants = [chemin for chemin in collecter_documents() if chemin.name == nom]
    if restants:
        print(f"   ⚠️  Le fichier est TOUJOURS dans le corpus ({restants[0].parent}).")
        print("      Il sera ré-ingéré au prochain passage : déplacez-le ou supprimez-le.")
    return 0


def analyser_arguments() -> argparse.Namespace:
    """Décrit la ligne de commande."""
    analyseur = argparse.ArgumentParser(
        description=(
            "Met à jour la base vectorielle à partir des documents "
            "(PDF, EPUB) des dossiers du corpus."
        ),
    )
    analyseur.add_argument(
        "--force",
        action="store_true",
        help="ré-ingère tout, même les documents inchangés",
    )
    analyseur.add_argument(
        "--status",
        action="store_true",
        help="affiche le registre et l'index, sans rien écrire",
    )

    # Options de tenue d'un LONG passage. Sur 1 000 documents, un seul fichier
    # illisible ne doit pas condamner les 999 autres, et une coupure réseau ne
    # doit pas coûter la fin de la course.
    analyseur.add_argument(
        "--continue-on-error",
        action="store_true",
        help="poursuit le passage malgré les échecs, et les récapitule à la fin",
    )
    analyseur.add_argument(
        "--retries",
        type=int,
        default=1,
        metavar="N",
        help="reprises par document en cas de panne passagère (défaut 1)",
    )

    # Ajouter et retirer s'excluent : demander les deux n'a aucun sens, et
    # argparse le refuse plutôt que d'en ignorer un en silence.
    actions = analyseur.add_mutually_exclusive_group()
    actions.add_argument(
        "--only",
        metavar="FICHIER",
        help="ne traiter qu'un seul document (nom de fichier, ex. cours.pdf)",
    )
    actions.add_argument(
        "--forget",
        metavar="FICHIER",
        help="retirer un document de l'index et du registre, sans toucher aux autres",
    )
    return analyseur.parse_args()


def main() -> int:
    args = analyser_arguments()

    if args.retries < 0:
        print(f"❌ --retries {args.retries} est invalide : la valeur doit être >= 0.")
        return 2

    # `--forget` passe AVANT tout le reste, pour deux raisons :
    #   · il ne dépend pas du corpus — retirer le dernier document d'un corpus
    #     vidé doit rester possible, alors que le contrôle « aucun document »
    #     refuserait de continuer ;
    #   · il ne dépend pas d'Ollama — supprimer des chunks n'embarque rien.
    if args.forget:
        create_db_and_tables()
        return oublier_un_document(args.forget)

    chemins = collecter_documents()
    doublons = noms_en_double(chemins)
    if doublons:
        print(f"❌ Nom(s) de fichier présent(s) dans plusieurs dossiers : {', '.join(doublons)}")
        print(
            "   Un document est identifié par son nom : ces fichiers se "
            "remplaceraient mutuellement dans l'index."
        )
        print("   Renommez-en un, puis relancez.")
        return 1

    if args.only:
        chemins = [chemin for chemin in chemins if chemin.name == args.only]
        if not chemins:
            print(f"❌ Aucun document nommé « {args.only} » dans les dossiers du corpus.")
            return 1

    if not chemins and not args.status:
        dossiers = " ou ".join(str(dossier) for dossier in config.CORPUS_DIRS)
        print(f"❌ Aucun document ({', '.join(FORMATS)}) trouvé dans {dossiers}")
        return 1

    create_db_and_tables()

    if args.status:
        # `--status` fonctionne même corpus vide : c'est précisément le moment
        # où l'on a besoin de voir les orphelins restés au registre.
        return afficher_etat(chemins, args)

    # Ollama AVANT de toucher à la base : sans lui, aucun embedding n'est
    # calculable, et tout travail commencé serait perdu.
    if not check_ollama():
        print("❌ Opération annulée : la base existante est intacte.")
        return 1

    collection = rag.get_collection()
    try:
        ingest.ensure_embedding_model(collection)
        ensure_decoupage(collection, force=args.force)
    except ingest.IngestError as erreur:
        print(f"❌ {erreur}")
        return 1

    # Une ré-ingestion causée par un changement de modèle n'est pas un défaut :
    # c'est la seule réponse honnête, faute de pouvoir prouver que les vecteurs
    # en place viennent du modèle courant. Mais elle doit être DITE — sinon un
    # passage « qui n'aurait dû rien faire » re-embarque tout le corpus, et on
    # cherche la panne là où il n'y en a pas.
    with Session(engine) as session:
        divergents = ingest.documents_au_modele_divergent(session, collection)
    if divergents:
        modele = ingest.modele_de_l_index(collection)
        print(
            f"⚠️  {len(divergents)} document(s) inscrit(s) sous un autre modèle que "
            f"l'index (« {modele} ») : ils seront ré-embarqués."
        )
        print("   Sans cela, leurs vecteurs ne seraient pas comparables aux questions.")

    resultats = []
    echecs = []
    total = len(chemins)

    with Session(engine) as session:
        for rang, chemin in enumerate(chemins, start=1):
            # Le compteur « i/N » n'est pas décoratif : sur un passage de
            # plusieurs heures, ne pas savoir où l'on en est est le premier
            # motif d'interrompre à tort.
            print(f"[{rang}/{total}] 📄 {chemin.name} …", end=" ", flush=True)
            try:
                resultat, note = ingerer_avec_reprises(session, collection, chemin, args)
            except Exception as erreur:
                echecs.append((chemin.name, erreur))
                print(f"❌ {type(erreur).__name__} : {erreur}")
                print(
                    "   Document NON inscrit au registre : il sera repris au "
                    "prochain passage."
                )
                if not args.continue_on_error:
                    print(
                        f"   Passage interrompu. « --continue-on-error » traiterait "
                        f"les {total - rang} document(s) restant(s)."
                    )
                    return 1
                continue

            resultats.append(resultat)
            print(resultat.libelle)
            if note:
                print(f"   {note}")

    afficher_resume(resultats)
    if echecs:
        afficher_echecs(echecs)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
