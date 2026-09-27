"""Extraction des EPUB : le corpus Shamela.

Pourquoi ce module existe
-------------------------
Le PDF de démonstration est corrompu par un mauvais OCR. Mesuré sur 52 chunks,
40 contiennent une forme abîmée du mot central : `ا،عتاال` au lieu de
`الاعتكاف`, `يب ل` au lieu de `يبطل` (qui n'apparaît JAMAIS sous sa forme
correcte). Aucun extracteur ne peut réparer cela : l'abîme est dans la couche
texte du PDF, et il n'y a pas d'image de page à ré-océriser.

Les EPUB publiés par shamela.ws contiennent, eux, du texte NUMÉRIQUE : lettres
liées, hamzas correctes, notes de bas de page en clair. C'est ce que ce module
lit.

Ce que l'extracteur garantit
----------------------------
1. **L'ordre de lecture vient du `spine` de l'OPF**, jamais de l'ordre des
   entrées de l'archive ZIP. Mesuré sur `12445.epub` : les fichiers y sont
   rangés dans le désordre (`P117.xhtml` avant `P11.xhtml`).
2. **Le numéro de page est la page IMPRIMÉE**, lue dans l'étiquette de bas de
   page (`الصفحة: 81`), pas le nom du fichier. Sur `12445.epub`, `P80.xhtml`
   porte la page 81 et `P161.xhtml` la page 154 : s'y fier donnerait une
   citation que le lecteur ne retrouverait pas dans son exemplaire.
3. **Une page imprimée peut être découpée en plusieurs fichiers** — 17 pages
   sur 154 dans `12445.epub`. Les morceaux consécutifs sont REFUSIONNÉS : sans
   cela, `(source, page)` ne désignerait pas un endroit unique, et la
   numérotation des lignes repartirait de 1 au milieu d'une page.
4. **Le texte produit la même forme que les PDF** : `(numéro, texte)`, lignes
   séparées par « \\n ». `chunk_pages` et toute la chaîne d'ingestion
   s'appliquent donc sans la moindre modification.

Limites assumées
----------------
- Sans étiquette de page, le repli est la POSITION dans l'ordre de lecture. Ce
  n'est pas une page imprimée : c'est un pis-aller, et `etiquetee` le signale.
- Les pages absentes de l'édition ne sont pas inventées. Sur `12445.epub`, 10
  pages manquent (6, 30-32, 137-142) : `pages_absentes` les nomme.
- Seul le contenu de `<div id="book-container">` est indexé quand ce bloc
  existe, car chez Shamela l'étiquette de page est un bloc VOISIN.
"""

import html.entities
import re
import xml.etree.ElementTree as ElementTree
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

# Point d'entrée imposé par la spécification EPUB : il désigne l'OPF, qui
# contient les métadonnées et l'ordre de lecture.
CONTENEUR = "META-INF/container.xml"

# Étiquette de bas de page des éditions Shamela : « الصفحة: 81 », précédée ou
# non de « الجزء: 1 - » et de « الحديث: 369 - ». Le chiffre peut être écrit en
# chiffres arabo-indiens (٨١) selon l'édition.
MOTIF_PAGE = re.compile(r"الصفحة\s*[:：]\s*([0-9٠-٩۰-۹]+)")

# Les deux plages de chiffres arabo-indiens → chiffres ASCII, pour `int()`.
CHIFFRES_ARABES = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

# Entités nommées reconnues par XML lui-même : ce sont les seules qu'ElementTree
# résout, et donc les seules à ne pas traduire.
ENTITES_XML = frozenset({"amp", "lt", "gt", "quot", "apos"})

# Référence d'entité nommée : « &nbsp; ». Le point-virgule est obligatoire — sans
# lui, un « & » isolé dans du texte ferait l'objet d'une réécriture hasardeuse.
MOTIF_ENTITE = re.compile(rb"&([A-Za-z][A-Za-z0-9]*);")

# Balises qui, en HTML, valent un saut de ligne une fois le texte aplati.
COUPURES = frozenset({"br", "hr"})

# Balises de bloc : elles encadrent une ligne, elles ne la coupent pas. Sans
# elles, un EPUB qui découpe ses paragraphes en <p> produirait une seule ligne
# géante et la citation « lignes 3-5 » ne pointerait plus rien de précis.
BLOCS = frozenset({
    "p", "div", "li", "tr", "td", "th", "table", "blockquote", "section",
    "article", "figure", "figcaption", "pre", "h1", "h2", "h3", "h4", "h5", "h6",
})


class EpubError(Exception):
    """EPUB illisible (message destiné à l'humain)."""


@dataclass(frozen=True)
class MetadonneesEpub:
    """Ce que l'EPUB déclare sur lui-même (Dublin Core).

    Un champ absent reste vide plutôt que de faire échouer la lecture : un EPUB
    sans auteur déclaré doit quand même s'ingérer.
    """

    titre: str = ""
    auteur: str = ""
    langue: str = ""
    editeur: str = ""
    identifiant: str = ""

    def pour_index(self) -> dict[str, str]:
        """Métadonnées à attacher à chaque chunk, champs vides écartés.

        ChromaDB refuse une valeur `None` dans les métadonnées : n'envoyer que
        ce qui existe évite de transformer un champ absent en erreur d'écriture.
        Les clés sont en anglais — ce sont des données, pas du code.
        """
        correspondance = {
            "title": self.titre,
            "author": self.auteur,
            "language": self.langue,
            "publisher": self.editeur,
            "identifier": self.identifiant,
        }
        return {cle: valeur for cle, valeur in correspondance.items() if valeur}


@dataclass(frozen=True)
class PageEpub:
    """Une page IMPRIMÉE, éventuellement reconstituée depuis plusieurs fichiers.

    `numero` est la page telle qu'elle est imprimée dans l'édition. Quand
    `etiquetee` est faux, l'EPUB ne portait aucune étiquette et `numero` n'est
    que la position dans l'ordre de lecture : ce n'est pas une page.
    """

    numero: int
    texte: str
    parties: int = 1
    etiquetee: bool = True


# --- Lecture de l'archive -------------------------------------------------

def _nom_local(balise: str) -> str:
    """Nom d'une balise sans son espace de noms : « {ns}div » → « div ».

    Les EPUB sortent de chaînes d'outils variées : la même balise apparaît
    tantôt dans l'espace de noms XHTML, tantôt sans lui. Comparer sur le nom
    local rend le lecteur insensible à cette différence.
    """
    return str(balise).rpartition("}")[2].lower()


def _tous(element: ElementTree.Element, nom: str) -> list[ElementTree.Element]:
    """Descendants portant ce nom, dans l'ordre du document."""
    return [noeud for noeud in element.iter() if _nom_local(noeud.tag) == nom]


def _premier(element: ElementTree.Element, nom: str) -> ElementTree.Element | None:
    """Premier descendant portant ce nom, ou None."""
    for noeud in element.iter():
        if _nom_local(noeud.tag) == nom:
            return noeud
    return None


def _entites_resolues(contenu: bytes) -> bytes:
    """Remplace les entités HTML nommées par des références numériques.

    ElementTree ne connaît que les cinq entités prédéfinies de XML
    (`&amp; &lt; &gt; &quot; &apos;`) : un `&nbsp;` — omniprésent dans les XHTML
    produits par des convertisseurs — fait échouer l'analyse. La DTD XHTML les
    définit bien, mais ElementTree ne charge pas les DTD externes.

    La substitution produit `&#160;` plutôt que le caractère lui-même : une
    référence numérique est de l'ASCII pur, donc valable quelle que soit
    l'encodage déclaré par le document. Une entité inconnue est laissée intacte,
    pour qu'ElementTree la signale au lieu de la faire disparaître en silence.
    """

    def remplacer(trouve: re.Match) -> bytes:
        nom = trouve.group(1).decode("ascii")
        if nom in ENTITES_XML:
            return trouve.group(0)

        valeur = html.entities.html5.get(f"{nom};")
        if valeur is None:
            return trouve.group(0)
        return "".join(f"&#{ord(caractere)};" for caractere in valeur).encode("ascii")

    return MOTIF_ENTITE.sub(remplacer, contenu)


def _analyser(archive: zipfile.ZipFile, nom: str) -> ElementTree.Element:
    """Lit et analyse un document XML de l'archive."""
    try:
        contenu = archive.read(nom)
    except KeyError as erreur:
        raise EpubError(f"Fichier déclaré mais absent de l'archive : {nom}") from erreur

    try:
        return ElementTree.fromstring(_entites_resolues(contenu))
    except ElementTree.ParseError as erreur:
        raise EpubError(f"XML illisible dans « {nom} » : {erreur}") from erreur


def _chemin_du_paquet(archive: zipfile.ZipFile) -> str:
    """Chemin de l'OPF, tel que déclaré dans META-INF/container.xml."""
    racine = _analyser(archive, CONTENEUR)
    rootfile = _premier(racine, "rootfile")
    chemin = rootfile.get("full-path") if rootfile is not None else None

    if not chemin:
        raise EpubError(f"« {CONTENEUR} » ne déclare aucun fichier OPF (full-path).")
    return chemin


def _ordre_de_lecture(paquet: ElementTree.Element) -> list[str]:
    """Chemins des documents XHTML, dans l'ordre du `spine`.

    Le `spine` est la SEULE source fiable de l'ordre de lecture. L'ordre des
    entrées d'une archive ZIP, lui, ne veut rien dire : sur `12445.epub`, il
    commence par `P117.xhtml` avant `P11.xhtml`.
    """
    manifeste = _premier(paquet, "manifest")
    if manifeste is None:
        raise EpubError("OPF invalide : aucun bloc <manifest>.")

    # Manifeste : identifiant → fichier. Seuls les documents se lisent ; la
    # feuille de styles et l'image de couverture sont donc écartées ici.
    par_id = {
        item.get("id"): item.get("href")
        for item in _tous(manifeste, "item")
        if "xhtml" in (item.get("media-type") or "").lower()
    }

    spine = _premier(paquet, "spine")
    if spine is None:
        raise EpubError("OPF invalide : aucun bloc <spine>.")

    ordre = []
    for itemref in _tous(spine, "itemref"):
        href = par_id.get(itemref.get("idref"))
        if href is None:
            raise EpubError(
                f"Le spine référence « {itemref.get('idref')} », absent du manifeste."
            )
        ordre.append(href)

    if not ordre:
        raise EpubError("Le spine est vide : aucun document à lire.")
    return ordre


def _metadonnees(paquet: ElementTree.Element) -> MetadonneesEpub:
    """Métadonnées Dublin Core de l'OPF, les absentes restant vides."""

    def valeur(nom: str) -> str:
        noeud = _premier(paquet, nom)
        return " ".join(_texte(noeud).split()) if noeud is not None else ""

    return MetadonneesEpub(
        titre=valeur("title"),
        auteur=valeur("creator"),
        langue=valeur("language"),
        editeur=valeur("publisher"),
        identifiant=valeur("identifier"),
    )


# --- Texte d'un document XHTML --------------------------------------------

def _texte(element: ElementTree.Element) -> str:
    """Texte d'un élément, les sauts de ligne HTML rendus par « \\n ».

    ElementTree expose, pour chaque nœud, son texte propre (`text`), le texte
    qui suit chaque enfant (`tail`) et ses enfants : de quoi aplatir un document
    en respectant `<br/>` et les paragraphes, sans dépendance externe.
    """
    morceaux: list[str] = []

    def parcourir(noeud: ElementTree.Element) -> None:
        if noeud.text:
            morceaux.append(noeud.text)
        for enfant in noeud:
            nom = _nom_local(enfant.tag)
            if nom in COUPURES:
                morceaux.append("\n")
            else:
                if nom in BLOCS:
                    morceaux.append("\n")
                parcourir(enfant)
                if nom in BLOCS:
                    morceaux.append("\n")
            if enfant.tail:
                morceaux.append(enfant.tail)

    parcourir(element)
    return "".join(morceaux)


def _lignes(element: ElementTree.Element) -> list[str]:
    """Lignes non vides du texte d'un élément, numérotables de 1 à n.

    Les lignes vides sont écartées : `line_start`/`line_end` doivent désigner
    des lignes RÉELLES, sinon « lignes 3-5 » ne pointerait nulle part.

    Les espaces internes sont normalisés au passage. Ce n'est pas cosmétique :
    les `&nbsp;` des convertisseurs XHTML laissent des espaces INSÉCABLES
    (U+00A0) — 135 mesurés dans `12445.epub` — qui ressemblent à des espaces
    sans en être. Deux textes visuellement identiques ne se comparent alors
    plus, et une recherche de citation échoue pour une raison invisible.
    """
    lignes = []
    for brute in _texte(element).split("\n"):
        ligne = " ".join(brute.split())
        if ligne:
            lignes.append(ligne)
    return lignes


def _conteneur_de_texte(document: ElementTree.Element) -> ElementTree.Element:
    """Élément qui porte le texte de la page.

    `<div id="book-container">` quand il existe : chez Shamela, l'étiquette de
    page est un bloc VOISIN, donc le lire écarte l'étiquette sans avoir à la
    retirer du texte. Sinon le `<body>` entier, dont l'étiquette sera retirée
    à la main — voir `_page`.
    """
    for bloc in _tous(document, "div"):
        if bloc.get("id") == "book-container":
            return bloc

    corps = _premier(document, "body")
    if corps is None:
        raise EpubError("Document XHTML sans <body> : rien à lire.")
    return corps


def _etiquette(document: ElementTree.Element) -> str | None:
    """Étiquette de bas de page (« الجزء: 1 - الصفحة: 81 »), si présente.

    Elle vit dans un bloc `<div class="center">`. Quand plusieurs candidats
    existent, celui qui porte un numéro de page est retenu : c'est le seul qui
    décrive la page.
    """
    candidats = []
    for bloc in _tous(document, "div"):
        if "center" in (bloc.get("class") or "").split():
            texte = " ".join(_texte(bloc).split())
            if texte:
                candidats.append(texte)

    for candidat in candidats:
        if MOTIF_PAGE.search(candidat):
            return candidat
    return candidats[0] if candidats else None


def _numero_depuis(etiquette: str | None) -> int | None:
    """Numéro de page imprimée lu dans l'étiquette, ou None si elle n'en porte pas."""
    if not etiquette:
        return None

    trouve = MOTIF_PAGE.search(etiquette)
    if trouve is None:
        return None
    return int(trouve.group(1).translate(CHIFFRES_ARABES))


def _page(document: ElementTree.Element, rang: int) -> PageEpub | None:
    """Page imprimée d'un document, ou None si le document ne porte aucun texte."""
    etiquette = _etiquette(document)
    lignes = _lignes(_conteneur_de_texte(document))

    # Ceinture et bretelles : quand le conteneur de texte est le <body> entier,
    # l'étiquette de bas de page s'y trouve. Elle DÉCRIT la page, elle ne fait
    # pas partie de son texte.
    if etiquette:
        lignes = [ligne for ligne in lignes if ligne != etiquette]

    if not lignes:
        # Un texte vide ne doit pas devenir un chunk : ChromaDB refuse un
        # document vide, et une page blanche n'a rien à indexer.
        return None

    numero = _numero_depuis(etiquette)
    return PageEpub(
        numero=numero if numero is not None else rang,
        texte="\n".join(lignes),
        etiquetee=numero is not None,
    )


# --- Entrée publique ------------------------------------------------------

def lire_epub(chemin: Path) -> tuple[MetadonneesEpub, list[PageEpub]]:
    """Lit un EPUB et renvoie ses métadonnées et ses pages, dans l'ordre de lecture.

    Les pages consécutives portant la MÊME page imprimée sont fusionnées : la
    page devient alors un endroit unique, et la numérotation des lignes y est
    continue. C'est une garantie de la citation, pas une optimisation.
    """
    try:
        with zipfile.ZipFile(chemin) as archive:
            chemin_opf = _chemin_du_paquet(archive)
            paquet = _analyser(archive, chemin_opf)
            base = PurePosixPath(chemin_opf).parent

            pages: list[PageEpub] = []
            for rang, href in enumerate(_ordre_de_lecture(paquet), start=1):
                document = _analyser(archive, str(base / href))
                page = _page(document, rang)
                if page is None:
                    continue

                precedente = pages[-1] if pages else None
                meme_page = (
                    precedente is not None
                    and page.etiquetee
                    and precedente.numero == page.numero
                )
                if meme_page:
                    pages[-1] = PageEpub(
                        numero=precedente.numero,
                        texte=f"{precedente.texte}\n{page.texte}",
                        parties=precedente.parties + page.parties,
                    )
                    continue
                pages.append(page)

    except zipfile.BadZipFile as erreur:
        raise EpubError(
            f"« {chemin.name} » n'est pas une archive ZIP lisible : {erreur}"
        ) from erreur

    return _metadonnees(paquet), pages


# --- Diagnostic -----------------------------------------------------------

def pages_absentes(pages: Sequence[PageEpub]) -> list[int]:
    """Numéros de page manquants entre la première et la dernière.

    Certaines éditions électroniques ne contiennent pas toutes les pages du
    papier (planches, pages blanches). Le signaler vaut mieux que de laisser
    croire à une pagination continue.
    """
    etiquetees = [page.numero for page in pages if page.etiquetee]
    if not etiquetees:
        return []

    presents = set(etiquetees)
    return [n for n in range(min(etiquetees), max(etiquetees) + 1) if n not in presents]


def formater_plages(nombres: Iterable[int]) -> str:
    """Résume des nombres en plages : « 6, 30-32, 137-142 ».

    Une suite contiguë devient « début-fin » : une liste de dix nombres isolés
    se lit mal, les mêmes dix en plages se lisent d'un coup d'œil.
    """
    tries = sorted(set(nombres))
    if not tries:
        return ""

    plages: list[tuple[int, int]] = []
    debut = precedent = tries[0]
    for nombre in tries[1:]:
        if nombre == precedent + 1:
            precedent = nombre
            continue
        plages.append((debut, precedent))
        debut = precedent = nombre
    plages.append((debut, precedent))

    return ", ".join(
        str(debut) if debut == fin else f"{debut}-{fin}" for debut, fin in plages
    )
