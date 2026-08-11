"""Lecture **en flux** d'un dump Wikipédia, sans jamais écrire le XML sur disque.

Un dump ``.xml.bz2`` gonfle d'un facteur 4 à 5 à la décompression : lire
l'arabe standard en entier coûterait ~9 Go de pic disque pour n'en garder que
~2,3 Go de texte. On décompresse donc à la volée, on parse en incrémental, et
on n'écrit que le texte retenu. Le pic disque retombe à la taille du résultat.

Deuxième conséquence, plus utile encore : on peut **couper le flux** dès qu'on a
assez de données. La connexion est interrompue, et on n'aura téléchargé que ce
qui a été lu — pas les 1,9 Go du dump complet.

Seule la bibliothèque standard est utilisée.
"""

from __future__ import annotations

import bz2
import re
from collections.abc import Iterator
from dataclasses import dataclass
from urllib.request import Request, urlopen
from xml.etree.ElementTree import ParseError, XMLPullParser

from ..normalize import Level, normalize

#: Modèle d'URL du dump des articles. On vise délibérément le fichier agrégé et
#: non les tranches numérotées (``…articles1.xml-p1p340838.bz2``) : leurs bornes
#: ``p<début>p<fin>`` changent à chaque dump, donc l'URL casse. Couper le flux
#: donne le même résultat sans dépendre d'un nom instable.
DUMP_URL = "https://dumps.wikimedia.org/{lang}wiki/latest/{lang}wiki-latest-pages-articles.xml.bz2"

_NS = re.compile(r"^\{[^}]+\}")
_ARABIC = re.compile(r"[؀-ۿ]")

_RX_COMMENT = re.compile(r"<!--.*?-->", re.S)
_RX_REF_SELF = re.compile(r"<ref[^>]*/>")
_RX_REF = re.compile(r"<ref[^>]*>.*?</ref>", re.S)
_RX_TAG = re.compile(r"<[^>]+>")
_RX_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
_RX_TABLE = re.compile(r"\{\|.*?\|\}", re.S)
_RX_NSLINK = re.compile(r"\[\[(?:[^\[\]|]*:)[^\[\]]*\]\]")
_RX_PIPED = re.compile(r"\[\[[^\[\]|]*\|([^\[\]]*)\]\]")
_RX_LINK = re.compile(r"\[\[([^\[\]]*)\]\]")
_RX_EXT_LABEL = re.compile(r"\[https?://\S+\s+([^\]]*)\]")
_RX_EXT = re.compile(r"\[https?://\S+\]")
_RX_HEADING = re.compile(r"^\s*={2,}.*?={2,}\s*$", re.M)
_RX_BULLET = re.compile(r"^[*#:;]+", re.M)


def strip_wikitext(text: str) -> str:
    """Retire le balisage MediaWiki pour ne garder que la prose.

    Volontairement approximatif : l'objectif est d'alimenter un modèle de
    n-grammes de caractères, pas de reconstruire l'article. Ce qui compte est de
    ne pas laisser passer de balisage, qui polluerait les n-grammes.
    """
    text = _RX_COMMENT.sub(" ", text)
    text = _RX_REF.sub(" ", text)
    text = _RX_REF_SELF.sub(" ", text)
    # Les modèles s'imbriquent ; on répète jusqu'au point fixe, bornes comprises.
    for _ in range(8):
        new = _RX_TEMPLATE.sub(" ", text)
        if new == text:
            break
        text = new
    text = _RX_TABLE.sub(" ", text)
    text = _RX_NSLINK.sub(" ", text)
    text = _RX_PIPED.sub(r"\1", text)
    text = _RX_LINK.sub(r"\1", text)
    text = _RX_EXT_LABEL.sub(r"\1", text)
    text = _RX_EXT.sub(" ", text)
    text = _RX_TAG.sub(" ", text)
    text = _RX_HEADING.sub(" ", text)
    text = text.replace("'''", "").replace("''", "")
    return _RX_BULLET.sub(" ", text)


def arabic_ratio(text: str) -> float:
    """Part de caractères arabes parmi les caractères non blancs."""
    dense = [c for c in text if not c.isspace()]
    if not dense:
        return 0.0
    return len(_ARABIC.findall(text)) / len(dense)


@dataclass
class DumpStats:
    """Ce qui a réellement été lu, pour rendre le coût vérifiable."""

    compressed_bytes: int = 0
    pages_seen: int = 0
    pages_kept: int = 0
    lines_kept: int = 0
    truncated: bool = False

    def as_dict(self) -> dict[str, object]:
        """Vue sérialisable."""
        return {
            "compressed_mb": round(self.compressed_bytes / (1024 * 1024), 2),
            "pages_seen": self.pages_seen,
            "pages_kept": self.pages_kept,
            "lines_kept": self.lines_kept,
            "truncated": self.truncated,
        }


def _stream_decompressed(
    url: str, max_bytes: int | None, stats: DumpStats, chunk: int = 1 << 20
) -> Iterator[bytes]:
    """Décompresse le dump à la volée, en coupant à ``max_bytes`` compressés."""
    req = Request(url, headers={"User-Agent": "darija-core/0.1 (dataset builder)"})
    decomp = bz2.BZ2Decompressor()
    with urlopen(req) as resp:  # noqa: S310 - URL fixe, schéma https
        while True:
            raw = resp.read(chunk)
            if not raw:
                break
            stats.compressed_bytes += len(raw)
            try:
                out = decomp.decompress(raw)
            except (OSError, EOFError):
                break
            if out:
                yield out
            # Certains dumps concatènent plusieurs flux bz2 bout à bout.
            if decomp.eof and decomp.unused_data:
                decomp = bz2.BZ2Decompressor()
                tail = decomp.decompress(decomp.unused_data or b"")
                if tail:
                    yield tail
            if max_bytes is not None and stats.compressed_bytes >= max_bytes:
                stats.truncated = True
                break


def iter_pages(
    lang: str, *, max_bytes: int | None = None, max_pages: int | None = None
) -> Iterator[tuple[str, str]]:
    """Itère les articles ``(titre, wikitexte)`` d'un dump.

    Ne rend que l'espace de noms principal (``ns == 0``) et saute les
    redirections, qui n'ont pas de contenu.
    """
    stats = DumpStats()
    parser = XMLPullParser(events=("start", "end"))
    root = None
    url = DUMP_URL.format(lang=lang)

    try:
        for data in _stream_decompressed(url, max_bytes, stats):
            parser.feed(data)
            for event, elem in parser.read_events():
                tag = _NS.sub("", elem.tag)
                if event == "start":
                    if root is None:
                        root = elem
                    continue
                if tag != "page":
                    continue

                stats.pages_seen += 1
                ns = elem.findtext("{*}ns")
                redirect = elem.find("{*}redirect")
                title = elem.findtext("{*}title") or ""
                body = elem.findtext("{*}revision/{*}text") or ""
                if root is not None:
                    root.clear()

                if ns != "0" or redirect is not None or not body:
                    continue
                stats.pages_kept += 1
                yield title, body
                if max_pages is not None and stats.pages_kept >= max_pages:
                    return
    except ParseError:
        # Attendu : on a coupé le flux au milieu du XML. Tout ce qui a déjà été
        # émis reste valide.
        return


def iter_lines(
    lang: str,
    *,
    max_bytes: int | None = None,
    max_pages: int | None = None,
    max_lines: int | None = None,
    min_words: int = 5,
    min_arabic: float = 0.6,
) -> Iterator[str]:
    """Itère des lignes de prose arabe propres, prêtes à l'entraînement.

    Args:
      lang: code du wiki (``ar``, ``arz``, ``ary``).
      max_bytes: plafond d'octets compressés lus.
      max_pages: plafond d'articles retenus.
      max_lines: plafond de lignes rendues.
      min_words: longueur minimale d'une ligne, en mots.
      min_arabic: part minimale de caractères arabes, pour écarter les lignes
        résiduelles de balisage, de translittération ou de listes.

    """
    n = 0
    for _title, body in iter_pages(lang, max_bytes=max_bytes, max_pages=max_pages):
        for raw in strip_wikitext(body).splitlines():
            line = re.sub(r"\s+", " ", raw).strip()
            if len(line.split()) < min_words or arabic_ratio(line) < min_arabic:
                continue
            if len(normalize(line, Level.STANDARD).split()) < min_words:
                continue
            yield line
            n += 1
            if max_lines is not None and n >= max_lines:
                return


__all__ = [
    "DUMP_URL",
    "DumpStats",
    "arabic_ratio",
    "iter_lines",
    "iter_pages",
    "strip_wikitext",
]
