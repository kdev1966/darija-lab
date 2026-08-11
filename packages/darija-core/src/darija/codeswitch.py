"""Segmentation de l'alternance codique arabe / français / Arabizi.

Le tunisien écrit alterne les codes en permanence, souvent à l'intérieur d'une
même phrase et parfois d'un même mot ::

    « ken 3andek le temps ajoutili chwaya de sucre »
    « المشكلة أنو le service ماهوش disponible توا »

Un pipeline qui traite ce texte comme monolingue se trompe partout : le
tokeniseur arabe bute sur le latin, le modèle français bute sur l'arabe, et le
détecteur de langue rend une moyenne qui ne décrit aucun segment.

Ce module découpe le texte en segments homogènes et étiquette chacun. Il ne
traduit rien et ne corrige rien.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Final

from .arabizi import ARABIZI_STOPWORDS, FRENCH_STOPWORDS

#: Un chiffre employé comme lettre : collé à une lettre latine, d'un côté ou de
#: l'autre. C'est le marqueur le plus sûr de l'Arabizi (``3andek``, ``bara7``),
#: et il ne se déclenche pas sur un nombre isolé (``iphone 13``).
_LETTER_DIGIT_WORD: Final = re.compile(
    r"(?<=[a-zà-ÿ])[2-9]'?|[2-9]'?(?=[a-zà-ÿ])", re.I
)

#: Runs de même nature : arabe, latin, chiffres seuls, ou le reste.
_RUN: Final = re.compile(
    r"(?P<arabic>[؀-ۿݐ-ݿ]+(?:[\sً-ٟ]+[؀-ۿݐ-ݿ]+)*)"
    r"|(?P<latin>[A-Za-zÀ-ÿ0-9']+(?:[ \t]+[A-Za-zÀ-ÿ0-9']+)*)"
    r"|(?P<other>[^\s]+)"
)

#: Diacritiques typiquement françaises. Leur seule présence tranche.
_FRENCH_ACCENT: Final = re.compile(r"[àâäçéèêëîïôöùûüÿœæ]", re.I)


@dataclass(frozen=True)
class Segment:
    """Un fragment homogène du texte."""

    text: str
    #: ``arabic`` · ``latin`` · ``other``
    script: str
    #: ``ar`` · ``fr`` · ``arabizi`` · ``unknown``
    lang: str
    start: int
    end: int

    @property
    def n_words(self) -> int:
        """Nombre de mots du segment."""
        return len(self.text.split())


#: Un mot latin, avec ses chiffres-lettres éventuels.
_WORD: Final = re.compile(r"[A-Za-zÀ-ÿ0-9']+")


def _label_word(word: str) -> str:
    """Étiquette **un seul mot** latin, sans contexte.

    Renvoie ``unknown`` quand rien ne tranche — la majorité des mots pleins,
    qui ne portent ni chiffre-lettre ni appartenance à une liste. Le contexte
    est résolu ensuite par :func:`_smooth`.
    """
    w = word.lower().strip("'")
    if not w:
        return "unknown"
    if _LETTER_DIGIT_WORD.search(w):
        return "arabizi"
    if w in ARABIZI_STOPWORDS:
        return "arabizi"
    if w in FRENCH_STOPWORDS or _FRENCH_ACCENT.search(w):
        return "fr"
    return "unknown"


def _smooth(labels: list[str]) -> list[str]:
    """Propage l'étiquette du plus proche voisin étiqueté, la gauche d'abord.

    Le code-switching se fait par syntagmes, pas mot à mot : un mot non décidé
    entouré de français est presque toujours français. Propager depuis la gauche
    reproduit ce comportement et évite d'éclater le texte en segments d'un mot.

    Les mots hybrides existent bel et bien (``ajoutili`` = radical français +
    clitique arabe) et sont rattachés à leur voisinage, faute de mieux.
    """
    out = list(labels)
    last = "unknown"
    for i, lab in enumerate(out):
        if lab != "unknown":
            last = lab
        elif last != "unknown":
            out[i] = last
    # Deuxième passe, vers la gauche, pour les mots situés avant tout indice.
    nxt = "unknown"
    for i in range(len(out) - 1, -1, -1):
        if out[i] != "unknown":
            nxt = out[i]
        elif nxt != "unknown":
            out[i] = nxt
    return out


def segment(text: str) -> list[Segment]:
    """Découpe ``text`` en segments homogènes étiquetés.

    Les segments sont rendus dans l'ordre du texte et leurs bornes
    ``start``/``end`` indexent la chaîne d'origine, donc
    ``text[s.start:s.end] == s.text``.
    """
    src = text or ""
    out: list[Segment] = []
    for m in _RUN.finditer(src):
        frag = m.group(0)
        if m.lastgroup == "arabic":
            out.append(Segment(frag, "arabic", "ar", m.start(), m.end()))
        elif m.lastgroup == "latin":
            out.extend(_segment_latin(src, frag, m.start()))
        else:
            # Ponctuation, emoji, symboles : ignorés, ils n'ont pas de langue.
            if all(unicodedata.category(c).startswith(("P", "S", "Z")) for c in frag):
                continue
            out.append(Segment(frag, "other", "unknown", m.start(), m.end()))
    return out


def _segment_latin(src: str, frag: str, offset: int) -> list[Segment]:
    """Découpe une suite latine en segments de langue homogène.

    Une suite latine contiguë mêle couramment français et Arabizi
    (« ken 3andek le temps ») : l'étiqueter d'un seul bloc ferait manquer
    précisément l'alternance qu'on cherche. On étiquette donc mot à mot, on
    lisse par le contexte, puis on fusionne les voisins de même langue.
    """
    words = list(_WORD.finditer(frag))
    if not words:
        return []
    labels = _smooth([_label_word(w.group(0)) for w in words])

    segments: list[Segment] = []
    start_i = 0
    for i in range(1, len(words) + 1):
        if i < len(words) and labels[i] == labels[start_i]:
            continue
        a = offset + words[start_i].start()
        b = offset + words[i - 1].end()
        segments.append(Segment(src[a:b], "latin", labels[start_i], a, b))
        start_i = i
    return segments


def profile(text: str) -> dict[str, float]:
    """Part de chaque langue, pondérée par le nombre de mots.

    Returns:
      ``{"ar": float, "fr": float, "arabizi": float, "unknown": float}``,
      de somme 1.0. Tout à 0.0 si le texte ne contient aucun mot.

    """
    counts: Counter[str] = Counter()
    for s in segment(text):
        counts[s.lang] += s.n_words
    total = sum(counts.values())
    keys = ("ar", "fr", "arabizi", "unknown")
    if not total:
        return dict.fromkeys(keys, 0.0)
    return {k: round(counts.get(k, 0) / total, 4) for k in keys}


def is_code_switched(text: str, threshold: float = 0.15) -> bool:
    """Vrai si au moins deux langues dépassent ``threshold`` chacune."""
    p = profile(text)
    return sum(1 for v in p.values() if v >= threshold) >= 2


def extract(text: str, lang: str) -> list[str]:
    """Tous les fragments d'une langue donnée, dans l'ordre."""
    return [s.text for s in segment(text) if s.lang == lang]


__all__ = [
    "Segment",
    "extract",
    "is_code_switched",
    "profile",
    "segment",
]
