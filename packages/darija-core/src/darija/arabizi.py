"""Arabizi tunisien ↔ écriture arabe.

L'Arabizi (``3arabi``, ``arabish``) est la façon dont le tunisien s'écrit
massivement en ligne : alphabet latin, chiffres pour les consonnes que le latin
n'a pas. C'est la forme écrite la plus répandue du dialecte, et pratiquement
aucun outil arabe standard ne la traite.

Les chiffres portent l'essentiel du signal, parce qu'ils sont *iconiques* — la
forme du chiffre imite la lettre::

    3 → ع    7 → ح    9 → ق    5 → خ    2 → ء    6 → ط
    3' → غ   7' → خ   6' → ظ   8 → غ    4 → ذ

La translittération est **intrinsèquement ambiguë** et ce module ne prétend pas
la résoudre : ``salem`` peut être ``سالم`` ou ``سلام``, les voyelles brèves ne
s'écrivent pas en arabe et l'Arabizi ne distingue pas brèves et longues.
:func:`to_arabic` produit une translittération déterministe, utile pour
l'indexation, la recherche approchée et la normalisation d'entrée — **pas** une
orthographe à publier telle quelle.

Le sens inverse (:func:`to_arabizi`) est, lui, presque sans perte, puisqu'il va
d'un alphabet plus riche vers un plus pauvre.
"""

from __future__ import annotations

import re
from typing import Final

#: Arabizi → arabe. Les clés les plus longues sont essayées d'abord, donc
#: ``3'`` l'emporte sur ``3`` et ``kh`` sur ``k``.
ARABIZI_TO_ARABIC: Final[dict[str, str]] = {
    # chiffres avec apostrophe — à tester avant les chiffres nus
    "3'": "غ", "7'": "خ", "6'": "ظ", "9'": "ظ", "2'": "ء",
    # chiffres
    "2": "ء", "3": "ع", "4": "ذ", "5": "خ", "6": "ط", "7": "ح", "8": "غ", "9": "ق",
    # trigrammes
    "sch": "ش",
    # digrammes
    "ch": "ش", "sh": "ش", "kh": "خ", "gh": "غ", "th": "ث", "dh": "ذ",
    "ph": "ف", "ou": "و", "aa": "ا", "ee": "ي", "ii": "ي", "oo": "و", "ss": "س",
    # lettres simples
    "a": "ا", "b": "ب", "c": "ك", "d": "د", "f": "ف", "g": "ڨ", "h": "ه",
    "i": "ي", "j": "ج", "k": "ك", "l": "ل", "m": "م", "n": "ن", "o": "و",
    "p": "ب", "q": "ق", "r": "ر", "s": "س", "t": "ت", "u": "و", "v": "ڥ",
    "w": "و", "x": "كس", "y": "ي", "z": "ز",
}

#: Arabe → Arabizi, pour le sens inverse.
ARABIC_TO_ARABIZI: Final[dict[str, str]] = {
    "ا": "a", "أ": "a", "إ": "i", "آ": "a", "ء": "2", "ؤ": "2", "ئ": "2",
    "ب": "b", "ت": "t", "ث": "th", "ج": "j", "ح": "7", "خ": "5", "د": "d",
    "ذ": "dh", "ر": "r", "ز": "z", "س": "s", "ش": "ch", "ص": "s", "ض": "dh",
    "ط": "t", "ظ": "dh", "ع": "3", "غ": "gh", "ف": "f", "ق": "9", "ك": "k",
    "ل": "l", "م": "m", "ن": "n", "ه": "h", "ة": "a", "و": "w", "ي": "i",
    "ى": "a", "ڨ": "g", "ڥ": "v", "پ": "p", "چ": "ch",
}

#: Clés triées par longueur décroissante : garantit le plus-long-d'abord.
_KEYS: Final[list[str]] = sorted(ARABIZI_TO_ARABIC, key=len, reverse=True)

#: Un chiffre-lettre entouré de lettres latines, ou en fin de mot après une
#: lettre. C'est le marqueur le plus fiable de l'Arabizi : ``3ala``, ``m3a``,
#: ``9alb``, ``bara7``. Un simple « du latin avec des chiffres » ne suffit pas —
#: ``iphone 13`` n'est pas de l'Arabizi.
_LETTER_DIGIT: Final = re.compile(r"(?<=[a-z])[234567892]'?|[234567892]'?(?=[a-z])", re.I)

#: Mots-outils tunisiens très fréquents en Arabizi. Sert de second signal quand
#: le texte ne contient aucun chiffre-lettre.
ARABIZI_STOPWORDS: Final[frozenset[str]] = frozenset({
    "chnowa", "chneya", "chnia", "kifach", "3lach", "alech", "9adech", "9adeh",
    "barcha", "barsha", "yezzi", "behi", "behy", "naam", "ey", "ekaka", "hakka",
    "taw", "tawa", "famma", "fama", "mahouch", "mouch", "mech", "manich",
    "enti", "enty", "houwa", "hiya", "a7na", "entouma", "houma", "eli", "elli",
    "ken", "wa9tech", "win", "chwaya", "chwiya", "labes", "labas",
    "sahha", "sa7a", "3aslema", "aslema", "brabi", "yaani", "ya3ni",
})

#: Mots-outils français fréquents dans le tunisien écrit — le code-switching est
#: la norme, pas l'exception. Utilisé par :mod:`darija.codeswitch`.
FRENCH_STOPWORDS: Final[frozenset[str]] = frozenset({
    "le", "la", "les", "un", "une", "des", "du", "de", "et", "ou", "mais",
    "je", "tu", "il", "elle", "nous", "vous", "ils", "elles", "on",
    "est", "sont", "was", "pour", "avec", "sans", "dans", "sur", "chez",
    "que", "qui", "quoi", "pas", "plus", "tres", "très", "bien", "bon",
    "merci", "bonjour", "salut", "voila", "voilà", "meme", "même", "aussi",
    "faire", "fait", "avoir", "etre", "être", "tout", "tous", "toute",
})


def to_arabic(text: str, *, g_as_qaf: bool = False) -> str:
    """Translittère de l'Arabizi vers l'écriture arabe.

    Args:
      text: texte en Arabizi.
      g_as_qaf: écrire /g/ ``ق`` au lieu de ``ڨ``. ``ڨ`` est la convention
        tunisienne (``ڨلب``), mais certains corpus n'emploient que ``ق``.

    Returns:
      La translittération. Les caractères déjà arabes, la ponctuation et les
      espaces traversent inchangés.

    Notes:
      ``e`` est traité comme une voyelle brève : rendu ``ا`` en début de mot
      (``enti`` → ``انتي``), supprimé ailleurs (``behi`` → ``بهي``), ce qui suit
      l'usage arabe de ne pas noter les brèves.

    """
    out: list[str] = []
    i = 0
    low = text.lower()
    n = len(text)
    while i < n:
        ch = low[i]

        # 'e' : voyelle brève, dépend de la position dans le mot
        if ch == "e":
            at_word_start = i == 0 or not low[i - 1].isalnum()
            out.append("ا" if at_word_start else "")
            i += 1
            continue

        for key in _KEYS:
            if low.startswith(key, i):
                mapped = ARABIZI_TO_ARABIC[key]
                if mapped == "ڨ" and g_as_qaf:
                    mapped = "ق"
                out.append(mapped)
                i += len(key)
                break
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def to_arabizi(text: str) -> str:
    """Translittère de l'arabe vers l'Arabizi. Quasi sans perte.

    Les diacritiques sont ignorées ; les caractères non arabes traversent
    inchangés.
    """
    from .normalize import strip_diacritics, strip_tatweel

    src = strip_diacritics(strip_tatweel(text or ""))
    return "".join(ARABIC_TO_ARABIZI.get(c, c) for c in src)


def arabizi_score(text: str) -> float:
    """Vraisemblance que ``text`` soit de l'Arabizi, dans ``[0, 1]``.

    Deux signaux, combinés :

    * la densité de **chiffres-lettres** (``3``, ``7``, ``9``… collés à des
      lettres) — le signal fort, quasi sans faux positif ;
    * la part de mots-outils tunisiens connus — le signal de repli, pour les
      phrases qui n'emploient aucun chiffre.

    Renvoie 0.0 pour du texte vide, arabe, ou purement latin sans marqueur.
    """
    if not text or not text.strip():
        return 0.0
    words = re.findall(r"[a-z0-9']+", text.lower())
    if not words:
        return 0.0

    digit_words = sum(1 for w in words if _LETTER_DIGIT.search(w))
    stop_words = sum(1 for w in words if w in ARABIZI_STOPWORDS)

    # Un seul chiffre-lettre est déjà très informatif : on sature vite.
    digit_signal = min(1.0, digit_words / max(1, len(words)) * 3.0)
    stop_signal = min(1.0, stop_words / max(1, len(words)) * 2.5)
    return max(0.0, min(1.0, max(digit_signal, stop_signal)))


def is_arabizi(text: str, threshold: float = 0.15) -> bool:
    """Vrai si ``text`` ressemble à de l'Arabizi tunisien."""
    return arabizi_score(text) >= threshold


__all__ = [
    "ARABIC_TO_ARABIZI",
    "ARABIZI_STOPWORDS",
    "ARABIZI_TO_ARABIC",
    "FRENCH_STOPWORDS",
    "arabizi_score",
    "is_arabizi",
    "to_arabic",
    "to_arabizi",
]
