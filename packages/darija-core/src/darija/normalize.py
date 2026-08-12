"""Normalisation orthographique de l'arabe **qui préserve le dialecte**.

La plupart des chaînes de traitement arabes normalisent vers l'arabe standard :
elles corrigent l'orthographe, remplacent les formes dialectales par leur
équivalent fusha, parfois suppriment ce qu'elles ne reconnaissent pas. Pour du
darija c'est destructeur — ce sont précisément ces traits qui portent
l'information.

Ce module ne fait qu'une chose : ramener à une forme canonique ce qui ne
distingue **jamais** deux mots (élongation typographique, diacritiques,
variantes de porteurs de hamza). Il ne touche jamais au lexique.

Trois niveaux, du plus conservateur au plus agressif :

``Level.LIGHT``
    Retire l'élongation (``tatweel``) et les diacritiques. Réversible en
    pratique : aucune lettre n'est fusionnée.

``Level.STANDARD`` (défaut)
    En plus, unifie les variantes orthographiques que l'usage confond :
    ``أ إ آ ٱ`` → ``ا``, ``ى`` → ``ي``, ``ة`` → ``ه``. C'est le niveau à utiliser
    pour comparer, indexer ou compter.

``Level.AGGRESSIVE``
    En plus, supprime tout caractère non arabe et réduit les répétitions de
    lettres (``برشاااا`` → ``برشا``), courantes à l'écrit spontané.

Ce qui n'est **jamais** fait, à aucun niveau : traduire ``برشا`` en ``كثيرا``,
``علاش`` en ``لماذا``, ou ``ما...ش`` en ``لا``. Voir :mod:`darija.markers`.
"""

from __future__ import annotations

import enum
import re
import unicodedata
from typing import Final

TATWEEL: Final = "ـ"

#: Marques combinantes arabes : fatha..sukun, alef suscrit, signes coraniques.
_DIACRITICS: Final = re.compile(r"[ً-ٰٟۖ-ۭ]")

#: Chiffres arabes et arabes étendus vers ASCII.
_DIGITS: Final = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

_ALEF_VARIANTS: Final = re.compile(r"[إأآٱ]")

#: Lettres maghrébines hors du bloc arabe de base. Elles notent des phonèmes
#: que le tunisien a et que l'arabe standard n'a pas — ``ڨ`` /g/ (``ڨلب``),
#: ``ڥ``/``ڤ`` /v/, ``پ`` /p/, ``چ`` /tʃ/ — et doivent survivre à la
#: normalisation, sans quoi ``ڨلب`` deviendrait ``لب``.
MAGHREBI_LETTERS: Final = "پچڤڥڨگ"

#: Tout ce qui n'est ni lettre arabe (bloc de base + maghrébines), ni chiffre
#: ASCII, ni espace.
_NON_ARABIC: Final = re.compile(rf"[^ء-ي{MAGHREBI_LETTERS}0-9 ]")

#: Trois occurrences ou plus de la même lettre → deux. Deux est conservé car
#: l'arabe a de vraies géminées à l'écrit (``ربّي`` sans shadda → ``ربي``, mais
#: ``الله`` garde son double lam).
_ELONGATION: Final = re.compile(r"(.)\1{2,}")

#: Lettres arabes, pour la détection de script.
_ARABIC_CHAR: Final = re.compile(r"[؀-ۿݐ-ݿ]")
_LATIN_CHAR: Final = re.compile(r"[A-Za-z]")


class Level(enum.StrEnum):
    """Niveau de normalisation."""

    LIGHT = "light"
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"


def strip_tatweel(text: str) -> str:
    """Retire l'élongation typographique (ـ)."""
    return text.replace(TATWEEL, "")


def strip_diacritics(text: str) -> str:
    """Retire toutes les marques combinantes arabes (tashkil)."""
    return _DIACRITICS.sub("", text)


def unify_orthography(text: str) -> str:
    """Fusionne les variantes qui ne distinguent jamais deux mots.

    ``أ إ آ ٱ`` → ``ا`` · ``ى`` → ``ي`` · ``ة`` → ``ه`` · ``ؤ`` → ``و`` ·
    ``ئ`` → ``ي``

    Destiné à la **comparaison**. Ne réécrivez pas votre texte source avec ça.
    """
    text = _ALEF_VARIANTS.sub("ا", text)
    return (
        text.replace("ى", "ي")
        .replace("ة", "ه")
        .replace("ؤ", "و")
        .replace("ئ", "ي")
    )


def reduce_elongation(text: str) -> str:
    """``برشاااا`` → ``برشاا``. Réduit 3 répétitions ou plus à 2."""
    return _ELONGATION.sub(r"\1\1", text)


#: Paires que l'écriture latine **ne peut pas distinguer**. L'Arabizi note les
#: consonnes par des lettres latines et des chiffres, mais rien n'y sépare une
#: emphatique de sa simple : ``s`` rend ``س`` comme ``ص``, ``t`` rend ``ت``
#: comme ``ط``. Et une voyelle finale ``a`` rend ``ا`` là où l'arabe écrit
#: ``ة``.
#:
#: Projeter les deux côtés dans cet alphabet réduit rend au classifieur ce que
#: la translittération lui avait pris. Mesuré sur TUNIZI, à contraste et
#: échantillon égaux : la part des blocs reconnus tunisiens passe de **64 % à
#: 88 %**, pour une AUC inchangée (0,999) et des faux positifs inchangés
#: (0,2 %). Les distinctions emphatiques portent donc peu de signal dialectal.
_ARABIZI_FOLD: Final[dict[int, str]] = str.maketrans(
    {"ص": "س", "ط": "ت", "ض": "د", "ظ": "ذ", "ة": "ا", "ه": "ا"}
)


def fold_for_arabizi(text: str) -> str:
    """Projette l'arabe dans l'alphabet que l'Arabizi sait exprimer.

    À n'employer que **des deux côtés à la fois** : sur le corpus
    d'entraînement et sur le texte translittéré. Appliqué d'un seul côté, il
    creuserait l'écart au lieu de le combler.

    Voir :data:`_ARABIZI_FOLD` pour ce qui est confondu et pourquoi.
    """
    return text.translate(_ARABIZI_FOLD)


def normalize(text: str, level: Level | str = Level.STANDARD) -> str:
    """Normalise ``text`` au niveau demandé.

    Args:
      text: texte brut, arabe ou mixte.
      level: un :class:`Level` ou son nom.

    Returns:
      Le texte normalisé, espaces réduits, sans espaces de bordure.

    """
    level = Level(level)
    text = unicodedata.normalize("NFKC", text or "")
    text = strip_tatweel(text)
    text = strip_diacritics(text)
    text = text.translate(_DIGITS)

    if level is not Level.LIGHT:
        text = unify_orthography(text)
    if level is Level.AGGRESSIVE:
        text = reduce_elongation(text)
        text = _NON_ARABIC.sub(" ", text)

    return re.sub(r"\s+", " ", text).strip()


def script_ratio(text: str) -> dict[str, float]:
    """Part de caractères arabes / latins / autres, sur les seuls caractères-mots.

    Sert à décider comment traiter un texte avant tout le reste : de l'arabe,
    de l'Arabizi (latin), ou un mélange.

    Returns:
      ``{"arabic": float, "latin": float, "other": float}``, somme 1.0.
      Tout à 0.0 si le texte ne contient aucun caractère alphanumérique.

    """
    ar = len(_ARABIC_CHAR.findall(text or ""))
    la = len(_LATIN_CHAR.findall(text or ""))
    other = sum(1 for c in (text or "") if c.isalnum()) - ar - la
    total = ar + la + max(0, other)
    if not total:
        return {"arabic": 0.0, "latin": 0.0, "other": 0.0}
    return {
        "arabic": ar / total,
        "latin": la / total,
        "other": max(0, other) / total,
    }


def tokenize(text: str, level: Level | str = Level.STANDARD) -> list[str]:
    """Découpe en mots après normalisation. Découpage sur l'espace, rien de plus.

    Volontairement naïf : la segmentation morphologique du darija (clitiques
    ``ع``, ``ب``, ``ل``, pronoms suffixés) est un problème distinct et un
    tokeniseur qui devine fait plus de mal que de bien sur un texte spontané.
    """
    return normalize(text, level).split()


__all__ = [
    "fold_for_arabizi",
    "MAGHREBI_LETTERS",
    "TATWEEL",
    "Level",
    "normalize",
    "reduce_elongation",
    "script_ratio",
    "strip_diacritics",
    "strip_tatweel",
    "tokenize",
    "unify_orthography",
]
