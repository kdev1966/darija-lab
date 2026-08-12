"""Marqueurs morphologiques et lexicaux du tunisien.

Le tunisien se signale d'abord **morphologiquement**, pas lexicalement : le
préfixe ``n-`` de première personne (``نمشي`` « je vais », là où la fusha dit
``أمشي``), le circumfixe de négation ``ما...ش``, le relativiseur ``اللي``, la
particule de futur ``باش``. Ces motifs sont sous-lexicaux et survivent à une
orthographe instable, ce qui les rend bien plus robustes qu'une liste de mots.

Tous les marqueurs ne se valent pas, et c'est mesuré : voir
:data:`DISCRIMINANT`. Trois d'entre eux — le préfixe ``ن-``, ``اللي``,
``علاش`` — sont aussi fréquents ailleurs qu'en tunisien, voire plus. Les
compter comme les autres rend la règle « au moins un marqueur » inopérante.

Un avertissement de calibrage, mesuré sur corpus réel : un score fondé sur le
**taux de marqueurs** atteint une AUC d'environ 0.77 pour séparer du tunisien
d'autre chose — honorable, mais nettement en deçà d'un classifieur contrastif
sur 4-grammes de caractères (~0.96). Utilisez ce module pour **expliquer** et
inspecter ; utilisez :mod:`darija.dialect` pour **décider**.

Tous les motifs s'appliquent au texte normalisé en
:data:`darija.normalize.Level.STANDARD` — donc après ``ة``→``ه`` et
``ى``→``ي``. ``شنوة`` est ainsi cherché sous la forme ``شنوه``.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Final

from .normalize import Level, normalize

#: Chaque entrée : nom → (motif, catégorie, glose).
#: Les motifs sont ancrés sur les frontières de mots là où c'est pertinent.
MARKERS: Final[dict[str, tuple[str, str, str]]] = {
    # --- morphologie verbale
    "n_prefix_1sg": (
        r"\bن[ابتثجحخدذرزسشصضطظعغفقكلمنهوي]{2,}\b",
        "morphologie",
        "préfixe n- de 1re personne du singulier (نمشي « je vais »)",
    ),
    "negation_ma_sh": (
        # L'espace après ``ما`` est optionnelle : les Tunisiens écrivent le
        # circumfixe détaché aussi souvent qu'attaché — « ما تبعدش » autant que
        # « ماتبعدش ». Le motif collé ne voyait que 53 blocs sur 400 de récit
        # tunisien là où celui-ci en voit 180, sans un seul faux positif sur
        # 400 blocs de fusha. Manquer les deux tiers d'un trait morphologique
        # contredisait la promesse de robustesse orthographique du module.
        r"\bما\s?\S{1,12}ش\b",
        "morphologie",
        "circumfixe de négation ما...ش (ماناكلش « je ne mange pas »)",
    ),
    "future_bash": (r"\bباش\b", "morphologie", "particule de futur باش"),
    "progressive_qaed": (r"\bقاعد\S*\b", "morphologie", "progressif قاعد"),
    # --- mots-outils
    "relativizer_elli": (r"\bاللي\b", "mot-outil", "relativiseur اللي « qui/que »"),
    "interrog_chnowa": (r"\bشنو[هاي]?\b", "mot-outil", "شنوة « quoi »"),
    "interrog_3lach": (r"\bعلاش\b", "mot-outil", "علاش « pourquoi »"),
    "interrog_9adech": (r"\bقداش\b", "mot-outil", "قداش « combien »"),
    "interrog_kifach": (r"\bكيفاش\b", "mot-outil", "كيفاش « comment »"),
    "interrog_win": (r"\bوين\b", "mot-outil", "وين « où »"),
    "interrog_waqtech": (r"\bوقتاش\b", "mot-outil", "وقتاش « quand »"),
    "existential_famma": (r"\bفم[اه]\b", "mot-outil", "فما « il y a »"),
    "now_tawa": (r"\bتو[ا]?\b", "mot-outil", "توا « maintenant »"),
    # --- lexique caractéristique
    # ``برشة`` se normalise en ``برشه`` : sans la classe finale, la graphie la
    # plus courante du marqueur le plus exclusivement tunisien échappait au
    # motif. Releve sur deux textes de modeles independants.
    "quant_barsha": (r"\bبرش[اه]\b", "lexique", "برشا « beaucoup »"),
    "num_zouz": (r"\bزوز\b", "lexique", "زوز « deux »"),
    "adj_behi": (r"\bباهي\b", "lexique", "باهي « bien »"),
    "enough_yezzi": (r"\bيزي\b", "lexique", "يزي « ça suffit »"),
    "little_chwaya": (r"\bشوي[اه]\b", "lexique", "شوية « un peu »"),
    "ok_yakhi": (r"\bياخي\b", "lexique", "ياخي « n'est-ce pas »"),
}

#: Marqueurs les plus discriminants du tunisien **contemporain**. Attention :
#: ``برشا`` et ``شنوة`` sont quasi absents du registre littéraire ancien
#: (2 occurrences sur ~400 000 mots de poésie), donc leur absence ne prouve pas
#: qu'un texte n'est pas tunisien — seulement qu'il n'est pas moderne.
MODERN_MARKERS: Final[frozenset[str]] = frozenset({
    "quant_barsha", "interrog_chnowa", "adj_behi", "ok_yakhi", "enough_yezzi",
})

#: Marqueurs dont la présence **informe** sur le tunisien. Les autres sont
#: gardés pour expliquer, jamais pour décider.
#:
#: Mesuré sur les corpus du dépôt — part des blocs de 60 mots où le marqueur
#: apparaît au moins une fois :
#:
#: ====================  ======  ======  ======  ======
#: marqueur              TN      MA      DZ      fusha
#: ====================  ======  ======  ======  ======
#: ``n_prefix_1sg``      67,7 %  72,0 %  42,9 %  66,6 %
#: ``relativizer_elli``   5,1 %  26,0 %   3,3 %   0,0 %
#: ``interrog_3lach``     4,4 %   5,8 %   0,6 %   0,0 %
#: ====================  ======  ======  ======  ======
#:
#: Ces trois-là ne discriminent pas : le préfixe ``ن-`` note le *je* en
#: tunisien et le *nous* en arabe classique, d'où 66,6 % côté fusha ; ``اللي``
#: est **cinq fois plus fréquent en marocain** qu'en tunisien ; ``علاش`` aussi
#: est plus marocain.
#:
#: Effet mesuré sur la règle « au moins un marqueur » :
#:
#: - avec les dix-neuf : 86,6 % du tunisien, **86,0 % du marocain**, 67,1 % de
#:   la fusha. Écart discriminant : **+0,6 %**. Autant tirer à pile ou face.
#: - sans ces trois : 66,8 % du tunisien, 37,0 % du marocain, **2,0 %** de la
#:   fusha. Écart : **+29,8 %**.
#:
#: Le gain porte surtout sur la fusha, et c'est ce qu'on demande à ce signal :
#: le classifieur écarte déjà très bien le marocain, il trébuche sur la fusha
#: conversationnelle (biais nº 7).
DISCRIMINANT: Final[frozenset[str]] = frozenset(
    set(MARKERS) - {"n_prefix_1sg", "relativizer_elli", "interrog_3lach"}
)

_COMPILED: Final[dict[str, re.Pattern[str]]] = {
    name: re.compile(pat) for name, (pat, _, _) in MARKERS.items()
}


@dataclass(frozen=True)
class Match:
    """Une occurrence de marqueur."""

    marker: str
    text: str
    start: int
    category: str


def find(text: str) -> list[Match]:
    """Toutes les occurrences de marqueurs, dans l'ordre d'apparition."""
    norm = normalize(text, Level.STANDARD)
    out: list[Match] = []
    for name, rx in _COMPILED.items():
        category = MARKERS[name][1]
        out.extend(
            Match(name, m.group(0), m.start(), category) for m in rx.finditer(norm)
        )
    return sorted(out, key=lambda m: m.start)


def profile(text: str) -> dict[str, int]:
    """Décompte par marqueur. Les marqueurs absents ne figurent pas."""
    return dict(Counter(m.marker for m in find(text)).most_common())


def rates(text: str, per: int = 10_000) -> dict[str, float]:
    """Taux par ``per`` mots, comparable entre textes de longueurs différentes."""
    n_words = len(normalize(text, Level.STANDARD).split())
    if not n_words:
        return {}
    return {k: round(per * v / n_words, 2) for k, v in profile(text).items()}


def score(text: str, *, modern: bool = False) -> float:
    """Score heuristique de « tunisianité », dans ``[0, 1]``.

    Fondé sur la **diversité** des marqueurs présents plutôt que sur leur
    nombre brut : dix ``اللي`` dans un texte prouvent moins que trois marqueurs
    distincts de catégories différentes.

    Args:
      text: le texte à évaluer.
      modern: n'utiliser que :data:`MODERN_MARKERS`, pour distinguer le
        tunisien contemporain du registre littéraire ancien.

    Returns:
      0.0 pour un texte vide ou sans aucun marqueur.

    Rappel : AUC ~0.77. Pour une décision, préférez :mod:`darija.dialect`.

    """
    universe = MODERN_MARKERS if modern else set(MARKERS)
    found = {m.marker for m in find(text)} & universe
    if not found:
        return 0.0
    n_words = len(normalize(text, Level.STANDARD).split())
    if not n_words:
        return 0.0

    # Diversité : part des marqueurs du référentiel effectivement rencontrés,
    # saturée — au-delà de 5 marqueurs distincts, le texte est clairement marqué.
    diversity = min(1.0, len(found) / 5.0)
    # Densité : les marqueurs ne peuvent pas être un accident sur un texte long.
    density = min(1.0, len(find(text)) / max(1.0, n_words / 20.0))
    return round(max(0.0, min(1.0, 0.7 * diversity + 0.3 * density)), 4)


def explain(text: str) -> str:
    """Rapport lisible : ce qui a été trouvé, et ce que cela veut dire."""
    found = find(text)
    if not found:
        return "aucun marqueur tunisien détecté"
    lines = [f"score={score(text):.3f}  ({len(found)} occurrences, "
             f"{len({m.marker for m in found})} marqueurs distincts)"]
    for marker, n in Counter(m.marker for m in found).most_common():
        _, category, gloss = MARKERS[marker]
        lines.append(f"  {marker:22s} x{n:<3d} [{category}] {gloss}")
    return "\n".join(lines)


__all__ = [
    "DISCRIMINANT",
    "MARKERS",
    "MODERN_MARKERS",
    "Match",
    "explain",
    "find",
    "profile",
    "rates",
    "score",
]
