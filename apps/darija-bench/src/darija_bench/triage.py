"""Trier une pile de textes : tunisien, pas tunisien, indécidable.

C'est l'usage pour lequel cet instrument est le mieux placé. Ses forces y
jouent — mesure agrégée, texte humain, blocs assez longs — et ses faiblesses
n'y mordent pas : le verdict individuel peu fiable sur une sortie de LLM
importe peu quand on trie des centaines de documents, et le minimum de 25 mots
n'est pas gênant sur du texte réel.

Quiconque construit un corpus tunisien doit séparer le tunisien du marocain,
de l'algérien et de la fusha dans une pile scrapée. Rien d'autre ne le fait
pour le tunisien.

**Le filtre de marqueurs n'est pas appliqué par défaut ici**, contrairement au
banc. Mesuré sur les corpus du dépôt, part des blocs de 60 mots reconnus :

===========  ==================  =============  ==============
corpus       classifieur seul    + marqueurs    coût du filtre
===========  ==================  =============  ==============
``linto``    93,0 %              83,0 %         10,0 %
``arbml_tn`` 85,9 %              56,6 %         29,2 %
``tsac``     86,8 %              49,8 %         37,0 %
négatifs     0 – 0,7 %           0 – 0,3 %      ~ 0
===========  ==================  =============  ==============

Le filtre coûte jusqu'à 37 points sur le tunisien et ne gagne **rien** sur les
contre-exemples : le classifieur seul les rejette déjà à 99,3-100 %. Il avait
été ajouté contre la fusha *conversationnelle*, un défaut propre aux sorties de
LLM (biais nº 7) — or un corpus de tweets tunisiens n'en contient pas.

``--strict`` le rétablit, pour trier une pile dont on soupçonne qu'elle
contient des textes générés.

**Un document n'est pas un bloc.** Les repères et le seuil ont été établis sur
des blocs d'environ 60 mots ; un texte long mesuré d'un seul tenant rend une
moyenne qui lisse ses variations. Chaque document est donc découpé, et son
verdict s'appuie sur la **médiane de ses blocs** — plus une part de blocs
tunisiens, qui dit si le document est homogène ou panaché.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from darija import markers
from darija.dialect import DialectModel

from . import anchors
from .scoring import MIN_DISTINCT_MARKERS, blocks, prepare

#: Extensions reconnues en entrée.
SUFFIXES = (".jsonl", ".txt", ".csv")


@dataclass(frozen=True)
class Verdict:
    """Le sort d'un document dans le tri."""

    ident: str
    n_words: int
    n_blocks: int
    verdict: str
    median: float | None = None
    position: float | None = None
    share_tunisian: float | None = None
    n_markers: int = 0
    transliterated: bool = False

    def as_dict(self) -> dict[str, object]:
        """Vue sérialisable."""
        return {
            "id": self.ident,
            "n_words": self.n_words,
            "n_blocks": self.n_blocks,
            "verdict": self.verdict,
            "median": self.median,
            "position": self.position,
            "share_tunisian": self.share_tunisian,
            "n_markers": self.n_markers,
            "transliterated": self.transliterated,
        }


def read_documents(path: Path, field: str = "text") -> Iterator[tuple[str, str]]:
    """Lit des documents ``(identifiant, texte)`` depuis un fichier ou un dossier.

    ``.jsonl`` prend ``field`` comme colonne de texte ; ``.txt`` traite **une
    ligne par document** ; un dossier prend chaque fichier comme un document.

    Raises:
      ValueError: extension inconnue.
      FileNotFoundError: chemin absent.

    """
    if not path.exists():
        raise FileNotFoundError(f"{path} introuvable")

    if path.is_dir():
        for f in sorted(path.rglob("*")):
            if f.is_file() and f.suffix.lower() in (".txt", ".md"):
                yield f.name, f.read_text(encoding="utf-8", errors="replace")
        return

    suffix = path.suffix.lower()
    if suffix not in SUFFIXES:
        raise ValueError(f"extension non reconnue {suffix!r} ; connues : {SUFFIXES}")

    if suffix == ".jsonl":
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            rec = json.loads(line)
            texte = rec.get(field)
            if texte:
                yield str(rec.get("id") or rec.get("uid") or i), str(texte)
        return

    if suffix == ".csv":
        import csv  # noqa: PLC0415

        with path.open(encoding="utf-8") as fh:
            for i, rec in enumerate(csv.DictReader(fh)):
                texte = rec.get(field)
                if texte:
                    yield str(rec.get("id") or i), str(texte)
        return

    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if line.strip():
            yield str(i), line


def group_documents(
    docs: Iterator[tuple[str, str]], target: int = 60
) -> Iterator[tuple[str, str]]:
    """Agrège des fragments consécutifs en unités mesurables.

    Beaucoup de corpus sont des piles de fragments — une ligne par tweet ou
    par commentaire — de huit à trente mots. Le classifieur en exige
    vingt-cinq : jugés un par un, 94 % de ces corpus ressortent
    ``indecidable``, et le tri ne dit rien.

    Grouper reproduit ce que fait déjà l'entraînement (``assemble.chunk``) :
    on mesure la **variété de langue du corpus**, pas chaque item.

    **Une unité groupée n'est donc pas un document.** Elle mêle des auteurs et
    des sujets différents. Le verdict porte sur le corpus, jamais sur un item
    en particulier — ne pas s'en servir pour retirer une ligne précise.
    """
    tampon: list[str] = []
    debut = ""
    mots = 0
    for ident, texte in docs:
        if not tampon:
            debut = ident
        tampon.append(texte)
        mots += len(texte.split())
        if mots >= target:
            yield f"{debut}+{len(tampon)}", "\n".join(tampon)
            tampon, mots = [], 0
    if tampon and mots >= target // 2:
        yield f"{debut}+{len(tampon)}", "\n".join(tampon)


def judge(
    text: str, model: DialectModel, *, ident: str = "", strict: bool = False
) -> Verdict:
    """Trie un document.

    Args:
      text: le document.
      model: le classifieur de dialecte.
      ident: identifiant reporté tel quel.
      strict: exiger aussi un marqueur discriminant. Faux par défaut — voir
        le module pour ce que ce filtre coûte sur du texte humain.

    ``indecidable`` n'est pas un rejet : c'est un document trop court pour que
    le classifieur se prononce. Le confondre avec un rejet fausserait tout
    décompte — un texte bref n'est pas un texte étranger.

    """
    # L'écriture latine passe par la translittération, comme partout ailleurs
    # dans l'application. L'oublier envoyait de l'Arabizi brut au classifieur,
    # qui n'en a jamais vu : TUNIZI — une source POSITIVE — ressortait à 0 %
    # de tunisien et une position de −41 %.
    scored, translit = prepare(text)
    n_words = len(scored.split())
    decoupe = blocks(scored)
    if not decoupe:
        return Verdict(ident, n_words, 0, "indecidable", transliterated=translit)

    scores = [model.score(b) for b in decoupe]
    median = statistics.median(scores)
    distinct = len({m.marker for m in markers.find(scored)} & markers.DISCRIMINANT)
    tunisien = [s for s in scores if s >= model.threshold]
    ok = median >= model.threshold and (not strict or distinct >= MIN_DISTINCT_MARKERS)
    return Verdict(
        ident=ident,
        n_words=n_words,
        n_blocks=len(decoupe),
        verdict="tunisien" if ok else "autre",
        median=round(median, 4),
        position=round(anchors.position(median), 4),
        share_tunisian=round(len(tunisien) / len(scores), 3),
        n_markers=distinct,
        transliterated=translit,
    )


def summarise(verdicts: list[Verdict]) -> str:
    """Rend le tableau de tri."""
    if not verdicts:
        return "aucun document"

    total = len(verdicts)
    par: dict[str, list[Verdict]] = {}
    for v in verdicts:
        par.setdefault(v.verdict, []).append(v)

    lignes = [
        f"{total} documents tries",
        "",
        f"  {'verdict':<13} {'n':>6} {'part':>7} {'position mediane':>18}",
        "  " + "-" * 48,
    ]
    for nom in ("tunisien", "autre", "indecidable"):
        groupe = par.get(nom, [])
        if not groupe:
            continue
        pos = [v.position for v in groupe if v.position is not None]
        med = f"{statistics.median(pos):.0%}" if pos else "—"
        lignes.append(f"  {nom:<13} {len(groupe):>6} {len(groupe) / total:>6.1%} {med:>18}")

    mesures = [v for v in verdicts if v.position is not None]
    if mesures:
        pos = sorted(v.position for v in mesures)
        lignes += [
            "",
            f"  position : min {pos[0]:.0%} · mediane {statistics.median(pos):.0%} "
            f"· max {pos[-1]:.0%}",
            f"  documents trop courts pour etre juges : {len(verdicts) - len(mesures)}",
        ]
    translit = sum(1 for v in verdicts if v.transliterated)
    if translit:
        lignes += [
            "",
            f"  {translit} document(s) en ecriture latine, translitteres avant mesure.",
            "  La conversion est approximative : lisez ces verdicts comme des indices.",
        ]
    lignes += [
        "",
        "  « indecidable » = moins de blocs mesurables que le minimum du",
        "                    classifieur. Ce n'est PAS un rejet.",
        "  Chaque document est decoupe en blocs d'environ 60 mots, la taille",
        "  sur laquelle le seuil et les reperes ont ete etablis.",
    ]
    return "\n".join(lignes)
