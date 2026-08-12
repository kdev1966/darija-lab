"""Trier une pile de textes : tunisien, pas tunisien, indécidable.

C'est l'usage pour lequel cet instrument est le mieux placé. Ses forces y
jouent — mesure agrégée, texte humain, blocs assez longs — et ses faiblesses
n'y mordent pas : le verdict individuel peu fiable sur une sortie de LLM
importe peu quand on trie des centaines de documents, et le minimum de 25 mots
n'est pas gênant sur du texte réel.

Quiconque construit un corpus tunisien doit séparer le tunisien du marocain,
de l'algérien et de la fusha dans une pile scrapée. Rien d'autre ne le fait
pour le tunisien.

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
from .scoring import MIN_DISTINCT_MARKERS, blocks

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


def judge(text: str, model: DialectModel, *, ident: str = "") -> Verdict:
    """Trie un document.

    ``indecidable`` n'est pas un rejet : c'est un document trop court pour que
    le classifieur se prononce. Le confondre avec un rejet fausserait tout
    décompte — un texte bref n'est pas un texte étranger.
    """
    n_words = len(text.split())
    decoupe = blocks(text)
    if not decoupe:
        return Verdict(ident, n_words, 0, "indecidable")

    scores = [model.score(b) for b in decoupe]
    median = statistics.median(scores)
    distinct = len({m.marker for m in markers.find(text)} & markers.DISCRIMINANT)
    tunisien = [s for s in scores if s >= model.threshold]
    ok = median >= model.threshold and distinct >= MIN_DISTINCT_MARKERS
    return Verdict(
        ident=ident,
        n_words=n_words,
        n_blocks=len(decoupe),
        verdict="tunisien" if ok else "autre",
        median=round(median, 4),
        position=round(anchors.position(median), 4),
        share_tunisian=round(len(tunisien) / len(scores), 3),
        n_markers=distinct,
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
    lignes += [
        "",
        "  « indecidable » = moins de blocs mesurables que le minimum du",
        "                    classifieur. Ce n'est PAS un rejet.",
        "  Chaque document est decoupe en blocs d'environ 60 mots, la taille",
        "  sur laquelle le seuil et les reperes ont ete etablis.",
    ]
    return "\n".join(lignes)
