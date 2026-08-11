"""Agrégation des mesures en un rapport lisible.

**Ce module ne publie plus de taux « % de réponses tunisiennes ».** Il l'a fait,
et le chiffre était faux. Deux mesures indépendantes l'ont établi, et chacune
aurait suffi :

1. **Le seuil coupe au milieu de la distribution.** Sur la première campagne
   réelle, 57 % des réponses en condition implicite tombaient à moins de 0,02
   du seuil, et les 12 rejets étaient *tous* dans cette bande — dont un à
   0,0001. Lus à la main, ils étaient du tunisien authentique. Un taux calculé
   sur une frontière que la moitié des données chevauche ne mesure rien.
2. **Le registre déplace le niveau de base de 0,048**, soit plus du double de
   cette bande : du quotidien (médiane 0,886) au récit (0,838). Le taux global
   dépend donc de la proportion de récits dans le jeu de prompts — un choix
   arbitraire de l'auteur du banc, pas une propriété du modèle évalué.

Restent trois vues qui ne dépendent d'aucun seuil, et qui sont publiées :

``position``
    Le score médian. Une position, pas un verdict binaire.

``écart apparié``
    Même prompt, deux conditions. Seul le **signe** de la différence compte, ce
    qui immunise contre toute erreur de calibration. Sur la première campagne :
    41 prompts sur 41 gagnent en tunisianité avec la consigne explicite.

``par registre``
    Là où le modèle tient, et là où il lâche. C'est l'information que le taux
    global détruisait en moyennant des registres incomparables.

La bande limite reste affichée : c'est le diagnostic de fiabilité de la mesure
elle-même, pas une note attribuée au modèle.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass

from darija.dialect import DialectModel

from .prompts import Prompt
from .runner import Reply
from .scoring import Verdict, evaluate

#: Largeur de la bande d'indécision autour du seuil, en score. Mesurée : les
#: 12 rejets de la première campagne y tenaient tous.
BORDERLINE: float = 0.02


@dataclass(frozen=True)
class Cell:
    """Les chiffres d'un couple (modèle, condition, écriture)."""

    model: str
    condition: str
    script: str
    n: int
    n_errors: int
    n_unscorable: int
    n_scored: int
    median: float | None
    #: Part des scores dans la bande d'indécision. Élevée = la mesure ne
    #: tranche pas ici, et aucun verdict individuel n'est fiable.
    borderline_rate: float | None = None

    @property
    def coverage(self) -> float:
        """Part des appels qui ont produit une mesure exploitable."""
        return self.n_scored / self.n if self.n else 0.0


@dataclass(frozen=True)
class Shift:
    """L'écart apparié d'un modèle entre les deux conditions.

    C'est la seule affirmation forte du banc, parce qu'elle ne dépend que du
    signe d'une différence : ni seuil, ni règle de décision, ni calibration.
    """

    model: str
    n_pairs: int
    n_up: int
    median_delta: float

    @property
    def rate_up(self) -> float:
        """Part des prompts où la consigne explicite améliore le score."""
        return self.n_up / self.n_pairs if self.n_pairs else 0.0


def score_all(replies: list[Reply], model: DialectModel) -> list[Verdict]:
    """Mesure toutes les réponses collectées, en ignorant celles en erreur."""
    return [
        evaluate(
            r.reply,
            model,
            prompt_id=r.prompt_id,
            model_name=r.model,
            condition=r.condition,
            script=r.script,
        )
        for r in replies
        if not r.error
    ]


def aggregate(
    replies: list[Reply], verdicts: list[Verdict], threshold: float
) -> list[Cell]:
    """Regroupe par (modèle, condition, écriture)."""
    errors: dict[tuple[str, str, str], int] = defaultdict(int)
    totals: dict[tuple[str, str, str], int] = defaultdict(int)
    for reply in replies:
        key = (reply.model, reply.condition, reply.script)
        totals[key] += 1
        if reply.error:
            errors[key] += 1

    grouped: dict[tuple[str, str, str], list[Verdict]] = defaultdict(list)
    for verdict in verdicts:
        grouped[(verdict.model, verdict.condition, verdict.script)].append(verdict)

    cells: list[Cell] = []
    for key in sorted(totals):
        model, condition, script = key
        group = grouped.get(key, [])
        scored = [v for v in group if v.scorable and v.score is not None]
        scores = [v.score for v in scored if v.score is not None]
        near = [s for s in scores if abs(s - threshold) < BORDERLINE]
        cells.append(
            Cell(
                model=model,
                condition=condition,
                script=script,
                n=totals[key],
                n_errors=errors[key],
                n_unscorable=len(group) - len(scored),
                n_scored=len(scored),
                median=round(statistics.median(scores), 3) if scores else None,
                borderline_rate=round(len(near) / len(scores), 3) if scores else None,
            )
        )
    return cells


def paired_shifts(verdicts: list[Verdict]) -> list[Shift]:
    """Écart implicite → explicite, prompt par prompt.

    Un prompt ne compte que s'il a été mesuré dans les **deux** conditions :
    comparer des ensembles différents réintroduirait exactement le biais de
    composition que ce rapport existe pour éviter.
    """
    by_prompt: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for v in verdicts:
        if v.scorable and v.score is not None:
            by_prompt[(v.model, v.prompt_id)][v.condition] = v.score

    per_model: dict[str, list[float]] = defaultdict(list)
    for (model, _), conds in by_prompt.items():
        if "implicite" in conds and "explicite" in conds:
            per_model[model].append(conds["explicite"] - conds["implicite"])

    return [
        Shift(
            model=model,
            n_pairs=len(deltas),
            n_up=sum(1 for d in deltas if d > 0),
            median_delta=round(statistics.median(deltas), 4),
        )
        for model, deltas in sorted(per_model.items())
        if deltas
    ]


def by_register(
    verdicts: list[Verdict], prompts: list[Prompt]
) -> dict[tuple[str, str, str], tuple[int, float]]:
    """Score médian par (modèle, condition, catégorie de prompt).

    C'est l'information que le taux global détruisait : il moyennait des
    registres dont les niveaux de base diffèrent de 0,048.
    """
    category = {p.id: p.category for p in prompts}
    groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for v in verdicts:
        if v.scorable and v.score is not None and v.prompt_id in category:
            groups[(v.model, v.condition, category[v.prompt_id])].append(v.score)
    return {k: (len(s), round(statistics.median(s), 3)) for k, s in sorted(groups.items())}


def render(
    cells: list[Cell],
    shifts: list[Shift],
    registers: dict[tuple[str, str, str], tuple[int, float]],
    threshold: float,
) -> str:
    """Rend les trois vues."""
    if not cells:
        return "aucun résultat"

    width = max(len(c.model) for c in cells)
    out = [
        f"seuil du classifieur : {threshold:.3f}",
        "",
        "== position ==",
        f"  {'modele':<{width}}  {'condition':<10} {'ecriture':<8} "
        f"{'n':>4} {'err':>4} {'n/scor':>7} {'mediane':>8} {'indecis':>8}",
        "  " + "-" * (width + 54),
    ]
    for c in cells:
        med = f"{c.median:.3f}" if c.median is not None else "—"
        ind = f"{c.borderline_rate:.0%}" if c.borderline_rate is not None else "—"
        out.append(
            f"  {c.model:<{width}}  {c.condition:<10} {c.script:<8} "
            f"{c.n:>4} {c.n_errors:>4} {c.n_unscorable:>7} {med:>8} {ind:>8}"
        )

    if shifts:
        out += ["", "== ecart appariee : implicite -> explicite ==",
                f"  {'modele':<{width}}  {'paires':>7} {'en hausse':>11} {'ecart median':>13}",
                "  " + "-" * (width + 35)]
        for s in shifts:
            out.append(
                f"  {s.model:<{width}}  {s.n_pairs:>7} "
                f"{s.n_up}/{s.n_pairs} ({s.rate_up:.0%})".ljust(width + 30)
                + f"{s.median_delta:>+8.4f}"
            )

    if registers:
        entete = (
            f"  {'modele':<{width}}  {'condition':<10} {'registre':<13} "
            f"{'n':>4} {'mediane':>8}"
        )
        out += ["", "== par registre (score median) ==", entete, "  " + "-" * (width + 40)]
        for (model, cond, cat), (n, med) in registers.items():
            out.append(f"  {model:<{width}}  {cond:<10} {cat:<13} {n:>4} {med:>8.3f}")

    out += [
        "",
        "  Aucun taux de « reponses tunisiennes » n'est publie, et c'est deliberé :",
        "  le seuil coupe au milieu de la distribution (voir la colonne indecis),",
        "  et le registre deplace le niveau de base plus que le bruit de mesure.",
        "  Un tel taux dependrait de la composition du jeu de prompts, pas du modele.",
        "",
        "  « indecis » = part des scores a moins de "
        f"{BORDERLINE} du seuil. Eleve = la mesure",
        "                ne tranche pas ici ; aucun verdict individuel n'est fiable.",
        "  « n/scor »  = reponses trop courtes pour le classifieur (min 25 mots).",
        "  Les lignes « arabizi » passent par une translitteration approximative.",
    ]
    return "\n".join(out)
