"""Agrégation des mesures en un tableau lisible.

Le rapport sépare systématiquement l'écriture arabe de l'Arabizi. Les mélanger
donnerait un chiffre unique flatteur et faux : les scores sur l'Arabizi passent
par une translittération approximative (voir :mod:`darija_bench.scoring`), donc
ils ne sont pas comparables aux autres.

Il sépare aussi les réponses **non scorables** — trop courtes pour le
classifieur — au lieu de les compter comme des échecs. Une réponse dont on ne
sait rien n'est pas une réponse fausse ; l'amalgame gonflerait le taux d'erreur
d'un modèle laconique.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass

from darija.dialect import DialectModel

from .runner import Reply
from .scoring import MIN_DISTINCT_MARKERS, Verdict, evaluate


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
    tunisian_rate: float | None
    #: Réponses au-dessus du seuil du classifieur mais sous le minimum de
    #: marqueurs. C'est la bande où se logeaient les faux positifs en fusha
    #: conversationnelle : un chiffre élevé ici signale soit un modèle qui
    #: glisse vers la fusha, soit une règle de conjonction mal calibrée.
    n_classifier_only: int = 0

    @property
    def coverage(self) -> float:
        """Part des appels qui ont produit une mesure exploitable."""
        return self.n_scored / self.n if self.n else 0.0


def score_all(replies: list[Reply], model: DialectModel) -> list[Verdict]:
    """Mesure toutes les réponses collectées, en ignorant celles en erreur."""
    out: list[Verdict] = []
    for reply in replies:
        if reply.error:
            continue
        out.append(
            evaluate(
                reply.reply,
                model,
                prompt_id=reply.prompt_id,
                model_name=reply.model,
                condition=reply.condition,
                script=reply.script,
            )
        )
    return out


def aggregate(replies: list[Reply], verdicts: list[Verdict]) -> list[Cell]:
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
        tunisian = [v for v in scored if v.is_tunisian]
        borderline = [v for v in scored if v.above_classifier and not v.is_tunisian]
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
                tunisian_rate=round(len(tunisian) / len(scored), 3) if scored else None,
                n_classifier_only=len(borderline),
            )
        )
    return cells


def render(cells: list[Cell], threshold: float) -> str:
    """Rend le tableau, dans le style de ``darija data validate``."""
    if not cells:
        return "aucun résultat"

    width = max(len(c.model) for c in cells)
    lines = [
        f"seuil du classifieur : {threshold:.3f}  ·  marqueurs distincts exiges : "
        f"{MIN_DISTINCT_MARKERS}",
        "",
        f"  {'modele':<{width}}  {'condition':<10} {'ecriture':<8} "
        f"{'n':>4} {'err':>4} {'n/scor':>7} {'mediane':>8} {'limite':>7} {'tunisien':>9}",
        "  " + "-" * (width + 56),
    ]
    for cell in cells:
        median = f"{cell.median:.3f}" if cell.median is not None else "—"
        rate = f"{cell.tunisian_rate:.1%}" if cell.tunisian_rate is not None else "—"
        lines.append(
            f"  {cell.model:<{width}}  {cell.condition:<10} {cell.script:<8} "
            f"{cell.n:>4} {cell.n_errors:>4} {cell.n_unscorable:>7} {median:>8} "
            f"{cell.n_classifier_only:>7} {rate:>9}"
        )

    lines += [
        "",
        "  « tunisien » = classifieur au-dessus du seuil ET au moins "
        f"{MIN_DISTINCT_MARKERS} marqueurs distincts.",
        "  « limite »   = au-dessus du seuil mais sans les marqueurs. C'est la bande ou",
        "                 se logeaient les faux positifs en fusha conversationnelle ;",
        "                 un chiffre eleve appelle une relecture a la main.",
        "  « n/scor »   = reponses trop courtes pour le classifieur (min 25 mots),",
        "                 comptees a part : indecidables, pas fausses.",
        "  Les lignes « arabizi » passent par une translitteration approximative :",
        "  a lire comme un indice, pas comme une mesure comparable aux autres.",
        "  La regle de conjonction est provisoire (6 textes par cote) : les deux",
        "  signaux restent separes dans le fichier de resultats pour la reviser.",
    ]
    return "\n".join(lines)
