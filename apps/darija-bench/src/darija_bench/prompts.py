"""Le jeu de prompts du banc.

Les prompts sont **en tunisien**, jamais en français ni en fusha : ce qu'on
mesure est la réaction du modèle à la langue qu'on lui adresse, donc la langue
de la question fait partie du protocole.

Deux contraintes ont dicté leur écriture :

- **Longueur.** ``DialectModel.min_words`` vaut 25 : en dessous, ``predict``
  renvoie ``None`` et la réponse est indécidable. Les prompts appellent donc
  tous une réponse développée — jamais un oui/non, jamais un mot unique.
- **Écriture.** Huit prompts sont en Arabizi. C'est la forme écrite majoritaire
  du tunisien, et aucun banc arabe existant ne l'évalue. Voir la réserve sur la
  translittération dans :mod:`darija_bench.scoring`.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

DATA = Path(__file__).parent / "data" / "prompts.jsonl"

#: Écritures reconnues. ``arabizi`` déclenche la translittération au scoring.
SCRIPTS = frozenset({"arabe", "arabizi"})


@dataclass(frozen=True)
class Prompt:
    """Une question posée au modèle."""

    id: str
    category: str
    script: str
    text: str


def load(path: Path = DATA) -> list[Prompt]:
    """Charge le jeu de prompts.

    Raises:
      ValueError: identifiant dupliqué, ou écriture inconnue. Les deux
        casseraient silencieusement l'agrégation du rapport, qui indexe par
        identifiant et sépare les résultats par écriture.

    """
    out: list[Prompt] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        prompt = Prompt(**rec)
        if prompt.id in seen:
            raise ValueError(f"identifiant dupliqué : {prompt.id!r}")
        if prompt.script not in SCRIPTS:
            raise ValueError(f"écriture inconnue {prompt.script!r} pour {prompt.id!r}")
        seen.add(prompt.id)
        out.append(prompt)
    return out


def by_script(prompts: list[Prompt]) -> Iterator[tuple[str, list[Prompt]]]:
    """Regroupe par écriture, dans un ordre stable."""
    for script in sorted(SCRIPTS):
        group = [p for p in prompts if p.script == script]
        if group:
            yield script, group


def sample(prompts: list[Prompt], n: int, *, seed: int = 0) -> list[Prompt]:
    """Tire ``n`` prompts, **stratifiés par écriture** et reproductibles.

    Mesuré sur le palier gratuit de Gemini : ``gemini-3.6-flash`` est plafonné
    à 20 requêtes par jour, et le protocole complet en demande 84. Sans
    échantillon, le banc est simplement hors d'atteinte d'un compte gratuit.

    Deux exigences, et l'ordre compte :

    - **Stratifié.** Un tirage uniforme sur 42 prompts dont 8 en Arabizi peut
      n'en retenir aucun. On perdrait précisément la forme écrite majoritaire
      du tunisien, et on ne le verrait pas dans le rapport.
    - **Reproductible.** À graine égale, même échantillon : deux modèles
      mesurés à des jours différents restent comparables. Un tirage
      non reproductible produirait un classement sans signification.

    Raises:
      ValueError: ``n`` non strictement positif.

    """
    if n <= 0:
        raise ValueError(f"taille d'échantillon invalide : {n}")
    if n >= len(prompts):
        return list(prompts)

    rng = random.Random(seed)
    groups = dict(by_script(prompts))
    total = len(prompts)

    # Quota proportionnel, au moins un par écriture présente.
    quotas = {s: max(1, round(n * len(g) / total)) for s, g in groups.items()}
    out: list[Prompt] = []
    for script in sorted(groups):
        pool = sorted(groups[script], key=lambda p: p.id)
        out.extend(rng.sample(pool, min(quotas[script], len(pool))))

    # L'arrondi peut dépasser ou manquer la cible ; on ajuste sur l'écriture
    # majoritaire, jamais sur celle qui n'a qu'un représentant.
    out.sort(key=lambda p: p.id)
    if len(out) > n:
        majority = max(groups, key=lambda s: len(groups[s]))
        surplus = len(out) - n
        for prompt in [p for p in reversed(out) if p.script == majority][:surplus]:
            out.remove(prompt)
    return out
