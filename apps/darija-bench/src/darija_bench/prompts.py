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
