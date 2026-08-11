"""Banc d'évaluation de la tunisianité des réponses d'un LLM.

L'application consomme les cinq modules de ``darija-core`` et n'ajoute aucune
mesure de son cru : tout ce qui décide vient du classifieur contrastif validé,
tout ce qui explique vient des marqueurs.

Voir :mod:`darija_bench.scoring` pour ce que la mesure vaut, et surtout pour ce
qu'elle ne vaut pas.
"""

from .prompts import Prompt, load
from .scoring import Verdict, evaluate

__all__ = ["Prompt", "Verdict", "evaluate", "load"]
