r"""Transformer les sorties du carnet Colab en corpus négatif utilisable.

Le carnet ``notebooks/colab_negatif_adversarial.ipynb`` produit un JSONL de
réponses brutes. Deux traitements les séparent d'un corpus d'entraînement, et
aucun des deux n'est cosmétique.

**Le nettoyage.** Qwen2.5-7B fuit : sur 9 réponses de 544 il ré-émet la
consigne après un jeton de gabarit (``\\nuser``), et il glisse du chinois. Or
les consignes de ce banc sont écrites *en tunisien* — la fuite injecte donc du
positif authentique au cœur de la classe négative. C'est la contamination la
plus coûteuse possible ici, et elle est invisible à l'œil sur un corpus arabe.

**Le partage par prompt.** Une même consigne est tirée 16 fois ; les réponses
sont des quasi-doublons. Les répartir au hasard entre entraînement et
validation ferait mesurer la mémorisation et non la généralisation — l'erreur
exacte qui avait fait annoncer « 8,7 % → 4,2 % » quand la mesure honnête, hors
des blocs vus, donnait « 66,7 % → 60,0 % ».
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Iterable
from pathlib import Path

from .scoring import blocks

#: Jeton de gabarit après lequel le modèle recommence à parler « en user ».
_FUITE = re.compile(r"\n\s*(?:user|assistant|system)\b")

#: Liste blanche : arabe, latin, chiffres, ponctuation courante. Une liste
#: *noire* aurait laissé passer la ponctuation pleine chasse (``，`` ``：``),
#: qui a effectivement produit 15 blocs identiques des deux côtés du partage.
_HORS_ALPHABET = re.compile(
    r"[^؀-ۿݐ-ݿࢠ-ࣿA-Za-z0-9\s.,;:!?()\[\]\-–—«»\"'/%+]"
)

#: En dessous, la réponse est un fragment : le modèle a été coupé ou a refusé.
MIN_MOTS: int = 20

#: Part des prompts réservée au jugement. Huit prompts sur trente-quatre.
PART_VALIDATION: float = 0.25


def clean(texte: str) -> str:
    """Retire la fuite de gabarit et tout alphabet étranger.

    Args:
      texte: réponse brute du modèle.

    Returns:
      Le texte tronqué au premier jeton de gabarit, réduit à l'alphabet retenu.

    """
    return re.sub(r"\s+", " ", _HORS_ALPHABET.sub(" ", _FUITE.split(texte, 1)[0])).strip()


def split_by_prompt(
    rows: Iterable[dict], *, seed: int = 0
) -> tuple[list[str], list[str]]:
    """Découpe en blocs d'entraînement et de validation, sans prompt commun.

    Args:
      rows: réponses brutes, chacune portant ``prompt_id`` et ``texte``.
      seed: graine du tirage, pour que le partage soit reproductible.

    Returns:
      ``(entraînement, validation)``, deux listes de blocs de 60 mots.

    """
    propres = [
        {**r, "texte": clean(r["texte"])}
        for r in rows
    ]
    propres = [r for r in propres if len(r["texte"].split()) >= MIN_MOTS]

    prompts = sorted({r["prompt_id"] for r in propres})
    random.Random(seed).shuffle(prompts)
    coupe = int(len(prompts) * PART_VALIDATION)
    val, train = set(prompts[:coupe]), set(prompts[coupe:])

    def _blocs(garder: set[str]) -> list[str]:
        return blocks("\n".join(r["texte"] for r in propres if r["prompt_id"] in garder))

    return _blocs(train), _blocs(val)


def write_corpus(source: Path, destination: Path, *, seed: int = 0) -> dict[str, int]:
    """Écrit ``llm_fusha.txt`` et ``llm_fusha_val.txt`` dans ``data/raw``.

    Args:
      source: le JSONL téléchargé depuis Colab.
      destination: le répertoire ``data/raw`` de ``darija-core``.
      seed: graine du partage.

    Returns:
      Le nombre de blocs écrits de chaque côté.

    """
    rows = [json.loads(ligne) for ligne in source.read_text(encoding="utf-8").splitlines() if ligne]
    train, val = split_by_prompt(rows, seed=seed)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "llm_fusha.txt").write_text("\n".join(train), encoding="utf-8")
    (destination / "llm_fusha_val.txt").write_text("\n".join(val), encoding="utf-8")
    return {"train": len(train), "validation": len(val)}
