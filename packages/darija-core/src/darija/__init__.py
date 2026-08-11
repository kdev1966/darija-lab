"""darija-core — socle de traitement du texte tunisien (الدارجة التونسية).

Quatre briques indépendantes, sans dépendance externe :

:mod:`darija.normalize`
    Normalisation orthographique **qui préserve le dialecte**.
:mod:`darija.arabizi`
    Translittération Arabizi ↔ arabe, et détection d'Arabizi.
:mod:`darija.codeswitch`
    Segmentation de l'alternance arabe / français / Arabizi.
:mod:`darija.markers`
    Marqueurs morphologiques et lexicaux du tunisien, pour inspection.
:mod:`darija.dialect`
    Classifieur contrastif entraînable, pour décision.

Prise en main::

    from darija import normalize, to_arabic, profile

    normalize("مـــاذا")             # -> "ماذا"
    to_arabic("chnowa a7welek")      # -> "شنوا احوالك"
    profile("ken 3andek le temps")   # -> {"arabizi": ..., "fr": ...}
"""

from __future__ import annotations

from .arabizi import arabizi_score, is_arabizi, to_arabic, to_arabizi
from .codeswitch import Segment, extract, is_code_switched, profile, segment
from .dialect import DialectModel, evaluate, train
from .markers import explain, find, rates
from .normalize import Level, normalize, script_ratio, tokenize

__version__ = "0.1.0"

__all__ = [
    "DialectModel",
    "Level",
    "Segment",
    "__version__",
    "arabizi_score",
    "evaluate",
    "explain",
    "extract",
    "find",
    "is_arabizi",
    "is_code_switched",
    "normalize",
    "profile",
    "rates",
    "script_ratio",
    "segment",
    "to_arabic",
    "to_arabizi",
    "tokenize",
    "train",
]
