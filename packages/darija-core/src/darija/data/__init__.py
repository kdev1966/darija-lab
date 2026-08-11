"""Récupération et assemblage des corpus pour ``darija.dialect``.

Chaîne complète, du réseau au modèle entraîné::

    darija data budget     # ce que ça va coûter, avant de lancer quoi que ce soit
    darija data fetch      # télécharge vers data/raw/
    darija data build      # assemble des jeux équilibrés
    darija data train      # entraîne, évalue, écrit le modèle

Trois choix structurent ce module :

* le **rôle** d'une source (positive = tunisien, negative = contre-exemple) est
  déclaré dans :mod:`~darija.data.sources`, parce que c'est le choix des
  négatifs qui détermine ce que le modèle saura faire ;
* les dumps Wikipédia sont lus **en flux** et coupés à un plafond d'octets, donc
  on ne télécharge et n'écrit jamais plus que nécessaire ;
* l'**équilibrage** a lieu à la construction, pas au téléchargement, pour que le
  cache reste complet et rééchantillonnable.
"""

from __future__ import annotations

from .assemble import (
    CONTRASTS,
    GENRE_CONFOUND,
    Contrast,
    Dataset,
    available,
    build,
    chunk,
)
from .entities import ENTITIES, count_entities, strip_entities
from .fetch import DEFAULT_CACHE, MissingExtra, fetch, fetch_all, load
from .sources import SOURCES, Source, budget, by_role, unlicensed

__all__ = [
    "CONTRASTS",
    "GENRE_CONFOUND",
    "PROVENANCE_MATTERS",
    "REGISTER_MATTERS",
    "DEFAULT_CACHE",
    "ENTITIES",
    "SOURCES",
    "Contrast",
    "Dataset",
    "MissingExtra",
    "Source",
    "available",
    "budget",
    "build",
    "by_role",
    "chunk",
    "count_entities",
    "fetch",
    "fetch_all",
    "load",
    "score_by_source",
    "strip_entities",
    "unlicensed",
]
