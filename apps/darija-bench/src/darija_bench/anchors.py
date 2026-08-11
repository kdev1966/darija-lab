"""Repères mesurés, pour qu'un score cesse d'être un nombre opaque.

« Médiane 0,850 » ne veut rien dire seul. Bon ? Mauvais ? Comparé à quoi ? En
retirant le taux de « réponses tunisiennes » — qui était faux — le banc avait
gagné en honnêteté et perdu en lisibilité.

Ce module rend la lisibilité, sans rendre le seuil. Un score n'est plus jugé
par rapport à une frontière, mais **situé entre deux textes humains** :

=====================================  ========  =======================
ancre                                   médiane  ce que c'est
=====================================  ========  =======================
``HAUT`` récit tunisien humain            0,9189  ``HkayetErwi``, 432 blocs
``BAS``  récit arabe classique            0,8292  Wikisource, 1 200 blocs
(hors échelle) arabe encyclopédique       0,7860  Wikipédia, 1 200 blocs
=====================================  ========  =======================

**Pourquoi Wikisource et non Wikipédia** — la troisième ligne est la raison
d'être de ce module. L'encyclopédie score 0,786, le récit classique 0,829 :
0,043 d'écart pour la même langue. Ancrer un conte sur l'encyclopédie aurait
ajouté l'écart de registre à l'écart de langue et gonflé toute mesure. Les deux
ancres retenues sont donc **du récit des deux côtés** — le contrôle de genre
que le biais nº 1 de ce dépôt exigeait déjà.

Une position de 0 % signifie « aussi peu tunisien qu'un conte en fusha », 100 %
« aussi tunisien qu'un conte tunisien humain ». Rien n'empêche de sortir de
l'intervalle, et rien ne doit l'empêcher : un modèle plus dialectal que le
corpus humain est une information, pas une anomalie à écrêter.
"""

from __future__ import annotations

#: Récit tunisien humain — ``HkayetErwi``, LinTO, CC BY-SA 4.0, 432 blocs.
HAUT: float = 0.9189

#: Récit arabe classique — Wikisource arabe, domaine public, 1 200 blocs.
BAS: float = 0.8292

#: Arabe encyclopédique — Wikipédia arabe, 1 200 blocs. **Hors échelle** :
#: conservé parce que c'est lui qui prouve pourquoi il ne fallait pas s'en
#: servir comme ancre.
ENCYCLOPEDIQUE: float = 0.7860


def position(score: float) -> float:
    """Situe un score entre les deux ancres narratives.

    Args:
      score: sortie de ``DialectModel.score``.

    Returns:
      0,0 au niveau du récit en fusha, 1,0 au niveau du récit tunisien humain.
      Les valeurs hors ``[0, 1]`` sont rendues telles quelles : un modèle peut
      légitimement se situer en deçà ou au-delà des deux repères.

    """
    return (score - BAS) / (HAUT - BAS)
