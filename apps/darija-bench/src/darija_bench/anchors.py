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

⚠️ **Ces trois valeurs appartiennent à `vs_maghreb`**, et à lui seul. Un score
n'a de sens que sur l'échelle du modèle qui l'a produit : `vs_maghreb_llm`
place le même récit tunisien à 0,9181 et déplace son seuil de 0,838 à 0,828.
L'employer avec ces ancres mélangerait deux échelles. Si le banc bascule un
jour sur lui, il faut re-mesurer les trois — la première demande le cache
``HkayetErwi``, les deux autres ``darija data fetch ar_source`` et ``ar``.
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

# --------------------------------------------------------------- dispersion
# :data:`HAUT` est une **médiane**, pas un plafond — et l'oublier fait lire
# « 57 % » comme « à moitié tunisien ». Mesuré sur les 432 blocs de
# ``HkayetErwi``, le corpus humain s'étale lui-même de 5 % à 151 % de position,
# écart-type 0,038. Un texte à 57 % est donc à un écart-type sous la médiane :
# bas de la fourchette normale, pas anomalie.
#
# C'est la même faute que celle corrigée sur les textes longs — un chiffre
# unique qui cache une dispersion. Sauf qu'ici la dispersion cachée est celle
# de l'ancre, pas celle du texte mesuré.

#: 10ᵉ centile du récit humain — position 34 %.
HUMAIN_P10: float = 0.8596

#: 1ᵉʳ quartile du récit humain — position 80 %.
HUMAIN_Q1: float = 0.9011

#: 3ᵉ quartile du récit humain — position 122 %.
HUMAIN_Q3: float = 0.9389

#: Écart-type des scores du récit humain, en score brut.
HUMAIN_ECART_TYPE: float = 0.0381


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


def qualify(pos: float) -> str:
    """Situe une position **par rapport à la dispersion humaine**, en clair.

    Sans ce repère, « 57 % » se lit « à moitié tunisien » alors que 13 % du
    récit tunisien authentique score plus bas. La phrase rendue ici est ce que
    la barre montre visuellement.

    Args:
      pos: sortie de :func:`position`.

    Returns:
      Une phrase courte, sans jargon.

    """
    if pos < position(HUMAIN_P10):
        return "sous le dixième le plus bas du récit humain"
    if pos < position(HUMAIN_Q1):
        return "sous la zone typique, mais dans la fourchette du récit humain"
    if pos <= position(HUMAIN_Q3):
        return "dans la moitié centrale du récit humain"
    return "au-dessus des trois quarts du récit humain"


def spread() -> dict[str, float]:
    """Les repères de dispersion, en position, pour l'affichage.

    Returns:
      ``{"p10", "q1", "q3"}`` — les bornes à dessiner sur la barre.

    """
    return {
        "p10": round(position(HUMAIN_P10), 4),
        "q1": round(position(HUMAIN_Q1), 4),
        "q3": round(position(HUMAIN_Q3), 4),
    }
