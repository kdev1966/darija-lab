"""Retrait des noms propres, pour que le modèle classe la langue et non le sujet.

Le problème, mesuré et non supposé. Un classifieur tunisien/marocain entraîné
sans ce filtre place ``مغرب`` et ``تونسي`` parmi ses traits les plus lourds. Ce
sont des **noms de pays**, pas des marques de dialecte : le modèle apprend en
partie « ce texte parle du Maroc » plutôt que « ce texte est écrit en marocain ».
Un texte tunisien évoquant le Maroc serait mal classé, et le score s'effondre dès
qu'on change de sujet.

On retire donc, avant l'entraînement :

* les pays et les gentilés qui en dérivent (``تونس`` → aussi ``تونسي``,
  ``التونسيين``…) ;
* les grandes villes du Maghreb et d'Égypte ;
* les noms de chaînes et de plateformes, qui trahissent la provenance du corpus
  plutôt que la langue de son auteur.

Le filtre travaille par **radical** : les clitiques (``ال ب ل ف ك و``) et les
suffixes de relation (``ي ية يين``) sont retirés avant comparaison, si bien
qu'une seule entrée couvre toutes les formes fléchies.

Appliqué à la construction du jeu, pas au téléchargement : le cache reste
intact et l'on peut mesurer l'effet du filtre en le désactivant.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from ..normalize import Level, normalize

#: Pays et régions. Les gentilés (``تونسي``, ``مغربية``) sont couverts par le
#: retrait de suffixe, il est inutile de les lister.
COUNTRIES: frozenset[str] = frozenset({
    "تونس", "مغرب", "جزاير", "جزائر", "ليبيا", "مصر", "سودان", "موريتانيا",
    "سوريا", "لبنان", "اردن", "فلسطين", "عراق", "سعوديه", "امارات", "قطر",
    "كويت", "عمان", "يمن", "بحرين", "مغارب",
})

#: Écartés délibérément, bien qu'ils désignent aussi des lieux ou des médias :
#: ce sont d'abord des mots courants, et les retirer détruirait du signal réel.
#: ``نهار`` (jour) et ``شمس`` (soleil) sont des chaînes algérienne et tunisienne ;
#: ``بيضا`` (blanc) est Casablanca ; ``وليدي`` (mon fils) est une ville ;
#: ``جده`` est à la fois Djeddah et « son grand-père ».
AMBIGUOUS_EXCLUDED: frozenset[str] = frozenset({
    "نهار", "شمس", "حوار", "بيان", "وطنيه", "سميره", "زيتونه", "جوهره",
    "شروق", "بيضا", "بيضاء", "وليدي", "جده", "سلا", "خليج", "مشرق",
})

#: Villes. Une mention de ville identifie le corpus aussi sûrement qu'un pays.
CITIES: frozenset[str] = frozenset({
    # Tunisie
    "صفاقس", "سوسه", "بنزرت", "قابس", "قيروان", "نابل", "منستير", "قفصه",
    "مدنين", "توزر", "جربه", "بنعروس", "اريانه", "منوبه", "زغوان", "سليانه",
    # Maroc
    "رباط", "مراكش", "فاس", "طنجه", "اكادير", "وجده",
    "مكناس", "تطوان", "ناظور",
    # Algérie
    "وهران", "قسنطينه", "عنابه", "سطيف", "باتنه", "تلمسان", "بجايه",
    "تيزي", "وزو", "بليده", "ورقله",
    # Libye
    "طرابلس", "بنغازي", "مصراته", "سبها", "زاويه", "درنه",
    # Égypte / autres
    "قاهره", "اسكندريه", "رياض", "دبي", "بيروت", "دمشق", "بغداد",
})

#: Chaînes, médias et plateformes. Ils marquent la **provenance du corpus** :
#: TSAC vient des pages de radios tunisiennes, le corpus algérien de Twitter.
MEDIA: frozenset[str] = frozenset({
    "موزاييك", "حنبعل", "هسبريس", "دزاير",
    "فيسبوك", "فايسبوك", "يوتيوب", "تويتر", "انستقرام", "انستغرام",
    "تيكتوك", "واتساب", "تغريد", "تغريده", "ريتويت", "هاشتاق",
})

#: Prénoms et patronymes. Mesuré : ``لطفي`` apparaît 126 fois dans TSAC et
#: **zéro fois** dans les quatre autres corpus — c'est une personnalité
#: tunisienne discutée sur les pages de radios dont TSAC est issu. Un
#: discriminateur parfait, et parfaitement étranger au dialecte.
#:
#: La liste est **volontairement courte et prudente**. Beaucoup de prénoms
#: arabes sont aussi des mots courants — ``كريم`` (généreux), ``امين``
#: (honnête), ``نور`` (lumière), ``سعيد`` (heureux), ``رشيد`` (sage) — et les
#: retirer détruirait du signal réel. Seuls les noms sans ambiguïté figurent
#: ici : la fuite par nom propre n'est donc que **partiellement** traitée.
PERSON_NAMES: frozenset[str] = frozenset({
    "لطفي", "جعفر", "مصطفي", "ياسين", "سفيان", "بلقاسم", "منصف", "الهادي",
    "نزار", "شكري", "حمادي", "منجي", "معز", "غازي", "زياد", "مروان",
    "عصام", "انيس", "هيثم", "سهيل", "وسيم", "بشير", "الطاهر", "صالحه",
})

#: L'ensemble effectivement filtré. ``AMBIGUOUS_EXCLUDED`` en est retranché par
#: construction, pour que l'oubli d'une purge ne passe pas inaperçu.
ENTITIES: frozenset[str] = (
    COUNTRIES | CITIES | MEDIA | PERSON_NAMES
) - AMBIGUOUS_EXCLUDED

#: Clitiques agglutinés en tête de mot, les plus longs d'abord.
_PREFIXES: tuple[str, ...] = ("وال", "فال", "بال", "كال", "لل", "ال",
                              "و", "ف", "ب", "ك", "ل")

#: Suffixes de relation et de pluriel, les plus longs d'abord.
_SUFFIXES: tuple[str, ...] = ("يين", "يات", "يه", "ين", "ات", "ي", "ه")

#: En deçà, retirer un affixe laisserait un radical trop court pour être fiable.
_MIN_STEM = 3

_TOKEN = re.compile(r"\S+")

#: Tout ce qui n'est ni lettre arabe, ni lettre latine, ni espace. La ponctuation
#: ne porte aucune information dialectale — c'est un **style de plateforme**.
#: Sans ce retrait, un classifieur tunisien/algérien s'appuie sur des n-grammes
#: comme ``...`` et ``ا...``, c'est-à-dire sur l'habitude de ponctuer d'une
#: communauté, pas sur sa langue.
_PUNCT = re.compile(r"[^\w\s\u0621-\u064A\u067E\u0686\u06A4\u06A5\u06A8\u06AF]|\d")


def strip_punctuation(text: str) -> str:
    """Retire ponctuation et chiffres, en préservant les lettres."""
    return re.sub(r"\s+", " ", _PUNCT.sub(" ", text or "")).strip()


def stem(token: str) -> str:
    """Retire un clitique initial et un suffixe de relation, au plus un de chaque.

    Volontairement minimal : ce n'est pas un analyseur morphologique, seulement
    de quoi rattacher ``التونسيين`` à ``تونس``.
    """
    for p in _PREFIXES:
        if token.startswith(p) and len(token) - len(p) >= _MIN_STEM:
            token = token[len(p):]
            break
    for s in _SUFFIXES:
        if token.endswith(s) and len(token) - len(s) >= _MIN_STEM:
            token = token[: -len(s)]
            break
    return token


def is_entity(token: str, entities: Iterable[str] = ENTITIES) -> bool:
    """Vrai si ``token`` est un nom propre à retirer, sous une forme fléchie."""
    # AGGRESSIVE, et non STANDARD : un token porte souvent sa ponctuation
    # (« لطفي!! »), et au niveau STANDARD elle reste collée, si bien que la
    # comparaison échoue silencieusement.
    norm = normalize(token, Level.AGGRESSIVE)
    if not norm:
        return False
    ent = entities if isinstance(entities, frozenset | set) else frozenset(entities)
    return norm in ent or stem(norm) in ent


def strip_entities(text: str, entities: Iterable[str] = ENTITIES) -> str:
    """Retire les noms propres de ``text``, en conservant tout le reste.

    Le texte n'est pas normalisé au passage : seuls les mots reconnus comme
    entités disparaissent, la casse et la ponctuation des autres sont
    préservées.
    """
    ent = entities if isinstance(entities, frozenset | set) else frozenset(entities)
    kept = [t for t in _TOKEN.findall(text or "") if not is_entity(t, ent)]
    return " ".join(kept)


def count_entities(lines: Iterable[str], entities: Iterable[str] = ENTITIES) -> dict[str, int]:
    """Décompte des entités rencontrées — pour vérifier ce que le filtre retire."""
    ent = entities if isinstance(entities, frozenset | set) else frozenset(entities)
    out: dict[str, int] = {}
    for line in lines:
        for tok in _TOKEN.findall(line or ""):
            norm = normalize(tok, Level.AGGRESSIVE)
            if not norm:
                continue
            key = norm if norm in ent else (stem(norm) if stem(norm) in ent else None)
            if key:
                out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def clean_for_training(text: str) -> str:
    """Retrait combiné : ponctuation et chiffres, puis noms propres.

    C'est ce que ``assemble.build`` applique aux contrastes contrôlés. Les deux
    filtres visent la même chose sous deux formes : empêcher le modèle de
    reconnaître **le corpus** au lieu de la langue.

    L'ordre compte : la ponctuation est retirée **avant** les noms propres, sans
    quoi un token comme ``لطفي!!`` n'est pas reconnu comme entité.
    """
    return strip_entities(strip_punctuation(text))


__all__ = [
    "AMBIGUOUS_EXCLUDED",
    "CITIES",
    "COUNTRIES",
    "ENTITIES",
    "MEDIA",
    "PERSON_NAMES",
    "clean_for_training",
    "count_entities",
    "is_entity",
    "stem",
    "strip_entities",
    "strip_punctuation",
]
