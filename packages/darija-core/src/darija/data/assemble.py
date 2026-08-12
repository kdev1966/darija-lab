"""Assemblage des jeux d'entraînement, à partir du cache local.

Deux traitements sont faits ici plutôt qu'au téléchargement, pour qu'on puisse
les rejouer sans retélécharger :

**Le regroupement en blocs.** ``dialect.MIN_WORDS`` vaut 25 : sous ce seuil un
score de n-grammes n'a pas convergé et sature. Or une ligne de Wikipédia ou un
commentaire Facebook font souvent moins. Les lignes consécutives d'une même
source sont donc agrégées en blocs d'environ :data:`TARGET_WORDS` mots.

**L'équilibrage.** Le corpus tunisien disponible pèse nettement moins que les
dumps Wikipédia. Entraîner sur 1 positif pour 5 négatifs biaise le modèle vers
la classe majoritaire ; on sous-échantillonne donc la classe la plus grosse au
volume de la plus petite. C'est fait ici, à la construction, précisément pour
que le cache reste complet et rééchantillonnable.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..normalize import Level, normalize
from .entities import clean_for_training
from .fetch import DEFAULT_CACHE, load
from .sources import SOURCES

#: Taille de bloc visée, en mots. Confortablement au-dessus de
#: ``dialect.MIN_WORDS`` (25), pour que tout échantillon soit décidable.
TARGET_WORDS: int = 60

#: ⚠️ BIAIS DE GENRE — à lire avant d'interpréter la moindre AUC.
#:
#: Les sources positives disponibles sont des **commentaires de réseaux
#: sociaux** (TSAC : Facebook ; TUNIZI : YouTube) et les négatives des
#: **articles d'encyclopédie** (Wikipédia). Ces deux populations diffèrent par
#: bien plus que le dialecte : longueur, registre, ponctuation, vocabulaire, et
#: l'alphabet même — TUNIZI est à 99,9 % en caractères latins, Wikipédia à 0 %.
#:
#: Mesuré sur ce montage : AUC de 1.000, y compris après avoir restreint les
#: deux classes à l'alphabet arabe. **Cela ne prouve pas que le modèle
#: distingue le tunisien du marocain** — il peut très bien séparer
#: « commentaire » de « article » sans rien avoir appris du dialecte.
#:
#: Une AUC parfaite sur ce montage est un signal d'alarme, pas un résultat. Pour
#: mesurer réellement la discrimination dialectale il faut le **même genre des
#: deux côtés** : des tweets tunisiens contre des tweets marocains, que
#: fournissent QADI ou NADI. Tant que ce n'est pas fait, traitez ces modèles
#: comme des détecteurs de domaine.
GENRE_CONFOUND = (
    "positifs = réseaux sociaux, négatifs = Wikipédia : une AUC élevée peut "
    "mesurer le genre plutôt que le dialecte. Voir CONTRASTS."
)

@dataclass(frozen=True)
class Contrast:
    """Une comparaison à entraîner.

    Attributes:
      description: ce que le modèle est censé apprendre.
      negatives: clés des sources servant de contre-exemples.
      positives: clés des sources tunisiennes à utiliser. ``None`` = toutes.
        Restreindre est ce qui permet de contrôler le genre.
      genre_controlled: vrai si les deux classes proviennent du **même type de
        support**. Une AUC n'est interprétable comme mesure de dialecte que
        dans ce cas.
      strip_entities: retirer les noms propres (pays, villes, chaînes) avant
        l'entraînement. Sans ce filtre, ``مغرب`` et ``تونسي`` figurent parmi les
        traits les plus lourds : le modèle classe alors le sujet autant que la
        langue.
      fold_arabizi: projeter les deux classes dans l'alphabet que l'Arabizi
        sait exprimer (:func:`darija.normalize.fold_for_arabizi`). Destiné aux
        modèles qui scoreront du texte **translittéré** : le latin ne distingue
        pas ``س``/``ص`` ni ``ت``/``ط``, et l'entraînement doit voir la même
        confusion que l'entrée. Mesuré : 64 % → 88 % de TUNIZI reconnu, à AUC
        et faux positifs inchangés.
      latin_only: le symétrique, pour les contrastes en Arabizi. Sans lui, un
        contraste en écriture latine laisserait entrer des lignes en alphabet
        arabe et le modèle apprendrait l'alphabet — le biais nº 2, dans
        l'autre sens.
      arabic_only: ne garder que les lignes majoritairement en alphabet arabe.
        Indispensable dès qu'une classe contient de l'Arabizi et pas l'autre :
        TUNIZI est à 99,9 % en caractères latins et OMCD à 0 %, donc sans ce
        filtre le modèle sépare les alphabets et non les dialectes.

    """

    description: str
    negatives: list[str]
    positives: list[str] | None = None
    genre_controlled: bool = False
    arabic_only: bool = False
    latin_only: bool = False
    fold_arabizi: bool = False
    strip_entities: bool = False


#: ⚠️ DIVERSITÉ DE PROVENANCE — le facteur le plus déterminant, et mesuré.
#:
#: Un modèle entraîné sur une **seule** provenance tunisienne apprend ce corpus,
#: pas la langue. Validé sur LinTO, une quatrième provenance jamais vue :
#:
#: ===========================  ==============  =============
#: positifs                     médiane LinTO   bien classés
#: ===========================  ==============  =============
#: TSAC seul                    0.576           89,6 %
#: TSAC + ARBML                 **0.695**       **99,8 %**
#: ===========================  ==============  =============
#:
#: Une AUC interne de 0,998 coexistait avec 70 % seulement sur du tunisien
#: d'ailleurs. Les contrastes contrôlés tirent donc leurs positifs de **toutes**
#: les provenances disponibles.
#:
#: À noter : TUNIZI ne contribue à aucun contraste ``arabic_only`` — il est
#: intégralement en Arabizi et le filtre d'alphabet l'élimine entièrement.
#: ⚠️ REGISTRE — le pendant de la provenance, mesuré de la même façon.
#:
#: Un modèle entraîné uniquement sur des réseaux sociaux ne transfère pas à la
#: prose formelle : il classait « tunisien » 25,6 % du marocain **encyclopédique**
#: (``ary``), contre 0,1 % du marocain de réseaux sociaux. La cause n'est pas le
#: dialecte mais le registre, jamais vu.
#:
#: Le correctif équilibre les registres **des deux côtés** — LinTO (parole
#: transcrite, prose formelle) chez les positifs, ``ary`` chez les négatifs :
#:
#: =========================  ==========  ==========
#: entraînement               ary classé  LinTO
#:                            tunisien
#: =========================  ==========  ==========
#: social seul                25,6 %      94,5 %
#: registres équilibrés       **0,0 %**   **99,6 %**
#: =========================  ==========  ==========
#:
#: Contrepartie mesurée : la poésie populaire (registre littéraire ancien) tombe
#: de 55,6 % à 43,2 %. Ce registre n'est dans aucune classe ; le modèle ne le
#: couvre pas, et ne le prétend pas.
REGISTER_MATTERS = (
    "entraîné sur des réseaux sociaux seuls, le modèle prend la prose formelle "
    "marocaine pour du tunisien (25,6 %) ; équilibrer les registres le corrige"
)

PROVENANCE_MATTERS = (
    "une seule provenance tunisienne = le modèle apprend le corpus, pas la "
    "langue (89,6 % contre 99,8 % sur une provenance tenue à l'écart)"
)

#: Les comparaisons disponibles. ``vs_moroccan_yt`` est la seule dont l'AUC
#: mesure réellement le dialecte : commentaires YouTube des deux côtés. Les
#: autres opposent des réseaux sociaux à une encyclopédie et sont donc sujettes
#: au biais décrit dans :data:`GENRE_CONFOUND`.
CONTRASTS: dict[str, Contrast] = {
    "vs_moroccan_yt": Contrast(
        "tunisien contre marocain, commentaires YouTube des deux côtés "
        "— la seule comparaison à genre contrôlé",
        negatives=["omcd"], positives=["tsac", "tunizi", "arbml_tn", "linto"],
        genre_controlled=True, arabic_only=True, strip_entities=True,
    ),
    "vs_moroccan_tw": Contrast(
        "tunisien contre marocain, réseaux sociaux des deux côtés, sujets variés "
        "— corrige la fuite thématique d'OMCD",
        negatives=["mac"], positives=["tsac", "tunizi", "arbml_tn", "linto"],
        genre_controlled=True, arabic_only=True, strip_entities=True,
    ),
    "vs_algerian": Contrast(
        "tunisien contre algérien — le voisin le plus proche, le test de finesse",
        negatives=["dz"], positives=["tsac", "tunizi", "arbml_tn", "linto"],
        genre_controlled=True, arabic_only=True, strip_entities=True,
    ),
    "vs_maghreb": Contrast(
        "tunisien contre marocain ET algérien, registres équilibrés des deux "
        "côtés — le contraste de référence",
        negatives=["omcd", "mac", "dz", "ary"], positives=["tsac", "tunizi", "arbml_tn", "linto"],
        genre_controlled=True, arabic_only=True, strip_entities=True,
    ),
    "vs_maghreb_arabizi": Contrast(
        "tunisien contre marocain et algérien, dans l'alphabet reduit que "
        "l'Arabizi sait exprimer — le modèle à employer sur du texte translittéré",
        negatives=["omcd", "mac", "dz", "ary"],
        positives=["tsac", "arbml_tn", "linto"],
        genre_controlled=True, arabic_only=True, strip_entities=True,
        fold_arabizi=True,
        # Identique au contraste de référence, à un détail près : les deux
        # classes sont projetées dans l'alphabet que le latin peut noter. Sans
        # ça, le classifieur voit à l'entraînement des distinctions que la
        # translittération lui retire à l'usage — et 24 points de TUNIZI se
        # perdent dans l'écart.
    ),
    "vs_maghreb_llm": Contrast(
        "le contraste de référence, plus de l'arabe standard écrit par un LLM "
        "sur des sujets du quotidien — le registre du biais nº 7",
        negatives=["omcd", "mac", "dz", "ary", "llm_fusha"],
        positives=["tsac", "tunizi", "arbml_tn", "linto"],
        genre_controlled=True, arabic_only=True, strip_entities=True,
        # Un premier essai avec 141 blocs n'avait rien donné : après équilibrage
        # ils pesaient 0,5 % de la classe négative. Le gain annoncé alors venait
        # d'une mesure faite sur les blocs vus à l'entraînement — une erreur, et
        # la raison pour laquelle `llm_fusha_val` existe désormais.
        # `llm_fusha_val` n'est PAS ici : c'est le juge, il ne s'entraîne pas.
    ),
    "vs_moroccan_latin": Contrast(
        "tunisien contre marocain, en Arabizi des deux côtés — le premier "
        "contraste qui mesure l'écriture latine au lieu de la traduire",
        negatives=["mar_latin"], positives=["tunizi"],
        latin_only=True, strip_entities=True,
        # `genre_controlled` reste FAUX, et il faut le lire comme un
        # avertissement : TUNIZI est du commentaire YouTube, le négatif de la
        # phrase traduite. Le découpage en blocs de 60 mots efface l'écart de
        # longueur, pas celui de registre. Une AUC élevée ici mesure donc
        # peut-être le registre autant que le dialecte — exactement le biais
        # nº 1. À valider provenance par provenance, jamais sur l'AUC seule.
    ),
    "vs_msa": Contrast("tunisien contre arabe standard", negatives=["ar"]),
    "vs_egyptian": Contrast("tunisien contre égyptien", negatives=["arz"]),
    "vs_maghrebi": Contrast(
        "tunisien contre marocain encyclopédique — biaisé par le genre",
        negatives=["ary"],
    ),
    "vs_all": Contrast(
        "tunisien contre tout le reste",
        negatives=["ar", "arz", "ary", "omcd", "mac", "dz"],
    ),
}


#: Part minimale de caractères arabes pour qu'une ligne passe le filtre
#: ``arabic_only``. 0,85 laisse entrer les emprunts isolés (« ok », « merci »)
#: sans laisser passer une ligne entière d'Arabizi.
MIN_ARABIC: float = 0.85

#: Part minimale de caractères latins pour ``latin_only``. Plus bas que
#: :data:`MIN_ARABIC` : l'Arabizi note des consonnes par des chiffres
#: (``3`` pour ع, ``7`` pour ح, ``9`` pour ق), que ``script_ratio`` classe
#: hors alphabet. Mesuré sur TUNIZI : part latine médiane de 0,96, mais la
#: queue descend bien plus bas sur les lignes riches en chiffres.
MIN_LATIN: float = 0.60


def _arabic_lines(lines: Sequence[str]) -> list[str]:
    """Ne garde que les lignes majoritairement en alphabet arabe."""
    from ..normalize import script_ratio  # noqa: PLC0415

    return [x for x in lines if script_ratio(x)["arabic"] >= MIN_ARABIC]


def _latin_lines(lines: Sequence[str]) -> list[str]:
    """Ne garde que les lignes majoritairement en alphabet latin.

    Le pendant de :func:`_arabic_lines`, pour les contrastes en Arabizi. Le
    seuil est plus bas parce que l'Arabizi mêle des chiffres-lettres — ``3``,
    ``7``, ``9`` — qui ne comptent pas comme latins alors qu'ils portent une
    part du signal.
    """
    from ..normalize import script_ratio  # noqa: PLC0415

    return [x for x in lines if script_ratio(x)["latin"] >= MIN_LATIN]


def chunk(lines: Sequence[str], target_words: int = TARGET_WORDS) -> list[str]:
    """Agrège des lignes consécutives en blocs d'environ ``target_words`` mots.

    Une ligne déjà assez longue est conservée telle quelle. Le dernier bloc est
    abandonné s'il n'atteint pas la moitié de la cible : un résidu trop court
    serait indécidable et ne ferait qu'ajouter du bruit.
    """
    out: list[str] = []
    buf: list[str] = []
    count = 0
    for line in lines:
        n = len(normalize(line, Level.STANDARD).split())
        if not n:
            continue
        buf.append(line)
        count += n
        if count >= target_words:
            out.append("\n".join(buf))
            buf, count = [], 0
    if buf and count >= target_words // 2:
        out.append("\n".join(buf))
    return out


@dataclass
class Dataset:
    """Un jeu prêt pour ``dialect.train`` puis ``dialect.evaluate``."""

    name: str
    description: str
    train_positive: list[str] = field(default_factory=list)
    train_negative: list[str] = field(default_factory=list)
    test_positive: list[str] = field(default_factory=list)
    test_negative: list[str] = field(default_factory=list)
    sources_positive: list[str] = field(default_factory=list)
    sources_negative: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, object]:
        """Vue sérialisable des effectifs."""
        return {
            "name": self.name,
            "description": self.description,
            "train": {"positive": len(self.train_positive),
                      "negative": len(self.train_negative)},
            "test": {"positive": len(self.test_positive),
                     "negative": len(self.test_negative)},
            "sources": {"positive": self.sources_positive,
                        "negative": self.sources_negative},
        }


#: Part maximale d'une classe qu'une seule provenance peut occuper.
#:
#: Le biais nº 5 — un modèle entraîné sur une seule provenance apprend le
#: corpus, pas la langue — était corrigé en *ajoutant* des provenances. Il
#: restait donc réintroductible sans que rien ne le signale : il a suffi que
#: le dépôt LinTO change d'adresse et passe de 80 000 à **2 020 697** lignes
#: pour que cette source seule pèse 98 % de la classe positive après
#: équilibrage. Les trois autres provenances tunisiennes disparaissaient, et
#: aucune AUC ne l'aurait montré.
#:
#: Le plafond rend l'accident impossible par construction, au lieu de dépendre
#: d'un ``--max-lines`` qu'il faut penser à passer.
MAX_SOURCE_SHARE: float = 0.5


def _capped(
    raw: dict[str, list[str]], target_words: int, rng: random.Random
) -> list[str]:
    """Découpe en blocs sans laisser une provenance écraser les autres.

    Args:
      raw: lignes par clé de source.
      target_words: taille de bloc visée.
      rng: générateur, pour que la troncature soit reproductible.

    Returns:
      Les blocs de toutes les sources, chacune bornée à
      :data:`MAX_SOURCE_SHARE` du total.

    """
    par_source = {k: chunk(v, target_words) for k, v in raw.items()}
    par_source = {k: v for k, v in par_source.items() if v}
    # Une part >= 1 ne borne rien, et la formule y divise par zéro. Sortir tôt
    # rend le plafond désactivable proprement — ce qui a servi à mesurer son
    # effet réel plutôt qu'à le supposer.
    if len(par_source) < 2 or MAX_SOURCE_SHARE >= 1.0:
        return [b for v in par_source.values() for b in v]

    # Le plafond se calcule sur le total des AUTRES sources : une source ne
    # peut pas dépasser ce que le reste du corpus apporte.
    for cle, blocs in par_source.items():
        autres = sum(len(v) for k, v in par_source.items() if k != cle)
        plafond = int(autres * MAX_SOURCE_SHARE / (1 - MAX_SOURCE_SHARE))
        if len(blocs) > plafond:
            rng.shuffle(blocs)
            par_source[cle] = blocs[:plafond]
    return [b for v in par_source.values() for b in v]


def build(
    contrast: str = "vs_all",
    cache: Path = DEFAULT_CACHE,
    *,
    target_words: int = TARGET_WORDS,
    balance: bool = True,
    holdout: float = 0.25,
    seed: int = 0,
) -> Dataset:
    """Construit un jeu équilibré pour l'une des comparaisons de :data:`CONTRASTS`.

    Args:
      contrast: clé de :data:`CONTRASTS`.
      cache: répertoire du cache alimenté par ``fetch_all``.
      target_words: taille de bloc visée.
      balance: sous-échantillonner la classe majoritaire au volume de l'autre.
      holdout: part réservée à l'évaluation. Le découpage précède l'équilibrage,
        donc le test n'est jamais contaminé par des blocs vus à l'entraînement.
      seed: graine, pour un découpage reproductible.

    Raises:
      KeyError: comparaison inconnue.
      FileNotFoundError: cache vide pour l'une des deux classes.

    """
    if contrast not in CONTRASTS:
        raise KeyError(f"comparaison inconnue {contrast!r} ; connues : {sorted(CONTRASTS)}")
    spec = CONTRASTS[contrast]
    description, neg_keys = spec.description, spec.negatives

    pos_all = load(cache, role="positive")
    pos_raw = (
        {k: v for k, v in pos_all.items() if k in spec.positives}
        if spec.positives
        else pos_all
    )
    neg_all = load(cache, role="negative")
    neg_raw = {k: v for k, v in neg_all.items() if k in neg_keys}

    if not pos_raw:
        raise FileNotFoundError(
            f"aucune source positive dans {cache} — lancez d'abord `darija data fetch`"
        )
    if not neg_raw:
        raise FileNotFoundError(
            f"aucune des sources négatives {neg_keys} dans {cache}"
        )

    if spec.strip_entities:
        pos_raw = {k: [clean_for_training(x) for x in v] for k, v in pos_raw.items()}
        neg_raw = {k: [clean_for_training(x) for x in v] for k, v in neg_raw.items()}

    if spec.fold_arabizi:
        from ..normalize import fold_for_arabizi  # noqa: PLC0415

        pos_raw = {k: [fold_for_arabizi(x) for x in v] for k, v in pos_raw.items()}
        neg_raw = {k: [fold_for_arabizi(x) for x in v] for k, v in neg_raw.items()}

    if spec.arabic_only or spec.latin_only:
        keep = _arabic_lines if spec.arabic_only else _latin_lines
        pos_raw = {k: keep(v) for k, v in pos_raw.items()}
        neg_raw = {k: keep(v) for k, v in neg_raw.items()}
        pos_raw = {k: v for k, v in pos_raw.items() if v}
        neg_raw = {k: v for k, v in neg_raw.items() if v}

    rng = random.Random(seed)
    pos = _capped(pos_raw, target_words, rng)
    neg = _capped(neg_raw, target_words, rng)
    rng.shuffle(pos)
    rng.shuffle(neg)

    def split(xs: list[str]) -> tuple[list[str], list[str]]:
        cut = int(len(xs) * (1 - holdout))
        return xs[:cut], xs[cut:]

    tr_pos, te_pos = split(pos)
    tr_neg, te_neg = split(neg)

    if balance:
        k = min(len(tr_pos), len(tr_neg))
        tr_pos, tr_neg = tr_pos[:k], tr_neg[:k]

    return Dataset(
        name=contrast, description=description,
        train_positive=tr_pos, train_negative=tr_neg,
        test_positive=te_pos, test_negative=te_neg,
        sources_positive=sorted(pos_raw), sources_negative=sorted(neg_raw),
    )


def score_by_source(
    model: object,
    cache: Path = DEFAULT_CACHE,
    *,
    target_words: int = TARGET_WORDS,
    arabic_only: bool = True,
    limit: int | None = 4000,
) -> dict[str, dict[str, object]]:
    """Score un modèle **source par source**, y compris celles jamais vues.

    C'est le seul diagnostic qui distingue « a appris la langue » de « a appris
    ce corpus ». Une AUC interne élevée peut parfaitement coexister avec un
    effondrement sur du tunisien d'une autre provenance — c'est exactement ce
    qui a été mesuré ici : 0,998 en interne, 70 % seulement sur ARBML.

    Args:
      model: un ``DialectModel`` entraîné.
      cache: répertoire du cache.
      target_words: taille des blocs évalués.
      arabic_only: n'évaluer que les lignes en alphabet arabe, comme à
        l'entraînement des contrastes contrôlés.
      limit: nombre maximal de blocs par source.

    Returns:
      ``{clé: {"role", "n", "median", "above_half"}}`` — ``above_threshold`` est la
      part de blocs classés du côté positif, au **seuil appris** du modèle.

    """
    import statistics  # noqa: PLC0415

    out: dict[str, dict[str, object]] = {}
    for key, lines in load(cache).items():
        cleaned = [clean_for_training(x) for x in lines]
        if arabic_only:
            cleaned = _arabic_lines(cleaned)
        blocks = chunk(cleaned, target_words)[:limit]
        if not blocks:
            continue
        scores = [model.score(b) for b in blocks]  # type: ignore[attr-defined]
        out[key] = {
            "role": SOURCES[key].role,
            "n": len(scores),
            "median": round(statistics.median(scores), 4),
            "above_threshold": round(
                sum(1 for s in scores if s >= getattr(model, "threshold", 0.5))
                / len(scores), 4
            ),
        }
    return out


def available(cache: Path = DEFAULT_CACHE) -> dict[str, object]:
    """Ce que contient le cache, et ce qu'on peut donc construire."""
    have = {k: len(v) for k, v in load(cache).items()}
    pos = [k for k in have if SOURCES[k].role == "positive"]
    return {
        "cached": have,
        "positive_sources": pos,
        "buildable": [
            name for name, c in CONTRASTS.items()
            if any(k in have for k in (c.positives or pos))
            and any(n in have for n in c.negatives)
        ],
        "genre_controlled": [n for n, c in CONTRASTS.items() if c.genre_controlled],
    }


__all__ = [
    "CONTRASTS",
    "GENRE_CONFOUND",
    "MIN_ARABIC",
    "PROVENANCE_MATTERS",
    "REGISTER_MATTERS",
    "Contrast",
    "TARGET_WORDS",
    "Dataset",
    "available",
    "score_by_source",
    "build",
    "chunk",
]
