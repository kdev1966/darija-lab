r"""Classifieur contrastif de dialecte, sur n-grammes de caractères.

La méthode
----------
Un rapport de vraisemblance logarithmique sur des n-grammes de **caractères** :

.. math::

    s(x) = \frac{1}{|G(x)|} \sum_{g \in G(x)}
           \log \frac{P(g \mid \text{positif})}{P(g \mid \text{négatif})}

avec lissage additif. Les 4-grammes de caractères l'emportent nettement sur les
unigrammes de mots parce que le tunisien se marque *morphologiquement* — préfixe
``n-``, circumfixe ``ما...ش``, relativiseur ``اللي``, clitique ``ع`` — et que ces
motifs sont sous-lexicaux, donc insensibles à l'orthographe instable du dialecte
écrit.

Discrimination mesurée sur corpus tunisien réel :

=========================  ==========
trait                      AUC
=========================  ==========
**4-grammes de car. (LLR)**  **0.960**
unigrammes de mots (LLR)     0.947
taux de marqueurs            0.771
longueur moyenne des mots    0.659
ratio type/token             0.223
=========================  ==========

Deux propriétés qui contraignent l'usage
----------------------------------------
**C'est un classifieur binaire contrastif, pas un détecteur absolu.** Il apprend
« positif plutôt que ce négatif-là ». Un modèle entraîné contre du MSA ne dit
rien d'utile face à de l'algérien. Choisissez la classe négative en fonction de
la décision que vous devez prendre, et réentraînez quand elle change.

**Il exige du texte.** En dessous de :data:`MIN_WORDS` mots, le score sature et
devient ininterprétable : trop peu de n-grammes pour que la moyenne se stabilise.
:meth:`DialectModel.predict` renvoie ``None`` dans ce cas plutôt qu'un chiffre
faussement confiant. C'est une erreur constatée sur un modèle réel, où un extrait
de 10 mots d'arabe standard obtenait le même score maximal qu'un texte
authentiquement dialectal.

Aucun modèle pré-entraîné n'est fourni : il faut vos données.
"""

from __future__ import annotations

import gzip
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .normalize import Level, normalize

#: Ordre des n-grammes. 4 gagne le compromis : 3 est trop générique, 5 commence
#: à mémoriser des mots entiers et s'emballe sur du texte recopié.
NGRAM: int = 4

#: Lissage additif pour les n-grammes non vus.
ALPHA: float = 0.5

#: Nombre maximal de traits conservés, par |poids| x support.
MAX_FEATURES: int = 60_000

#: Longueur minimale, en mots, pour qu'un score soit interprétable.
#: En deçà, la moyenne des LLR n'a pas convergé et sature contre les bornes de
#: calibration.
MIN_WORDS: int = 25


def char_ngrams(text: str, n: int = NGRAM) -> list[str]:
    """n-grammes de caractères du texte normalisé, frontières de mots marquées.

    Les espaces deviennent ``_`` et le tout est encadré d'espaces, si bien qu'un
    n-gramme peut représenter un préfixe (``_نم``) ou un suffixe (``ش_``) —
    exactement les positions où la morphologie dialectale se manifeste.
    """
    s = " " + normalize(text, Level.STANDARD).replace(" ", "_") + " "
    if len(s) < n:
        return []
    return [s[i : i + n] for i in range(len(s) - n + 1)]


@dataclass(frozen=True)
class DialectModel:
    """Poids LLR sur n-grammes de caractères, plus la calibration.

    Attributes:
      weights: n-gramme → log P(g|positif) − log P(g|négatif).
      n: ordre des n-grammes.
      lo, hi: ancres de calibration ramenant le LLR brut dans ``[0, 1]``.
      threshold: seuil de décision **appris**, et non 0,5 codé en dur. Les
        ancres ``lo``/``hi`` préservent l'ordre mais pas la position de la
        frontière : ``lo`` dépend du minimum de la classe négative, donc d'une
        seule valeur extrême. Un seul négatif atypique suffisait à décaler tout
        le reste au-dessus de 0,5 et à faire classer « tunisien » l'intégralité
        des contre-exemples — AUC intacte, ``predict`` inutilisable.
      min_words: seuil sous lequel :meth:`predict` refuse de conclure.
      labels: ``(nom_positif, nom_négatif)``, pour des sorties lisibles.
      meta: provenance (effectifs, AUC à la construction).

    """

    weights: Mapping[str, float]
    n: int
    lo: float
    hi: float
    threshold: float
    min_words: int
    labels: tuple[str, str]
    meta: Mapping[str, object]

    def raw(self, text: str) -> float:
        """LLR moyen sur les n-grammes. Les n-grammes inconnus comptent 0."""
        grams = char_ngrams(text, self.n)
        if not grams:
            return self.lo
        w = self.weights
        return sum(w.get(g, 0.0) for g in grams) / len(grams)

    def score(self, text: str) -> float:
        """Score calibré dans ``[0, 1]`` : 1 = classe positive.

        Ne vérifie **pas** la longueur. Pour une décision, utilisez
        :meth:`predict`.
        """
        if self.hi <= self.lo:
            return 0.0
        return max(0.0, min(1.0, (self.raw(text) - self.lo) / (self.hi - self.lo)))

    def predict(self, text: str) -> tuple[str, float] | None:
        """Étiquette et confiance, ou ``None`` si le texte est trop court.

        Returns:
          ``(label, confiance)`` où ``confiance`` est dans ``[0, 1]``, ou
          ``None`` si le texte fait moins de :attr:`min_words` mots — auquel cas
          le score n'est pas interprétable et aucune décision ne doit être prise.

        """
        if len(normalize(text, Level.STANDARD).split()) < self.min_words:
            return None
        s = self.score(text)
        pos, neg = self.labels
        t = self.threshold
        if s >= t:
            return (pos, 0.5 + 0.5 * (s - t) / max(1e-9, 1.0 - t))
        return (neg, 0.5 + 0.5 * (t - s) / max(1e-9, t))

    def save(self, path: str | Path) -> None:
        """Écrit le modèle en JSON gzippé."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(p, "wt", encoding="utf-8") as fh:
            json.dump(
                {
                    "weights": dict(self.weights), "n": self.n,
                    "lo": self.lo, "hi": self.hi, "threshold": self.threshold,
                    "min_words": self.min_words,
                    "labels": list(self.labels), "meta": dict(self.meta),
                },
                fh, ensure_ascii=False,
            )

    @classmethod
    def load(cls, path: str | Path) -> DialectModel:
        """Relit un modèle écrit par :meth:`save`."""
        with gzip.open(Path(path), "rt", encoding="utf-8") as fh:
            b = json.load(fh)
        return cls(
            weights=b["weights"], n=b["n"], lo=b["lo"], hi=b["hi"],
            threshold=b.get("threshold", 0.5),
            min_words=b.get("min_words", MIN_WORDS),
            labels=tuple(b.get("labels", ("positive", "negative"))),
            meta=b.get("meta", {}),
        )


def _best_threshold(pos: Sequence[float], neg: Sequence[float]) -> float:
    """Seuil maximisant l'indice de Youden (sensibilité + spécificité − 1).

    Balayage sur les scores observés eux-mêmes : sans hypothèse de forme, et
    exact sur l'échantillon d'entraînement.
    """
    if not pos or not neg:
        return 0.5
    best, best_j = 0.5, -2.0
    for t in sorted({*pos, *neg}):
        tpr = sum(1 for x in pos if x >= t) / len(pos)
        fpr = sum(1 for x in neg if x >= t) / len(neg)
        if (j := tpr - fpr) > best_j:
            best, best_j = t, j
    return best


def _percentile(xs: Sequence[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, int(q * len(s))))]


def train(
    positive: Sequence[str],
    negative: Sequence[str],
    *,
    labels: tuple[str, str] = ("positive", "negative"),
    n: int = NGRAM,
    alpha: float = ALPHA,
    max_features: int = MAX_FEATURES,
    min_words: int = MIN_WORDS,
    meta: Mapping[str, object] | None = None,
) -> DialectModel:
    """Ajuste le modèle contrastif.

    Args:
      positive: textes de la classe à reconnaître.
      negative: textes dont il faut la distinguer. **Ce choix définit ce que le
        modèle sait faire** — voir la note du module.
      labels: noms des deux classes, pour des sorties lisibles.
      n: ordre des n-grammes.
      alpha: lissage additif.
      max_features: n-grammes conservés, les plus informatifs d'abord.
      min_words: seuil de refus de :meth:`DialectModel.predict`.
      meta: provenance enregistrée dans l'artefact.

    Returns:
      Un :class:`DialectModel` calibré.

    Raises:
      ValueError: si l'une des deux classes est vide.

    """
    if not positive or not negative:
        raise ValueError("les deux classes doivent être non vides")

    cp: Counter[str] = Counter()
    cn: Counter[str] = Counter()
    for t in positive:
        cp.update(char_ngrams(t, n))
    for t in negative:
        cn.update(char_ngrams(t, n))

    vocab = set(cp) | set(cn)
    np_, nn_ = sum(cp.values()), sum(cn.values())
    v = len(vocab)
    weights: dict[str, float] = {
        g: math.log((cp[g] + alpha) / (np_ + alpha * v))
        - math.log((cn[g] + alpha) / (nn_ + alpha * v))
        for g in vocab
    }

    if len(weights) > max_features:
        ranked = sorted(
            weights.items(),
            key=lambda kv: abs(kv[1]) * math.log1p(cp[kv[0]] + cn[kv[0]]),
            reverse=True,
        )
        weights = dict(ranked[:max_features])

    probe = DialectModel(weights, n, 0.0, 1.0, 0.5, min_words, labels, {})
    pos_raw = [probe.raw(t) for t in positive]
    neg_raw = [probe.raw(t) for t in negative]

    # Le plancher est placé sous toute la distribution négative, pas à sa
    # médiane : le clipping est la seule étape qui détruit de l'information de
    # rang, et le placer sous min(neg) préserve exactement l'AUC.
    lo_raw = min(neg_raw)
    hi = _percentile(pos_raw, 0.90)
    lo = lo_raw - 0.10 * max(hi - lo_raw, 1e-6)

    calibrated = DialectModel(weights, n, lo, hi, 0.5, min_words, labels, {})
    threshold = _best_threshold(
        [calibrated.score(t) for t in positive], [calibrated.score(t) for t in negative]
    )

    m = dict(meta or {})
    m.update({
        "n_positive": len(positive), "n_negative": len(negative),
        "n_features": len(weights), "ngram": n,
        "labels": list(labels), "threshold": round(threshold, 4),
    })
    return DialectModel(weights, n, lo, hi, threshold, min_words, labels, m)


def auc(pos: Iterable[float], neg: Iterable[float]) -> float:
    """AUC ROC par les rangs ; les ex æquo comptent 0.5.

    Renvoie ``nan`` si l'une des deux listes est vide.
    """
    p, q = list(pos), list(neg)
    if not p or not q:
        return float("nan")
    wins = sum((1.0 if a > b else 0.5 if a == b else 0.0) for a in p for b in q)
    return wins / (len(p) * len(q))


def evaluate(
    model: DialectModel, positive: Sequence[str], negative: Sequence[str]
) -> dict[str, object]:
    """Évalue un modèle sur des données **tenues à l'écart de l'entraînement**.

    Rapporte aussi combien de textes sont trop courts pour être décidés : une
    AUC calculée sur des textes qu'on refuserait de classer en production est
    trompeuse.
    """
    ps = [model.score(t) for t in positive]
    ns = [model.score(t) for t in negative]
    t = model.threshold
    too_short = sum(
        1 for t in list(positive) + list(negative)
        if len(normalize(t, Level.STANDARD).split()) < model.min_words
    )
    return {
        "auc": round(auc(ps, ns), 4),
        "n_positive": len(ps),
        "n_negative": len(ns),
        "median_positive": round(sorted(ps)[len(ps) // 2], 4) if ps else None,
        "median_negative": round(sorted(ns)[len(ns) // 2], 4) if ns else None,
        "threshold": round(t, 4),
        "accuracy": round(
            (sum(1 for x in ps if x >= t) + sum(1 for x in ns if x < t))
            / max(1, len(ps) + len(ns)), 4
        ),
        "too_short": too_short,
        "too_short_pct": round(100 * too_short / max(1, len(ps) + len(ns)), 1),
    }


def train_test_split(
    texts: Sequence[str], holdout: float = 0.25, seed: int = 0
) -> tuple[list[str], list[str]]:
    """Découpe reproductible, pour ne jamais évaluer sur les données vues."""
    import random

    xs = list(texts)
    random.Random(seed).shuffle(xs)
    cut = int(len(xs) * (1 - holdout))
    return xs[:cut], xs[cut:]


__all__ = [
    "ALPHA",
    "MAX_FEATURES",
    "MIN_WORDS",
    "NGRAM",
    "DialectModel",
    "auc",
    "char_ngrams",
    "evaluate",
    "train",
    "train_test_split",
]
