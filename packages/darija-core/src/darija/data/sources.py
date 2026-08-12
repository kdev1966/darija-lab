"""Registre des sources de données, avec leur rôle, leur licence et leur budget.

Une source a un **rôle**, et c'est la partie qui compte :

``positive``
    Du tunisien. Ce que le modèle doit apprendre à reconnaître.

``negative``
    Ce qui n'est **pas** du tunisien. Indispensable : un classifieur contrastif
    n'apprend pas « voici du tunisien », il apprend « ceci plutôt que cela ». Un
    modèle qui n'aurait jamais vu autre chose répondrait « tunisien » à tout.

Le choix des négatifs détermine exactement ce que le modèle saura faire. Avec du
MSA seul, il apprend « dialecte contre arabe standard » et étiquettera du
marocain comme tunisien — les deux dialectes sont proches. C'est pourquoi
``ary`` (marocain) est ici la source la plus utile bien qu'elle soit la plus
petite : c'est elle qui rend la tâche difficile, donc le modèle discriminant.

Les licences sont reportées telles que déclarées à la source. **Une licence
absente n'est pas une licence permissive** : par défaut, tous droits réservés.
Voir :data:`SOURCES` et le champ ``license``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Role = Literal["positive", "negative"]
Kind = Literal["wikipedia", "hf", "url", "local"]


@dataclass(frozen=True)
class Source:
    """Une source de texte à récupérer.

    Attributes:
      key: identifiant court, sert de nom de fichier.
      role: ``positive`` (tunisien) ou ``negative`` (contre-exemple).
      kind: mécanisme de récupération. ``local`` = produit par ce dépôt et non
        téléchargé ; ``fetch`` le saute.
      locator: langue du wiki, dépôt Hugging Face, ou URL directe.
      max_bytes: plafond d'octets **compressés** lus. Le flux est interrompu
        au-delà, ce qui borne à la fois la bande passante et le disque.
        ``None`` = tout lire.
      include: pour ``kind="hf"``, préfixes de chemin à retenir dans le dépôt.
        Vide = tout prendre. Nécessaire depuis que LinTO est distribué comme
        agrégat de dix-sept sous-corpus : en avaler l'intégralité ferait entrer
        TSAC — déjà une source distincte ici — des deux côtés du découpage
        train/test. C'est le biais nº 5 par la porte de derrière.
      license: licence déclarée, ou ``None`` si la source n'en déclare aucune.
      note: précision utile avant usage.

    """

    key: str
    role: Role
    kind: Kind
    locator: str
    max_bytes: int | None = None
    include: tuple[str, ...] = ()
    license: str | None = None
    note: str = ""

    @property
    def is_licensed(self) -> bool:
        """Vrai si la source déclare explicitement une licence."""
        return bool(self.license)


MB = 1024 * 1024

#: Le registre. Les plafonds correspondent au profil « complet » (~600 Mo) :
#: tout le tunisien disponible, plus des négatifs abondants qu'on rééchantillonne
#: à la construction plutôt qu'au téléchargement.
SOURCES: dict[str, Source] = {
    # ------------------------------------------------ positif : le tunisien
    "linto": Source(
        key="linto", role="positive", kind="hf",
        # L'ancien dépôt `linagora/linto-dataset-text-ar-tn` a disparu : le
        # `fetch` échouait donc sur toute machine neuve, rendant le modèle de
        # référence irreproductible. Le dépôt vivant agrège dix-sept
        # sous-corpus, d'où le filtre ci-dessous.
        locator="linagora/Tunisian_Derja_Dataset",
        include=(
            "Derja_tunsi/", "TunBERT/", "TuDiCOI/", "Tweet_TN/",
            "TunSwitchTunisiaOnly/", "TunSwitchCodeSwitching/",
            "Sentiment_Derja/", "TunisianSentimentAnalysis/",
            "TA_Segmentation/", "Tunisian_Dialectic_English_Derja/",
            "MADAR_TunisianDialect/",
        ),
        # `TSAC/` est exclu : le dépôt le récupère déjà comme source distincte,
        # et le laisser entrer ici mettrait les mêmes textes des deux côtés du
        # découpage train/test. `QADI_TunisianDialect/` est exclu aussi — QADI
        # ne distribue que des identifiants de tweets, voir docs/HANDOVER.md.
        #
        # ⚠️ `HkayetErwi/` est exclu pour la MÊME raison, et l'oubli a coûté
        # cher : c'est le corpus de vérité terrain du banc, celui qui a servi à
        # recalibrer `MIN_DISTINCT_MARKERS` et qui fournit l'ancre haute. Le
        # laisser ici en faisait un positif d'entraînement. Mesuré avant
        # correction : 16,8 % de ses lignes étaient dans `linto.txt`, et les
        # blocs concernés étaient reconnus à 96,0 % contre 92,7 % pour les
        # blocs jamais vus. L'ancre, elle, n'a pas bougé — 0,9187 sur les seuls
        # blocs propres contre 0,9189 publié. Un corpus de validation ne
        # s'entraîne pas, même de 17 %.
        license="CC BY-SA 4.0",
        note="Agrégat de sous-corpus tunisiens. La licence est CC BY-**SA** : "
             "le partage à l'identique s'impose aux œuvres dérivées, ce qui "
             "engage ce que ce dépôt pourra publier. `HkayetErwi/` est du "
             "récit — le registre où le classifieur décroche le plus.",
    ),
    "arbml_tn": Source(
        key="arbml_tn", role="positive", kind="hf",
        locator="arbml/Tunisian_Dialect_Corpus",
        license=None,
        note="~50k lignes de Twitter, arabe et Arabizi mêlés. Licence non "
             "documentée sur la fiche du jeu.",
    ),
    "tsac": Source(
        key="tsac", role="positive", kind="url",
        locator=" ".join(
            f"https://raw.githubusercontent.com/fbougares/TSAC/master/{f}.txt"
            for f in ("train_pos", "train_neg", "test_pos", "test_neg")
        ),
        license="LGPL-3.0",
        note="~17k commentaires Facebook (radios et TV tunisiennes, 2015-2016), "
             "annotés en polarité. Corpus éclaté en quatre fichiers. Citer "
             "Medhaffar et al., WANLP 2017.",
    ),
    "tunizi": Source(
        key="tunizi", role="positive", kind="url",
        locator="https://raw.githubusercontent.com/chaymafourati/"
                "TUNIZI-Sentiment-Analysis-Tunisian-Arabizi-Dataset/master/TUNIZI-Dataset.txt",
        license=None,
        note="Arabizi pur, commentaires YouTube. Aucune licence dans le dépôt. "
             "L'article annonce 9k+ phrases, le fichier livré fait 160 ko — "
             "vérifier ce qui est réellement obtenu.",
    ),
    # ------------------------------------ négatif : ce qui n'est pas tunisien
    "mar_latin": Source(
        key="mar_latin", role="negative", kind="hf",
        locator="ilias-brh/english-darija-arabizi-sentence-pairs",
        license="CC BY-NC-SA 4.0",
        note="Marocain en écriture latine — le seul négatif d'Arabizi trouvé "
             "sous licence déclarée. 6,7 M de mots, Arabizi authentique "
             "(95,7 % des lignes portent des chiffres-lettres, densité 0,242 "
             "par mot, contre 0,209 pour TUNIZI). ⚠️ **NC** : usage non "
             "commercial, la contrainte la plus stricte du dépôt. ⚠️ Registre "
             "non contrôlé : 23 mots par ligne (phrases traduites) contre 8 "
             "pour TUNIZI (commentaires YouTube).",
    ),
    "llm_fusha": Source(
        key="llm_fusha", role="negative", kind="local",
        locator="produit par apps/darija-bench/notebooks/colab_negatif_adversarial.ipynb",
        license="sorties de modèles, produites par ce dépôt",
        note="Négatif adversarial : de l'arabe standard écrit par un LLM "
             "(Qwen2.5-7B, T4 gratuit) sur des sujets du quotidien — le registre "
             "exact du biais nº 7, que ni `ar` (encyclopédique) ni `ary` "
             "(marocain) ne représentent.\n"
             "**631 blocs, 26 prompts sur 34.** Les 8 autres sont "
             "`llm_fusha_val` et ne servent qu'à juger. Le partage est fait par "
             "prompt, pas par réponse : 16 réponses tirées d'une même consigne "
             "sont des quasi-doublons, les répartir des deux côtés mesurerait la "
             "mémorisation plutôt que la généralisation.\n"
             "Nettoyage obligatoire, voir `darija_bench.adversarial` : le modèle "
             "ré-émet parfois la consigne — **écrite en tunisien** — après un "
             "jeton de gabarit.",
    ),
    "llm_fusha_val": Source(
        key="llm_fusha_val", role="negative", kind="local",
        locator="même carnet, les 8 prompts tenus à l'écart",
        license="sorties de modèles, produites par ce dépôt",
        note="**Jeu de validation. N'apparaît dans aucun contraste, et c'est "
             "délibéré** — un test le verrouille. C'est le seul témoin du "
             "registre qui a révélé le biais nº 7 ; s'il entrait à "
             "l'entraînement, plus rien ne mesurerait le progrès sur ce "
             "registre.\n"
             "224 blocs. Verdict mesuré sur lui seul : le modèle de référence "
             "en classe **4,9 % comme tunisiens**, `vs_maghreb_llm` **0,0 %**.",
    ),
    "omcd": Source(
        key="omcd", role="negative", kind="url",
        locator=" ".join(
            "https://raw.githubusercontent.com/kabilessefar/"
            f"OMCD-Offensive-Moroccan-Comments-Dataset/main/{f}.csv"
            for f in ("train", "test")
        ),
        license=None,
        note="~8k commentaires YouTube marocains. LE négatif de contrôle : même "
             "genre que TUNIZI (YouTube) et TSAC (Facebook), donc une AUC "
             "obtenue contre lui mesure le dialecte et non le registre. "
             "Réserve : le jeu est orienté langage offensant, donc un biais "
             "thématique subsiste — bien plus faible que commentaire/encyclopédie. "
             "Aucune licence dans le dépôt.",
    ),
    "mac": Source(
        key="mac", role="negative", kind="url",
        locator="https://raw.githubusercontent.com/LeMGarouani/MAC/main/MAC%20corpus.csv",
        license=None,
        note="18k tweets marocains annotés en sentiment. Thématiquement varié, "
             "là où OMCD est concentré sur le langage offensant : c'est le "
             "négatif qui réduit la fuite thématique. Aucune licence déclarée.",
    ),
    "dz": Source(
        key="dz", role="negative", kind="url",
        locator="https://raw.githubusercontent.com/kinmokusu/oea_algd/master/data/dataset/data.csv",
        license=None,
        note="~6k messages algériens (Twitter). LE test de finesse : l'algérien "
             "est le voisin le plus proche du tunisien. Le CSV source contient "
             "des données personnelles (nom, pseudo, âge) — seule la colonne "
             "Post est extraite. Aucune licence déclarée.",
    ),
    "ary": Source(
        key="ary", role="negative", kind="wikipedia", locator="ary",
        max_bytes=None, license="CC BY-SA 4.0",
        note="Wikipédia en darija marocaine, ~18 Mo, lue intégralement. "
             "LE négatif décisif : c'est le plus proche du tunisien, donc "
             "celui qui force le modèle à apprendre le tunisien et non "
             "« du maghrébin ».",
    ),
    "arz": Source(
        key="arz", role="negative", kind="wikipedia", locator="arz",
        max_bytes=294 * MB, license="CC BY-SA 4.0",
        note="Wikipédia en arabe égyptien. Dialecte, mais oriental : sépare "
             "le maghrébin de l'égyptien.",
    ),
    "ar": Source(
        key="ar", role="negative", kind="wikipedia", locator="ar",
        max_bytes=183 * MB, license="CC BY-SA 4.0",
        note="Wikipédia en arabe standard. Le flux est coupé à 183 Mo "
             "compressés — inutile d'avaler les 1,9 Go du dump complet, un "
             "classifieur sur n-grammes sature bien avant.",
    ),
    "ar_source": Source(
        key="ar_source", role="negative", kind="wikipedia", locator="arwikisource",
        max_bytes=60 * MB, license="domaine public / CC BY-SA 4.0",
        note="Wikisource en arabe : prose classique, en grande partie "
             "narrative. C'est l'ancre négative du **registre du récit**, que "
             "`ar` ne pouvait pas fournir — l'encyclopédie est hors registre, "
             "donc un repère faussé pour juger un conte. Les textes sont "
             "anciens et dans le domaine public ; seul l'appareil du wiki "
             "relève de CC BY-SA.",
    ),
}

#: Il n'existe **pas** de Wikipédia en tunisien. Seuls l'arabe standard (``ar``),
#: l'égyptien (``arz``) et le marocain (``ary``) disposent d'un wiki. C'est une
#: contrainte réelle pour la classe positive, mais elle se retourne en atout :
#: ces trois wikis sont des négatifs libres, volumineux et déjà propres.
NO_TUNISIAN_WIKIPEDIA = True


def by_role(role: Role) -> list[Source]:
    """Toutes les sources d'un rôle donné."""
    return [s for s in SOURCES.values() if s.role == role]


def unlicensed() -> list[Source]:
    """Sources sans licence déclarée — à vérifier avant tout usage public."""
    return [s for s in SOURCES.values() if not s.is_licensed]


def budget() -> dict[str, object]:
    """Budget de téléchargement, en octets compressés.

    ``None`` pour une source dont la taille n'est pas bornée a priori (dépôts
    Hugging Face, fichiers GitHub).
    """
    known = {k: s.max_bytes for k, s in SOURCES.items() if s.max_bytes}
    return {
        "capped_bytes": sum(known.values()),
        "capped_mb": round(sum(known.values()) / MB, 1),
        "per_source": {k: round(v / MB, 1) for k, v in known.items()},
        "uncapped": [k for k, s in SOURCES.items() if not s.max_bytes],
    }


__all__ = [
    "MB",
    "NO_TUNISIAN_WIKIPEDIA",
    "SOURCES",
    "Kind",
    "Role",
    "Source",
    "budget",
    "by_role",
    "unlicensed",
]
