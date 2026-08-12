"""Vérité terrain : du récit tunisien authentique, sous licence.

Jusqu'ici l'instrument n'avait été calibré que sur six textes écrits par son
auteur et sur des sorties de modèles sans étiquette. Les deux sont de mauvais
juges : les premiers reflètent la main qui les a écrits, les secondes sont
précisément ce qu'on cherche à évaluer.

``HkayetErwi`` est du récit tunisien humain, publié dans l'agrégat LinTO sous
**CC BY-SA 4.0**. C'est le premier positif de référence dont ce banc dispose.

**Le texte n'est pas versé dans le dépôt, et c'est délibéré.** Le partage à
l'identique s'imposerait alors aux œuvres dérivées, donc à ce dépôt entier.
Les tests se sautent quand le corpus n'est pas déjà en cache local — la CI ne
télécharge rien et n'a rien à redistribuer.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from darija_bench.scoring import MIN_DISTINCT_MARKERS

REPO = "linagora/Tunisian_Derja_Dataset"
FICHIER = "HkayetErwi/train-00000-of-00001.parquet"

MODEL_PATH = Path(
    os.environ.get(
        "DARIJA_MODEL",
        Path(__file__).resolve().parents[3] / "packages/darija-core/models/vs_maghreb.json.gz",
    )
)

#: Part minimale de récit tunisien authentique que la règle doit conserver.
#: En dessous, la règle rejette la langue qu'elle est censée reconnaître.
#:
#: Abaissé de 0,85 à 0,72 en connaissance de cause. La décision ne compte plus
#: que les marqueurs **discriminants** : le rappel mesuré passe de 88,0 % à
#: 76,9 %, mais le taux de déclenchement sur la fusha tombe de 67,1 % à 2,0 %.
#: Onze points de rappel contre soixante-cinq de précision sur le registre où
#: le classifieur trébuche — voir markers.DISCRIMINANT.
GARDE_MINIMALE = 0.72


def _blocs():
    """Charge le corpus depuis le cache local, ou saute le test."""
    try:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
    except ImportError:  # pragma: no cover
        pytest.skip("extra [data] de darija-core absent")

    from darija.data.assemble import chunk

    try:
        chemin = hf_hub_download(
            REPO, FICHIER, repo_type="dataset", local_files_only=True
        )
    except Exception:  # noqa: BLE001 - toute indisponibilité vaut saut
        pytest.skip(
            f"{REPO}:{FICHIER} absent du cache local. "
            "Le corpus n'est pas versé dans le dépôt (CC BY-SA)."
        )
    lignes = [str(x) for x in pq.read_table(chemin).column("text").to_pylist() if x]
    return chunk(lignes)


@pytest.fixture(scope="module")
def terrain():
    """Blocs de récit authentique et leurs mesures."""
    from darija import markers
    from darija.dialect import DialectModel

    if not MODEL_PATH.exists():
        pytest.skip(f"modèle absent : {MODEL_PATH}")
    blocs = _blocs()
    modele = DialectModel.load(MODEL_PATH)
    scores = [modele.score(b) for b in blocs]
    # Meme regle que la decision : seuls les discriminants comptent.
    marqueurs = [
        len({m.marker for m in markers.find(b)} & markers.DISCRIMINANT) for b in blocs
    ]
    return modele, scores, marqueurs


def test_le_classifieur_reconnait_le_recit_authentique(terrain):
    # Contredit le diagnostic qui avait cours avant cette mesure. On croyait
    # que le classifieur decrochait sur le registre narratif, d'apres les
    # sorties de modeles. Sur du recit humain il tient tres bien : c'est donc
    # la production des modeles qui est reellement moins tunisienne, pas
    # l'instrument qui serait aveugle.
    modele, scores, _ = terrain
    passe = sum(1 for s in scores if s >= modele.threshold) / len(scores)
    assert passe >= 0.90, f"seulement {passe:.1%} du recit authentique reconnu"


def test_la_regle_ne_rejette_pas_la_langue_quelle_mesure(terrain):
    # Le minimum de marqueurs avait ete fixe a 2 sur six textes de fusha ecrits
    # a la main. Sur 432 blocs authentiques, ce reglage rejetait 37 % du
    # tunisien reel pour eviter un unique faux positif. Ce test empeche de
    # reserrer la vis sans revenir a la verite terrain.
    modele, scores, marqueurs = terrain
    garde = sum(
        1
        for s, k in zip(scores, marqueurs, strict=True)
        if s >= modele.threshold and k >= MIN_DISTINCT_MARKERS
    ) / len(scores)
    assert garde >= GARDE_MINIMALE, (
        f"MIN_DISTINCT_MARKERS={MIN_DISTINCT_MARKERS} ne conserve que "
        f"{garde:.1%} du tunisien authentique"
    )


def test_le_recit_authentique_est_loin_de_la_frontiere(terrain):
    # Le pendant du diagnostic : si le recit humain se pressait sur le seuil
    # comme les sorties de modeles, l'instrument serait en cause. Il ne s'y
    # presse pas — moins de 15 % dans la bande d'indecision.
    from darija_bench.report import BORDERLINE

    modele, scores, _ = terrain
    proches = sum(1 for s in scores if abs(s - modele.threshold) < BORDERLINE)
    assert proches / len(scores) < 0.15
