"""Ce que la page doit dire, et pourquoi c'est elle qui doit le dire.

Ce module n'avait aucun test, et c'est là que les deux derniers défauts du banc
se sont logés — tous deux invisibles à la mesure, qui était juste :

1. la barre présentait l'ancre haute comme un plafond alors que c'est une
   médiane, faisant lire « 57 % » comme « à moitié tunisien » ;
2. le tableau listait cinq marqueurs quand le compte au-dessus en annonçait
   trois, sans dire lesquels.

Une erreur de présentation ne casse aucun calcul. Seul un test qui interroge la
**vue** l'attrape.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from darija_bench import anchors, web

MODEL_PATH = Path(
    os.environ.get(
        "DARIJA_MODEL",
        Path(__file__).resolve().parents[3] / "packages/darija-core/models/vs_maghreb.json.gz",
    )
)

# Le texte qui a révélé les deux défauts : du tunisien manifeste, mais soutenu.
SOUTENU = (
    "فمّا شيخ كبير في الحومة، نسمّوه ديما عمّ صلاح الدين. وجهو ديما ضاحك "
    "و البشاشة ما تفارقش جبينو. الحومة لكلها تعتبرو كما الأب، الصغير والكبير "
    "يمشيلو بش يستشيروه في أكبر مشاكلهم. ديما قاعد على كرسي خشب قديم قدام "
    "الدار، بفنجان قهوة عربي وسبحة في يدّو. كلامو رزين وموزون، كل كلمة يقولها "
    "فيها حكمة تخليك تخمّم لأيّامات. هو الروح متاع الحومة اللّي بلاش بيه تضيع "
    "الأصالة متاعنا."
)


@pytest.fixture(scope="module")
def modele():
    """Le classifieur de référence, ou saut si ``models/`` n'est pas construit."""
    from darija.dialect import DialectModel

    if not MODEL_PATH.exists():
        pytest.skip(f"modèle absent : {MODEL_PATH} (voir `darija data train`)")
    return DialectModel.load(MODEL_PATH)


def test_le_tableau_dit_quels_marqueurs_decident(modele):
    # Le tableau listait 5 marqueurs quand le compte au-dessus disait 3, sans
    # aucun moyen de savoir lesquels. `ن-` note le « je » tunisien mais aussi
    # le « nous » de la fusha, et `اللي` est cinq fois plus frequent en
    # marocain : ils sont reconnus, ils ne decident pas.
    d = web.Measure(SOUTENU, modele).as_dict()
    decident = [m["name"] for m in d["markers"] if m["decides"]]
    assert d["n_markers"] == len(decident), "le compte affiche doit etre celui du tableau"
    assert len(d["markers"]) > len(decident), "ce texte porte des marqueurs non decisifs"
    assert all(
        not m["decides"] for m in d["markers"] if m["name"] == "relativizer_elli"
    )


def test_la_dispersion_de_l_ancre_est_transmise_a_la_page(modele):
    # Sans ces bornes, la barre laisse lire « 57 % » comme « a moitie
    # tunisien » alors que 13 % du recit tunisien AUTHENTIQUE score plus bas.
    d = web.Measure(SOUTENU, modele).as_dict()
    assert d["spread"] == anchors.spread()
    assert d["qualifier"] == anchors.qualify(d["position"])


def test_un_texte_tunisien_soutenu_reste_classe_tunisien(modele):
    # Le fond n'a jamais ete en cause : trois marqueurs discriminants, score
    # au-dessus du seuil. C'est l'affichage qui trompait. Ce test verrouille le
    # fait qu'aucune correction d'affichage n'a touche au verdict.
    d = web.Measure(SOUTENU, modele).as_dict()
    assert d["status"] == "mesure"
    assert d["is_tunisian"] is True
    assert d["n_markers"] >= 3


def test_un_texte_trop_court_est_indecidable_pas_faux(modele):
    # `predict` rend None sous `min_words`. La page doit le dire, pas inventer.
    d = web.Measure("برشا باهي", modele).as_dict()
    assert d["status"] == "trop_court"
    assert "is_tunisian" not in d
    assert d["min_words"] == modele.min_words


def test_un_texte_vide_ne_produit_aucun_score(modele):
    assert web.Measure("   ", modele).as_dict()["status"] == "vide"


def test_la_reponse_est_serialisable(modele):
    # La page consomme du JSON ; un objet non sérialisable ferait une page
    # blanche sans message d'erreur lisible.
    json.dumps(web.Measure(SOUTENU, modele).as_dict())


def test_la_page_expose_les_reperes_qu_elle_promet():
    # La legende annonce une bande et un dixieme ; s'ils disparaissaient du
    # gabarit, le texte resterait et mentirait.
    assert 'class="band"' in web.PAGE
    assert 'class="p10"' in web.PAGE
    assert "médiane" in web.PAGE
    assert "décide" in web.PAGE
