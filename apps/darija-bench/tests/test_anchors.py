"""Les repères qui rendent un score interprétable."""

from __future__ import annotations

from darija_bench import anchors


def test_les_ancres_bornent_lechelle():
    assert anchors.position(anchors.BAS) == 0.0
    assert anchors.position(anchors.HAUT) == 1.0


def test_lechelle_nest_pas_ecretee():
    # Un modèle plus dialectal que le corpus humain est une information, pas
    # une anomalie : écrêter à 100 % effacerait le fait. Gemini 3.6 flash en
    # condition explicite mesure 106 %.
    assert anchors.position(0.95) > 1.0
    assert anchors.position(0.80) < 0.0


def test_lancre_basse_est_du_recit_pas_de_lencyclopedie():
    # C'est la raison d'être du module. Wikipédia arabe score 0,786 et le récit
    # classique 0,829 : 0,043 d'écart pour la même langue. Ancrer un conte sur
    # l'encyclopédie ajouterait l'écart de registre à l'écart de langue et
    # gonflerait toute mesure — le biais nº 1 du dépôt, à l'envers.
    assert anchors.BAS > anchors.ENCYCLOPEDIQUE
    assert anchors.BAS - anchors.ENCYCLOPEDIQUE > 0.03


def test_les_deux_ancres_encadrent_le_seuil_du_classifieur():
    # Le seuil appris (0,838) doit tomber entre les deux repères humains,
    # sinon l'échelle et la décision parleraient de choses différentes.
    assert anchors.BAS < 0.8381 < anchors.HAUT
