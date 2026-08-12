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


# ---------------------------------------- la dispersion de l'ancre elle-meme
def test_l_ancre_haute_est_une_mediane_pas_un_plafond():
    # 57 % se lisait « a moitie tunisien » alors que 13 % du recit tunisien
    # AUTHENTIQUE score plus bas. Le quartile superieur du corpus humain est a
    # 122 % : si ces bornes tombaient sous 100 %, la barre redeviendrait
    # trompeuse dans l'autre sens.
    s = anchors.spread()
    assert s["q3"] > 1.0, "la moitie du corpus humain est au-dela de la mediane"
    assert s["q1"] < 1.0 < s["q3"]
    assert s["p10"] < s["q1"]


def test_un_texte_a_57_pourcent_reste_dans_la_fourchette_humaine():
    # Le cas qui a motive tout ceci : un texte tunisien manifeste (3 marqueurs
    # discriminants, 0,8801 pour un seuil de 0,8381) affichait 57 % et se
    # lisait comme un demi-echec. Il est a un ecart-type sous la mediane.
    phrase = anchors.qualify(0.57)
    assert "fourchette" in phrase
    assert "dixième" not in phrase


def test_les_bornes_sont_ordonnees_comme_les_scores():
    assert anchors.HUMAIN_P10 < anchors.HUMAIN_Q1 < anchors.HAUT < anchors.HUMAIN_Q3


def test_chaque_zone_a_sa_phrase():
    # Quatre zones, quatre lectures distinctes : sans ca, la legende ne
    # traduirait pas ce que la barre montre.
    s = anchors.spread()
    phrases = {
        anchors.qualify(s["p10"] - 0.1),
        anchors.qualify((s["p10"] + s["q1"]) / 2),
        anchors.qualify(1.0),
        anchors.qualify(s["q3"] + 0.1),
    }
    assert len(phrases) == 4
