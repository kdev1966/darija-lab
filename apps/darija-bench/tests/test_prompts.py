"""Le jeu de prompts est le protocole : une erreur dedans invalide la campagne."""

from __future__ import annotations

import pytest

from darija_bench import prompts as prompts_mod


def test_le_jeu_se_charge():
    items = prompts_mod.load()
    assert len(items) >= 40


def test_identifiants_uniques():
    # `load` indexe par identifiant et le rapport agrège dessus : un doublon
    # écraserait silencieusement une mesure au lieu de lever.
    ids = [p.id for p in prompts_mod.load()]
    assert len(ids) == len(set(ids))


def test_arabizi_represente():
    # L'Arabizi est la forme écrite majoritaire du tunisien et le trou le plus
    # net du classifieur (0 bloc à l'entraînement, `arabic_only` ayant éliminé
    # 99,9 % de TUNIZI). Un banc qui ne l'évaluerait pas raterait le cas le
    # plus fréquent.
    items = prompts_mod.load()
    arabizi = [p for p in items if p.script == "arabizi"]
    assert len(arabizi) >= 5


def test_aucun_prompt_en_francais_ni_en_fusha():
    # Ce qu'on mesure est la réaction du modèle à la langue qu'on lui adresse.
    # Un prompt en français mesurerait autre chose.
    for prompt in prompts_mod.load():
        assert prompt.text.strip()
        assert not prompt.text.strip().startswith(("Peux-tu", "Comment", "Pourquoi"))


# --- défaut 2 : le protocole complet dépasse les plafonds gratuits ---


def test_echantillon_garde_toujours_de_larabizi():
    # Mesuré : le protocole complet demande 84 appels, le palier gratuit en
    # accorde 20 par jour. Il faut donc pouvoir échantillonner — mais un
    # tirage uniforme sur 42 prompts dont 8 en Arabizi peut n'en retenir
    # aucun, et on perdrait la forme écrite majoritaire du tunisien sans
    # que le rapport le signale.
    items = prompts_mod.load()
    for n in (4, 6, 10, 20):
        for seed in range(5):
            tire = prompts_mod.sample(items, n, seed=seed)
            scripts = {p.script for p in tire}
            assert "arabizi" in scripts, f"n={n} seed={seed} : aucun Arabizi"
            assert "arabe" in scripts


def test_echantillon_reproductible():
    # Deux modèles mesurés à des jours différents — obligatoire quand le
    # plafond est journalier — doivent voir le MÊME échantillon, sinon leur
    # comparaison n'a aucun sens.
    items = prompts_mod.load()
    a = [p.id for p in prompts_mod.sample(items, 10, seed=7)]
    b = [p.id for p in prompts_mod.sample(items, 10, seed=7)]
    c = [p.id for p in prompts_mod.sample(items, 10, seed=8)]
    assert a == b
    assert a != c


def test_echantillon_respecte_la_taille():
    items = prompts_mod.load()
    for n in (3, 5, 12, 33):
        assert len(prompts_mod.sample(items, n)) == n


def test_echantillon_plus_grand_que_le_jeu():
    items = prompts_mod.load()
    assert len(prompts_mod.sample(items, 999)) == len(items)


def test_echantillon_taille_invalide():
    with pytest.raises(ValueError, match="invalide"):
        prompts_mod.sample(prompts_mod.load(), 0)


def test_ecriture_inconnue_refusee(tmp_path):
    # Une écriture non reconnue passerait le scoring sans translittération et
    # produirait une ligne de rapport muette.
    path = tmp_path / "p.jsonl"
    path.write_text(
        '{"id": "x-1", "category": "test", "script": "cyrillique", "text": "..."}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="écriture inconnue"):
        prompts_mod.load(path)


def test_doublon_refuse(tmp_path):
    path = tmp_path / "p.jsonl"
    ligne = '{"id": "x-1", "category": "test", "script": "arabe", "text": "شنوة"}\n'
    path.write_text(ligne * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="dupliqué"):
        prompts_mod.load(path)
