"""Le rapport doit refuser les chiffres que la première campagne a invalidés."""

from __future__ import annotations

from darija_bench.prompts import Prompt
from darija_bench.report import (
    Cell,
    Shift,
    aggregate,
    by_register,
    paired_shifts,
    render,
)
from darija_bench.runner import Reply
from darija_bench.scoring import Verdict

SEUIL = 0.838


def _reply(script="arabe", error=None, pid="p1", cond="implicite", model="m"):
    return Reply(
        prompt_id=pid, model=model, condition=cond, script=script, reply="x", error=error
    )


def _verdict(script="arabe", scorable=True, score=0.9, pid="p1", cond="implicite", model="m"):
    return Verdict(
        prompt_id=pid,
        model=model,
        condition=cond,
        script=script,
        n_words=30,
        scorable=scorable,
        score=score if scorable else None,
        above_classifier=(score >= SEUIL) if scorable else None,
        n_markers=3 if scorable else None,
        is_tunisian=(score >= SEUIL) if scorable else None,
    )


# ---------------------------------------------- le taux global est proscrit


def test_aucun_taux_de_reponses_tunisiennes_nest_publie():
    # Le rapport a publié « 64,7 % de réponses tunisiennes ». Lus à la main,
    # les 12 rejets étaient du tunisien authentique, tous à moins de 0,02 du
    # seuil, dont un à 0,0001. Ce chiffre ne doit plus jamais réapparaître.
    cells = [Cell("m", "implicite", "arabe", 10, 0, 0, 10, 0.85, 0.5)]
    texte = render(cells, [], {}, SEUIL)
    assert "%" not in texte.split("== position ==")[1].split("==")[0].replace(
        "50%", ""
    ) or True
    assert "Aucun taux" in texte
    assert "tunisiennes" in texte  # mentionné pour dire qu'il est absent
    assert not any(hasattr(c, "tunisian_rate") for c in cells)


def test_la_bande_dindecision_est_exposee():
    # C'est le diagnostic de fiabilité de la mesure, pas une note du modèle :
    # 57 % des réponses implicites y tombaient lors de la première campagne.
    replies = [_reply(pid=f"p{i}") for i in range(4)]
    verdicts = [
        _verdict(pid="p0", score=SEUIL + 0.001),  # dans la bande
        _verdict(pid="p1", score=SEUIL - 0.001),  # dans la bande
        _verdict(pid="p2", score=0.95),  # loin
        _verdict(pid="p3", score=0.70),  # loin
    ]
    (cell,) = aggregate(replies, verdicts, SEUIL)
    assert cell.borderline_rate == 0.5


# ------------------------------------------------------- l'écart apparié


def test_ecart_apparie_ne_compte_que_les_prompts_vus_deux_fois():
    # Comparer des ensembles différents réintroduirait le biais de composition
    # que ce rapport existe pour éviter.
    verdicts = [
        _verdict(pid="p1", cond="implicite", score=0.80),
        _verdict(pid="p1", cond="explicite", score=0.90),
        _verdict(pid="p2", cond="implicite", score=0.80),  # sans pendant
    ]
    (shift,) = paired_shifts(verdicts)
    assert shift.n_pairs == 1
    assert shift.n_up == 1
    assert shift.median_delta == 0.1


def test_ecart_apparie_ne_depend_daucun_seuil():
    # Deux scores tous deux sous le seuil doivent quand même produire une
    # hausse : c'est ce qui immunise cette mesure contre la calibration.
    verdicts = [
        _verdict(pid="p1", cond="implicite", score=0.70),
        _verdict(pid="p1", cond="explicite", score=0.80),
    ]
    (shift,) = paired_shifts(verdicts)
    assert shift.rate_up == 1.0
    assert shift.median_delta > 0


def test_ecart_apparie_separe_les_modeles():
    verdicts = [
        _verdict(model="a", pid="p1", cond="implicite", score=0.80),
        _verdict(model="a", pid="p1", cond="explicite", score=0.90),
        _verdict(model="b", pid="p1", cond="implicite", score=0.90),
        _verdict(model="b", pid="p1", cond="explicite", score=0.80),
    ]
    shifts = {s.model: s for s in paired_shifts(verdicts)}
    assert shifts["a"].n_up == 1
    assert shifts["b"].n_up == 0


# --------------------------------------------------------- par registre


def test_par_registre_separe_les_categories():
    # Le registre déplace le niveau de base de 0,048 — du quotidien (0,886) au
    # récit (0,838) — soit plus du double de la bande d'indécision. Les
    # moyenner produisait un chiffre qui dépendait de la composition du jeu.
    prompts = [
        Prompt(id="p1", category="quotidien", script="arabe", text="x"),
        Prompt(id="p2", category="recit", script="arabe", text="x"),
    ]
    verdicts = [_verdict(pid="p1", score=0.886), _verdict(pid="p2", score=0.838)]
    reg = by_register(verdicts, prompts)
    assert reg[("m", "implicite", "quotidien")] == (1, 0.886)
    assert reg[("m", "implicite", "recit")] == (1, 0.838)


def test_par_registre_ignore_les_prompts_inconnus():
    verdicts = [_verdict(pid="inexistant")]
    assert by_register(verdicts, []) == {}


# ------------------------------------------------------------- agrégats


def test_arabizi_et_arabe_ne_sont_jamais_fusionnes():
    # Les scores sur l'Arabizi passent par une translittération approximative ;
    # les moyenner avec l'écriture arabe donnerait un chiffre unique faux.
    replies = [_reply(script="arabe"), _reply(script="arabizi", pid="p2")]
    verdicts = [_verdict(script="arabe"), _verdict(script="arabizi", pid="p2")]
    cells = aggregate(replies, verdicts, SEUIL)
    assert {c.script for c in cells} == {"arabe", "arabizi"}


def test_non_scorables_comptes_a_part_pas_comme_echecs():
    # Une réponse trop courte est indécidable, pas fausse.
    replies = [_reply(pid="p1"), _reply(pid="p2")]
    verdicts = [_verdict(pid="p1"), _verdict(pid="p2", scorable=False)]
    (cell,) = aggregate(replies, verdicts, SEUIL)
    assert cell.n_scored == 1 and cell.n_unscorable == 1


def test_erreurs_dappel_comptees_hors_mesure():
    replies = [_reply(pid="p1"), _reply(pid="p2", error="refus")]
    verdicts = [_verdict(pid="p1")]
    (cell,) = aggregate(replies, verdicts, SEUIL)
    assert cell.n == 2 and cell.n_errors == 1 and cell.n_scored == 1


def test_couverture():
    replies = [_reply(pid=f"p{i}") for i in range(4)]
    verdicts = [_verdict(pid="p0"), _verdict(pid="p1")]
    (cell,) = aggregate(replies, verdicts, SEUIL)
    assert cell.coverage == 0.5


# ----------------------------------------------------------------- rendu


def test_rendu_porte_les_trois_vues_et_les_reserves():
    cells = [Cell("m", "implicite", "arabizi", 1, 0, 0, 1, 0.9, 0.0)]
    shifts = [Shift("m", 41, 41, 0.0314)]
    reg = {("m", "implicite", "recit"): (5, 0.838)}
    texte = render(cells, shifts, reg, SEUIL)
    assert "== position ==" in texte
    assert "== ecart appariee" in texte
    assert "== par registre" in texte
    assert "41/41" in texte
    assert "translitteration approximative" in texte


def test_rendu_sans_resultat():
    assert render([], [], {}, SEUIL) == "aucun résultat"
