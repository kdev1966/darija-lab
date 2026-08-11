"""L'agrégation doit refuser les amalgames qui produiraient un chiffre flatteur."""

from __future__ import annotations

from darija_bench.report import Cell, aggregate, render
from darija_bench.runner import Reply
from darija_bench.scoring import Verdict


def _reply(script="arabe", error=None, pid="p1"):
    return Reply(
        prompt_id=pid,
        model="m",
        condition="implicite",
        script=script,
        reply="x",
        error=error,
    )


def _verdict(script="arabe", scorable=True, score=0.9, above=True, markers=3, pid="p1"):
    return Verdict(
        prompt_id=pid,
        model="m",
        condition="implicite",
        script=script,
        n_words=30,
        scorable=scorable,
        score=score if scorable else None,
        above_classifier=above if scorable else None,
        n_markers=markers if scorable else None,
        is_tunisian=(above and markers >= 2) if scorable else None,
    )


def test_arabizi_et_arabe_ne_sont_jamais_fusionnes():
    # Les scores sur l'Arabizi passent par une translittération approximative
    # (`barcha` → `بارشا`, que le regex des marqueurs ne reconnaît même pas).
    # Les moyenner avec l'écriture arabe donnerait un chiffre unique faux.
    replies = [_reply(script="arabe"), _reply(script="arabizi", pid="p2")]
    verdicts = [_verdict(script="arabe"), _verdict(script="arabizi", pid="p2")]
    cells = aggregate(replies, verdicts)
    assert {c.script for c in cells} == {"arabe", "arabizi"}
    assert all(c.n == 1 for c in cells)


def test_non_scorables_comptes_a_part_pas_comme_echecs():
    # Une réponse trop courte est indécidable, pas fausse. La compter comme un
    # échec pénaliserait un modèle laconique pour une raison qui n'est pas la
    # langue.
    replies = [_reply(pid="p1"), _reply(pid="p2")]
    verdicts = [_verdict(pid="p1"), _verdict(pid="p2", scorable=False)]
    (cell,) = aggregate(replies, verdicts)
    assert cell.n == 2
    assert cell.n_scored == 1
    assert cell.n_unscorable == 1
    assert cell.tunisian_rate == 1.0  # calculé sur les scorables uniquement


def test_erreurs_dappel_comptees_mais_hors_taux():
    # Un refus ou une panne réseau n'est pas une performance linguistique.
    replies = [_reply(pid="p1"), _reply(pid="p2", error="refus")]
    verdicts = [_verdict(pid="p1")]
    (cell,) = aggregate(replies, verdicts)
    assert cell.n == 2
    assert cell.n_errors == 1
    assert cell.tunisian_rate == 1.0


def test_bande_limite_isolee():
    # Une réponse au-dessus du seuil du classifieur mais sans marqueurs est
    # exactement le profil des deux fusha conversationnelles qui ont franchi
    # le seuil (0,842 et 0,867). Le rapport doit la compter à part pour
    # signaler soit une dérive vers la fusha, soit une règle mal calibrée.
    replies = [_reply(pid="p1"), _reply(pid="p2")]
    verdicts = [_verdict(pid="p1", markers=4), _verdict(pid="p2", markers=0)]
    (cell,) = aggregate(replies, verdicts)
    assert cell.n_scored == 2
    assert cell.n_classifier_only == 1
    assert cell.tunisian_rate == 0.5


def test_sous_le_seuil_nest_pas_dans_la_bande_limite():
    # La bande « limite » ne concerne que ce qui passe le classifieur. Un texte
    # rejeté par le classifieur est simplement non tunisien, pas litigieux.
    replies = [_reply(pid="p1")]
    verdicts = [_verdict(pid="p1", above=False, markers=0)]
    (cell,) = aggregate(replies, verdicts)
    assert cell.n_classifier_only == 0
    assert cell.tunisian_rate == 0.0


def test_couverture():
    replies = [_reply(pid=f"p{i}") for i in range(4)]
    verdicts = [_verdict(pid="p0"), _verdict(pid="p1")]
    (cell,) = aggregate(replies, verdicts)
    assert cell.coverage == 0.5


def test_rendu_signale_la_reserve_arabizi():
    # La réserve doit voyager avec le tableau : un chiffre recopié sans elle
    # serait sur-interprété.
    cells = [Cell("m", "implicite", "arabizi", 1, 0, 0, 1, 0.9, 1.0)]
    texte = render(cells, threshold=0.838)
    assert "translitteration approximative" in texte
    assert "0.838" in texte


def test_rendu_sans_resultat():
    assert render([], threshold=0.838) == "aucun résultat"
