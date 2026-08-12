"""Le tri est l'usage principal de l'instrument : ses erreurs comptent double."""

from __future__ import annotations

import json

import pytest

from darija_bench.triage import Verdict, judge, read_documents, summarise


class _FauxModele:
    """Modele scriptable : un score fixe, un seuil fixe."""

    threshold = 0.84
    min_words = 25

    def __init__(self, score=0.95):
        self._score = score

    def score(self, _text):
        return self._score


TUNISIEN = (
    "برشا علاش قداش اللي باش نمشي للدار متاعي توا و ناكل شوية طعام باهي "
    "مع خويا و اختي و من بعد نرقد شوية قبل ما نخرج للخدمة و ياخي هكاكا "
    "تعدى النهار متاعي كامل بلا ما نحس بيه ياخي"
)


def test_un_document_trop_court_est_indecidable_pas_rejete():
    # La confusion serait grave : un texte bref n'est pas un texte etranger,
    # et l'amalgame fausserait tout decompte de tri.
    v = judge("برشا باهي", _FauxModele())
    assert v.verdict == "indecidable"
    assert v.median is None


def test_un_document_tunisien_est_reconnu():
    v = judge(TUNISIEN, _FauxModele(0.95))
    assert v.verdict == "tunisien"
    assert v.n_markers >= 1


def test_le_classifieur_peut_rejeter_malgre_les_marqueurs():
    # Le classifieur decide en premier. Un texte riche en marqueurs mais que
    # le modele place sous le seuil reste rejete : c'est ce qui a permis de
    # rejeter un recit marocain portant six marqueurs.
    v = judge(TUNISIEN, _FauxModele(0.50))
    assert v.verdict == "autre"


def test_le_verdict_sappuie_sur_la_mediane_des_blocs():
    # Les reperes et le seuil ont ete etablis sur des blocs de 60 mots. Juger
    # un long document d'un seul tenant comparerait des choses differentes.
    v = judge(TUNISIEN * 6, _FauxModele())
    assert v.n_blocks >= 2
    assert v.share_tunisian == 1.0


# ----------------------------------------------------------------- lecture
def test_lecture_jsonl(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text(
        json.dumps({"uid": "a", "text": "شنوة"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    assert list(read_documents(p)) == [("a", "شنوة")]


def test_lecture_txt_une_ligne_par_document(tmp_path):
    p = tmp_path / "c.txt"
    p.write_text("premier\n\nsecond\n", encoding="utf-8")
    assert [t for _, t in read_documents(p)] == ["premier", "second"]


def test_lecture_dossier(tmp_path):
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.txt").write_text("beta", encoding="utf-8")
    assert sorted(n for n, _ in read_documents(tmp_path)) == ["a.txt", "b.txt"]


def test_extension_inconnue_refusee(tmp_path):
    p = tmp_path / "c.parquet"
    p.write_bytes(b"x")
    with pytest.raises(ValueError, match="non reconnue"):
        list(read_documents(p))


def test_chemin_absent_signale(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(read_documents(tmp_path / "absent.jsonl"))


# ------------------------------------------------------------------ rendu
def test_le_rendu_distingue_indecidable_de_rejete():
    texte = summarise([
        Verdict("a", 100, 2, "tunisien", 0.9, 0.8, 1.0, 3),
        Verdict("b", 100, 2, "autre", 0.7, -0.2, 0.0, 0),
        Verdict("c", 5, 0, "indecidable"),
    ])
    assert "tunisien" in texte and "autre" in texte and "indecidable" in texte
    assert "PAS un rejet" in texte


def test_rendu_vide():
    assert summarise([]) == "aucun document"


def test_l_ecriture_latine_est_translitteree_avant_mesure():
    """TUNIZI est une source POSITIVE, et le tri la rejetait en bloc.

    `judge` envoyait le texte latin brut au classifieur, qui n'a jamais vu
    d'Arabizi — le filtre `arabic_only` l'ayant elimine a l'entrainement.
    Resultat mesure avant correctif : 0 % de tunisien et une position mediane
    de -41 % sur du tunisien authentique. Le reste de l'application appelait
    deja `prepare` ; le tri ne le faisait pas.
    """
    v = judge("chnowa a7welek ya sa7bi, rani mrigel barcha w enti chneya "
              "el jdid 3andek, 9ouli chnowa 3malt lbare7 fi el 3achiya "
              "w win msit m3a s7abek el kol", _FauxModele())
    assert v.transliterated, "l'ecriture latine doit etre signalee"
    assert v.n_words > 0


def test_le_rendu_signale_la_translitteration():
    # La reserve doit voyager avec le chiffre : une position obtenue apres
    # translitteration n'est pas comparable aux autres.
    texte = summarise([
        Verdict("a", 100, 2, "tunisien", 0.9, 0.8, 1.0, 3, transliterated=True),
    ])
    assert "translitteres" in texte and "approximative" in texte


def test_le_filtre_de_marqueurs_nest_pas_applique_par_defaut():
    """Il coute 10 a 37 points sur du texte humain et ne gagne rien sur les negatifs.

    Mesure sur les corpus du depot, part des blocs de 60 mots reconnus :
    linto 93,0 % -> 83,0 %, arbml_tn 85,9 % -> 56,6 %, tsac 86,8 % -> 49,8 %.
    Les contre-exemples, eux, restent a 0-0,7 % avec ou sans. Le filtre visait
    la fusha CONVERSATIONNELLE, propre aux sorties de LLM ; un corpus de
    tweets tunisiens n'en contient pas.
    """
    sans_marqueur = "الطقس اليوم جميل و الشمس طالعة و الناس خرجوا " * 6
    v = judge(sans_marqueur, _FauxModele(0.95))
    assert v.n_markers == 0
    assert v.verdict == "tunisien", "le classifieur seul doit suffire par defaut"


def test_strict_retablit_le_filtre():
    sans_marqueur = "الطقس اليوم جميل و الشمس طالعة و الناس خرجوا " * 6
    assert judge(sans_marqueur, _FauxModele(0.95), strict=True).verdict == "autre"
