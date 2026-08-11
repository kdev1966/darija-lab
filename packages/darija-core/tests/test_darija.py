"""Tests du socle darija-core."""

import json

import pytest

from darija import arabizi, codeswitch, dialect, markers
from darija.normalize import Level, normalize, script_ratio


# ------------------------------------------------------------------ normalize
def test_strip_tatweel_and_diacritics():
    assert normalize("مـــاذا") == "ماذا"
    assert normalize("مَاذَا") == "ماذا"


def test_levels_differ_as_documented():
    text = "أنا مِن تُونس"
    light = normalize(text, Level.LIGHT)
    standard = normalize(text, Level.STANDARD)
    # LIGHT garde le porteur de hamza, STANDARD l'unifie.
    assert "أ" in light
    assert "أ" not in standard and standard.startswith("انا")


def test_dialect_words_survive_every_level():
    """Le coeur du module : ne jamais corriger le dialecte vers la fusha."""
    for word in ("برشا", "علاش", "قداش", "كيفاش", "اللي", "باهي"):
        for level in Level:
            assert word in normalize(word, level), (word, level)


def test_maghrebi_letters_survive_aggressive():
    """ڨ note /g/ en tunisien ; le supprimer casserait le mot."""
    assert "ڨ" in normalize("ڨلب", Level.AGGRESSIVE)
    assert "ڥ" in normalize("ڥيديو", Level.AGGRESSIVE)


def test_aggressive_reduces_elongation():
    assert normalize("برشاااااا", Level.AGGRESSIVE) == "برشاا"


def test_script_ratio_sums_to_one():
    r = script_ratio("مرحبا hello")
    assert abs(sum(r.values()) - 1.0) < 1e-9
    assert r["arabic"] > 0 and r["latin"] > 0


def test_script_ratio_empty():
    assert script_ratio("...") == {"arabic": 0.0, "latin": 0.0, "other": 0.0}


# -------------------------------------------------------------------- arabizi
@pytest.mark.parametrize(
    ("src", "expected"),
    [("9alb", "قالب"), ("7ob", "حوب"), ("enti", "انتي"), ("kha", "خا")],
)
def test_to_arabic_basic(src, expected):
    assert arabizi.to_arabic(src) == expected


def test_longest_match_wins():
    """3' doit l'emporter sur 3, et kh sur k+h."""
    assert arabizi.to_arabic("3'") == "غ"
    assert arabizi.to_arabic("3") == "ع"
    assert arabizi.to_arabic("kh") == "خ"
    assert arabizi.to_arabic("k") == "ك"


def test_e_is_positional():
    """e initial est une voyelle longue, e interne est bref donc non noté."""
    assert arabizi.to_arabic("enti").startswith("ا")
    assert "ا" not in arabizi.to_arabic("bhe")[1:]


def test_g_convention_is_configurable():
    assert arabizi.to_arabic("gal") == "ڨال"
    assert arabizi.to_arabic("gal", g_as_qaf=True) == "قال"


def test_to_arabizi_roundtrip_is_stable():
    az = arabizi.to_arabizi("قلب")
    assert az == "9lb"
    assert arabizi.to_arabic(az) == "قلب"


def test_arabizi_detection():
    assert arabizi.is_arabizi("chnowa a7welek ya sa7bi")
    assert arabizi.is_arabizi("3aslema kifach enti")
    assert not arabizi.is_arabizi("bonjour comment allez vous")
    assert not arabizi.is_arabizi("مرحبا كيف حالك")


def test_digits_alone_are_not_arabizi():
    """« iphone 13 » ne doit pas passer pour de l'Arabizi."""
    assert not arabizi.is_arabizi("i bought an iphone 13 yesterday")


def test_arabizi_score_bounds():
    for t in ("", "   ", "!!!", "abc", "3ala", "مرحبا"):
        assert 0.0 <= arabizi.arabizi_score(t) <= 1.0


# ----------------------------------------------------------------- codeswitch
def test_segments_index_back_into_source():
    text = "المشكلة أنو le service ماهوش disponible"
    for s in codeswitch.segment(text):
        assert text[s.start : s.end] == s.text


def test_french_and_arabic_are_separated():
    langs = {s.lang for s in codeswitch.segment("المشكلة أنو le service disponible")}
    assert "ar" in langs and "fr" in langs


def test_latin_run_is_split_between_french_and_arabizi():
    """Le cas d'usage central : alterner sans changer d'alphabet.

    « ken 3andek le temps ... chwaya de sucre » est une seule suite latine mais
    contient deux langues. L'étiqueter d'un bloc ferait manquer l'alternance.
    """
    segs = codeswitch.segment("ken 3andek le temps ajoutili chwaya de sucre")
    langs = [s.lang for s in segs]
    assert "arabizi" in langs and "fr" in langs
    assert len(segs) > 1, "la suite latine doit être découpée, pas étiquetée d'un bloc"
    assert segs[0].lang == "arabizi" and segs[0].text.startswith("ken")
    assert segs[-1].lang == "fr" and segs[-1].text.endswith("sucre")


def test_split_segments_still_index_back():
    text = "ken 3andek le temps de sucre"
    for s in codeswitch.segment(text):
        assert text[s.start : s.end] == s.text


def test_arabizi_beats_french_when_digits_present():
    """Un chiffre-lettre tranche, même à côté de mots français."""
    segs = codeswitch.segment("ken 3andek le temps")
    assert any(s.lang == "arabizi" for s in segs)


def test_profile_sums_to_one():
    p = codeswitch.profile("المشكلة أنو le service ماهوش disponible")
    assert abs(sum(p.values()) - 1.0) < 1e-6


def test_profile_empty_text():
    assert codeswitch.profile("") == {"ar": 0.0, "fr": 0.0, "arabizi": 0.0, "unknown": 0.0}


def test_code_switching_detected():
    assert codeswitch.is_code_switched("المشكلة أنو le service ماهوش disponible")
    assert not codeswitch.is_code_switched("المشكلة أنو الخدمة ماهيش موجودة")


def test_extract_by_language():
    got = codeswitch.extract("المشكلة le service ماهوش", "ar")
    assert got and all("le" not in g for g in got)


# -------------------------------------------------------------------- markers
def test_finds_negation_circumfix():
    found = {m.marker for m in markers.find("ماناكلش الخبز")}
    assert "negation_ma_sh" in found


def test_finds_relativizer_and_lexicon():
    found = {m.marker for m in markers.find("الراجل اللي جا برشا باهي")}
    assert {"relativizer_elli", "quant_barsha", "adj_behi"} <= found


def test_markers_work_after_normalization():
    """شنوة s'écrit شنوه une fois normalisé ; le motif doit suivre."""
    assert "interrog_chnowa" in {m.marker for m in markers.find("شنوة أحوالك")}


def test_rates_are_per_10k_words():
    text = " ".join(["اللي"] + ["كلمة"] * 99)
    assert markers.rates(text)["relativizer_elli"] == pytest.approx(100.0)


def test_score_bounds_and_empty():
    assert markers.score("") == 0.0
    assert markers.score("hello world") == 0.0
    assert 0.0 < markers.score("شنوة أحوالك برشا باهي اللي ماناكلش") <= 1.0


def test_modern_flag_separates_registers():
    """Le registre littéraire ancien n'emploie ni برشا ni شنوة."""
    ancient = "الراجل اللي جا وقالي ماناكلش وباش نمشي"
    assert markers.score(ancient) > 0
    assert markers.score(ancient, modern=True) == 0.0


def test_explain_is_readable():
    out = markers.explain("شنوة أحوالك برشا")
    assert "score=" in out and "quant_barsha" in out
    assert markers.explain("hello") == "aucun marqueur tunisien détecté"


# -------------------------------------------------------------------- dialect
def _corpus():
    pos = [
        "اللي جا لعندنا قالي باش نمشيو للدار وماناكلش قبل ما نوصلو وقتاش تحب تجي "
        "معانا للسوق نشريو شوية خضرة وحاجات أخرى برشا ناس تحب تجي معانا اليوم",
        "شنوة أحوالك اليوم يا صاحبي راني نحب نحكيلك على قصة صارتلي البارح كي كنت "
        "ماشي للخدمة ولقيت برشا ناس واقفين يستناو في الكار وماجاش حتى واحد",
        "علاش ماجيتش البارح كنا نستناو فيك برشا وقتاش باش تجي المرة الجاية قولنا "
        "باش نحضرو روحنا ونجيو معاك للمحل اللي حكيتلنا عليه توا",
        "قداش تحب في هالحاجة هاذي راني نحب نشريها أما ماعنديش برشا فلوس توا "
        "كيفاش نجم ندبر روحي باش نجيب الباقي في الأيام الجاية",
    ] * 6
    neg = [
        "إن الحمد لله نحمده ونستعينه ونستغفره ونعوذ بالله من شرور أنفسنا ومن سيئات "
        "أعمالنا من يهده الله فلا مضل له ومن يضلل فلا هادي له وأشهد أن لا إله إلا الله",
        "لقد أرسل الله الرسل مبشرين ومنذرين لكي لا يكون للناس على الله حجة بعد الرسل "
        "وكان الله عزيزا حكيما وهو الذي خلق السماوات والأرض بالحق",
        "يجب على الطلاب أن يجتهدوا في دراستهم حتى يحققوا النجاح الذي يصبون إليه "
        "وأن يحترموا أساتذتهم ويلتزموا بقواعد المؤسسة التعليمية التي ينتمون إليها",
        "تعتبر اللغة العربية من أقدم اللغات وأكثرها انتشارا في العالم وهي لغة القرآن "
        "الكريم التي حفظها الله من التحريف على مر العصور والأزمان",
    ] * 6
    return pos, neg


def test_train_and_separate():
    pos, neg = _corpus()
    m = dialect.train(pos, neg, labels=("tunisien", "msa"))
    assert dialect.auc([m.score(t) for t in pos], [m.score(t) for t in neg]) > 0.9


def test_predict_labels_correctly():
    pos, neg = _corpus()
    m = dialect.train(pos, neg, labels=("tunisien", "msa"))
    assert m.predict(pos[0])[0] == "tunisien"
    assert m.predict(neg[0])[0] == "msa"


def test_predict_refuses_short_text():
    """Le garde-fou : sous MIN_WORDS le score sature et ne veut rien dire."""
    pos, neg = _corpus()
    m = dialect.train(pos, neg, labels=("tunisien", "msa"))
    assert m.predict("شنوة أحوالك") is None
    assert m.score("شنوة أحوالك") is not None  # score() ne bloque pas, lui


def test_empty_class_rejected():
    with pytest.raises(ValueError, match="non vides"):
        dialect.train([], ["نص"])


def test_save_load_roundtrip(tmp_path):
    pos, neg = _corpus()
    m = dialect.train(pos, neg, labels=("tunisien", "msa"))
    p = tmp_path / "m.json.gz"
    m.save(p)
    back = dialect.DialectModel.load(p)
    assert back.labels == ("tunisien", "msa")
    assert back.n == m.n and back.min_words == m.min_words
    assert back.score(pos[0]) == pytest.approx(m.score(pos[0]))


def test_evaluate_reports_short_texts():
    pos, neg = _corpus()
    m = dialect.train(pos, neg)
    rep = dialect.evaluate(m, pos, [*neg, "قصير"])
    assert rep["auc"] >= 0.9
    assert rep["too_short"] >= 1
    json.dumps(rep)  # doit rester sérialisable


def test_auc_edge_cases():
    assert dialect.auc([1.0], [0.0]) == 1.0
    assert dialect.auc([0.5], [0.5]) == 0.5
    assert dialect.auc([], [1.0]) != dialect.auc([], [1.0])  # nan


def test_char_ngrams_mark_word_boundaries():
    grams = dialect.char_ngrams("نمشي", 4)
    assert any(g.startswith(" ") or "_" in g or g.endswith(" ") for g in grams)


def test_split_is_reproducible():
    a = dialect.train_test_split([str(i) for i in range(100)], seed=42)
    b = dialect.train_test_split([str(i) for i in range(100)], seed=42)
    assert a == b


# ------------------------------------------------------- seuil de décision
def test_threshold_is_learned_not_hardcoded():
    """Régression : 0.5 en dur classait « positif » l'intégralité des négatifs.

    `lo` s'ancre sous le minimum de la classe négative, donc une seule valeur
    extrême décale toute l'échelle. L'AUC reste intacte, mais la frontière se
    déplace : elle doit être apprise.
    """
    pos, neg = _corpus()
    m = dialect.train(pos, neg, labels=("tunisien", "msa"))
    assert 0.0 < m.threshold < 1.0
    assert m.predict(pos[0])[0] == "tunisien"
    assert m.predict(neg[0])[0] == "msa"


def test_threshold_survives_a_negative_outlier():
    """Un négatif atypique ne doit pas faire basculer toute la classe."""
    pos, neg = _corpus()
    m = dialect.train(pos, [*neg, "ZZZZ " * 40], labels=("tunisien", "msa"))
    wrong = sum(1 for t in neg if m.predict(t) and m.predict(t)[0] == "tunisien")
    assert wrong == 0, f"{wrong} négatifs classés positifs malgré l'aberrant"


def test_threshold_round_trips():
    pos, neg = _corpus()
    m = dialect.train(pos, neg)
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "m.json.gz"
        m.save(p)
        assert dialect.DialectModel.load(p).threshold == m.threshold


def test_evaluate_reports_threshold_and_accuracy():
    pos, neg = _corpus()
    m = dialect.train(pos, neg)
    rep = dialect.evaluate(m, pos, neg)
    assert "threshold" in rep and 0.0 <= rep["accuracy"] <= 1.0
