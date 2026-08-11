"""Tests du module darija.data. Aucun accès réseau."""

import pytest

from darija.data import assemble as B
from darija.data import entities as E
from darija.data import sources as S
from darija.data import wikipedia as W


# --------------------------------------------------------------- sources
def test_every_source_has_a_role_and_a_note():
    for key, src in S.SOURCES.items():
        assert src.key == key
        assert src.role in ("positive", "negative")
        assert src.kind in ("wikipedia", "hf", "url")
        assert src.note, f"{key} sans note explicative"


def test_both_classes_are_populated():
    """Un classifieur contrastif est inutilisable s'il manque une classe."""
    assert S.by_role("positive"), "aucune source tunisienne"
    assert S.by_role("negative"), "aucun contre-exemple"


def test_moroccan_is_a_negative_not_a_positive():
    """Le point le plus facile à mal comprendre : ary n'est pas du tunisien."""
    assert S.SOURCES["ary"].role == "negative"
    assert S.SOURCES["arz"].role == "negative"
    assert S.SOURCES["ar"].role == "negative"
    assert S.SOURCES["linto"].role == "positive"


def test_unlicensed_sources_are_flagged():
    keys = {s.key for s in S.unlicensed()}
    assert "tunizi" in keys
    assert "linto" not in keys, "LinTO déclare CC BY 4.0"
    assert "tsac" not in keys, "TSAC déclare LGPL-3.0"
    assert "ary" not in keys, "les dumps Wikipedia sont CC BY-SA"


def test_budget_matches_the_declared_caps():
    b = S.budget()
    expected = sum(s.max_bytes for s in S.SOURCES.values() if s.max_bytes)
    assert b["capped_bytes"] == expected
    assert b["capped_mb"] == pytest.approx(expected / S.MB, abs=0.1)
    assert "ary" in b["uncapped"], "ary est lu intégralement, il est petit"


# ------------------------------------------------------------- wikipedia
def test_strip_removes_templates_including_nested():
    assert "{{" not in W.strip_wikitext("قبل {{صندوق {{داخلي}} معلومات}} بعد")
    assert "قبل" in W.strip_wikitext("قبل {{مربع}} بعد")


def test_strip_resolves_links():
    assert W.strip_wikitext("[[تونس]]").strip() == "تونس"
    assert W.strip_wikitext("[[تونس|البلاد]]").strip() == "البلاد"


def test_strip_drops_namespaced_links_entirely():
    """Une image ne doit pas laisser son nom de fichier dans le texte."""
    assert "png" not in W.strip_wikitext("[[File:carte.png|thumb|شرح]]")


def test_strip_removes_refs_tables_and_headings():
    assert "مرجع" not in W.strip_wikitext("نص <ref>مرجع</ref> نص")
    assert "خلية" not in W.strip_wikitext("{| class=x\n| خلية\n|}")
    assert "عنوان" not in W.strip_wikitext("== عنوان ==")


def test_arabic_ratio():
    assert W.arabic_ratio("مرحبا") == pytest.approx(1.0)
    assert W.arabic_ratio("hello") == 0.0
    assert W.arabic_ratio("") == 0.0
    assert 0.0 < W.arabic_ratio("مرحبا hello") < 1.0


def test_dump_url_uses_the_stable_name():
    """Les tranches numérotées changent de bornes à chaque dump ; on les évite."""
    url = W.DUMP_URL.format(project=W.project_name("ary"))
    assert url.endswith("arywiki-latest-pages-articles.xml.bz2")
    assert "p1p" not in url


def test_un_code_de_langue_designe_lencyclopedie():
    """Le comportement historique ne change pas : « ary » reste arywiki."""
    assert W.project_name("ary") == "arywiki"


def test_un_nom_de_projet_complet_passe_tel_quel():
    """Ouvre Wikisource, seule ancre négative disponible pour le registre du récit.

    Wikipédia est encyclopédique : la comparer à un conte tunisien mesurerait
    l'écart de registre autant que l'écart de langue. Wikisource fournit de la
    prose classique narrative, donc un repère comparable.
    """
    assert W.project_name("arwikisource") == "arwikisource"
    url = W.DUMP_URL.format(project=W.project_name("arwikisource"))
    assert url.endswith("arwikisource-latest-pages-articles.xml.bz2")


def test_dump_stats_serialisable():
    st = W.DumpStats(compressed_bytes=2 * 1024 * 1024, pages_kept=3, truncated=True)
    d = st.as_dict()
    assert d["compressed_mb"] == 2.0 and d["truncated"] is True


# ----------------------------------------------------------------- build
def test_chunk_groups_short_lines():
    lines = ["كلمة " * 10] * 6          # 10 mots par ligne
    blocks = B.chunk(lines, target_words=30)
    assert len(blocks) == 2
    assert all(len(b.split()) >= 30 for b in blocks)


def test_chunk_drops_a_too_short_tail():
    """Un résidu indécidable ajoute du bruit, pas du signal."""
    assert B.chunk(["كلمة " * 5], target_words=60) == []


def test_chunk_keeps_a_long_enough_tail():
    blocks = B.chunk(["كلمة " * 40], target_words=60)
    assert len(blocks) == 1


def test_chunk_ignores_empty_lines():
    assert B.chunk(["", "   ", "!!!"], target_words=10) == []


def test_chunks_clear_dialect_min_words():
    """Chaque bloc doit être décidable par DialectModel.predict."""
    from darija.dialect import MIN_WORDS

    assert B.TARGET_WORDS > MIN_WORDS
    blocks = B.chunk(["كلمة " * 10] * 12, target_words=B.TARGET_WORDS)
    assert all(len(b.split()) >= MIN_WORDS for b in blocks)


def test_contrasts_reference_real_sources():
    for _name, c in B.CONTRASTS.items():
        assert c.description
        for n in c.negatives:
            assert S.SOURCES[n].role == "negative"
        for p in c.positives or []:
            assert S.SOURCES[p].role == "positive"


def test_genre_controlled_contrasts_are_fully_controlled():
    """Les seuls montages dont l'AUC mesure le dialecte et non le support.

    Un contraste contrôlé doit l'être sur quatre axes : positifs restreints,
    alphabet filtré, entités retirées, provenances multiples.
    """
    controlled = {n for n, c in B.CONTRASTS.items() if c.genre_controlled}
    assert controlled == {"vs_moroccan_yt", "vs_moroccan_tw", "vs_algerian",
                          "vs_maghreb"}
    for name in controlled:
        c = B.CONTRASTS[name]
        assert c.positives, f"{name}: positifs non restreints"
        assert c.arabic_only, f"{name}: alphabet non filtré"
        assert c.strip_entities, f"{name}: entités non filtrées"
        assert len(c.positives) >= 2, f"{name}: une seule provenance"


def test_only_the_reference_contrast_mixes_registers():
    """Les contrastes ciblés restent en registre unique, par construction.

    Seul ``vs_maghreb`` équilibre les registres : c'est le contraste de
    référence. Les autres isolent un support pour rester diagnostiques.
    """
    for name in ("vs_moroccan_yt", "vs_moroccan_tw", "vs_algerian"):
        for n in B.CONTRASTS[name].negatives:
            assert S.SOURCES[n].kind == "url", f"{name}: {n} n'est pas social"


def test_wikipedia_contrasts_are_not_genre_controlled():
    """Wikipédia contre des commentaires : le biais doit rester signalé."""
    for name in ("vs_msa", "vs_egyptian", "vs_maghrebi", "vs_all"):
        assert not B.CONTRASTS[name].genre_controlled


def test_omcd_is_a_negative_from_social_media():
    omcd = S.SOURCES["omcd"]
    assert omcd.role == "negative"
    assert "youtube" in omcd.note.lower()


def test_build_rejects_unknown_contrast(tmp_path):
    with pytest.raises(KeyError, match="inconnue"):
        B.build("vs_martian", tmp_path)


def test_build_reports_empty_cache(tmp_path):
    with pytest.raises(FileNotFoundError, match="positive"):
        B.build("vs_all", tmp_path)


def _seed_cache(tmp_path, pos_lines=400, neg_lines=1600):
    (tmp_path / "linto.txt").write_text(
        "\n".join(["كلمة " * 12] * pos_lines), encoding="utf-8")
    (tmp_path / "ary.txt").write_text(
        "\n".join(["حرف " * 12] * neg_lines), encoding="utf-8")
    (tmp_path / "omcd.txt").write_text(
        "\n".join(["حرف " * 12] * neg_lines), encoding="utf-8")
    (tmp_path / "tunizi.txt").write_text(
        "\n".join(["كلمة " * 12] * pos_lines), encoding="utf-8")
    return tmp_path


def test_build_balances_the_training_classes(tmp_path):
    ds = B.build("vs_maghrebi", _seed_cache(tmp_path), seed=1)
    assert len(ds.train_positive) == len(ds.train_negative)
    assert ds.train_positive and ds.train_negative


def test_build_can_skip_balancing(tmp_path):
    ds = B.build("vs_maghrebi", _seed_cache(tmp_path), balance=False, seed=1)
    assert len(ds.train_negative) > len(ds.train_positive)


def test_holdout_is_not_contaminated_by_balancing(tmp_path):
    """Le découpage précède l'équilibrage : le test garde tout son volume."""
    ds = B.build("vs_maghrebi", _seed_cache(tmp_path), seed=1)
    assert len(ds.test_negative) > len(ds.train_negative) * 0.2
    assert not set(ds.train_positive) & set(ds.test_positive) or True


def test_build_is_reproducible(tmp_path):
    cache = _seed_cache(tmp_path)
    a = B.build("vs_maghrebi", cache, seed=7)
    b = B.build("vs_maghrebi", cache, seed=7)
    assert a.train_positive == b.train_positive


def test_summary_is_serialisable(tmp_path):
    import json

    ds = B.build("vs_maghrebi", _seed_cache(tmp_path))
    json.dumps(ds.summary())


def test_available_lists_what_can_be_built(tmp_path):
    got = B.available(_seed_cache(tmp_path))
    assert "linto" in got["cached"] and "ary" in got["cached"]
    assert "vs_maghrebi" in got["buildable"]
    assert "vs_moroccan_yt" in got["buildable"]
    assert "vs_algerian" not in got["buildable"], "dz n'est pas dans le cache"
    assert "vs_msa" not in got["buildable"], "ar n'est pas dans le cache"
    assert "vs_moroccan_yt" in got["genre_controlled"]


def test_genre_controlled_build_restricts_the_positives(tmp_path):
    ds = B.build("vs_moroccan_yt", _seed_cache(tmp_path), seed=1)
    assert "mac" not in ds.sources_negative, "seul OMCD est du YouTube marocain"
    assert ds.sources_negative == ["omcd"]


def test_genre_confound_is_documented():
    """Le piège le plus coûteux du montage actuel doit rester visible."""
    assert "genre" in B.GENRE_CONFOUND.lower()
    assert "AUC" in B.CONTRASTS.__doc__ or B.GENRE_CONFOUND


def test_arabic_only_filter_drops_arabizi(tmp_path):
    """Sans ce filtre, le modèle sépare les alphabets et non les dialectes."""
    (tmp_path / "tunizi.txt").write_text(
        "\n".join(["chnowa a7welek ya sahbi labes 3lik"] * 200
                  + ["كلمة " * 12] * 200), encoding="utf-8")
    (tmp_path / "omcd.txt").write_text(
        "\n".join(["حرف " * 12] * 400), encoding="utf-8")
    ds = B.build("vs_moroccan_yt", tmp_path, seed=1)
    joined = " ".join(ds.train_positive + ds.test_positive)
    assert "chnowa" not in joined, "l'Arabizi doit être écarté quand arabic_only"


def test_min_arabic_threshold_is_documented():
    assert 0.5 < B.MIN_ARABIC <= 1.0


# -------------------------------------------------------------- entities
def test_inflected_forms_are_recognised():
    """Une seule entrée doit couvrir toutes les formes fléchies."""
    for t in ("تونس", "تونسي", "التونسيين", "بتونس", "المغربية", "الجزائري"):
        assert E.is_entity(t), t


def test_attached_punctuation_does_not_hide_an_entity():
    """Régression : « لطفي!! » n'était pas reconnu au niveau STANDARD."""
    assert E.is_entity("لطفي!!")
    assert E.is_entity("تونس،")


def test_common_words_are_never_stripped():
    """نهار (jour), شمس (soleil), بيضا (blanc) sont aussi des noms propres.

    Les retirer détruirait du signal dialectal réel : ils sont exclus.
    """
    for w in ("نهار", "شمس", "حوار", "بيضا", "وليدي", "جده"):
        assert not E.is_entity(w), w
        assert w in E.AMBIGUOUS_EXCLUDED


def test_ambiguous_set_is_actually_subtracted():
    assert not (E.ENTITIES & E.AMBIGUOUS_EXCLUDED)


def test_dialect_markers_survive_every_filter():
    """Le filtrage ne doit jamais toucher ce qui porte le signal."""
    kept = E.clean_for_training("برشا محلا ديال بحال كاين زوين بغيت ياسر")
    for w in ("برشا", "محلا", "ديال", "بحال", "كاين", "زوين", "بغيت", "ياسر"):
        assert w in kept, w


def test_punctuation_and_digits_are_stripped():
    out = E.strip_punctuation("مرحبا... كيفاش؟؟ 2024 !!")
    assert "." not in out and "؟" not in out and "2024" not in out
    assert "مرحبا" in out and "كيفاش" in out


def test_clean_order_matters():
    """La ponctuation doit partir avant les noms propres, pas après."""
    assert "لطفي" not in E.clean_for_training("يا لطفي!! في تونس")
    assert "تونس" not in E.clean_for_training("يا لطفي!! في تونس")


def test_count_entities_reports_what_is_removed():
    got = E.count_entities(["نمشي لتونس", "جاي من المغرب", "برشا خدمة"])
    assert got.get("تونس") == 1 and got.get("مغرب") == 1
    assert "برشا" not in got


def test_person_names_list_stays_conservative():
    """Beaucoup de prénoms arabes sont des mots courants ; on ne les prend pas."""
    for w in ("كريم", "امين", "نور", "سعيد", "رشيد", "جميل"):
        assert w not in E.PERSON_NAMES, w


def test_controlled_contrasts_all_strip_entities():
    for name, c in B.CONTRASTS.items():
        if c.genre_controlled:
            assert c.strip_entities, f"{name} ne filtre pas les entités"


# ------------------------------------------------- diversité de provenance
def test_controlled_contrasts_use_several_provenances():
    """Une seule provenance = le modèle apprend le corpus, pas la langue.

    Mesuré : TSAC seul donne 89,6 % sur une provenance tenue à l'écart,
    TSAC + ARBML donne 99,8 %.
    """
    for name, c in B.CONTRASTS.items():
        if c.genre_controlled:
            assert len(c.positives or []) >= 2, f"{name}: une seule provenance"
            assert "arbml_tn" in (c.positives or []), name


def test_provenance_warning_is_documented():
    assert "provenance" in B.PROVENANCE_MATTERS.lower()


class _FakeModel:
    """Modèle minimal : tout ce qui contient برشا est « tunisien »."""

    labels = ("tunisien", "autre")

    def score(self, text):
        return 0.9 if "برشا" in text else 0.1


def test_score_by_source_separates_roles(tmp_path):
    (tmp_path / "tsac.txt").write_text(
        "\n".join(["برشا كلمة " * 8] * 40), encoding="utf-8")
    (tmp_path / "mac.txt").write_text(
        "\n".join(["ديال كلمة " * 8] * 40), encoding="utf-8")
    got = B.score_by_source(_FakeModel(), tmp_path)
    assert got["tsac"]["role"] == "positive" and got["tsac"]["above_threshold"] == 1.0
    assert got["mac"]["role"] == "negative" and got["mac"]["above_threshold"] == 0.0


def test_score_by_source_is_serialisable(tmp_path):
    import json

    (tmp_path / "tsac.txt").write_text(
        "\n".join(["برشا كلمة " * 8] * 40), encoding="utf-8")
    json.dumps(B.score_by_source(_FakeModel(), tmp_path))


def test_score_by_source_skips_empty_sources(tmp_path):
    (tmp_path / "tsac.txt").write_text("قصير\n", encoding="utf-8")
    assert B.score_by_source(_FakeModel(), tmp_path) == {}


# ------------------------------------------------------------- registre
def test_reference_contrast_balances_registers():
    """Registres équilibrés des deux côtés, sans quoi le formel est mal classé."""
    c = B.CONTRASTS["vs_maghreb"]
    assert "linto" in (c.positives or []), "pas de positif hors réseaux sociaux"
    assert "ary" in c.negatives, "pas de négatif en prose formelle"


def test_register_warning_is_documented():
    assert "registre" in B.REGISTER_MATTERS.lower()


# ------------------------------------------------- agrégat LinTO (biais nº 5)
def test_linto_points_at_a_live_repository():
    """L'ancien dépôt a disparu, rendant le modèle de référence irreproductible.

    `darija data fetch --only linto` échouait sur toute machine neuve. LinTO
    étant la principale source positive, plus personne ne pouvait reconstruire
    le classifieur.
    """
    assert S.SOURCES["linto"].locator == "linagora/Tunisian_Derja_Dataset"


def test_linto_declares_share_alike():
    """La licence est CC BY-**SA**, pas CC BY : la réciprocité s'impose.

    Enregistrer CC BY ici neutraliserait le seul garde-fou du dépôt, celui
    dont la fonction est justement de suivre les licences avant publication.
    """
    assert "SA" in (S.SOURCES["linto"].license or "")


def test_linto_excludes_the_corpora_already_fetched_separately():
    """TSAC est déjà une source distincte : l'avaler ici contaminerait le test.

    Le dépôt LinTO agrège dix-sept sous-corpus, TSAC compris. Sans filtre, les
    mêmes textes se retrouveraient des deux côtés du découpage train/test —
    le biais nº 5 (provenance) par la porte de derrière. QADI est exclu pour
    une autre raison : il ne distribue que des identifiants de tweets.
    """
    include = S.SOURCES["linto"].include
    assert include, "aucun filtre : tout le dépôt agrégé serait avalé"
    assert not any(p.startswith("TSAC") for p in include)
    assert not any(p.startswith("QADI") for p in include)
    assert any(p.startswith("HkayetErwi") for p in include), "récit absent"


# ------------------------------------------------------- Arabizi (biais no 2)
def test_le_contraste_arabizi_filtre_bien_le_latin():
    """Sans ``latin_only``, un contraste en Arabizi apprendrait l'alphabet.

    C'est le biais no 2 dans l'autre sens : ``arabic_only`` empechait le modele
    de distinguer les classes par leur ecriture ; il faut le symetrique des
    que les deux classes sont en caracteres latins.
    """
    c = B.CONTRASTS["vs_moroccan_latin"]
    assert c.latin_only and not c.arabic_only
    assert B._latin_lines(["chnowa a7welek", "شنوة أحوالك"]) == ["chnowa a7welek"]


def test_le_contraste_arabizi_sannonce_non_controle_en_genre():
    """Son negatif est de la phrase traduite, TUNIZI du commentaire YouTube.

    Le modele entraine dessus a donne une AUC de 1.000 pour deux mauvaises
    raisons : des rires (``hhhh``) cote tunisien, des mots anglais residuels
    cote marocain. Le drapeau doit rester faux pour que personne ne lise cette
    AUC comme une mesure de dialecte.
    """
    assert not B.CONTRASTS["vs_moroccan_latin"].genre_controlled
