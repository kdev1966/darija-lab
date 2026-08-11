"""Ce que la mesure doit garantir, et pourquoi chaque garantie existe."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from darija_bench import scoring

MODEL_PATH = Path(
    os.environ.get(
        "DARIJA_MODEL",
        Path(__file__).resolve().parents[3] / "packages/darija-core/models/vs_maghreb.json.gz",
    )
)

# Réponse tunisienne réaliste, au-dessus des 25 mots exigés par le classifieur.
TUNISIEN = (
    "أنا في العادة نفيق الصباح بكري، نشرب قهوة و نخرج للخدمة. "
    "الطريق تكون معمرة برشا و الميترو ياخذلي وقت. "
    "و كي نرجع للدار في العشية نتعشى مع العايلة و نرتاح شوية قبل ما نرقد. "
    "علاش نعمل هكاكا؟ خاطر اللي يبدا نهارو بكري يلحق كل حاجة."
)

# La même intention, en arabe standard. C'est la dérive typique d'un LLM.
FUSHA = (
    "أستيقظ في الصباح الباكر ثم أتناول فنجاناً من القهوة قبل الذهاب إلى العمل. "
    "تكون الطرقات مزدحمة للغاية في تلك الساعة، ويستغرق قطار الأنفاق وقتاً طويلاً. "
    "وعندما أعود إلى المنزل في المساء أتناول العشاء مع أفراد أسرتي "
    "ثم أستريح قليلاً قبل أن أخلد إلى النوم في وقت مبكر."
)


@pytest.fixture(scope="module")
def model():
    """Le classifieur de référence.

    ``models/`` est gitignoré — reproductible par ``darija data fetch`` puis
    ``darija data train``. En son absence on saute plutôt que d'échouer : un
    test rouge pour un fichier manquant masquerait les vraies régressions.
    """
    from darija.dialect import DialectModel

    if not MODEL_PATH.exists():
        pytest.skip(f"modèle absent : {MODEL_PATH} (voir `darija data train`)")
    return DialectModel.load(MODEL_PATH)


def _verdict(text, model, **kw):
    defaults = {
        "prompt_id": "t-1",
        "model_name": "test",
        "condition": "implicite",
        "script": "arabe",
    }
    return scoring.evaluate(text, model, **{**defaults, **kw})


def test_le_tunisien_passe(model):
    verdict = _verdict(TUNISIEN, model)
    assert verdict.scorable
    assert verdict.is_tunisian, (verdict.score, verdict.n_markers)


def test_la_fusha_conversationnelle_ne_passe_pas(model):
    # LE test de régression du banc, et celui qui a dicté son design.
    #
    # `darija data validate` mesure 0,4 % de faux positifs sur `ar`, mais `ar`
    # est encyclopédique. Sur six passages de fusha au registre d'assistant,
    # le classifieur seul en a classé DEUX comme tunisiens — celui-ci à 0,842
    # pour un seuil de 0,838. C'est le biais nº 6 qui se répète : un agrégat
    # rassurant qui ne généralise pas au registre qui compte.
    #
    # La conjonction avec les marqueurs le rattrape : la fusha n'en utilise
    # aucun (0 ou 1 distinct sur les six), le tunisien en a 2 à 5.
    verdict = _verdict(FUSHA, model)
    assert verdict.scorable
    assert verdict.above_classifier, "le classifieur seul laisse passer ce texte"
    assert verdict.n_markers is not None and verdict.n_markers < 2
    assert not verdict.is_tunisian, (verdict.score, verdict.n_markers)


def test_les_deux_signaux_restent_lisibles_separement(model):
    # La règle de conjonction est provisoire (six textes par côté). Les deux
    # signaux doivent rester exposés pour qu'on puisse la réviser sur des
    # données déjà collectées, sans repayer les appels d'API.
    verdict = _verdict(TUNISIEN, model)
    assert verdict.above_classifier is not None
    assert verdict.n_markers is not None
    assert verdict.is_tunisian == (
        verdict.above_classifier and verdict.n_markers >= scoring.MIN_DISTINCT_MARKERS
    )


def test_reponse_courte_non_scorable_et_non_fausse(model):
    # `min_words` vaut 25 : en dessous, `predict` renvoie None. Compter ces
    # réponses comme fausses gonflerait le taux d'erreur d'un modèle laconique
    # alors qu'on ne sait simplement rien de sa langue.
    verdict = _verdict("برشا باهي", model)
    assert not verdict.scorable
    assert verdict.is_tunisian is None
    assert "trop court" in (verdict.skipped or "")


def test_reponse_vide_non_scorable(model):
    verdict = _verdict("   ", model)
    assert not verdict.scorable
    assert verdict.skipped == "réponse vide"


def test_arabizi_translittere_avant_scoring():
    # Le filtre `arabic_only` a éliminé 99,9 % de TUNIZI : le classifieur n'a
    # jamais vu d'Arabizi. Le lui soumettre brut ne mesurerait rien du tout.
    texte, translit = scoring.prepare("chnowa a7welek? rani mrigel barcha w enti?")
    assert translit
    assert texte != "chnowa a7welek? rani mrigel barcha w enti?"


def test_arabe_avec_emprunts_francais_non_translittere():
    # Seuil haut à dessein : une réponse arabe contenant « merci » ou « ok »
    # ne doit pas partir en translittération, ça la détruirait.
    texte, translit = scoring.prepare("نمشي للخدمة ok و من بعد نرجع merci برشا")
    assert not translit
    assert texte.startswith("نمشي")


def test_les_marqueurs_ne_decident_jamais_seuls(model):
    # Les marqueurs ne séparent pas le tunisien du marocain, qui partage علاش
    # كيفاش وين اللي — mesuré sur les corpus du dépôt : médiane 0,520 sur le
    # marocain encyclopédique contre 0,240 sur le tunisien des réseaux.
    # Ils ne servent donc qu'à écarter la fusha, en conjonction, jamais seuls.
    # Ce test verrouille le fait qu'un texte riche en marqueurs mais rejeté
    # par le classifieur reste rejeté.
    verdict = _verdict(TUNISIEN, model)
    assert verdict.n_markers is not None and verdict.n_markers >= 2
    assert verdict.is_tunisian is (verdict.above_classifier and True)
    assert verdict.explanation
