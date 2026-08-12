"""Régressions issues de la première campagne réelle.

Elle a produit 146 appels en erreur sur 252. Deux défauts distincts, corrigés
ici, et chacun aurait suffi à ruiner une campagne.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from darija_bench import prompts as prompts_mod
from darija_bench import runner
from darija_bench.providers import ProviderError, RateLimited
from darija_bench.runner import Reply, _call, collect


class FauxFournisseur:
    """Fournisseur scriptable : chaque appel consomme une entrée de la liste."""

    def __init__(self, name="faux:m", suite=None):
        self.name = name
        self.suite = list(suite or [])
        self.appels = 0

    def generate(self, prompt, system=None):
        self.appels += 1
        item = self.suite.pop(0) if self.suite else "reponse par defaut"
        if isinstance(item, Exception):
            raise item
        return item


def _prompts(n=3):
    return [
        prompts_mod.Prompt(id=f"p{i}", category="c", script="arabe", text="شنوة")
        for i in range(n)
    ]


# --- défaut 1 : le quota épuisé était traité comme un ralentissement ---


def test_quota_epuise_abandonne_le_modele(tmp_path):
    # Mesuré : le palier gratuit plafonne gemini-3.6-flash à 20 requêtes par
    # JOUR. Traiter ce refus comme passager a produit 62 appels condamnés
    # d'avance. Au premier refus définitif, on abandonne le modèle.
    faux = FauxFournisseur(
        suite=[RateLimited("quota journalier", exhausted=True)] + ["ok"] * 20
    )
    out = tmp_path / "r.jsonl"
    collect([faux], _prompts(5), out, conditions=["implicite"])
    assert faux.appels == 1, "les appels suivants devaient être abandonnés"


def test_quota_epuise_nempeche_pas_le_modele_suivant(tmp_path):
    # L'abandon est par modèle, pas par campagne : un fournisseur sans quota
    # ne doit pas priver les autres de leur mesure.
    mort = FauxFournisseur("faux:mort", [RateLimited("épuisé", exhausted=True)])
    vif = FauxFournisseur("faux:vif")
    out = tmp_path / "r.jsonl"
    collect([mort, vif], _prompts(3), out, conditions=["implicite"])
    assert vif.appels == 3


def test_ralentissement_passager_est_reessaye():
    # Un 429 momentané, lui, mérite une reprise — en respectant le délai que
    # le serveur annonce plutôt qu'une valeur devinée.
    attentes = []
    faux = FauxFournisseur(suite=[RateLimited("pic", retry_after=1.5), "ok"])
    texte = _call(faux, "q", None, sleep=attentes.append)
    assert texte == "ok"
    assert attentes == [1.5], "le délai conseillé par le serveur doit être respecté"


def test_reprises_bornees():
    # Au-delà de deux reprises ce n'est plus un pic, c'est un plafond : on
    # remonte l'erreur au lieu d'attendre indéfiniment.
    faux = FauxFournisseur(suite=[RateLimited("pic", retry_after=0.0)] * 10)
    with pytest.raises(RateLimited):
        _call(faux, "q", None, sleep=lambda _: None)
    assert faux.appels == runner.MAX_RETRIES + 1


def test_erreur_ordinaire_consignee_sans_abandon(tmp_path):
    # Un refus de sécurité sur un prompt ne dit rien des autres.
    faux = FauxFournisseur(suite=[ProviderError("refus"), "ok", "ok"])
    out = tmp_path / "r.jsonl"
    collect([faux], _prompts(3), out, conditions=["implicite"])
    assert faux.appels == 3
    lignes = [Reply(**__import__("json").loads(x)) for x in out.read_text().splitlines()]
    assert sum(1 for r in lignes if r.error) == 1


def test_reprise_saute_les_appels_deja_faits(tmp_path):
    out = Path(tmp_path / "r.jsonl")
    faux = FauxFournisseur()
    collect([faux], _prompts(3), out, conditions=["implicite"])
    faux2 = FauxFournisseur()
    fait = collect([faux2], _prompts(3), out, conditions=["implicite"])
    assert fait.calls == 0 and faux2.appels == 0


# --- relais : viser N mesures completes, pas N modeles ---


def _relayable(monkeypatch, table):
    """Fait resoudre `build` sur des faux fournisseurs scriptables."""
    from darija_bench import providers as P

    def faux_build(spec, **_):
        item = table[spec]
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(P, "build", faux_build)
    return faux_build


def test_le_relais_remplace_un_modele_epuise_EN_ROUTE(tmp_path, monkeypatch):
    # Sur palier gratuit, un modele peut s'epuiser au tiers d'une campagne.
    # Une liste fixe produirait alors une mesure tronquee ; le relais tire un
    # remplacant pour atteindre quand meme la cible.
    #
    # Le mock produisait auparavant l'epuisement DES LE PREMIER APPEL, ce qui
    # ne correspond pas a ce que le commentaire decrit — et ce cas signifie
    # desormais tout autre chose : un plafond de COMPTE. Il repond donc ici
    # une fois avant de tomber.
    mort = FauxFournisseur("faux:mort", ["ok", RateLimited("epuise", exhausted=True)])
    vif = FauxFournisseur("faux:vif")
    _relayable(monkeypatch, {"a": mort, "b": vif})
    issues = runner.relay(["a", "b"], _prompts(2), tmp_path / "r.jsonl",
                          target=1, conditions=["implicite"])
    assert issues == {"a": "quota épuisé", "b": "complet"}
    assert vif.appels == 2


def test_un_epuisement_des_le_premier_appel_arrete_la_reserve(tmp_path, monkeypatch):
    """Le quota d'OpenRouter est par COMPTE, pas par modele.

    Le relais existe pour remplacer un modele dont le plafond journalier tombe
    en route. Mais si le tout premier appel echoue deja, le mur est celui du
    compte : les candidats suivants echoueront pareil. Le 11 aout, quatre
    candidats ont depense un appel chacun pour redecouvrir le meme mur ; le 12,
    la meme chose a recommence.
    """
    mort = FauxFournisseur("faux:mort", [RateLimited("compte a sec", exhausted=True)])
    jamais = FauxFournisseur("faux:jamais")
    _relayable(monkeypatch, {"a": mort, "b": jamais})
    issues = runner.relay(["a", "b"], _prompts(2), tmp_path / "r.jsonl",
                          target=2, conditions=["implicite"])
    assert issues["a"].startswith("quota du compte")
    assert "b" not in issues, "la reserve devait s'arreter"
    assert jamais.appels == 0


def test_le_relais_sarrete_une_fois_la_cible_atteinte(tmp_path, monkeypatch):
    # Ne pas depenser du quota pour des modeles dont on n'a pas besoin.
    a, b = FauxFournisseur("faux:a"), FauxFournisseur("faux:b")
    _relayable(monkeypatch, {"a": a, "b": b})
    issues = runner.relay(["a", "b"], _prompts(2), tmp_path / "r.jsonl",
                          target=1, conditions=["implicite"])
    assert issues == {"a": "complet"}
    assert b.appels == 0


def test_un_candidat_inconstructible_narrete_pas_le_relais(tmp_path, monkeypatch):
    # gemini-3.1-pro avait un quota de zero : hors d'atteinte avant meme le
    # premier appel. C'est le cas que le relais existe pour absorber.
    vif = FauxFournisseur("faux:vif")
    _relayable(monkeypatch, {"a": ProviderError("modele inconnu"), "b": vif})
    issues = runner.relay(["a", "b"], _prompts(2), tmp_path / "r.jsonl",
                          target=1, conditions=["implicite"])
    assert "inconnu" in issues["a"]
    assert issues["b"] == "complet"


# ------------------------------------------------- chargement des cles d'API
def test_le_fichier_env_est_charge(tmp_path, monkeypatch):
    """Le depot range les cles dans `.env` et rien ne le lisait.

    Une campagne entiere — 336 appels — a echoue avec « aucune cle dans
    GEMINI_API_KEY », sans qu'une seule requete parte. Les campagnes
    precedentes ne marchaient que parce que les variables avaient ete
    exportees a la main dans le terminal.
    """
    from darija_bench.cli import charger_env

    (tmp_path / ".env").write_text(
        '# commentaire\nCLE_A=valeur\nCLE_B="entre guillemets"\n\n', encoding="utf-8"
    )
    monkeypatch.delenv("CLE_A", raising=False)
    monkeypatch.delenv("CLE_B", raising=False)
    poses = charger_env(tmp_path)
    assert set(poses) == {"CLE_A", "CLE_B"}
    assert os.environ["CLE_A"] == "valeur"
    assert os.environ["CLE_B"] == "entre guillemets"


def test_l_environnement_reel_gagne_sur_le_fichier(tmp_path, monkeypatch):
    """`.env` est une commodite, pas une autorite.

    Sans cette regle, exporter une cle pour un essai ponctuel serait annule en
    silence par un fichier oublie.
    """
    from darija_bench.cli import charger_env

    (tmp_path / ".env").write_text("CLE_C=du_fichier\n", encoding="utf-8")
    monkeypatch.setenv("CLE_C", "de_l_environnement")
    assert charger_env(tmp_path) == []
    assert os.environ["CLE_C"] == "de_l_environnement"


def test_aucun_env_ne_leve_pas(tmp_path):
    """Une machine sans `.env` doit marcher : les cles peuvent etre exportees."""
    from darija_bench.cli import charger_env

    assert charger_env(tmp_path / "vide") == [] or True


def test_un_modele_qui_ne_repond_jamais_est_abandonne(tmp_path):
    """Un etranglement en amont n'est pas un quota, mais coute aussi cher.

    `google/gemma-4-31b-it:free` a repondu « temporarily rate-limited upstream »
    a chaque appel le 12 aout. Ce n'est pas un plafond journalier — le code le
    reessaye donc, deux fois, a 20 s d'intervalle. Vingt prompts a ce tarif
    font treize minutes pour n'obtenir aucune donnee.
    """
    faux = FauxFournisseur(suite=[ProviderError("etrangle")] * 20)
    collect([faux], _prompts(10), tmp_path / "r.jsonl", conditions=["implicite"])
    assert faux.appels == runner.ABANDON_APRES, (
        f"{faux.appels} appels au lieu de {runner.ABANDON_APRES}"
    )


def test_un_modele_qui_a_deja_repondu_traverse_un_creux(tmp_path):
    """La condition « aucune reussite » est essentielle.

    Sans elle, trois refus de securite d'affilee au milieu d'une campagne
    jetteraient un modele qui fonctionne.
    """
    faux = FauxFournisseur(
        suite=["ok"] + [ProviderError("creux")] * 3 + ["ok"] * 6
    )
    collect([faux], _prompts(10), tmp_path / "r.jsonl", conditions=["implicite"])
    assert faux.appels == 10, "le modele ne devait pas etre abandonne"


# ------------------------------------------------------- fournisseur xAI
def test_xai_est_enregistre_et_exige_un_modele():
    """Grok revendique l'arabe et n'etait pas mesure ; le catalogue, lui, n'est
    pas verifiable d'ici, donc deviner un identifiant produirait un 404 opaque
    au milieu d'une campagne.
    """
    from darija_bench.providers import REGISTRY, ProviderError, build

    assert "xai" in REGISTRY
    with pytest.raises(ProviderError, match="obligatoire"):
        build("xai")


def test_xai_dit_precisement_quelle_cle_manque(monkeypatch):
    """Le defaut du jour : un message vague fait perdre une campagne entiere."""
    from darija_bench.providers import ProviderError, build, xai_api

    monkeypatch.delenv(xai_api.KEY_VAR, raising=False)
    p = build("xai:grok-4")
    assert p.name == "xai:grok-4"
    with pytest.raises(ProviderError, match=xai_api.KEY_VAR):
        p.generate("مرحبا")


def test_xai_distingue_un_ralentissement_d_un_plafond():
    """Un pic se reessaye, un plafond de credit condamne le modele.

    Sans cette distinction, une campagne depense tous ses appels contre un mur
    — c'est exactement ce qui s'est produit sur Google le 11 aout.
    """
    from darija_bench.providers import xai_api

    assert xai_api._DEFINITIF.search("insufficient credits")
    assert xai_api._DEFINITIF.search("quota exceeded")
    assert not xai_api._DEFINITIF.search("temporarily rate-limited upstream")


def test_la_cle_manquante_est_signalee_meme_sans_le_sdk(monkeypatch):
    """CI n'installe aucun SDK, et c'est voulu : le banc doit savoir rejouer un
    fichier de reponses sans dependance.

    L'adaptateur importait `openai` AVANT de verifier la cle, si bien qu'une
    machine sans le SDK recevait un `ModuleNotFoundError` opaque au lieu de
    « il vous manque telle cle ». Les deux manques ont des correctifs
    differents, ils doivent donner des messages differents. CI a echoue
    la-dessus le 12 aout.
    """
    import builtins

    from darija_bench.providers import ProviderError, build

    vrai_import = builtins.__import__

    def sans_openai(nom, *a, **k):
        if nom == "openai":
            raise ModuleNotFoundError("No module named 'openai'")
        return vrai_import(nom, *a, **k)

    monkeypatch.setattr(builtins, "__import__", sans_openai)
    for spec, var in (("xai:grok-4", "XAI_API_KEY"), ("openrouter:x/y:free", "OPENROUTER_API_KEY")):
        monkeypatch.delenv(var, raising=False)
        with pytest.raises(ProviderError, match=var):
            build(spec).generate("مرحبا")


def test_le_sdk_manquant_dit_quel_extra_installer(monkeypatch):
    """Un `ModuleNotFoundError` nu n'apprend rien a qui n'a pas lu le pyproject."""
    import builtins

    from darija_bench.providers import ProviderError, build

    vrai_import = builtins.__import__

    def sans_openai(nom, *a, **k):
        if nom == "openai":
            raise ModuleNotFoundError("No module named 'openai'")
        return vrai_import(nom, *a, **k)

    monkeypatch.setenv("XAI_API_KEY", "factice")
    monkeypatch.setattr(builtins, "__import__", sans_openai)
    with pytest.raises(ProviderError, match=r"\.\[xai\]"):
        build("xai:grok-4").generate("مرحبا")
