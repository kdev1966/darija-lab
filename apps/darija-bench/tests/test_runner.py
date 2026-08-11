"""Régressions issues de la première campagne réelle.

Elle a produit 146 appels en erreur sur 252. Deux défauts distincts, corrigés
ici, et chacun aurait suffi à ruiner une campagne.
"""

from __future__ import annotations

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


def test_le_relais_remplace_un_modele_epuise(tmp_path, monkeypatch):
    # Sur palier gratuit, un modele peut s'epuiser au tiers d'une campagne.
    # Une liste fixe produirait alors une mesure tronquee ; le relais tire un
    # remplacant pour atteindre quand meme la cible.
    mort = FauxFournisseur("faux:mort", [RateLimited("epuise", exhausted=True)])
    vif = FauxFournisseur("faux:vif")
    _relayable(monkeypatch, {"a": mort, "b": vif})
    issues = runner.relay(["a", "b"], _prompts(2), tmp_path / "r.jsonl",
                          target=1, conditions=["implicite"])
    assert issues == {"a": "quota épuisé", "b": "complet"}
    assert vif.appels == 2


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
