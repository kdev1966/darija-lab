"""Adaptateurs vers les fournisseurs de modèles.

Un adaptateur fait une seule chose : envoyer un prompt, rendre du texte. Tout
ce qui relève de la mesure est ailleurs — le banc doit pouvoir évaluer un
fournisseur qu'il ne connaissait pas sans qu'une ligne de scoring change.

Chaque fournisseur utilise **son SDK officiel**, importé à l'appel et non au
chargement : installer l'extra ``[anthropic]`` seul doit suffire pour évaluer
Claude, sans tirer les SDK des autres.

Les identifiants de modèle ne sont devinés pour aucun fournisseur sauf
Anthropic, où ``claude-opus-5`` est vérifiable. Ailleurs, l'identifiant est
obligatoire dans la spécification — inventer une chaîne produirait un 404
opaque plutôt qu'une erreur lisible.
"""

from __future__ import annotations

from typing import Protocol


class ProviderError(RuntimeError):
    """Échec d'appel à un fournisseur, ou refus du modèle."""


class Provider(Protocol):
    """Ce que le runner attend d'un fournisseur."""

    name: str

    def generate(self, prompt: str, system: str | None = None) -> str:
        """Envoie ``prompt`` et rend le texte de la réponse."""
        ...


#: Fournisseurs connus → module d'implémentation et modèle par défaut.
#: ``None`` en défaut = identifiant obligatoire dans la spécification.
REGISTRY: dict[str, tuple[str, str | None]] = {
    "anthropic": ("darija_bench.providers.anthropic_api", "claude-opus-5"),
    "openai": ("darija_bench.providers.openai_api", None),
    "google": ("darija_bench.providers.google_api", None),
}


def build(spec: str, **options: object) -> Provider:
    """Construit un fournisseur à partir d'une spécification ``kind:model``.

    Args:
      spec: ``anthropic``, ``anthropic:claude-opus-5``, ``openai:<modèle>``…
      options: passées telles quelles au constructeur de l'adaptateur.

    Raises:
      ProviderError: fournisseur inconnu, ou identifiant de modèle manquant
        alors qu'aucun défaut vérifiable n'existe pour ce fournisseur.

    """
    from importlib import import_module  # noqa: PLC0415

    kind, _, model = spec.partition(":")
    kind = kind.strip().lower()
    if kind not in REGISTRY:
        known = ", ".join(sorted(REGISTRY))
        raise ProviderError(f"fournisseur inconnu {kind!r} ; connus : {known}")

    module_name, default = REGISTRY[kind]
    model = model.strip() or (default or "")
    if not model:
        raise ProviderError(
            f"{kind} : identifiant de modèle obligatoire (aucun défaut vérifiable). "
            f"Écrivez « {kind}:<modèle> »."
        )
    return import_module(module_name).make(model, **options)
