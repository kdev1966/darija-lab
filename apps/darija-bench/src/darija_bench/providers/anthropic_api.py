"""Adaptateur Anthropic, via le SDK officiel ``anthropic``.

Choix d'appel, et pourquoi :

- ``effort`` est réglé bas par défaut. Ce banc mesure la **langue** d'une
  réponse ordinaire, pas la profondeur d'un raisonnement ; faire réfléchir le
  modèle longuement coûterait sans rien changer à ce qu'on observe. Le
  paramètre reste réglable pour qui veut vérifier que l'effort ne déplace pas
  la tunisianité — c'est une question ouverte, pas une évidence.
- ``max_tokens`` est large parce que la réflexion et le texte se partagent le
  même plafond : une valeur serrée tronquerait la réponse au milieu et
  fausserait la mesure de longueur, dont dépend la scorabilité.
- Un refus de sécurité revient en HTTP 200 avec ``stop_reason == "refusal"`` et
  un ``content`` vide. Lire ``content[0]`` sans vérifier planterait ; on lève
  une erreur explicite, que le runner consigne comme échec du prompt.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import ProviderError

#: Assez large pour que la réflexion adaptative ne mange pas la réponse.
MAX_TOKENS: int = 4000


@dataclass
class AnthropicProvider:
    """Un modèle Claude interrogé via l'API Messages."""

    model: str
    effort: str = "low"
    max_tokens: int = MAX_TOKENS

    @property
    def name(self) -> str:
        """Nom reporté dans les résultats."""
        return f"anthropic:{self.model}"

    def generate(self, prompt: str, system: str | None = None) -> str:
        """Envoie ``prompt`` et rend le texte de la réponse."""
        import anthropic  # noqa: PLC0415

        client = anthropic.Anthropic()
        kwargs: dict[str, object] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "output_config": {"effort": self.effort},
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        try:
            response = client.messages.create(**kwargs)
        except anthropic.APIError as exc:  # pragma: no cover - dépend du réseau
            raise ProviderError(f"{self.name} : {exc}") from exc

        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None)
            raise ProviderError(f"{self.name} : refus ({category})")

        return "".join(b.text for b in response.content if b.type == "text")


def make(model: str, **options: object) -> AnthropicProvider:
    """Construit l'adaptateur. Appelé par :func:`darija_bench.providers.build`."""
    return AnthropicProvider(model=model, **options)  # type: ignore[arg-type]
