"""Adaptateur OpenRouter, via le SDK ``openai`` pointé sur un autre hôte.

OpenRouter parle le dialecte Chat Completions d'OpenAI ; on réutilise donc le
SDK officiel plutôt que de reparler HTTP à la main. Seuls changent l'hôte et
la variable de clé.

Son intérêt ici est direct : **une seule clé donne accès à des modèles de
plusieurs fournisseurs**, ce qui est la condition d'un classement. La campagne
Google avait échoué là-dessus — deux des trois modèles étaient hors d'atteinte
d'un compte gratuit, et un banc à un seul modèle n'est pas un banc.

Les identifiants sont de la forme ``editeur/modele:free``. Ils ne sont pas
devinés : ``list_models`` interroge le catalogue réel de la clé.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from . import ProviderError, RateLimited

BASE_URL: str = "https://openrouter.ai/api/v1"
KEY_VAR: str = "OPENROUTER_API_KEY"
MAX_TOKENS: int = 4000

#: Un quota journalier épuisé ne se rouvre pas en attendant quelques secondes ;
#: il faut abandonner le modèle. Voir :class:`RateLimited`.
_DAILY = re.compile(r"per day|daily|free-models-per-day", re.I)

#: Délai conseillé, quand le fournisseur en donne un.
_RETRY = re.compile(r"(?:retry|try again) (?:in|after) ([\d.]+)\s*s", re.I)


def _client():
    """Construit le client, ou dit précisément ce qui manque."""
    import openai  # noqa: PLC0415

    key = os.environ.get(KEY_VAR)
    if not key:
        raise ProviderError(
            f"openrouter : aucune clé dans {KEY_VAR}. "
            "Créez-la sur openrouter.ai/keys, puis exportez-la avant de lancer."
        )
    return openai.OpenAI(api_key=key, base_url=BASE_URL)


def list_models(free_only: bool = True) -> list[str]:
    """Catalogue réellement atteignable par la clé.

    Args:
      free_only: ne garder que les modèles dont le tarif est nul.

    """
    client = _client()
    out: list[str] = []
    for model in client.models.list().data:
        ident = model.id
        if not free_only:
            out.append(ident)
            continue
        pricing = (getattr(model, "pricing", None) or {}) if not isinstance(model, dict) else {}
        gratuit = ident.endswith(":free") or all(
            float(pricing.get(k, 0) or 0) == 0 for k in ("prompt", "completion")
        )
        if gratuit:
            out.append(ident)
    return sorted(out)


@dataclass
class OpenRouterProvider:
    """Un modèle servi par OpenRouter."""

    model: str
    max_tokens: int = MAX_TOKENS

    @property
    def name(self) -> str:
        """Nom reporté dans les résultats."""
        return f"openrouter:{self.model}"

    def generate(self, prompt: str, system: str | None = None) -> str:
        """Envoie ``prompt`` et rend le texte de la réponse.

        Raises:
          ProviderError: clé absente, échec d'appel, ou réponse vide.
          RateLimited: quota dépassé — momentané ou épuisé.

        """
        import openai  # noqa: PLC0415

        client = _client()
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            response = client.chat.completions.create(
                model=self.model, messages=messages, max_tokens=self.max_tokens
            )
        except openai.RateLimitError as exc:
            message = str(exc)
            delay = _RETRY.search(message)
            raise RateLimited(
                f"{self.name} : {message}",
                retry_after=float(delay.group(1)) if delay else None,
                exhausted=bool(_DAILY.search(message)),
            ) from exc
        except openai.OpenAIError as exc:  # pragma: no cover - dépend du réseau
            raise ProviderError(f"{self.name} : {exc}") from exc

        # OpenRouter route vers des fournisseurs tiers : une réponse sans choix
        # arrive quand le fournisseur en aval a refusé. C'est une non-réponse,
        # pas une réponse hors sujet — elle doit être consignée comme erreur.
        if not response.choices:
            err = getattr(response, "error", None)
            raise ProviderError(f"{self.name} : réponse sans contenu ({err})")
        text = response.choices[0].message.content
        if not text:
            reason = response.choices[0].finish_reason
            raise ProviderError(f"{self.name} : réponse vide (finish_reason={reason})")
        return text


def make(model: str, **options: object) -> OpenRouterProvider:
    """Construit l'adaptateur. Appelé par :func:`darija_bench.providers.build`."""
    return OpenRouterProvider(model=model, **options)  # type: ignore[arg-type]
