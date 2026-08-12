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

#: Repères d'un message d'erreur exploitable : qui, où créer la clé,
#: et quel extra installer. Les deux manques ont des correctifs différents.
NOM: str = "openrouter"
CONSOLE: str = "openrouter.ai/keys"
EXTRA: str = "openrouter"

#: Un quota journalier épuisé ne se rouvre pas en attendant quelques secondes ;
#: il faut abandonner le modèle. Voir :class:`RateLimited`.
_DAILY = re.compile(r"per day|daily|free-models-per-day", re.I)

#: Délai conseillé, quand le fournisseur en donne un.
_RETRY = re.compile(r"(?:retry|try again) (?:in|after) ([\d.]+)\s*s", re.I)


def _client():
    """Construit le client, ou dit précisément ce qui manque.

    La clé est vérifiée **avant** l'import du SDK : sans cet ordre, une machine
    sans ``openai`` installé remonte un ``ModuleNotFoundError`` opaque au lieu
    de « il vous manque telle clé ». Les deux manques ont des correctifs
    différents, ils doivent donner des messages différents.
    """
    key = os.environ.get(KEY_VAR)
    if not key:
        raise ProviderError(
            f"{NOM} : aucune clé dans {KEY_VAR}. Créez-la sur {CONSOLE}, "
            "puis mettez-la dans apps/darija-bench/.env avant de lancer."
        )
    try:
        import openai  # noqa: PLC0415
    except ModuleNotFoundError as exc:  # pragma: no cover - dépend de l'install
        raise ProviderError(
            f"{NOM} : SDK absent. Installez l'extra — pip install -e '.[{EXTRA}]'"
        ) from exc
    return openai.OpenAI(base_url=BASE_URL, api_key=key)


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
        client = _client()
        import openai  # noqa: PLC0415

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
