"""Adaptateur xAI (Grok), via le SDK ``openai`` pointé sur un autre hôte.

xAI parle le dialecte Chat Completions d'OpenAI ; on réutilise donc le SDK
officiel plutôt que de reparler HTTP à la main. Seuls changent l'hôte et la
variable de clé — la même mécanique que pour OpenRouter.

Son intérêt pour ce banc : Grok revendique l'arabe, aucun modèle xAI n'était
mesuré, et un classement gagne à ne pas se limiter à un seul écosystème.

⚠️ **L'API xAI est entièrement payante, vérifié le 12 août 2026.** Une clé
fraîchement créée ne donne accès à rien du tout — pas même au catalogue :

    403 permission-denied — "Your newly created team doesn't have any
    credits or licenses yet."

Contrairement à Google AI Studio (20 requêtes/jour sur Flash) et à OpenRouter
(~50/jour tous modèles confondus), il n'y a ici **aucun palier gratuit**. Cet
adaptateur est donc écrit et testé mais inutilisable sans achat de crédit ; une
collecte complète fait 84 appels par modèle, 20 avec l'échantillon réduit.

Les identifiants de modèle sont **obligatoires** : le catalogue n'est pas
vérifiable depuis ce dépôt, et une chaîne devinée produirait un 404 opaque au
milieu d'une campagne plutôt qu'une erreur lisible avant de commencer.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from . import ProviderError, RateLimited

BASE_URL: str = "https://api.x.ai/v1"
KEY_VAR: str = "XAI_API_KEY"
MAX_TOKENS: int = 4000

#: Un plafond de facturation ou de compte ne se rouvre pas en attendant
#: quelques secondes ; il faut abandonner le modèle. Voir :class:`RateLimited`.
_DEFINITIF = re.compile(r"per day|daily|quota|credit|billing|insufficient", re.I)

#: Délai conseillé, quand le fournisseur en donne un.
_RETRY = re.compile(r"(?:retry|try again) (?:in|after) ([\d.]+)\s*s", re.I)


def _client():
    """Construit le client, ou dit précisément ce qui manque."""
    import openai  # noqa: PLC0415

    key = os.environ.get(KEY_VAR)
    if not key:
        raise ProviderError(
            f"xai : aucune clé dans {KEY_VAR}. Créez-la sur console.x.ai, "
            "puis mettez-la dans apps/darija-bench/.env avant de lancer."
        )
    return openai.OpenAI(base_url=BASE_URL, api_key=key)


@dataclass
class XAIProvider:
    """Un modèle xAI interrogé via l'API Chat Completions."""

    model: str
    max_tokens: int = MAX_TOKENS

    @property
    def name(self) -> str:
        """Nom reporté dans les résultats."""
        return f"xai:{self.model}"

    def generate(self, prompt: str, system: str | None = None) -> str:
        """Envoie ``prompt`` et rend le texte de la réponse.

        Raises:
          RateLimited: ralentissement ou plafond, selon ce que dit le message.
          ProviderError: tout autre échec, y compris une réponse vide.

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
                exhausted=bool(_DEFINITIF.search(message)),
            ) from exc
        except openai.OpenAIError as exc:  # pragma: no cover - dépend du réseau
            raise ProviderError(f"{self.name} : {exc}") from exc

        if not response.choices:
            err = getattr(response, "error", None)
            raise ProviderError(f"{self.name} : réponse sans contenu ({err})")
        text = response.choices[0].message.content
        if not text:
            reason = response.choices[0].finish_reason
            raise ProviderError(f"{self.name} : réponse vide (finish_reason={reason})")
        return text


def make(model: str, **options: object) -> XAIProvider:
    """Construit l'adaptateur. Appelé par :func:`darija_bench.providers.build`."""
    return XAIProvider(model=model, **options)  # type: ignore[arg-type]
