"""Adaptateur OpenAI, via le SDK officiel ``openai``.

L'identifiant de modèle est **obligatoire** : le catalogue OpenAI n'est pas
vérifiable depuis ce dépôt, et une chaîne devinée produirait un 404 opaque au
milieu d'une campagne plutôt qu'une erreur lisible avant de commencer.

Non exercé contre l'API réelle faute de clé au moment de l'écriture — la forme
d'appel suit ``chat.completions``, la surface la plus stable du SDK.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import ProviderError

MAX_TOKENS: int = 4000


@dataclass
class OpenAIProvider:
    """Un modèle OpenAI interrogé via l'API Chat Completions."""

    model: str
    max_tokens: int = MAX_TOKENS

    @property
    def name(self) -> str:
        """Nom reporté dans les résultats."""
        return f"openai:{self.model}"

    def generate(self, prompt: str, system: str | None = None) -> str:
        """Envoie ``prompt`` et rend le texte de la réponse."""
        import openai  # noqa: PLC0415

        client = openai.OpenAI()
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=self.max_tokens,
            )
        except openai.OpenAIError as exc:  # pragma: no cover - dépend du réseau
            raise ProviderError(f"{self.name} : {exc}") from exc

        return response.choices[0].message.content or ""


def make(model: str, **options: object) -> OpenAIProvider:
    """Construit l'adaptateur. Appelé par :func:`darija_bench.providers.build`."""
    return OpenAIProvider(model=model, **options)  # type: ignore[arg-type]
