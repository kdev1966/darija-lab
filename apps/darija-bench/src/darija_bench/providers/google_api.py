"""Adaptateur Google, via le SDK officiel ``google-genai``.

Comme pour OpenAI, l'identifiant de modèle est obligatoire.

Non exercé contre l'API réelle faute de clé au moment de l'écriture.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import ProviderError

MAX_TOKENS: int = 4000


@dataclass
class GoogleProvider:
    """Un modèle Gemini interrogé via ``google-genai``."""

    model: str
    max_tokens: int = MAX_TOKENS

    @property
    def name(self) -> str:
        """Nom reporté dans les résultats."""
        return f"google:{self.model}"

    def generate(self, prompt: str, system: str | None = None) -> str:
        """Envoie ``prompt`` et rend le texte de la réponse."""
        from google import genai  # noqa: PLC0415
        from google.genai import types  # noqa: PLC0415

        client = genai.Client()
        config = types.GenerateContentConfig(
            max_output_tokens=self.max_tokens,
            system_instruction=system or None,
        )
        try:
            response = client.models.generate_content(
                model=self.model, contents=prompt, config=config
            )
        except Exception as exc:  # pragma: no cover - dépend du réseau
            raise ProviderError(f"{self.name} : {exc}") from exc

        return response.text or ""


def make(model: str, **options: object) -> GoogleProvider:
    """Construit l'adaptateur. Appelé par :func:`darija_bench.providers.build`."""
    return GoogleProvider(model=model, **options)  # type: ignore[arg-type]
