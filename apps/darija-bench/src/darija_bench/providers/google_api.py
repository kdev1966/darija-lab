"""Adaptateur Google, via le SDK officiel ``google-genai``.

Comme pour OpenAI, l'identifiant de modèle est obligatoire : le catalogue
Gemini bouge, et une chaîne devinée produirait un 404 au milieu d'une
campagne. ``client.models.list()`` dit ce que la clé atteint réellement.

Le SDK lit la clé dans ``GEMINI_API_KEY`` ou ``GOOGLE_API_KEY`` (vérifié sur
``google-genai`` 2.17.0).

Deux cas que le chemin nominal ne traite pas et qui arrivent en pratique :

- **Clé absente.** Le SDK lève une erreur peu lisible au moment de construire
  le client. On la rattrape pour dire quoi faire.
- **Réponse vide.** Gemini peut bloquer une génération sur ses propres filtres
  et rendre une réponse sans texte. Ce n'est pas une réponse en fusha, c'est
  une non-réponse : il faut qu'elle soit consignée comme erreur, sinon elle
  compterait comme un échec linguistique du modèle.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from . import ProviderError, RateLimited

MAX_TOKENS: int = 4000

#: Variables lues par le SDK, dans cet ordre.
KEY_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")

#: Délai conseillé par le serveur, en fin de message : « Please retry in 47.8s ».
_RETRY = re.compile(r"retry in ([\d.]+)s")

#: Identifiant de quota journalier. Mesuré sur le palier gratuit :
#: ``GenerateRequestsPerDayPerProjectPerModel-FreeTier`` avec ``limit: 20``
#: pour ``gemini-3.6-flash``, et ``limit: 0`` pour ``gemini-3.1-pro`` — qui
#: n'a donc aucun accès gratuit. Aucune attente ne rouvre ces quotas.
_DAILY = re.compile(r"PerDay|limit: 0\b")


def _classify(message: str) -> ProviderError | None:
    """Reconnaît un refus de quota dans le message d'erreur du SDK.

    Le SDK ne type pas les 429 : le code et les métriques ne sont lisibles que
    dans le texte. On lit donc le texte, faute de mieux.
    """
    if "RESOURCE_EXHAUSTED" not in message and "429" not in message:
        return None
    delay = _RETRY.search(message)
    return RateLimited(
        message,
        retry_after=float(delay.group(1)) if delay else None,
        exhausted=bool(_DAILY.search(message)),
    )


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
        """Envoie ``prompt`` et rend le texte de la réponse.

        Raises:
          ProviderError: clé absente, échec d'appel, ou génération bloquée.

        """
        from google import genai  # noqa: PLC0415
        from google.genai import types  # noqa: PLC0415

        if not any(os.environ.get(var) for var in KEY_VARS):
            raise ProviderError(
                f"{self.name} : aucune clé dans {' ni '.join(KEY_VARS)}. "
                "Créez-la sur aistudio.google.com, puis exportez-la avant de lancer."
            )

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
            quota = _classify(str(exc))
            if quota is not None:
                raise type(quota)(
                    f"{self.name} : {exc}",
                    retry_after=getattr(quota, "retry_after", None),
                    exhausted=getattr(quota, "exhausted", False),
                ) from exc
            raise ProviderError(f"{self.name} : {exc}") from exc

        text = response.text
        if not text:
            # Génération bloquée ou vide : une non-réponse, pas une réponse en
            # fusha. La consigner comme erreur évite de la compter comme un
            # échec linguistique du modèle.
            reason = None
            candidates = getattr(response, "candidates", None) or []
            if candidates:
                reason = getattr(candidates[0], "finish_reason", None)
            raise ProviderError(f"{self.name} : réponse vide (finish_reason={reason})")
        return text


def make(model: str, **options: object) -> GoogleProvider:
    """Construit l'adaptateur. Appelé par :func:`darija_bench.providers.build`."""
    return GoogleProvider(model=model, **options)  # type: ignore[arg-type]
