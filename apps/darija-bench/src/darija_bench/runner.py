"""Collecte des réponses des modèles évalués.

La collecte et la mesure sont **séparées** : ce module n'écrit que des réponses
brutes, et ``report`` les score ensuite. Les appels d'API coûtent de l'argent ;
améliorer le scorer ne doit jamais obliger à les repayer. C'est aussi ce qui
rend le banc rejouable par quelqu'un qui n'a aucune clé — il lui suffit du
fichier de réponses.

Deux conditions sont mesurées, et l'écart entre les deux est l'observation
intéressante :

``implicite``
    Aucune consigne. Le modèle voit une question en tunisien. Répond-il dans la
    même langue, ou glisse-t-il vers la fusha ? C'est le comportement réel que
    rencontre un utilisateur.

``explicite``
    On lui demande de répondre en tunisien. Mesure la capacité, pas le réflexe.

Un modèle qui échoue en implicite et réussit en explicite ne manque pas de
compétence : il manque de calibration. Les deux défauts appellent des réponses
différentes, d'où la séparation.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .prompts import Prompt
from .providers import Provider, ProviderError, RateLimited

#: Nombre de reprises sur un quota momentané. Deux suffisent : au-delà, ce
#: n'est plus un pic de trafic, c'est un plafond.
MAX_RETRIES: int = 2

#: Attente de repli quand le serveur ne conseille aucun délai.
FALLBACK_DELAY: float = 20.0

#: Consignes par condition. ``None`` = aucun prompt système.
CONDITIONS: dict[str, str | None] = {
    "implicite": None,
    "explicite": (
        "جاوب ديما بالدارجة التونسية، موش بالفصحى و موش بلهجة مغاربية أخرى. "
        "استعمل الكلام اللي يستعملو التوانسة في حياتهم اليومية."
    ),
}


@dataclass(frozen=True)
class Reply:
    """Une réponse brute, avant toute mesure."""

    prompt_id: str
    model: str
    condition: str
    script: str
    reply: str
    error: str | None = None

    @property
    def key(self) -> tuple[str, str, str]:
        """Clé d'unicité, pour la reprise après interruption."""
        return (self.prompt_id, self.model, self.condition)


def existing_keys(path: Path) -> set[tuple[str, str, str]]:
    """Clés déjà présentes dans un fichier de réponses.

    Permet de reprendre une campagne interrompue sans repayer les appels déjà
    faits. Un fichier absent ou une ligne illisible ne sont pas des erreurs :
    on reprend ce qu'on peut.
    """
    if not path.exists():
        return set()
    keys: set[tuple[str, str, str]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            keys.add((rec["prompt_id"], rec["model"], rec["condition"]))
        except (json.JSONDecodeError, KeyError):
            continue
    return keys


def _call(provider: Provider, prompt: str, system: str | None, *, sleep=time.sleep) -> str:
    """Appelle un fournisseur, en réessayant les ralentissements passagers.

    Un quota **épuisé** n'est pas réessayé : il remonte immédiatement pour que
    l'appelant abandonne ce modèle. C'est ce qui manquait à la première
    campagne, où 62 appels ont été passés contre un plafond journalier déjà
    atteint.

    Raises:
      ProviderError: échec définitif, quota épuisé compris.

    """
    for attempt in range(MAX_RETRIES + 1):
        try:
            return provider.generate(prompt, system)
        except RateLimited as exc:
            if exc.exhausted or attempt == MAX_RETRIES:
                raise
            # Le serveur sait mieux que nous combien attendre.
            sleep(exc.retry_after if exc.retry_after is not None else FALLBACK_DELAY)
    raise AssertionError("inatteignable")  # pragma: no cover


def collect(
    providers: Iterable[Provider],
    prompts: Iterable[Prompt],
    out: Path,
    *,
    conditions: Iterable[str] = ("implicite", "explicite"),
    resume: bool = True,
    on_progress: object = None,
) -> int:
    """Interroge chaque modèle sur chaque prompt, dans chaque condition.

    Les réponses sont écrites **au fil de l'eau**, une ligne JSON par appel :
    une interruption au milieu d'une campagne ne perd que l'appel en cours.

    Args:
      providers: fournisseurs déjà construits.
      prompts: prompts à poser.
      out: fichier de sortie, en ajout.
      conditions: clés de :data:`CONDITIONS`.
      resume: sauter les couples déjà présents dans ``out``.
      on_progress: appelable optionnel reçevant chaque :class:`Reply` écrite.

    Returns:
      Le nombre d'appels réellement effectués.

    Raises:
      KeyError: condition inconnue.

    """
    conditions = list(conditions)
    for condition in conditions:
        if condition not in CONDITIONS:
            raise KeyError(f"condition inconnue {condition!r} ; connues : {sorted(CONDITIONS)}")

    done = existing_keys(out) if resume else set()
    out.parent.mkdir(parents=True, exist_ok=True)
    calls = 0

    providers = list(providers)
    prompts = list(prompts)
    with out.open("a", encoding="utf-8") as fh:
        for provider in providers:
            # Un quota journalier épuisé condamne tous les appels restants de
            # ce modèle. On l'abandonne au premier refus définitif au lieu de
            # dérouler la liste entière contre un mur.
            abandoned: str | None = None
            for condition in conditions:
                system = CONDITIONS[condition]
                for prompt in prompts:
                    key = (prompt.id, provider.name, condition)
                    if key in done:
                        continue
                    if abandoned:
                        continue
                    try:
                        record = Reply(
                            prompt_id=prompt.id,
                            model=provider.name,
                            condition=condition,
                            script=prompt.script,
                            reply=_call(provider, prompt.text, system),
                        )
                    except ProviderError as exc:
                        # Un échec est consigné, pas propagé : une campagne ne
                        # doit pas s'arrêter parce qu'un prompt a été refusé.
                        if isinstance(exc, RateLimited) and exc.exhausted:
                            abandoned = str(exc)
                        record = Reply(
                            prompt_id=prompt.id,
                            model=provider.name,
                            condition=condition,
                            script=prompt.script,
                            reply="",
                            error=str(exc),
                        )
                    fh.write(json.dumps(record.__dict__, ensure_ascii=False) + "\n")
                    fh.flush()
                    calls += 1
                    if callable(on_progress):
                        on_progress(record)
            if abandoned and callable(on_progress):
                on_progress(
                    Reply(
                        prompt_id="—",
                        model=provider.name,
                        condition="—",
                        script="—",
                        reply="",
                        error=f"quota epuise, modele abandonne : {abandoned[:120]}",
                    )
                )
    return calls


def load_replies(path: Path) -> list[Reply]:
    """Relit un fichier de réponses collectées."""
    out: list[Reply] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        out.append(Reply(**json.loads(line)))
    return out
