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
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from .prompts import Prompt
from .providers import Provider, ProviderError, RateLimited

#: Nombre de reprises sur un quota momentané. Deux suffisent : au-delà, ce
#: n'est plus un pic de trafic, c'est un plafond.
MAX_RETRIES: int = 2

#: Attente de repli quand le serveur ne conseille aucun délai.
FALLBACK_DELAY: float = 20.0

#: Échecs consécutifs, **sans une seule réussite**, avant d'abandonner un modèle.
#:
#: Un quota épuisé s'annonce (« per day ») et est traité à part. Ce garde-fou
#: vise l'autre cas : un modèle étranglé en amont, qui répond
#: « temporarily rate-limited upstream » sans jamais céder. Mesuré le 12 août
#: sur ``google/gemma-4-31b-it:free`` — chaque prompt coûtait alors deux
#: reprises à 20 s avant d'être consigné, soit 13 minutes pour découvrir que le
#: modèle ne répondrait pas du tout.
#:
#: La condition « aucune réussite » est essentielle : un modèle qui a déjà
#: produit des réponses traverse un creux, il ne faut pas le jeter.
ABANDON_APRES: int = 3

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


@dataclass
class CollectReport:
    """Ce qu'une collecte a réellement produit.

    ``abandoned`` recense les modèles dont le quota s'est épuisé en route. Sans
    cette information, l'appelant ne peut pas distinguer « ce modèle est
    mesuré » de « ce modèle a été coupé au tiers », et publierait les deux
    comme équivalents.
    """

    calls: int = 0
    abandoned: dict[str, str] = field(default_factory=dict)
    #: Appels REUSSIS par modele. Zero reussite avant l'abandon distingue
    #: « ce modele a son quota epuise » de « le COMPTE a son quota epuise ».
    succeeded: dict[str, int] = field(default_factory=lambda: defaultdict(int))


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
) -> CollectReport:
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
      Un :class:`CollectReport` : le nombre d'appels et les modèles abandonnés.

    Raises:
      KeyError: condition inconnue.

    """
    conditions = list(conditions)
    for condition in conditions:
        if condition not in CONDITIONS:
            raise KeyError(f"condition inconnue {condition!r} ; connues : {sorted(CONDITIONS)}")

    done = existing_keys(out) if resume else set()
    out.parent.mkdir(parents=True, exist_ok=True)
    report = CollectReport()

    providers = list(providers)
    prompts = list(prompts)
    with out.open("a", encoding="utf-8") as fh:
        for provider in providers:
            # Un quota journalier épuisé condamne tous les appels restants de
            # ce modèle. On l'abandonne au premier refus définitif au lieu de
            # dérouler la liste entière contre un mur.
            abandoned: str | None = None
            echecs = 0
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
                    report.calls += 1
                    if record.error:
                        echecs += 1
                        # Jamais rien produit et deja `ABANDON_APRES` echecs :
                        # ce modele ne repondra pas, inutile de lui payer ses
                        # vingt appels a 40 s chacun.
                        if not report.succeeded[provider.name] and echecs >= ABANDON_APRES:
                            abandoned = f"{echecs} echecs d'affilee sans une seule reponse"
                    else:
                        report.succeeded[provider.name] += 1
                        echecs = 0
                    if callable(on_progress):
                        on_progress(record)
            if abandoned:
                report.abandoned[provider.name] = abandoned
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
    return report


def load_replies(path: Path) -> list[Reply]:
    """Relit un fichier de réponses collectées."""
    out: list[Reply] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        out.append(Reply(**json.loads(line)))
    return out


def relay(
    pool: Iterable[str],
    prompts: Iterable[Prompt],
    out: Path,
    *,
    target: int,
    conditions: Iterable[str] = ("implicite", "explicite"),
    resume: bool = True,
    on_progress: object = None,
) -> dict[str, str]:
    """Mesure ``target`` modèles en puisant dans une réserve de candidats.

    Sur un palier gratuit, un modèle peut être hors d'atteinte avant le premier
    appel — ``gemini-3.1-pro`` avait un quota de **zéro** — ou s'épuiser au
    tiers d'une campagne. Passer une liste fixe de modèles produit alors des
    mesures tronquées qu'on ne peut pas comparer entre elles.

    Le relais renverse la logique : on ne demande pas « mesure ces trois
    modèles » mais « obtiens-moi trois mesures complètes », et le prochain
    candidat remplace celui qui tombe. La réserve est parcourue dans l'ordre
    donné, donc les candidats préférés se placent en tête.

    Args:
      pool: spécifications ``fournisseur:modele``, par ordre de préférence.
      prompts: prompts à poser — le même jeu pour tous, sans quoi les mesures
        ne seraient pas comparables.
      out: fichier de réponses, en ajout.
      target: nombre de mesures complètes visé.
      conditions: clés de :data:`CONDITIONS`.
      resume: sauter les couples déjà présents dans ``out``.
      on_progress: appelable optionnel reçevant chaque :class:`Reply`.

    Returns:
      Le sort de chaque candidat essayé : ``complet``, ``quota épuisé``, ou le
      message d'erreur s'il n'a pas pu être construit. Les candidats jamais
      atteints — parce que la cible était déjà remplie — sont absents.

    """
    from .providers import ProviderError as _PE  # noqa: PLC0415
    from .providers import build  # noqa: PLC0415

    prompts = list(prompts)
    conditions = list(conditions)
    issues: dict[str, str] = {}
    complets = 0

    for spec in pool:
        if complets >= target:
            break
        try:
            provider = build(spec)
        except _PE as exc:
            # Un candidat inconstructible ne doit pas arrêter le relais : c'est
            # exactement le cas qu'il existe pour absorber.
            issues[spec] = str(exc)
            continue

        report = collect(
            [provider], prompts, out,
            conditions=conditions, resume=resume, on_progress=on_progress,
        )
        if provider.name in report.abandoned:
            # ⚠️ Épuisé DÈS LE PREMIER APPEL = le plafond est celui du COMPTE,
            # pas du modèle. C'est le cas d'OpenRouter (~50 requêtes par jour,
            # tous modèles confondus). Continuer la réserve ne fait alors que
            # dépenser un appel par candidat pour heurter le même mur — ce qui
            # est arrivé quatre fois de suite le 11 août.
            if not report.succeeded.get(provider.name):
                issues[spec] = "quota du compte épuisé — réserve interrompue"
                break
            issues[spec] = "quota épuisé"
        else:
            issues[spec] = "complet"
            complets += 1

    return issues
