"""Interface en ligne de commande du banc.

Trois sous-commandes, qui suivent la séparation collecte / mesure ::

    darija-bench prompts                                  # inspecter le jeu
    darija-bench run --model anthropic:claude-opus-5 ...  # collecter (payant)
    darija-bench report --replies replies.jsonl ...       # mesurer (gratuit)

``report`` ne touche à aucune API : il se relance autant de fois qu'on veut,
y compris après avoir modifié le scorer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from darija.dialect import DialectModel

from . import prompts as prompts_mod
from . import report as report_mod
from .providers import ProviderError, build
from .runner import CONDITIONS, Reply, collect, load_replies


def _cmd_prompts(args: argparse.Namespace) -> int:
    """Affiche le jeu de prompts."""
    items = prompts_mod.load()
    for script, group in prompts_mod.by_script(items):
        print(f"\n=== {script} ({len(group)}) ===")
        for prompt in group:
            print(f"  {prompt.id}  [{prompt.category:<11}] {prompt.text}")
    print(f"\n{len(items)} prompts, {len(list(prompts_mod.by_script(items)))} ecritures")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    """Collecte les réponses des modèles."""
    items = prompts_mod.load()
    if args.limit:
        items = items[: args.limit]

    try:
        providers = [build(spec) for spec in args.model]
    except ProviderError as exc:
        print(f"erreur : {exc}", file=sys.stderr)
        return 2

    conditions = args.condition or list(CONDITIONS)
    planned = len(providers) * len(conditions) * len(items)
    print(
        f"{len(providers)} modele(s) x {len(conditions)} condition(s) x {len(items)} prompts "
        f"= {planned} appels au maximum (les deja-faits sont sautes)."
    )
    if not args.yes:
        answer = input("Ces appels sont factures. Continuer ? [o/N] ").strip().lower()
        if answer not in {"o", "oui", "y", "yes"}:
            print("annule.")
            return 1

    def progress(record: Reply) -> None:
        state = "ERREUR" if record.error else f"{len(record.reply.split()):>4} mots"
        print(f"  {record.model}  {record.condition:<10} {record.prompt_id}  {state}")

    made = collect(
        providers,
        items,
        Path(args.out),
        conditions=conditions,
        resume=not args.no_resume,
        on_progress=progress,
    )
    print(f"\n{made} appels effectues, ecrits dans {args.out}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    """Mesure des réponses déjà collectées."""
    path = Path(args.replies)
    if not path.exists():
        print(f"erreur : {path} introuvable — lancez d'abord `darija-bench run`", file=sys.stderr)
        return 2

    model = DialectModel.load(args.dialect_model)
    replies = load_replies(path)
    verdicts = report_mod.score_all(replies, model)
    cells = report_mod.aggregate(replies, verdicts)
    print(report_mod.render(cells, model.threshold))

    if args.details:
        print("\n=== reponses jugees non tunisiennes ===")
        for verdict in verdicts:
            if not verdict.scorable or verdict.is_tunisian:
                continue
            cause = (
                "classifieur sous le seuil"
                if not verdict.above_classifier
                else f"seuil franchi mais {verdict.n_markers} marqueur(s) distinct(s)"
            )
            print(
                f"\n  {verdict.model}  {verdict.condition}  {verdict.prompt_id}"
                f"  score={verdict.score}  → {cause}"
            )
            print("  " + verdict.explanation.replace("\n", "\n  "))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée."""
    parser = argparse.ArgumentParser(
        prog="darija-bench",
        description="Un LLM repond-il vraiment en tunisien ?",
    )
    sub = parser.add_subparsers(required=True)

    p_prompts = sub.add_parser("prompts", help="afficher le jeu de prompts")
    p_prompts.set_defaults(fn=_cmd_prompts)

    p_run = sub.add_parser("run", help="collecter les reponses (appels factures)")
    p_run.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="SPEC",
        help="fournisseur:modele, repetable (ex: anthropic:claude-opus-5)",
    )
    p_run.add_argument("--out", default="replies.jsonl", help="fichier de reponses")
    p_run.add_argument(
        "--condition", action="append", choices=sorted(CONDITIONS), help="par defaut : les deux"
    )
    p_run.add_argument("--limit", type=int, help="n'utiliser que les N premiers prompts")
    p_run.add_argument("--no-resume", action="store_true", help="ne pas sauter les deja-faits")
    p_run.add_argument("--yes", action="store_true", help="ne pas demander confirmation")
    p_run.set_defaults(fn=_cmd_run)

    p_report = sub.add_parser("report", help="mesurer des reponses collectees")
    p_report.add_argument("--replies", default="replies.jsonl")
    p_report.add_argument(
        "--dialect-model", required=True, help="chemin du modele .json.gz de darija-core"
    )
    p_report.add_argument(
        "--details", action="store_true", help="detailler les reponses jugees non tunisiennes"
    )
    p_report.set_defaults(fn=_cmd_report)

    args = parser.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
