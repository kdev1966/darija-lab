"""Interface en ligne de commande du banc.

Trois sous-commandes, qui suivent la séparation collecte / mesure ::

    darija-bench prompts                                  # inspecter le jeu
    darija-bench run --model anthropic:claude-opus-5 ...  # collecter (payant)
    darija-bench report --replies replies.jsonl ...       # mesurer (gratuit)
    darija-bench serve --dialect-model ...                # interface locale
    darija-bench triage --input corpus.jsonl ...          # trier une pile

``report`` ne touche à aucune API : il se relance autant de fois qu'on veut,
y compris après avoir modifié le scorer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from darija.dialect import DialectModel

from . import prompts as prompts_mod
from . import report as report_mod
from .providers import ProviderError, build
from .runner import CONDITIONS, Reply, collect, load_replies, relay
from .scoring import Scorer


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
    if args.sample:
        items = prompts_mod.sample(items, args.sample, seed=args.seed)
    elif args.limit:
        # `--limit` prend les N premiers, donc jamais d'Arabizi : c'est un
        # raccourci de mise au point, pas un protocole. `--sample` est ce
        # qu'il faut pour une mesure sous plafond de quota.
        items = items[: args.limit]

    conditions = args.condition or list(CONDITIONS)
    vise = args.target or len(args.model)
    planned = vise * len(conditions) * len(items)
    mode = "relais" if args.target else "liste fixe"
    print(
        f"{mode} : {vise} mesure(s) x {len(conditions)} condition(s) x {len(items)} prompts "
        f"= {planned} appels au maximum (les deja-faits sont sautes)."
    )
    if args.target:
        print(f"reserve de {len(args.model)} candidat(s) ; un modele epuise est remplace.")
    if not args.yes:
        answer = input("Ces appels sont factures. Continuer ? [o/N] ").strip().lower()
        if answer not in {"o", "oui", "y", "yes"}:
            print("annule.")
            return 1

    def progress(record: Reply) -> None:
        state = "ERREUR" if record.error else f"{len(record.reply.split()):>4} mots"
        print(f"  {record.model}  {record.condition:<10} {record.prompt_id}  {state}")

    if args.target:
        issues = relay(
            args.model, items, Path(args.out),
            target=args.target, conditions=conditions,
            resume=not args.no_resume, on_progress=progress,
        )
        print("\nsort des candidats :")
        for spec, etat in issues.items():
            print(f"  {spec:<48} {etat}")
        return 0

    try:
        providers = [build(spec) for spec in args.model]
    except ProviderError as exc:
        print(f"erreur : {exc}", file=sys.stderr)
        return 2
    made = collect(
        providers, items, Path(args.out),
        conditions=conditions, resume=not args.no_resume, on_progress=progress,
    )
    print(f"\n{made.calls} appels effectues, ecrits dans {args.out}")
    return 0


def _scorer(args: argparse.Namespace) -> Scorer:
    """Construit le scorer, avec le modele Arabizi s'il est fourni."""
    arabizi = DialectModel.load(args.arabizi_model) if args.arabizi_model else None
    return Scorer(DialectModel.load(args.dialect_model), arabizi)


def _arabizi_option(parser: argparse.ArgumentParser) -> None:
    """Ajoute le drapeau du modele Arabizi."""
    parser.add_argument(
        "--arabizi-model",
        help="modele entraine dans l'alphabet reduit (contraste vs_maghreb_arabizi), "
        "employe pour le texte translittere. Sans lui, tout passe par "
        "--dialect-model : 77 %% de TUNIZI reconnu au lieu de 87 %%.",
    )


def _cmd_report(args: argparse.Namespace) -> int:
    """Mesure des réponses déjà collectées."""
    path = Path(args.replies)
    if not path.exists():
        print(f"erreur : {path} introuvable — lancez d'abord `darija-bench run`", file=sys.stderr)
        return 2

    model = _scorer(args)
    replies = load_replies(path)
    verdicts = report_mod.score_all(replies, model)
    print(
        report_mod.render(
            report_mod.aggregate(replies, verdicts, model.model.threshold),
            report_mod.paired_shifts(verdicts),
            report_mod.by_register(verdicts, prompts_mod.load()),
            model.model.threshold,
        )
    )

    if args.details:
        # Les réponses de la bande d'indécision, celles où la mesure ne tranche
        # pas. Ce sont elles qu'il faut lire à la main — c'est ainsi qu'on a
        # découvert que les « rejets » étaient du tunisien authentique.
        print(f"\n=== reponses dans la bande d'indecision (+/-{report_mod.BORDERLINE}) ===")
        for v in verdicts:
            if not v.scorable or v.score is None:
                continue
            if abs(v.score - model.model.threshold) >= report_mod.BORDERLINE:
                continue
            print(
                f"\n  {v.model}  {v.condition}  {v.prompt_id}"
                f"  score={v.score}  ecart={v.score - model.model.threshold:+.4f}"
                f"  marqueurs={v.n_markers}"
            )
            print("  " + v.explanation.replace("\n", "\n  "))
    return 0


def _cmd_triage(args: argparse.Namespace) -> int:
    """Trie une pile de textes."""
    from .triage import group_documents, judge, read_documents, summarise  # noqa: PLC0415

    model = _scorer(args)
    try:
        docs = list(read_documents(Path(args.input), args.field))
    except (FileNotFoundError, ValueError) as exc:
        print(f"erreur : {exc}", file=sys.stderr)
        return 2
    if args.group:
        docs = list(group_documents(iter(docs), args.group))
    if args.limit:
        docs = docs[: args.limit]

    verdicts = [
        judge(texte, model, ident=ident, strict=args.strict) for ident, texte in docs
    ]
    print(summarise(verdicts))

    if args.out:
        with Path(args.out).open("w", encoding="utf-8") as fh:
            for v in verdicts:
                fh.write(json.dumps(v.as_dict(), ensure_ascii=False) + "\n")
        print(f"\nverdicts ecrits dans {args.out}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    """Ouvre l'interface locale."""
    from .web import serve  # noqa: PLC0415 - import tardif : pas de coût si inutilisé

    serve(DialectModel.load(args.dialect_model), port=args.port)
    return 0


#: Nom du fichier de cles, cherche depuis le repertoire courant vers la racine.
FICHIER_ENV = ".env"


def charger_env(depart: Path | None = None) -> list[str]:
    """Charge `.env` dans l'environnement, sans dependance.

    Le depot range les cles d'API dans `apps/darija-bench/.env` (gitignore),
    mais **rien ne le lisait** : une campagne entiere de 336 appels a echoue
    avec « aucune cle dans GEMINI_API_KEY », sans qu'une seule requete parte.
    Les campagnes precedentes ne marchaient que parce que les variables avaient
    ete exportees a la main dans le terminal.

    Une variable deja presente dans l'environnement gagne : `.env` est une
    commodite, pas une autorite.

    Args:
      depart: repertoire ou commencer la recherche. Defaut : le courant.

    Returns:
      Les noms des variables effectivement posees.

    """
    dossier = (depart or Path.cwd()).resolve()
    for base in (dossier, *dossier.parents):
        chemin = base / FICHIER_ENV
        if not chemin.is_file():
            continue
        poses = []
        for ligne in chemin.read_text(encoding="utf-8").splitlines():
            ligne = ligne.strip()
            if not ligne or ligne.startswith("#") or "=" not in ligne:
                continue
            cle, _, valeur = ligne.partition("=")
            cle, valeur = cle.strip(), valeur.strip().strip("\"'")
            if cle and cle not in os.environ:
                os.environ[cle] = valeur
                poses.append(cle)
        return poses
    return []


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée."""
    charger_env()
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
        help="fournisseur:modele, repetable (ex: anthropic:claude-opus-5). "
        "Avec --target, cette liste devient une reserve de candidats.",
    )
    p_run.add_argument(
        "--target",
        type=int,
        metavar="N",
        help="viser N mesures COMPLETES en puisant dans la reserve --model. "
        "Un modele dont le quota s'epuise est remplace par le suivant.",
    )
    p_run.add_argument("--out", default="replies.jsonl", help="fichier de reponses")
    p_run.add_argument(
        "--condition", action="append", choices=sorted(CONDITIONS), help="par defaut : les deux"
    )
    p_run.add_argument(
        "--sample",
        type=int,
        metavar="N",
        help="tirer N prompts, stratifies par ecriture et reproductibles "
        "(pour tenir sous un plafond de quota)",
    )
    p_run.add_argument("--seed", type=int, default=0, help="graine du tirage (defaut 0)")
    p_run.add_argument(
        "--limit",
        type=int,
        help="les N premiers prompts — mise au point seulement, jamais d'arabizi",
    )
    p_run.add_argument("--no-resume", action="store_true", help="ne pas sauter les deja-faits")
    p_run.add_argument("--yes", action="store_true", help="ne pas demander confirmation")
    p_run.set_defaults(fn=_cmd_run)

    p_report = sub.add_parser("report", help="mesurer des reponses collectees")
    p_report.add_argument("--replies", default="replies.jsonl")
    p_report.add_argument(
        "--dialect-model", required=True, help="chemin du modele .json.gz de darija-core"
    )
    _arabizi_option(p_report)
    p_report.add_argument(
        "--details", action="store_true", help="detailler les reponses jugees non tunisiennes"
    )
    p_report.set_defaults(fn=_cmd_report)

    p_serve = sub.add_parser("serve", help="interface locale pour mesurer un texte colle")
    p_serve.add_argument(
        "--dialect-model", required=True, help="chemin du modele .json.gz de darija-core"
    )
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(fn=_cmd_serve)

    p_triage = sub.add_parser("triage", help="trier une pile de textes")
    p_triage.add_argument("--input", required=True, help="fichier .jsonl/.txt/.csv, ou dossier")
    p_triage.add_argument("--field", default="text", help="colonne de texte (jsonl/csv)")
    p_triage.add_argument(
        "--dialect-model", required=True, help="chemin du modele .json.gz de darija-core"
    )
    _arabizi_option(p_triage)
    p_triage.add_argument("--out", help="ecrire un verdict par document, en jsonl")
    p_triage.add_argument(
        "--group",
        type=int,
        nargs="?",
        const=60,
        metavar="N",
        help="agreger les fragments consecutifs en unites de N mots (defaut 60). "
        "Pour les corpus faits de lignes courtes, ou 94 %% ressortent indecidables. "
        "Le verdict porte alors sur le corpus, pas sur un item.",
    )
    p_triage.add_argument(
        "--strict",
        action="store_true",
        help="exiger aussi un marqueur discriminant. Coute 10 a 37 points sur du "
        "texte humain sans rien gagner sur les contre-exemples : a reserver aux "
        "piles soupconnees de contenir du texte genere.",
    )
    p_triage.add_argument("--limit", type=int, help="n'examiner que les N premiers documents")
    p_triage.set_defaults(fn=_cmd_triage)

    args = parser.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
