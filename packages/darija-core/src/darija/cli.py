"""Interface en ligne de commande de ``darija-core``.

Chaque sous-commande lit un fichier (``--file``) ou l'entrée standard ::

    echo "chnowa a7welek" | darija translit
    darija markers --file post.txt
    echo "ken 3andek le temps" | darija segment
    darija normalize --level aggressive --file texte.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .arabizi import arabizi_score, to_arabic, to_arabizi
from .codeswitch import profile, segment
from .markers import explain, rates
from .normalize import Level, normalize, script_ratio


def _read(args: argparse.Namespace) -> str:
    if args.file:
        return Path(args.file).read_text(encoding="utf-8")
    return sys.stdin.read()


def _cmd_normalize(args: argparse.Namespace) -> int:
    print(normalize(_read(args), Level(args.level)))
    return 0


def _cmd_translit(args: argparse.Namespace) -> int:
    text = _read(args)
    if args.to == "arabic":
        print(to_arabic(text, g_as_qaf=args.g_as_qaf))
    else:
        print(to_arabizi(text))
    return 0


def _cmd_segment(args: argparse.Namespace) -> int:
    text = _read(args)
    if args.json:
        print(json.dumps(
            {
                "profile": profile(text),
                "segments": [
                    {"text": s.text, "lang": s.lang, "script": s.script,
                     "start": s.start, "end": s.end}
                    for s in segment(text)
                ],
            },
            ensure_ascii=False, indent=2,
        ))
        return 0
    for s in segment(text):
        print(f"  [{s.lang:8s}] {s.text}")
    print("\nprofil : " + "  ".join(f"{k}={v:.2f}" for k, v in profile(text).items()))
    return 0


def _cmd_markers(args: argparse.Namespace) -> int:
    text = _read(args)
    if args.json:
        print(json.dumps(rates(text), ensure_ascii=False, indent=2))
        return 0
    print(explain(text))
    return 0


def _cmd_detect(args: argparse.Namespace) -> int:
    text = _read(args)
    out = {
        "script": script_ratio(text),
        "arabizi_score": round(arabizi_score(text), 4),
        "code_switching": profile(text),
        "marker_rates": rates(text),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


# ------------------------------------------------------------------ data
def _cmd_data_budget(args: argparse.Namespace) -> int:
    from .data import SOURCES, budget, unlicensed

    b = budget()
    print("sources\n")
    for s in SOURCES.values():
        cap = f"{s.max_bytes / (1024 * 1024):.0f} Mo" if s.max_bytes else "—"
        lic = s.license or "NON DECLAREE"
        print(f"  [{s.role:8s}] {s.key:10s} {cap:>8s}  {lic}")
    print(f"\nplafond cumule des sources bornees : {b['capped_mb']} Mo")
    print(f"non bornees a priori : {', '.join(b['uncapped'])}")

    if missing := unlicensed():
        print("\nATTENTION — sources sans licence declaree :")
        for s in missing:
            print(f"  {s.key}: {s.note}")
        print("  Absence de licence != domaine public. A verifier avant usage public.")
    return 0


def _cmd_data_fetch(args: argparse.Namespace) -> int:
    from .data import fetch_all

    keys = args.only.split(",") if args.only else None
    metas = fetch_all(Path(args.cache), keys=keys,
                      max_lines=args.max_lines, force=args.force)
    total = sum(m.get("mb", 0) or 0 for m in metas)
    lines = sum(m.get("lines", 0) or 0 for m in metas)
    print(f"\n{lines:,} lignes · {total:.1f} Mo dans {args.cache}")
    if failed := [m["key"] for m in metas if m.get("error")]:
        print(f"echecs : {', '.join(failed)}")
        return 1
    return 0


def _cmd_data_build(args: argparse.Namespace) -> int:
    from .data.assemble import available, build

    if args.list:
        print(json.dumps(available(Path(args.cache)), ensure_ascii=False, indent=2))
        return 0
    ds = build(args.contrast, Path(args.cache), balance=not args.no_balance,
               holdout=args.holdout, seed=args.seed)
    print(json.dumps(ds.summary(), ensure_ascii=False, indent=2))
    return 0


def _cmd_data_train(args: argparse.Namespace) -> int:
    from .data.assemble import build
    from .dialect import evaluate, train

    ds = build(args.contrast, Path(args.cache), balance=not args.no_balance,
               holdout=args.holdout, seed=args.seed)
    print(f"{ds.description}\n  entrainement : {len(ds.train_positive):,} positifs / "
          f"{len(ds.train_negative):,} negatifs")

    model = train(ds.train_positive, ds.train_negative,
                  labels=("tunisien", ds.name.replace("vs_", "")),
                  meta={"contrast": ds.name, "sources": ds.summary()["sources"]})
    report = evaluate(model, ds.test_positive, ds.test_negative)
    print("\nevaluation sur donnees tenues a l'ecart :")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    out = Path(args.out or f"models/{ds.name}.json.gz")
    model.save(out)
    print(f"\nmodele ecrit : {out}")
    if report["auc"] < 0.85:
        print("AUC faible — verifier le volume et la proprete des deux classes.")
        return 1
    return 0


def _cmd_data_validate(args: argparse.Namespace) -> int:
    from .data.assemble import score_by_source
    from .dialect import DialectModel

    model = DialectModel.load(args.model)
    rows = score_by_source(model, Path(args.cache))
    print(f"modele {args.model}  (labels {model.labels})\n")
    print(f"  {'source':10s} {'role':9s} {'n':>6s} {'mediane':>9s} {'>=seuil':>8s}")
    print("  " + "-" * 46)
    for k, v in sorted(rows.items(), key=lambda kv: (kv[1]["role"], -kv[1]["median"])):
        print(f"  {k:10s} {v['role']:9s} {v['n']:6d} {v['median']:9.3f} "
              f"{100 * v['above_threshold']:7.1f}%")
    pos = [v for v in rows.values() if v["role"] == "positive"]
    if pos:
        worst = min(v["above_threshold"] for v in pos)
        print(f"\n  pire provenance tunisienne : {100 * worst:.1f}% bien classes")
        if worst < 0.9:
            print("  -> generalisation insuffisante : le modele a appris un corpus, "
                  "pas la langue. Ajoutez une provenance a l'entrainement.")
            return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    """Analyse les arguments et exécute la sous-commande."""
    ap = argparse.ArgumentParser(prog="darija", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name: str, help_: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_)
        p.add_argument("--file", help="fichier d'entrée (défaut : stdin)")
        return p

    p = add("normalize", "normaliser du texte arabe")
    p.add_argument("--level", default="standard",
                   choices=[x.value for x in Level])
    p.set_defaults(fn=_cmd_normalize)

    p = add("translit", "translittérer Arabizi <-> arabe")
    p.add_argument("--to", default="arabic", choices=["arabic", "arabizi"])
    p.add_argument("--g-as-qaf", action="store_true",
                   help="écrire /g/ ق plutôt que ڨ")
    p.set_defaults(fn=_cmd_translit)

    p = add("segment", "segmenter l'alternance codique")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=_cmd_segment)

    p = add("markers", "rapport sur les marqueurs tunisiens")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=_cmd_markers)

    p = add("detect", "rapport complet, en JSON")
    p.set_defaults(fn=_cmd_detect)

    # ------------------------------------------------------------- data
    data = sub.add_parser("data", help="récupérer les corpus et entraîner un modèle")
    dsub = data.add_subparsers(dest="data_cmd", required=True)

    def add_data(name: str, help_: str) -> argparse.ArgumentParser:
        q = dsub.add_parser(name, help=help_)
        q.add_argument("--cache", default="data/raw")
        return q

    q = dsub.add_parser("budget", help="coût et licences, avant de télécharger")
    q.set_defaults(fn=_cmd_data_budget)

    q = add_data("fetch", "télécharger les sources vers le cache")
    q.add_argument("--only", help="clés séparées par des virgules (ex: linto,ary)")
    q.add_argument("--max-lines", type=int, help="plafond de lignes par source")
    q.add_argument("--force", action="store_true", help="retélécharger même si en cache")
    q.set_defaults(fn=_cmd_data_fetch)

    q = add_data("validate", "scorer un modèle source par source")
    q.add_argument("--model", required=True)
    q.set_defaults(fn=_cmd_data_validate)

    for name, helptext, fn in (
        ("build", "assembler un jeu équilibré", _cmd_data_build),
        ("train", "entraîner et évaluer un modèle de dialecte", _cmd_data_train),
    ):
        q = add_data(name, helptext)
        from .data.assemble import CONTRASTS
        q.add_argument("--contrast", default="vs_moroccan_yt",
                       choices=sorted(CONTRASTS))
        q.add_argument("--no-balance", action="store_true",
                       help="ne pas sous-échantillonner la classe majoritaire")
        q.add_argument("--holdout", type=float, default=0.25)
        q.add_argument("--seed", type=int, default=0)
        if name == "build":
            q.add_argument("--list", action="store_true",
                           help="montrer le contenu du cache, ne rien construire")
        else:
            q.add_argument("--out", help="chemin du modèle (défaut: models/<contrast>.json.gz)")
        q.set_defaults(fn=fn)

    args = ap.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
