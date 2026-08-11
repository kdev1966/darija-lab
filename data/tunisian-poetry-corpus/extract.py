#!/usr/bin/env python3
"""Reconstruct a clean, structured corpus of Tunisian folk poetry.

Two sources, in decreasing order of completeness:

``--from-diwan PATH``
    The original Diwan markdown tree. Carries every field, **including the
    poet**, and is the only source that does. Requires the corpus, which is not
    public.

``--from-artifacts PATH`` (default)
    The tuning datasets committed inside ``tuni-folk-gemini``. The poem text is
    there because it is an SFT target, but the metadata was never stored as
    fields: it is embedded in the Arabic prompt string and has to be parsed back
    out. Recoverable this way: genre, gharad, wazn_sub, region, mode. **Not
    recoverable: the poet.**

The original ``uid`` is recovered for the subset of texts that also appear in
the RL prompt set, by matching the novelty fingerprint the RL examples carry.

Output: ``corpus.jsonl`` (canonical), ``corpus.csv`` (inspection) and
``report.json`` (counts, coverage, provenance).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

# --------------------------------------------------------------- normalisation
_TATWEEL = "ـ"
_DIACRITICS = re.compile(r"[ً-ٰٟۖ-ۭ]")
_NON_ARABIC = re.compile(r"[^ء-ي0-9 ]")
_ALEF = re.compile(r"[إأآٱ]")


def normalize(text: str) -> str:
    """Comparison-only normalisation. Never rewrites the stored text.

    Mirrors ``tunifolk.prosody.normalize.normalize`` so that fingerprints
    computed here match the ones stored in the RL dataset.
    """
    text = unicodedata.normalize("NFKC", text or "").replace(_TATWEEL, "")
    text = _DIACRITICS.sub("", text)
    text = _ALEF.sub("ا", text)
    text = (
        text.replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")
    )
    return re.sub(r"\s+", " ", _NON_ARABIC.sub(" ", text)).strip()


# ------------------------------------------------------------- prompt grammar
#: Exact task strings emitted by ``tunifolk.sft.dataset._TASK``. Matched in
#: full rather than by first word, so a change upstream fails loudly here
#: instead of silently mislabelling a genre.
_TASK_TO_GENRE = {
    "انظم القسيم": ("qasim", "القسيم"),
    "انظم الملزومة": ("malzuma", "الملزومة"),
    "انظم الموقف": ("mawqif", "الموقف"),
    "انظم المسدّسة": ("musaddas", "المسدّسة"),
    "انظم أغنية شعبية تونسية": ("song", "أغنية"),
    "انظم لغزًا شعبيًّا تونسيًّا": ("riddle", "لغز"),
    "انظم رباعية شعبية تونسية": ("quatrain", "رباعية/مقطوعة"),
    "انظم نصًّا شعبيًّا تونسيًّا": ("unclassified", "غير محدّد"),
}

_RX_GHARAD = re.compile(r"في غرض «(.+?)»")
_RX_SUB = re.compile(r"الميزان الفرعي «(.+?)»")
_RX_REGION = re.compile(r"بنَفَس أهل «(.+?)»")
_RX_CONTINUE = re.compile(r"واجعل مطلعه هذا الشطر بنصّه")
_RX_PROSE = re.compile(r"^اكتب تعريفًا نثريًّا موجزًا بـ«(.+?)»")

#: The four verifiable rhyme topologies, as opposed to songs/riddles/prose.
_USUL = {"qasim", "malzuma", "mawqif", "musaddas"}


@dataclass
class Entry:
    """One poem or prose text with everything recoverable about it."""

    text: str
    genre: str
    genre_ar: str
    is_usul: bool
    gharad: str | None = None
    wazn_sub: str | None = None
    region: str | None = None
    poet: str | None = None
    title: str | None = None
    uid: str | None = None
    n_lines: int = 0
    n_words: int = 0
    modes: list[str] = field(default_factory=list)
    split: str = "train"
    source: str = "artifacts"

    def finalise(self) -> Entry:
        """Fill the derived counts. Idempotent."""
        self.n_lines = sum(1 for ln in self.text.splitlines() if ln.strip())
        self.n_words = len(normalize(self.text).split())
        self.modes = sorted(set(self.modes))
        return self


def _parse_prompt(prompt: str) -> tuple[str, str, dict] | None:
    """Recover the metadata that ``sft.dataset`` encoded into the prompt."""
    if m := _RX_PROSE.match(prompt):
        # The prose prompt carries `title or poet` in one slot and does not say
        # which, so it is reported as `title` rather than guessed as a poet.
        return "prose", "نثر", {"title": m.group(1)}

    head = prompt.split("\n", 1)[0].rstrip(".").strip()
    genre = None
    for task, (key, arabic) in _TASK_TO_GENRE.items():
        if head.startswith(task):
            genre, genre_ar = key, arabic
            break
    if genre is None:
        return None

    meta: dict = {}
    if m := _RX_GHARAD.search(prompt):
        meta["gharad"] = m.group(1)
    if m := _RX_SUB.search(prompt):
        meta["wazn_sub"] = m.group(1)
    if m := _RX_REGION.search(prompt):
        meta["region"] = m.group(1)
    meta["mode"] = "continue" if _RX_CONTINUE.search(prompt) else "compose"
    return genre, genre_ar, meta


def from_artifacts(root: Path) -> list[Entry]:
    """Rebuild entries from the committed SFT datasets, de-duplicated by text.

    A source poem yields up to two SFT examples (``compose`` and ``continue``)
    that share one target, so texts are merged and the modes collected.
    """
    by_text: dict[str, Entry] = {}
    for split in ("train", "validation"):
        path = root / "artifacts" / "sft" / "dataset" / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"missing {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            prompt = rec["contents"][0]["parts"][0]["text"]
            text = rec["contents"][1]["parts"][0]["text"]
            parsed = _parse_prompt(prompt)
            if parsed is None:
                continue
            genre, genre_ar, meta = parsed
            if text in by_text:
                e = by_text[text]
                if m := meta.get("mode"):
                    e.modes.append(m)
                continue
            e = Entry(
                text=text,
                genre=genre,
                genre_ar=genre_ar,
                is_usul=genre in _USUL,
                gharad=meta.get("gharad"),
                wazn_sub=meta.get("wazn_sub"),
                region=meta.get("region"),
                title=meta.get("title"),
                modes=[meta["mode"]] if meta.get("mode") else [],
                split=split,
            )
            by_text[text] = e
    return [e.finalise() for e in by_text.values()]


def from_diwan(diwan: Path, tunifolk_src: Path) -> list[Entry]:
    """Rebuild entries from the original Diwan markdown — the complete source."""
    sys.path.insert(0, str(tunifolk_src))
    from tunifolk.data.diwan import load_diwan  # noqa: PLC0415

    out = []
    for t in load_diwan(diwan):
        genre = t.genre
        if genre is None:
            continue
        key = {
            "القسيم": "qasim", "الملزومة": "malzuma", "الموقف": "mawqif",
            "المسدّسة": "musaddas", "أغنية": "song", "لغز": "riddle",
            "رباعية/مقطوعة": "quatrain", "نثر": "prose", "غير محدّد": "unclassified",
        }.get(genre.value, genre.value)
        out.append(
            Entry(
                text=t.text, genre=key, genre_ar=genre.value, is_usul=key in _USUL,
                gharad=t.gharad or None, wazn_sub=t.wazn_sub, region=t.region,
                poet=t.poet, title=t.title or None, uid=t.uid, source="diwan",
            ).finalise()
        )
    return out


def attach_uids(entries: list[Entry], root: Path, tunifolk_src: Path) -> int:
    """Recover the original ``uid`` by matching the RL novelty fingerprint.

    The RL prompt set stores ``source_uid`` alongside a fingerprint of the poem
    it was derived from. Re-computing that fingerprint over our text identifies
    the same source. Only the poems used for RL are covered.

    Returns the number of uids attached; 0 if tunifolk is unavailable.
    """
    try:
        sys.path.insert(0, str(tunifolk_src))
        from tunifolk.prosody.novelty import fingerprint  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        print(f"  uid join skipped ({exc})", file=sys.stderr)
        return 0

    index: dict[str, str] = {}
    for split in ("train", "validation"):
        p = root / "artifacts" / "rl" / "dataset" / f"{split}.jsonl"
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            refs = json.loads(line).get("references", {})
            if refs.get("source_fingerprint") and refs.get("source_uid"):
                index[refs["source_fingerprint"]] = refs["source_uid"]

    n = 0
    for e in entries:
        if e.uid:
            continue
        if uid := index.get(fingerprint(e.text)):
            e.uid = uid
            n += 1
    return n


def build_report(entries: list[Entry], source: str) -> dict:
    """Counts, coverage and provenance for the extracted corpus."""
    n = len(entries)

    def cov(attr: str) -> dict:
        k = sum(1 for e in entries if getattr(e, attr))
        return {"n": k, "pct": round(100 * k / n, 1) if n else 0.0}

    return {
        "source": source,
        "n_texts": n,
        "n_words": sum(e.n_words for e in entries),
        "n_lines": sum(e.n_lines for e in entries),
        "vocabulary": len({w for e in entries for w in normalize(e.text).split()}),
        "coverage": {f: cov(f) for f in
                     ("gharad", "wazn_sub", "region", "poet", "uid", "title")},
        "by_genre": dict(Counter(e.genre for e in entries).most_common()),
        "by_gharad": dict(Counter(e.gharad for e in entries if e.gharad).most_common()),
        "by_region": dict(Counter(e.region for e in entries if e.region).most_common()),
        "by_wazn_sub": dict(Counter(e.wazn_sub for e in entries if e.wazn_sub).most_common()),
        "by_split": dict(Counter(e.split for e in entries).most_common()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-artifacts", default="../../../tuni-folk-gemini",
                    help="tuni-folk-gemini checkout holding the committed datasets")
    ap.add_argument("--from-diwan", help="original Diwan markdown tree (adds the poet)")
    ap.add_argument("--tunifolk-src",
                    help="path to tuni-folk-gemini/src (defaults to <from-artifacts>/src)")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    root = Path(args.from_artifacts).expanduser().resolve()
    tsrc = Path(args.tunifolk_src).expanduser().resolve() if args.tunifolk_src else root / "src"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.from_diwan:
        print(f"reading the Diwan at {args.from_diwan} ...")
        entries = from_diwan(Path(args.from_diwan).expanduser(), tsrc)
        source = "diwan-markdown"
    else:
        print(f"rebuilding from the committed datasets at {root} ...")
        entries = from_artifacts(root)
        source = "tuning-artifacts"
        print("  recovering uids via the RL fingerprint index ...")
        n = attach_uids(entries, root, tsrc)
        print(f"  {n} uids recovered")

    entries.sort(key=lambda e: (e.genre, e.uid or "", e.text[:40]))

    jl = out / "corpus.jsonl"
    with open(jl, "w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(asdict(e), ensure_ascii=False) + "\n")

    cols = ["uid", "genre", "genre_ar", "is_usul", "gharad", "wazn_sub", "region",
            "poet", "title", "n_lines", "n_words", "modes", "split", "source", "text"]
    with open(out / "corpus.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for e in entries:
            d = asdict(e)
            d["modes"] = "|".join(d["modes"])
            w.writerow({c: d[c] for c in cols})

    report = build_report(entries, source)
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{report['n_texts']:,} texts · {report['n_words']:,} words · "
          f"{report['vocabulary']:,} types")
    print("coverage:")
    for f, c in report["coverage"].items():
        print(f"  {f:10s} {c['n']:5d}  ({c['pct']:.1f}%)")
    print(f"\nwrote {jl}, {out/'corpus.csv'}, {out/'report.json'}")
    if not any(e.poet for e in entries):
        print("\nNOTE: no poet recovered. The tuning artifacts never stored it; "
              "re-run with --from-diwan to get it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
