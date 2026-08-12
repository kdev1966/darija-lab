"""Récupération des sources vers un cache local, une ligne de texte par ligne.

Chaque source produit ``<cache>/<key>.txt`` plus ``<cache>/<key>.meta.json``
consignant ce qui a réellement été lu — nombre de lignes, octets, troncature.
Le coût annoncé reste ainsi vérifiable après coup.

Les dumps Wikipédia ne demandent que la bibliothèque standard. Les jeux hébergés
sur Hugging Face exigent l'extra ``data``::

    pip install -e ".[data]"
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from urllib.request import Request, urlopen

from ..normalize import Level, normalize
from . import wikipedia
from .sources import SOURCES, Source

#: Cache par défaut. Volontairement hors du paquet : ce sont des données, pas du code.
DEFAULT_CACHE = Path("data/raw")

#: Extensions reconnues dans un dépôt Hugging Face, par ordre de préférence.
_HF_EXTS = (".parquet", ".jsonl", ".json", ".txt", ".csv", ".tsv")

#: Étiquette de polarité en tête de ligne, tous séparateurs confondus.
_RX_LABEL = re.compile(r"^\s*-?[01]\s*[;,\t]\s*|^\s*-?[01]\s+(?=\S)")


class MissingExtra(RuntimeError):
    """L'extra ``data`` n'est pas installé."""


def _require_hf() -> tuple[object, object]:
    try:
        import pyarrow.parquet as pq  # noqa: PLC0415
        from huggingface_hub import HfApi, hf_hub_download  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - dépend de l'environnement
        raise MissingExtra(
            "les sources Hugging Face demandent l'extra 'data' : "
            'pip install -e ".[data]"'
        ) from exc
    return (HfApi, hf_hub_download), pq


def _clean(line: str, min_words: int) -> str | None:
    line = _RX_NOISE.sub(" ", line or "")
    line = _RX_EMOJI.sub(" ", line)
    line = _RX_PUNCT_RUN.sub(r"\1", line)
    line = re.sub(r"\s+", " ", line).strip()
    if not line or len(normalize(line, Level.STANDARD).split()) < min_words:
        return None
    return line


#: Noms de colonne portant le texte, par ordre de préférence. Les jeux issus de
#: Twitter embarquent souvent des colonnes de **données personnelles** (nom,
#: pseudo, âge, nombre d'abonnés). Cibler la colonne de texte par son nom, et
#: ne retenir qu'elle, est aussi ce qui évite de recopier ces données.
_TEXT_COLUMNS = (
    "comment", "text", "tweet", "tweets", "post", "sentence", "content", "body",
)

#: Mentions @pseudo, URLs et RT : bruit pour un modèle de n-grammes, et des
#: données personnelles qu'on n'a aucune raison de conserver.
#:
#: Les **hashtags** et les **emoji** sont retirés pour une raison distincte et
#: mesurée : ce sont des marqueurs de *plateforme*, pas de langue. Un modèle
#: entraîné sans ce filtre a appris à séparer le tunisien de l'algérien sur
#: « 😂😂😂😂 » et « #الج » — c'est-à-dire Twitter contre Facebook, avec une AUC
#: de 0,9995 qui ne mesurait rien de linguistique.
_RX_NOISE = re.compile(r"@\w+|#\S+|https?://\S+|www\.\S+|\bRT\b")

#: Emoji et symboles décoratifs.
_RX_EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0000FE00-\U0000FE0F"
    "\U0001F1E6-\U0001F1FF\U00002190-\U000021FF\U00002B00-\U00002BFF]+"
)

#: Répétitions de ponctuation (« .... », « !!!! ») : bruit de plateforme.
_RX_PUNCT_RUN = re.compile(r"([^\w\s\u0621-\u064A])\1{2,}")


def _csv_column(raw: str) -> list[str]:
    """Extrait la colonne de texte d'un CSV, sans connaître son schéma.

    On tente d'abord les noms usuels, puis on retombe sur la colonne dont les
    valeurs sont en moyenne les plus longues — dans ces jeux annotés, les autres
    colonnes sont des étiquettes ou des identifiants.
    """
    import csv  # noqa: PLC0415
    from io import StringIO  # noqa: PLC0415

    rows = list(csv.DictReader(StringIO(raw)))
    if not rows:
        return []
    fields = [f for f in rows[0] if f]
    for want in _TEXT_COLUMNS:
        for f in fields:
            if f.strip().lower() == want:
                return [r.get(f) or "" for r in rows]
    best, best_len = None, 0.0
    for f in fields:
        vals = [r.get(f) or "" for r in rows[:200]]
        avg = sum(len(v) for v in vals) / max(1, len(vals))
        if avg > best_len:
            best, best_len = f, avg
    return [r.get(best) or "" for r in rows] if best else []


def _iter_url(src: Source, min_words: int) -> Iterator[str]:
    """Lit une ou plusieurs URLs.

    ``locator`` peut en contenir plusieurs, séparées par des espaces — TSAC, par
    exemple, éclate son corpus en quatre fichiers.
    """
    for url in src.locator.split():
        req = Request(url, headers={"User-Agent": "darija-core/0.1"})
        with urlopen(req) as resp:  # noqa: S310 - URLs https fixes du registre
            raw = resp.read().decode("utf-8", errors="replace")
        lines = _csv_column(raw) if url.lower().endswith(".csv") else raw.splitlines()
        for line in lines:
            # Ces corpus préfixent parfois une étiquette de polarité, avec des
            # séparateurs qui diffèrent d'un jeu à l'autre : TUNIZI écrit
            # « 1;texte », TSAC sépare par une tabulation ou un espace. On ne
            # garde que le texte — la polarité ne sert pas à un classifieur de
            # langue.
            line = _RX_LABEL.sub("", line)
            if cleaned := _clean(line, min_words):
                yield cleaned


def _reservoir(lines: Iterator[str], k: int, seed: int = 0) -> list[str]:
    """Échantillon uniforme de ``k`` lignes en une passe, mémoire bornée.

    Tronquer en tête serait un piège mesuré : LinTO concatène ses sources par
    blocs, et ses 40 premiers pourcents sont de l'arabe standard. Un
    ``--max-lines 40000`` naïf n'aurait ramené aucun dialecte, sans le signaler.
    """
    import random  # noqa: PLC0415

    rng = random.Random(seed)
    out: list[str] = []
    for i, line in enumerate(lines):
        if i < k:
            out.append(line)
        else:
            j = rng.randint(0, i)
            if j < k:
                out[j] = line
    return out


def _iter_hf(src: Source, min_words: int, max_lines: int | None) -> Iterator[str]:
    (HfApi, hf_hub_download), pq = _require_hf()
    files = HfApi().list_repo_files(src.locator, repo_type="dataset")
    picked = [f for f in files if f.lower().endswith(_HF_EXTS)]
    if src.include:
        picked = [f for f in picked if f.startswith(src.include)]
    if not picked:
        return
    picked.sort(key=lambda f: (_HF_EXTS.index(Path(f).suffix.lower()), f))

    def all_lines() -> Iterator[str]:
        for name in picked:
            path = Path(hf_hub_download(src.locator, name, repo_type="dataset"))
            for line in _iter_hf_file(path, pq):
                if cleaned := _clean(line, min_words):
                    yield cleaned

    if max_lines is None:
        yield from all_lines()
    else:
        # Échantillon uniforme sur tout le dépôt, jamais une troncature en tête.
        yield from _reservoir(all_lines(), max_lines)


def _iter_hf_file(path: Path, pq: object) -> Iterator[str]:
    """Extrait le texte d'un fichier de jeu de données, quel que soit son schéma."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        table = pq.read_table(path)  # type: ignore[attr-defined]
        # On ne connaît pas le schéma a priori : on retient la colonne texte
        # la plus « longue » en moyenne, qui est le contenu dans tous les jeux
        # visés (les autres sont des étiquettes ou des identifiants).
        best, best_len = None, 0.0
        for name in table.column_names:
            col = table.column(name).to_pylist()[:200]
            strings = [x for x in col if isinstance(x, str)]
            if not strings:
                continue
            avg = sum(len(x) for x in strings) / len(strings)
            if avg > best_len:
                best, best_len = name, avg
        if best is None:
            return
        yield from (x for x in table.column(best).to_pylist() if isinstance(x, str))
        return

    text = path.read_text(encoding="utf-8", errors="replace")
    if suffix in (".jsonl", ".json"):
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, str):
                yield obj
            elif isinstance(obj, dict):
                strings = [v for v in obj.values() if isinstance(v, str)]
                if strings:
                    yield max(strings, key=len)
        return

    for line in text.splitlines():
        yield line.split("\t")[-1] if suffix == ".tsv" else line


def fetch(
    src: Source,
    cache: Path = DEFAULT_CACHE,
    *,
    max_lines: int | None = None,
    min_words: int = 4,
    force: bool = False,
) -> dict[str, object]:
    """Récupère une source vers ``<cache>/<key>.txt``.

    Ne retélécharge pas si le fichier existe, sauf ``force=True``.

    Returns:
      Le dictionnaire de métadonnées, également écrit à côté du fichier.

    """
    cache.mkdir(parents=True, exist_ok=True)
    out = cache / f"{src.key}.txt"
    meta_path = cache / f"{src.key}.meta.json"

    if out.exists() and not force and meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))

    if src.kind == "local":
        # Produit par ce dépôt, pas téléchargé. On sort AVANT d'ouvrir le
        # fichier en écriture : `open(out, "w")` le tronquerait, donc un
        # `fetch --force` détruirait le corpus au lieu de le régénérer.
        if not out.exists():
            raise FileNotFoundError(
                f"{src.key} est produit localement et absent de {cache}. "
                f"Voir : {src.locator}"
            )
        return {
            "key": src.key, "role": src.role, "kind": src.kind,
            "locator": src.locator, "license": src.license,
            "lines": sum(1 for _ in out.open(encoding="utf-8")),
            "bytes": out.stat().st_size, "path": str(out),
        }

    stats: dict[str, object] = {}
    n = 0
    with open(out, "w", encoding="utf-8") as fh:
        if src.kind == "wikipedia":
            wiki_stats = wikipedia.DumpStats()
            lines = wikipedia.iter_lines(
                src.locator, max_bytes=src.max_bytes, max_lines=max_lines,
                min_words=min_words,
            )
            for line in lines:
                fh.write(line + "\n")
                n += 1
            stats = wiki_stats.as_dict()
        elif src.kind == "hf":
            for line in _iter_hf(src, min_words, max_lines):
                fh.write(line + "\n")
                n += 1
        else:
            for line in _iter_url(src, min_words):
                fh.write(line + "\n")
                n += 1
                if max_lines is not None and n >= max_lines:
                    break

    meta = {
        "key": src.key, "role": src.role, "kind": src.kind,
        "locator": src.locator, "license": src.license,
        "lines": n, "bytes": out.stat().st_size,
        "mb": round(out.stat().st_size / (1024 * 1024), 2),
        "path": str(out), **stats,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def fetch_all(
    cache: Path = DEFAULT_CACHE,
    *,
    keys: list[str] | None = None,
    max_lines: int | None = None,
    force: bool = False,
) -> list[dict[str, object]]:
    """Récupère toutes les sources (ou celles listées dans ``keys``).

    Une source qui échoue n'interrompt pas les autres : son erreur est consignée
    dans son entrée de métadonnées. Perdre un jeu de données ne doit pas coûter
    le téléchargement des six autres.
    """
    todo = [SOURCES[k] for k in (keys or SOURCES)]
    out = []
    for src in todo:
        print(f"[{src.role:8s}] {src.key} ...", flush=True)
        try:
            meta = fetch(src, cache, max_lines=max_lines, force=force)
            print(f"           {meta['lines']:,} lignes · {meta['mb']} Mo")
        except Exception as exc:  # noqa: BLE001 - on continue malgré tout
            meta = {"key": src.key, "role": src.role, "error": f"{type(exc).__name__}: {exc}"}
            print(f"           ECHEC : {meta['error']}")
        out.append(meta)
    return out


def load(cache: Path = DEFAULT_CACHE, *, role: str | None = None) -> dict[str, list[str]]:
    """Relit le cache. ``{clé: lignes}``, filtré par rôle si demandé."""
    out: dict[str, list[str]] = {}
    for key, src in SOURCES.items():
        if role and src.role != role:
            continue
        p = cache / f"{key}.txt"
        if p.exists():
            out[key] = p.read_text(encoding="utf-8").splitlines()
    return out


__all__ = [
    "DEFAULT_CACHE",
    "MissingExtra",
    "fetch",
    "fetch_all",
    "load",
]
