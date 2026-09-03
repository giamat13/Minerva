"""Download Project Gutenberg's own catalogue and pick a large book set.

Run once, offline from training. Writes two artefacts under `data/cache/`:

* ``pg_catalog.csv``      - the catalogue exactly as Gutenberg publishes it.
* ``gutenberg_ids.json``  - the selected ebook ids, with title and language.

Selection is by the catalogue's own metadata rather than by a hand-typed
list, because at this scale a hand-typed list is not reviewable and would be
the very "hit a row count" behaviour CLAUDE.md forbids. What is chosen is a
*category* decision that can be stated in one sentence and checked: English
and Hebrew prose and verse, `Type == Text`, excluding the periodical and
index material that is boilerplate rather than writing.
"""

from __future__ import annotations

import csv
import io
import json
import sys
import urllib.request
from pathlib import Path

UA = {"User-Agent": "Minerva/0.5 corpus build (github.com/giamat13/Minerva)"}
CATALOGUE = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"

#: Subject/title markers for material that is not continuous prose: indexes of
#: other books, periodical runs, and the catalogue's own bookkeeping entries.
_SKIP_TITLE = (
    "index of the project gutenberg",
    "complete project gutenberg",
    "project gutenberg's",
)
_SKIP_SUBJECT = ("periodicals", "indexes", "bibliography", "catalogs", "encyclopedias")


def fetch_catalogue(cache: Path) -> list[dict[str, str]]:
    target = cache / "pg_catalog.csv"
    if not (target.exists() and target.stat().st_size > 1_000_000):
        print(f"downloading catalogue from {CATALOGUE}", flush=True)
        request = urllib.request.Request(CATALOGUE, headers=UA)
        with urllib.request.urlopen(request, timeout=600) as response:
            target.write_bytes(response.read())
    print(f"catalogue: {target.stat().st_size / 1e6:.1f} MB", flush=True)
    text = target.read_text(encoding="utf-8", errors="replace")
    return list(csv.DictReader(io.StringIO(text)))


def wanted(row: dict[str, str]) -> bool:
    if row.get("Type") != "Text":
        return False
    if row.get("Language") not in {"en", "he"}:
        return False
    title = (row.get("Title") or "").lower()
    if not title or any(marker in title for marker in _SKIP_TITLE):
        return False
    subjects = (row.get("Subjects") or "").lower()
    return not any(marker in subjects for marker in _SKIP_SUBJECT)


def main() -> int:
    cache = Path("data/cache")
    cache.mkdir(parents=True, exist_ok=True)
    rows = fetch_catalogue(cache)
    print(f"catalogue rows: {len(rows):,}", flush=True)

    picked: list[dict[str, object]] = []
    for row in rows:
        if not wanted(row):
            continue
        try:
            gid = int(row["Text#"])
        except (KeyError, ValueError):
            continue
        picked.append(
            {"id": gid, "title": (row.get("Title") or "").strip()[:120],
             "language": row["Language"]}
        )

    languages: dict[str, int] = {}
    for item in picked:
        languages[str(item["language"])] = languages.get(str(item["language"]), 0) + 1

    out = cache / "gutenberg_ids.json"
    out.write_text(json.dumps(picked, ensure_ascii=False), encoding="utf-8")
    print(f"selected {len(picked):,} texts -> {out}", flush=True)
    print(f"by language: {languages}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
