"""Download a large, catalogue-selected slice of Project Gutenberg.

Reads the ids chosen by ``fetch_gutenberg_catalogue.py`` and fetches them into
``data/cache/gutenberg_bulk/``. Politeness is not optional here: Gutenberg's
robot policy asks automated clients to use a mirror, and the main site
enforces it (a GitHub runner was refused outright with RemoteDisconnected),
so the mirror is tried first and a delay is taken between books.

Resumable by design - a book already on disk is skipped - because at this
volume the run will be interrupted.

    python scripts/fetch_gutenberg_books.py --target-gb 1.7
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

UA = {"User-Agent": "Minerva/0.5 corpus build (github.com/giamat13/Minerva)"}
MIRRORS = (
    "https://gutenberg.pglaf.org/cache/epub/{id}/pg{id}.txt",
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt",
)
# Eight streams with a per-request pause is well inside what the mirror
# serves comfortably, and turns a ~9-hour serial fetch into about one hour.
WORKERS = 8
DELAY = 0.4
ATTEMPTS = 2
#: Below this a "book" is usually a stub, a contents page or a broken record.
MIN_BYTES = 30_000


def fetch(gid: int) -> bytes | None:
    for template in MIRRORS:
        for attempt in range(ATTEMPTS):
            try:
                request = urllib.request.Request(template.format(id=gid), headers=UA)
                with urllib.request.urlopen(request, timeout=90) as response:
                    return bytes(response.read())
            except Exception:
                time.sleep(DELAY * (2**attempt))
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-gb", type=float, default=1.7)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args(argv)

    cache = Path("data/cache")
    ids = json.loads((cache / "gutenberg_ids.json").read_text(encoding="utf-8"))
    # Shuffled with a fixed seed: a reproducible sample across the whole
    # catalogue rather than the lowest ids, which skew to one era and to the
    # books already in the corpus.
    random.Random(args.seed).shuffle(ids)

    books = cache / "gutenberg_bulk"
    books.mkdir(parents=True, exist_ok=True)
    have = sum(f.stat().st_size for f in books.glob("*.txt"))
    target = int(args.target_gb * 1e9)
    print(f"target {target/1e9:.2f} GB, already have {have/1e9:.3f} GB", flush=True)

    kept = len(list(books.glob("*.txt")))
    skipped = failed = 0

    def grab(gid: int) -> int:
        """Fetch one book. Returns bytes written, 0 if skipped, -1 if failed."""
        path = books / f"pg{gid}.txt"
        if path.exists():
            return 0
        payload = fetch(gid)
        time.sleep(DELAY)
        if payload is None:
            return -1
        if len(payload) < MIN_BYTES:
            return 0
        path.write_bytes(payload)
        return len(payload)

    pending = [int(item["id"]) for item in ids]
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        # Submitted in blocks so the target check actually stops the run
        # instead of queueing all 57,000 ids up front.
        for start in range(0, len(pending), 200):
            if have >= target:
                break
            for size in pool.map(grab, pending[start:start + 200]):
                if size > 0:
                    have += size
                    kept += 1
                elif size < 0:
                    failed += 1
                else:
                    skipped += 1
            print(f"  {kept:5d} books  {have/1e9:.3f} GB  "
                  f"(skipped {skipped}, failed {failed})", flush=True)

    print(f"done: {kept} books, {have/1e9:.3f} GB, skipped {skipped}, failed {failed}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
