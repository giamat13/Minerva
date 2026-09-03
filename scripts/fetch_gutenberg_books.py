"""Download a large, catalogue-selected slice of Project Gutenberg.

Fetches books into ``data/cache/gutenberg_bulk/``. By default it uses the
committed id list at ``data/gutenberg_bulk_ids.json``, which names the exact
books in the shipped corpus so a fresh clone rebuilds the same one; pass
``--recatalogue`` to re-select from the full catalogue instead.

Politeness is not optional here: Gutenberg's
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


def download(ids: list[int], cache: Path, target: int | None) -> int:
    """Fetch each id into the cache, stopping early once `target` bytes exist."""
    books = cache / "gutenberg_bulk"
    books.mkdir(parents=True, exist_ok=True)
    have = sum(f.stat().st_size for f in books.glob("*.txt"))
    kept = len(list(books.glob("*.txt")))
    skipped = failed = 0
    if target:
        print(f"target {target/1e9:.2f} GB, already have {have/1e9:.3f} GB", flush=True)

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

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        # Submitted in blocks so the target check actually stops the run
        # instead of queueing every id up front.
        for start in range(0, len(ids), 200):
            if target and have >= target:
                break
            for size in pool.map(grab, ids[start:start + 200]):
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-gb", type=float, default=1.7)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--recatalogue",
        action="store_true",
        help="ignore the committed id list and re-select from the full catalogue",
    )
    args = parser.parse_args(argv)

    cache = Path("data/cache")

    # Prefer the committed lockfile: it names the exact books in the shipped
    # corpus, so a fresh clone rebuilds the same corpus rather than a similar
    # one. Without it the set depends on transient fetch failures and on the
    # 30 KB stub threshold, neither of which is reproducible.
    lock = Path("data/gutenberg_bulk_ids.json")
    if lock.exists() and not args.recatalogue:
        locked = [int(g) for g in json.loads(lock.read_text(encoding="utf-8"))["ids"]]
        print(f"using the committed id list: {len(locked)} books from {lock}", flush=True)
        return download(locked, cache, None)

    catalogue = cache / "gutenberg_ids.json"
    if not catalogue.exists():
        print(
            f"{catalogue} not found. Run "
            f"'python scripts/fetch_gutenberg_catalogue.py' first.",
            file=sys.stderr,
        )
        return 1
    ids = json.loads(catalogue.read_text(encoding="utf-8"))
    # Shuffled with a fixed seed: a reproducible sample across the whole
    # catalogue rather than the lowest ids, which skew to one era and to the
    # books already in the corpus.
    random.Random(args.seed).shuffle(ids)
    return download([int(item["id"]) for item in ids], cache, int(args.target_gb * 1e9))


if __name__ == "__main__":
    sys.exit(main())
