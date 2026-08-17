"""The Swift pretraining corpus.

Every source here is **real, human-written text** with a named origin and a
known licence. Nothing in this module generates text, templates text, or
augments a dataset to inflate its size - see `CLAUDE.md`, which forbids it.

The corpus is *not* vendored into this repository. This module downloads each
source from its original distributor, so provenance stays verifiable and we do
not redistribute anyone's corpus. Run it with::

    python -m minerva.training.data --out data/

Deliberate exclusions
---------------------
Two corpora that were available were rejected on quality grounds, which is the
kind of decision `CLAUDE.md` asks for explicitly:

* **Brown corpus** - distributed POS-tagged (``The/at Fulton/np-tl``).
  De-tagging is mechanical but leaves unnatural spacing around punctuation,
  which teaches the model a typography that no real text uses.
* **Movie reviews (Pang & Lee)** - distributed lowercased and pre-tokenised
  (``films adapted from comic books , whether they 're``). Casing and spacing
  are destroyed, and a language model would learn the damage.

Both are real text; neither is *good* text. Volume was not a good enough
reason to include them.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import unicodedata
import urllib.request
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["SOURCES", "CorpusSource", "build_corpus", "download_source"]

# NLTK's data distribution is a stable, well-documented mirror of these corpora
# and is reachable without an account.
_NLTK_BASE = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora"


@dataclass(frozen=True, slots=True)
class CorpusSource:
    """One real corpus, with the provenance `CLAUDE.md` requires."""

    name: str
    url: str
    #: Glob(s), relative to the extracted archive root, selecting text files.
    patterns: tuple[str, ...]
    licence: str
    origin: str
    description: str
    #: Files whose names match these are metadata, not text.
    skip_names: frozenset[str] = field(default=frozenset({"README", "CONTENTS", "cats.txt"}))
    #: Optional per-source cleaner applied to each file's text.
    cleaner: str = "generic"


# --- THE CORPUS ------------------------------------------------------------
# Roughly 28 MB of clean English prose across four registers: literature,
# newswire, spoken/political oratory, and informal web writing. The mix is
# deliberate - a model trained only on 19th-century novels writes only
# 19th-century novels.
SOURCES: tuple[CorpusSource, ...] = (
    CorpusSource(
        name="gutenberg",
        url=f"{_NLTK_BASE}/gutenberg.zip",
        patterns=("gutenberg/*.txt",),
        licence="Public domain",
        origin="Project Gutenberg (18 complete books), via the NLTK data distribution",
        description=(
            "Literary prose and verse: Austen, Melville, Milton, Shakespeare, "
            "Chesterton, Whitman, the King James Bible. The backbone of the "
            "corpus - long-form, carefully edited, correctly typeset English."
        ),
        cleaner="gutenberg",
    ),
    CorpusSource(
        name="reuters",
        url=f"{_NLTK_BASE}/reuters.zip",
        patterns=("reuters/training/*", "reuters/test/*"),
        licence="Reuters-21578, Distribution 1.0 - free for research use",
        origin="Reuters newswire (1987), via the NLTK data distribution",
        description=(
            "Newswire: factual, dense, present-tense reporting. Balances the "
            "literary sources with contemporary declarative prose."
        ),
        cleaner="reuters",
    ),
    CorpusSource(
        name="webtext",
        url=f"{_NLTK_BASE}/webtext.zip",
        patterns=("webtext/*.txt",),
        licence="Freely redistributable, see the corpus README",
        origin="NLTK web text sample: forum posts, reviews, film script",
        description=(
            "Informal, contemporary, conversational writing. The only source "
            "here with a modern casual register."
        ),
    ),
    CorpusSource(
        name="inaugural",
        url=f"{_NLTK_BASE}/inaugural.zip",
        patterns=("inaugural/*.txt",),
        licence="Public domain (US federal government work)",
        origin="US presidential inaugural addresses, 1789-2021",
        description="Formal oratory: rhetorical, structured, spoken-for-an-audience prose.",
    ),
    CorpusSource(
        name="state_union",
        url=f"{_NLTK_BASE}/state_union.zip",
        patterns=("state_union/*.txt",),
        licence="Public domain (US federal government work)",
        origin="US State of the Union addresses, 1945-2006",
        description="More formal oratory, and a second register of political language.",
    ),
    CorpusSource(
        name="europarl_en",
        url=f"{_NLTK_BASE}/europarl_raw.zip",
        patterns=("europarl_raw/english/*",),
        licence="European Parliament proceedings - freely available",
        origin="European Parliament proceedings (English portion), via NLTK",
        description=(
            "Transcribed parliamentary debate: long, subordinate-clause-heavy "
            "sentences that none of the other sources supply."
        ),
        cleaner="europarl",
    ),
)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def download_source(source: CorpusSource, cache_dir: Path) -> Path:
    """Fetch a source archive into ``cache_dir``, reusing an existing copy."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"{source.name}.zip"
    if target.exists() and target.stat().st_size > 0:
        return target

    print(f"  downloading {source.name} from {source.url}")
    with urllib.request.urlopen(source.url, timeout=120) as response:
        payload = response.read()
    target.write_bytes(payload)
    print(f"  {source.name}: {len(payload) / 1e6:.2f} MB")
    return target


def _iter_source_texts(archive: Path, source: CorpusSource) -> Iterator[tuple[str, str]]:
    """Yield ``(member_name, decoded_text)`` for every text file in the archive."""
    with zipfile.ZipFile(archive) as zf:
        names = sorted(zf.namelist())
        for pattern in source.patterns:
            for name in names:
                if name.endswith("/"):
                    continue
                if not _matches(name, pattern):
                    continue
                if Path(name).name in source.skip_names:
                    continue
                raw = zf.read(name)
                # These corpora predate universal UTF-8; latin-1 is the
                # documented fallback and never raises.
                try:
                    text = raw.decode("utf-8")
                except UnicodeDecodeError:
                    text = raw.decode("latin-1")
                yield name, text


def _matches(name: str, pattern: str) -> bool:
    """Match ``a/b/*`` style patterns against archive member names."""
    from fnmatch import fnmatch

    return fnmatch(name, pattern)


# ---------------------------------------------------------------------------
# Cleaning
#
# Each cleaner removes corpus-specific markup and nothing else. We are not
# "improving" the text - only stripping artefacts of how it was distributed,
# so the model learns English rather than someone's file format.
# ---------------------------------------------------------------------------

_GUTENBERG_HEADER = re.compile(r"^\[[^\]\n]{0,120}\]\s*$", re.MULTILINE)
_MULTI_BLANK = re.compile(r"\n{3,}")
_TRAILING_WS = re.compile(r"[ \t]+$", re.MULTILINE)
_EUROPARL_TAG = re.compile(r"^<[^>\n]*>\s*$", re.MULTILINE)


def _clean_generic(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _TRAILING_WS.sub("", text)
    return _MULTI_BLANK.sub("\n\n", text).strip()


def _clean_gutenberg(text: str) -> str:
    # NLTK's copies carry a "[Emma by Jane Austen 1816]" line instead of the
    # full Gutenberg licence header. Drop it: it is metadata, not prose.
    return _clean_generic(_GUTENBERG_HEADER.sub("", text))


def _clean_reuters(text: str) -> str:
    # Every article opens with an ALL-CAPS headline on its own line. Keeping it
    # would teach the model that sentences routinely shout; the body is the
    # part worth learning.
    lines = text.split("\n")
    if lines and lines[0].strip() and lines[0].strip() == lines[0].strip().upper():
        lines = lines[1:]
    # Reuters wraps at ~70 columns with leading indentation on continuation
    # lines. Unwrap so paragraphs are real paragraphs.
    body = "\n".join(line.strip() for line in lines)
    body = re.sub(r"\n(?=[a-z,;])", " ", body)
    return _clean_generic(body)


def _clean_europarl(text: str) -> str:
    # Europarl carries <SPEAKER> / <CHAPTER> SGML tags on their own lines.
    return _clean_generic(_EUROPARL_TAG.sub("", text))


_CLEANERS = {
    "generic": _clean_generic,
    "gutenberg": _clean_gutenberg,
    "reuters": _clean_reuters,
    "europarl": _clean_europarl,
}

# A document shorter than this after cleaning is almost always a stub, a stray
# index page, or a truncated file. Real prose survives it easily.
_MIN_DOCUMENT_CHARS = 200

# Documents in this corpus range from a 1 KB newswire item to a 700 KB novel.
# Left alone, that wrecks the train/val split: any split by document count is
# dominated by whichever source has the most files. Long documents are
# therefore cut into chunks at paragraph boundaries, which makes the split
# unit roughly uniform without ever cutting mid-sentence.
_MAX_CHUNK_CHARS = 16_384


def _chunk_document(text: str) -> list[str]:
    """Split an over-long document at paragraph boundaries.

    Paragraphs longer than the limit on their own are emitted whole rather than
    cut mid-sentence - a truncated sentence is worse training signal than an
    oversized chunk.
    """
    if len(text) <= _MAX_CHUNK_CHARS:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for paragraph in text.split("\n\n"):
        if current and size + len(paragraph) > _MAX_CHUNK_CHARS:
            chunks.append("\n\n".join(current))
            current, size = [], 0
        current.append(paragraph)
        size += len(paragraph) + 2
    if current:
        chunks.append("\n\n".join(current))
    return [chunk for chunk in chunks if len(chunk) >= _MIN_DOCUMENT_CHARS]

#: Separates documents in the packed corpus. The tokenizer learns it as a real
#: token, which is how the model learns that documents end.
DOCUMENT_SEPARATOR = "\n\n<|endoftext|>\n\n"


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_corpus(
    out_dir: Path,
    *,
    cache_dir: Path | None = None,
    val_fraction: float = 0.005,
    seed: int = 1729,
) -> dict[str, object]:
    """Download, clean and pack every source into ``train.txt`` / ``val.txt``.

    The split is held out **per source and by character count**, so the
    validation set carries every register in the same proportion as training
    and no chunk straddles the boundary. Splitting by document count instead
    would hand validation almost entirely to Reuters, which contributes 8,578
    of the corpus's documents but only a quarter of its text.

    Returns a manifest describing exactly what was built - it is written next
    to the corpus as ``manifest.json`` and is the provenance record
    `CLAUDE.md` requires.
    """
    import json
    import random

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = cache_dir or out_dir / "cache"

    rng = random.Random(seed)
    train_docs: list[str] = []
    val_docs: list[str] = []
    per_source: list[dict[str, object]] = []

    for source in SOURCES:
        archive = download_source(source, cache_dir)
        cleaner = _CLEANERS[source.cleaner]

        chunks: list[str] = []
        skipped = 0
        for _name, text in _iter_source_texts(archive, source):
            cleaned = cleaner(text)
            if len(cleaned) < _MIN_DOCUMENT_CHARS:
                skipped += 1
                continue
            chunks.extend(_chunk_document(cleaned))

        characters = sum(len(chunk) for chunk in chunks)

        # Hold out `val_fraction` of THIS source's characters, so every
        # register is represented in validation in proportion to its size.
        rng.shuffle(chunks)
        target = characters * val_fraction
        source_val: list[str] = []
        held = 0
        for chunk in chunks:
            if held >= target:
                break
            source_val.append(chunk)
            held += len(chunk)
        source_train = chunks[len(source_val) :]

        val_docs.extend(source_val)
        train_docs.extend(source_train)
        per_source.append(
            {
                "name": source.name,
                "chunks": len(chunks),
                "chunks_train": len(source_train),
                "chunks_val": len(source_val),
                "skipped_too_short": skipped,
                "characters": characters,
                "characters_val": held,
                "licence": source.licence,
                "origin": source.origin,
                "description": source.description,
                "url": source.url,
            }
        )
        print(
            f"  {source.name:<14} {len(chunks):6d} chunks  {characters / 1e6:6.2f} MB"
            f"  (val {held / 1e3:6.1f} KB)"
        )

    rng.shuffle(train_docs)
    rng.shuffle(val_docs)

    train_text = DOCUMENT_SEPARATOR.join(train_docs)
    val_text = DOCUMENT_SEPARATOR.join(val_docs)

    (out_dir / "train.txt").write_text(train_text, encoding="utf-8")
    (out_dir / "val.txt").write_text(val_text, encoding="utf-8")

    manifest = {
        "sources": per_source,
        "chunks": {"train": len(train_docs), "val": len(val_docs)},
        "max_chunk_chars": _MAX_CHUNK_CHARS,
        "characters": {"train": len(train_text), "val": len(val_text)},
        "split": {
            "val_fraction": val_fraction,
            "seed": seed,
            "unit": "chunk, held out per source by character count",
        },
        "sha256": {
            "train": hashlib.sha256(train_text.encode("utf-8")).hexdigest(),
            "val": hashlib.sha256(val_text.encode("utf-8")).hexdigest(),
        },
        "excluded": {
            "brown": "distributed POS-tagged; de-tagging leaves unnatural punctuation spacing",
            "movie_reviews": (
                "distributed lowercased and pre-tokenised; casing and spacing destroyed"
            ),
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Swift pretraining corpus.")
    parser.add_argument("--out", type=Path, default=Path("data"), help="output directory")
    parser.add_argument("--val-fraction", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args(argv)

    print("Building the Swift pretraining corpus from real sources:")
    manifest = build_corpus(args.out, val_fraction=args.val_fraction, seed=args.seed)

    chars = manifest["characters"]
    assert isinstance(chars, dict)
    print(
        f"\ntrain: {chars['train'] / 1e6:.2f} MB   "
        f"val: {chars['val'] / 1e6:.2f} MB   -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
