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
Corpora that were available were rejected on quality grounds, which is the
kind of decision `CLAUDE.md` asks for explicitly:

* **Movie reviews (Pang & Lee)** - distributed lowercased and pre-tokenised
  (``films adapted from comic books , whether they 're``). Casing and spacing
  are destroyed, and a language model would learn the damage.
* **OSCAR-2301 (Hebrew)** - a raw Common Crawl web scrape: gated access,
  no per-document quality signal, and the usual web-crawl duplication and
  boilerplate. Project Ben-Yehuda gives curated, individually attributed
  Hebrew prose instead of unfiltered web text - the same trade this project
  already made for English by choosing Gutenberg over a web dump.
* **Sefaria's Hebrew library** - real, carefully edited Hebrew, but a GPL-3.0
  licence on a text dataset (a licence written for software, whose terms for
  a trained model's weights are genuinely unclear) and a register - biblical,
  halakhic and liturgical source text segmented paragraph-by-paragraph, not
  continuous prose - that would not teach the same thing Project Ben-Yehuda's
  literature does.
* **Full English Wikipedia (`wikimedia/wikipedia`, ``20231101.en``)** - 6.4M
  articles, tens of gigabytes. Nobody could read a representative sample of
  that and vouch for it.

None of these are *bad* text, exactly; they are text this project could not
honestly claim to have read and stand behind at the volume they come in.
Volume was not a good enough reason to include any of them.

Removed in v0.3.0: encyclopedic and news content
--------------------------------------------------
Two sources that *were* in the corpus - Simple English Wikipedia (encyclopedic
general knowledge) and Reuters newswire (event reporting) - were removed, not
for a quality reason but a capability one: Swift is 9.9M parameters. It has
nowhere to reliably store "the capital of X" or "what happened on date Y", and
training on text whose whole point is dense factual claims does not give it
that capacity - it gives it fluent confabulation, a model that states wrong
facts exactly as confidently as right ones. That is worse than not knowing.
The honest fix is not more parameters; it is not asking the weights to be a
fact database at all. `minerva.tools.builtin.web_search` is the replacement:
a real tool, so the model looks a fact up instead of guessing at one it was
never big enough to remember correctly.

Added back in v0.3.0: Brown corpus, for register not facts
------------------------------------------------------------
Removing Reuters and Wikipedia cost the corpus more than their facts: they
were also its only *short, declarative, factual-register* prose. What
remained (Gutenberg, oratory, parliamentary debate, Hebrew literature) is
long-form and rhetorical, and a held-out measurement after the v0.3.0 retrain
showed the cost was real - `swift-instruct` finetuned on the resulting base
model scored far below the numbers `docs/TRAINING.md` records for earlier
rounds (routing accuracy fell from the 90s into the 60-70% range, and
argument accuracy - producing a clean, correct `calculate` call - collapsed
to single digits), even on the *unchanged* instruct set from before this
change, isolating the regression to the base model rather than the instruct
data.

The Brown corpus was rejected in v0.1.0/v0.2.0 for a **formatting** reason,
not a content one: NLTK distributes it POS-tagged (``The/at Fulton/np-tl``),
and the module docstring at the time said de-tagging "leaves unnatural
spacing around punctuation." That turned out to be a solvable cleaning
problem (see `_clean_brown`), not a reason to reject the text - and the text
itself is exactly the missing register: c.1961 American press reportage,
editorial and fiction, short and declarative, real and licensed for
redistribution. It is not a database of facts to memorise the way Wikipedia
or 1987 Reuters newswire is - it is old enough, and general enough in
subject, to teach sentence structure without teaching anything worth
confabulating. The original rejection was reversed, not ignored: the reason
it no longer applies is written down here, per `CLAUDE.md`'s rule that a
judgement call gets recorded, not just changed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import random
import re
import sys
import time
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
    #: Empty for sources whose ``kind`` selects text some other way.
    patterns: tuple[str, ...]
    licence: str
    origin: str
    description: str
    #: Files whose names match these are metadata, not text.
    skip_names: frozenset[str] = field(default=frozenset({"README", "CONTENTS", "cats.txt"}))
    #: Optional per-source cleaner applied to each file's text.
    cleaner: str = "generic"
    #: How to fetch and select this source's documents. "archive" (the
    #: default) downloads `url` as a zip and globs `patterns` inside it - see
    #: `_iter_source_texts`. A source that cannot be selected by filename glob
    #: alone gets its own kind and iterator function instead, dispatched in
    #: `build_corpus` - see `_iter_benyehuda_texts`.
    kind: str = "archive"


# --- THE CORPUS ------------------------------------------------------------
# English registers - literary, short-declarative press/fiction, informal
# web writing, spoken/political oratory - plus Swift's non-English source:
# curated Hebrew literature. No encyclopedic or news *content* (see "Removed
# in v0.3.0" above): Brown supplies the short-sentence register that used to
# come from Reuters and Wikipedia, without their factual payload. The mix is
# deliberate - a model trained only on 19th-century novels writes only
# 19th-century novels, and a model that never sees Hebrew cannot write it.
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
        name="gutenberg_extended",
        url="https://gutenberg.pglaf.org/cache/epub/{id}/pg{id}.txt",
        patterns=(),  # selected by id, see _iter_gutenberg_extended_texts
        licence="Public domain (Project Gutenberg licence stripped with the header)",
        origin=(
            "Project Gutenberg, fetched per-book by ebook id from an "
            "official mirror (gutenberg.pglaf.org, falling back to "
            "gutenberg.org) - 119 curated titles, see _GUTENBERG_EXTENDED_IDS"
        ),
        description=(
            "The volume that makes fluency possible: 119 canonical English "
            "books across novels, gothic, detective, adventure, early science "
            "fiction, children's literature, essays, philosophy, drama and "
            "translated classics. Chosen for register diversity rather than "
            "raw size, verified to resolve and de-duplicated against the 18 "
            "books the `gutenberg` source above already ships."
        ),
        cleaner="gutenberg_extended",
        kind="gutenberg_ids",
    ),
    CorpusSource(
        name="gutenberg_bulk",
        url="https://gutenberg.pglaf.org/cache/epub/{id}/pg{id}.txt",
        patterns=(),  # whatever scripts/fetch_gutenberg_books.py placed on disk
        licence="Public domain (Project Gutenberg licence stripped with the header)",
        origin=(
            "Project Gutenberg, selected from Gutenberg's own published "
            "catalogue (Type == Text, Language in {en, he}, periodical and "
            "index material excluded) by scripts/fetch_gutenberg_catalogue.py "
            "and downloaded by scripts/fetch_gutenberg_books.py"
        ),
        description=(
            "The bulk of the corpus. The 119 hand-picked titles in "
            "`gutenberg_extended` above chose register diversity by hand; this "
            "source adds volume, because a model large enough to hold a real "
            "conversation needs roughly twenty tokens per parameter and the "
            "hand-picked set cannot supply them. The selection is still a "
            "stated category decision rather than a row count: Gutenberg's own "
            "metadata, not a generator. Books under 30 KB are skipped at "
            "download time (stubs and contents pages), and ids already in "
            "`gutenberg_extended` are skipped here so nothing is duplicated."
        ),
        cleaner="gutenberg_extended",
        kind="gutenberg_bulk",
    ),
    CorpusSource(
        name="brown",
        url=f"{_NLTK_BASE}/brown.zip",
        patterns=("brown/????",),
        licence="Distributed with the permission of the copyright holder (Brown "
        "University); redistribution permitted",
        origin=(
            "The Brown Corpus (Francis & Kucera, 1964/1979), via the NLTK data "
            "distribution - 500 samples of c.1961 American English"
        ),
        description=(
            "Short, declarative press reportage, editorial and fiction - the "
            "register Reuters and Wikipedia used to supply, restored here "
            "without their factual content: 1961 news is neither current nor "
            "worth memorising, only worth imitating the sentence structure of. "
            "Rejected in earlier versions for a formatting reason (distributed "
            "POS-tagged) that turned out to be a solvable cleaning problem, not "
            "a reason to exclude the text - see the module docstring."
        ),
        cleaner="brown",
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
    CorpusSource(
        name="benyehuda",
        url=(
            "https://github.com/projectbenyehuda/public_domain_dump/releases/"
            "download/2026-03/txt.zip"
        ),
        patterns=(),  # curated from the catalogue by _iter_benyehuda_texts
        licence="Public domain",
        origin=(
            "Project Ben-Yehuda, https://benyehuda.org - the release tagged "
            "2026-03 of https://github.com/projectbenyehuda/public_domain_dump"
        ),
        description=(
            "Hebrew literature, Swift's first non-English source: original "
            "(non-translated) poetry, prose, drama and essays by seven "
            "canonical figures of modern Hebrew writing - Bialik, Rachel "
            "Bluwstein, Brenner, Ahad Ha'am, Mendele Mocher Sforim, "
            "Tchernichovsky and Frishman. The Hebrew analogue of the "
            "Gutenberg entry above: curated authors, not the whole library - "
            "see _iter_benyehuda_texts for the selection criteria."
        ),
        cleaner="benyehuda",
        kind="benyehuda",
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


# --- Sources that cannot be selected by filename glob alone ----------------
#
# `download_source` + `_iter_source_texts` cover every source whose wanted
# documents are just "every file matching this glob inside this zip". These
# two are not: which document belongs in the corpus depends on metadata that
# does not live in a filename, so each gets its own small iterator instead of
# forcing the glob abstraction to do something it was not built for.

#: The seven person-ids (Project Ben-Yehuda's folder scheme) whose *original*
#: (non-translated) work was curated for `benyehuda`. Chosen and measured by
#: hand: filtering to original_language == "" and excluding the "letters" and
#: "reference" genres took Bialik's contribution from 10.2 MB (which includes
#: his translations and correspondence) down to 2.9 MB of his own poetry,
#: articles and prose. Across all seven, the same filter is the difference
#: between a ~40 MB pull of the whole catalogue and a 17.6 MB curated one.
#: Ben-Yehuda person-ids for the curated authors. Expanded in v0.4.0 from
#: seven to thirty-five, all of them **modern** Hebrew writers (Haskalah and
#: the Hebrew revival onward, roughly 1850+).
#:
#: The exclusion that matters: Ben-Yehuda's largest contributors by work count
#: are medieval poets - Samuel HaNagid (p49, 1,856 works), Ibn Gabirol (p180),
#: Judah Halevi (p161), Abraham and Moses Ibn Ezra (p20, p170), Shalom Shabazi
#: (p146). They are real, canonical Hebrew, and they are deliberately left out:
#: 11th-17th century liturgical and courtly verse is to modern conversational
#: Hebrew roughly what Chaucer is to spoken English. Swift is meant to hold a
#: conversation, so the corpus buys modern register rather than raw volume,
#: even though including them would have more than doubled the Hebrew side.
_BENYEHUDA_AUTHOR_IDS = frozenset(
    {
        # The original seven (v0.2.0).
        "p89",    # Bialik
        "p141",   # Rachel Bluwstein
        "p66",    # Brenner
        "p23",    # Ahad Ha'am
        "p44",    # Mendele Mocher Sforim
        "p57",    # Tchernichovsky
        "p142",   # Frishman
        # Added v0.4.0 - modern poetry, prose, essays and journalism.
        "p609",   # Yehuda Karni
        "p440",   # Yitzhak Katzenelson
        "p55",    # Berl Katznelson
        "p1274",  # Asher Barash
        "p388",   # Yaakov Steinberg
        "p46",    # Y. L. Gordon
        "p78",    # David Vogel
        "p111",   # A. Z. Rabinovitz
        "p503",   # Yitzhak Lufban
        "p115",   # Menachem Mivashan
        "p1449",  # Yaakov Klatzkin
        "p164",   # Berdyczewski
        "p904",   # David Remez
        "p726",   # Chaim Lensky
        "p814",   # Fania Bergstein
        "p181",   # Moshe Beilinson
        "p720",   # Shlomo Mandelkern
        "p135",   # Moshe Glickson
        "p1975",  # David Smilansky
        "p117",   # Yehuda Steinberg
        "p1367",  # Zvi Hirsch Masliansky
        "p30",    # Aharon Liebushitzky
        "p123",   # Yeshayahu Karniel
        "p24",    # Alter Druyanov
        "p32",    # Itamar Ben-Avi
        "p155",   # Y. L. Peretz
        "p41",    # Naftali Herz Imber
        "p87",    # Azriel Nathan Frank
    }
)
_BENYEHUDA_EXCLUDED_GENRE_SUBSTRINGS = ("letters", "reference")
#: Project Gutenberg ebook ids for `gutenberg_extended`. Every one was
#: fetched and checked before being written here (see the "Added in v0.4.0"
#: note in the module docstring): the id resolves, the text is a real
#: Gutenberg ebook of substantial length, and it is mostly-Latin script -
#: Hebrew comes from Ben-Yehuda, not from here. Two candidates were caught
#: and dropped by that check because they duplicated books the `gutenberg`
#: source above already ships: id 1522 is Julius Caesar (NLTK has
#: shakespeare-caesar) and id 19033 is Alice in Wonderland (NLTK has
#: carroll-alice). Titles below are the ones Gutenberg itself reports, not
#: the ones this file assumed.
_GUTENBERG_EXTENDED_IDS: dict[int, str] = {
    16: "Peter Pan",
    27: "Far from the Madding Crowd",
    33: "The Scarlet Letter",
    35: "The Time Machine",
    36: "The war of the worlds",
    43: "The strange case of Dr. Jekyll and Mr. Hyde",
    45: "Anne of Green Gables",
    46: "A Christmas Carol in Prose; Being a Ghost Story of Christmas",
    55: "The Wonderful Wizard of Oz",
    74: "The Adventures of Tom Sawyer, Complete",
    76: "Adventures of Huckleberry Finn",
    77: "The House of the Seven Gables",
    84: "Frankenstein; or, the modern prometheus",
    86: "A Connecticut Yankee in King Arthur's Court",
    98: "A Tale of Two Cities",
    103: "Around the World in Eighty Days",
    108: "The Return of Sherlock Holmes",
    110: "Tess of the d'Urbervilles: A Pure Woman",
    113: "The Secret Garden",
    119: "A Tramp Abroad",
    120: "Treasure Island",
    121: "Northanger Abbey",
    132: "The Art of War",
    141: "Mansfield Park",
    145: "Middlemarch",
    155: "The Moonstone",
    159: "The island of Doctor Moreau",
    160: "The Awakening, and Selected Short Stories",
    164: "Twenty Thousand Leagues under the Sea",
    174: "The Picture of Dorian Gray",
    203: "Uncle Tom's Cabin",
    205: "Walden, and On The Duty Of Civil Disobedience",
    209: "The Turn of the Screw",
    215: "The call of the wild",
    219: "Heart of Darkness",
    236: "The Jungle Book",
    244: "A Study in Scarlet",
    284: "The House of Mirth",
    289: "The Wind in the Willows",
    345: "Dracula",
    394: "Cranford",
    421: "Kidnapped",
    432: "The Ambassadors",
    482: "The Woodlanders",
    507: "Adam Bede",
    514: "Little Women",
    521: "The Life and Adventures of Robinson Crusoe",
    526: "Heart of Darkness",
    541: "The Age of Innocence",
    550: "Silas Marner",
    580: "The Pickwick Papers",
    583: "The Woman in White",
    599: "Vanity Fair",
    600: "Notes from the Underground",
    696: "The Castle of Otranto",
    730: "Oliver Twist",
    766: "David Copperfield",
    768: "Wuthering Heights",
    786: "Hard Times",
    829: "Gulliver's Travels into Several Remote Nations of the World",
    834: "The Memoirs of Sherlock Holmes",
    844: "The Importance of Being Earnest: A Trivial Comedy for Serious People",
    910: "White Fang",
    963: "Little Dorrit",
    967: "Nicholas Nickleby",
    969: "The Tenant of Wildfell Hall",
    974: "The Secret Agent: A Simple Tale",
    996: "Don Quixote",
    1023: "Bleak House",
    1081: "Dead Souls",
    1164: "The iron heel",
    1184: "The Count of Monte Cristo",
    1232: "The Prince",
    1250: "Anthem",
    1257: "The three musketeers",
    1260: "Jane Eyre: An Autobiography",
    1342: "Pride and Prejudice",
    1399: "Anna Karenina",
    1400: "Great Expectations",
    1497: "The Republic",
    1513: "Romeo and Juliet",
    1526: "Twelfth Night",
    1531: "Othello",
    1661: "The Adventures of Sherlock Holmes",
    1727: "The Odyssey",
    1837: "The Prince and the Pauper",
    1952: "The Yellow Wallpaper",
    1998: "Thus Spake Zarathustra: A Book for All and None",
    2005: "Piccadilly Jim",
    2097: "The Sign of the Four",
    2147: "The Works of Edgar Allan Poe - Volume 1",
    2148: "The Works of Edgar Allan Poe - Volume 2",
    2153: "Mary Barton",
    2226: "Kim",
    2413: "Madame Bovary",
    2554: "Crime and Punishment",
    2591: "Grimms' Fairy Tales",
    2600: "War and Peace",
    2638: "The Idiot",
    2680: "Meditations",
    2814: "Dubliners",
    2833: "The Portrait of a Lady - Volume 1",
    2852: "The Hound of the Baskervilles",
    3207: "Leviathan",
    3268: "The Mysteries of Udolpho",
    3300: "An Inquiry into the Nature and Causes of the Wealth of Nations",
    3600: "Essays of Michel de Montaigne - Complete",
    4217: "A Portrait of the Artist as a Young Man",
    4363: "Beyond Good and Evil",
    4507: "As a man thinketh",
    5200: "Metamorphosis",
    5230: "The Invisible Man: A Grotesque Romance",
    5658: "Lord Jim",
    5827: "The Problems of Philosophy",
    6130: "The Iliad",
    7849: "The Trial",
    8800: "The divine comedy",
    18857: "A Journey to the Centre of the Earth",
    28054: "The Brothers Karamazov",
}


_BENYEHUDA_CATALOGUE_URL = (
    "https://github.com/projectbenyehuda/public_domain_dump/releases/"
    "download/2026-03/pseudocatalogue.csv"
)


#: Sent on every Gutenberg request. gutenberg.org asks automated clients to
#: identify themselves rather than pretend to be a browser.
_GUTENBERG_UA = "Minerva/0.4 corpus build (github.com/giamat13/Minerva)"

#: Tried in order, per book. The **mirror comes first on purpose**: Project
#: Gutenberg's robot policy asks automated clients to use a mirror rather than
#: hammer the main site, and the main site enforces that - fetching these 119
#: books from a GitHub Actions runner failed on the very first request with
#: `http.client.RemoteDisconnected`, the datacenter IP being refused outright,
#: while the identical code succeeded from a home connection. Politeness here
#: is also the thing that makes the build work at all.
_GUTENBERG_MIRRORS: tuple[str, ...] = (
    "https://gutenberg.pglaf.org/cache/epub/{id}/pg{id}.txt",
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt",
)
_GUTENBERG_ATTEMPTS_PER_MIRROR = 3
#: Seconds between downloads, and the base for exponential retry backoff.
_GUTENBERG_DELAY = 0.5


def _fetch_gutenberg_book(gid: int) -> bytes:
    """Download one book, trying each mirror with backoff.

    Raises if every mirror fails: a corpus quietly missing a third of its
    text would still train, and would still be wrong.
    """
    last: Exception | None = None
    for template in _GUTENBERG_MIRRORS:
        url = template.format(id=gid)
        for attempt in range(_GUTENBERG_ATTEMPTS_PER_MIRROR):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": _GUTENBERG_UA})
                with urllib.request.urlopen(request, timeout=120) as response:
                    return bytes(response.read())
            except Exception as exc:  # retried below, and re-raised if all fail
                last = exc
                time.sleep(_GUTENBERG_DELAY * (2**attempt))
    raise RuntimeError(
        f"could not download Project Gutenberg ebook {gid} from any mirror "
        f"({', '.join(_GUTENBERG_MIRRORS)}); last error: {last!r}"
    ) from last


def _iter_gutenberg_extended_texts(source: CorpusSource, cache_dir: Path) -> Iterator[str]:
    """Fetch each curated Gutenberg book by id, caching it on disk.

    One file per book rather than one big archive, because Gutenberg has no
    bulk endpoint for an arbitrary curated set - and because a per-book cache
    means an interrupted build resumes instead of starting the whole download
    again.
    """
    books_dir = cache_dir / "gutenberg_extended"
    books_dir.mkdir(parents=True, exist_ok=True)

    for gid in sorted(_GUTENBERG_EXTENDED_IDS):
        target = books_dir / f"pg{gid}.txt"
        if not (target.exists() and target.stat().st_size > 0):
            target.write_bytes(_fetch_gutenberg_book(gid))
            # Only when something was actually fetched - a warm cache should
            # not sit through two minutes of sleeping for no reason.
            time.sleep(_GUTENBERG_DELAY)
        yield target.read_text(encoding="utf-8", errors="replace")


def _iter_gutenberg_bulk_texts(source: CorpusSource, cache_dir: Path) -> Iterator[str]:
    """Read the bulk book set fetched by ``scripts/fetch_gutenberg_books.py``.

    Deliberately does *not* download: at this volume the fetch is a long,
    polite, resumable job that belongs in its own script, and a corpus build
    that silently started a multi-hour download would be a bad surprise. An
    empty directory is a loud error rather than a silently smaller corpus.
    """
    books_dir = cache_dir / "gutenberg_bulk"
    paths = sorted(books_dir.glob("*.txt")) if books_dir.is_dir() else []
    if not paths:
        raise RuntimeError(
            f"no books found in {books_dir}. Fetch them first with "
            f"'python scripts/fetch_gutenberg_books.py --target-gb 1.7' "
            f"(and, if data/cache/gutenberg_ids.json is missing, run "
            f"'python scripts/fetch_gutenberg_catalogue.py' before it)."
        )
    already = {f"pg{gid}.txt" for gid in _GUTENBERG_EXTENDED_IDS}
    for path in paths:
        if path.name in already:
            continue
        yield path.read_text(encoding="utf-8", errors="replace")


def _iter_benyehuda_texts(source: CorpusSource, cache_dir: Path) -> Iterator[str]:
    """Curate Project Ben-Yehuda's library down to seven authors' own work.

    The release this source downloads is the *entire* public-domain Hebrew
    library - over 26,000 works, ~250 MB - the Hebrew analogue of "all of
    Project Gutenberg," not something anyone could read and vouch for.
    `pseudocatalogue.csv` records each work's author (as a person-id folder,
    e.g. Bialik is ``p89``), genre and original language, which is enough to
    keep only original Hebrew poetry, prose, drama, memoir and essays by the
    seven curated authors and drop their translations and correspondence.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    catalogue_path = cache_dir / "benyehuda_pseudocatalogue.csv"
    if not (catalogue_path.exists() and catalogue_path.stat().st_size > 0):
        print(f"  downloading benyehuda catalogue from {_BENYEHUDA_CATALOGUE_URL}")
        with urllib.request.urlopen(_BENYEHUDA_CATALOGUE_URL, timeout=60) as response:
            catalogue_path.write_bytes(response.read())

    with catalogue_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    wanted_paths: set[str] = set()
    for row in rows:
        person_id = row["path"].split("/")[1]
        if person_id not in _BENYEHUDA_AUTHOR_IDS:
            continue
        if (row["original_language"] or "").strip():
            continue  # a translation, not this author's own Hebrew
        if any(s in row["genre"] for s in _BENYEHUDA_EXCLUDED_GENRE_SUBSTRINGS):
            continue  # correspondence or a bibliographic stub, not prose
        wanted_paths.add(row["path"].strip("/") + ".txt")

    archive = download_source(source, cache_dir)
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            member_path = name.removeprefix("txt/")
            if member_path not in wanted_paths:
                continue
            yield zf.read(name).decode("utf-8")


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
    #
    # They are *mostly* pre-stripped of the licence, but not entirely: a
    # whole-corpus scan after the v0.4.0 rebuild found the Chesterton files
    # still carrying an "*** END OF THE PROJECT GUTENBERG EBOOK ... ***"
    # footer and the foundation's donation address behind it. Small (a few
    # KB) but real, and pre-existing since v0.1.0 - so the same footer strip
    # the full-text source uses runs here too.
    return _clean_generic(_strip_gutenberg_markers(_GUTENBERG_HEADER.sub("", text)))


# A full Project Gutenberg ebook wraps the real text in a licence header and
# footer, marked by these lines. Everything outside them is boilerplate -
# identical across all 119 books, so leaving it in would teach the model to
# recite the Gutenberg licence. NLTK's `gutenberg` source ships pre-stripped
# copies, which is why it needs a different (much smaller) cleaner.
_PG_START = re.compile(r"\*\*\* ?START OF TH[EIS]+ PROJECT GUTENBERG EBOOK.*?\*\*\*", re.S)
_PG_END = re.compile(r"\*\*\* ?END OF TH[EIS]+ PROJECT GUTENBERG EBOOK.*?\*\*\*", re.S)
#: Transcriber's notes and produced-by credits sit inside the markers but are
#: apparatus, not prose.
_PG_PRODUCED_BY = re.compile(r"^\s*(Produced by|E-?text prepared by|Transcribed from).*$", re.M)


def _strip_gutenberg_markers(text: str) -> str:
    """Cut everything outside the START/END licence markers, if present."""
    start = _PG_START.search(text)
    if start:
        text = text[start.end() :]
    # Searched *after* the header trim, not before: the trim shifts every
    # offset, and reusing a stale one silently keeps the whole footer.
    end = _PG_END.search(text)
    if end:
        text = text[: end.start()]
    return text


def _clean_gutenberg_extended(text: str) -> str:
    return _clean_generic(_PG_PRODUCED_BY.sub("", _strip_gutenberg_markers(text)))


def _clean_europarl(text: str) -> str:
    # Europarl carries <SPEAKER> / <CHAPTER> SGML tags on their own lines.
    return _clean_generic(_EUROPARL_TAG.sub("", text))


# Brown is distributed word-by-word POS-tagged: "The/at Fulton/np-tl said/vbd".
# Stripping the /TAG suffix is mechanical (split on the last '/', since tags
# never contain one - contractions like "didn't/dod*" and "he'd/pps+md" are
# single tokens already). The harder part, and the reason this source was
# rejected before, is that naively joining the bare words back with spaces
# puts a space in front of every comma and period ("election , the jury
# said ."), which is not how English is written. This detokenizes properly:
# no space before closing punctuation, none after an opening bracket, and the
# ``/'' tag pair (Brown's own opening/closing quote marks) becomes a real
# paired '"', converted before the punctuation-spacing pass runs so a quote
# immediately followed by a comma collapses correctly instead of leaving a
# stray space between them.
_BROWN_TAG = re.compile(r"/[^/\s]+$")
_NO_SPACE_BEFORE = re.compile(r"\s+([.,;:!?%)\]}])")
_NO_SPACE_AFTER = re.compile(r"([(\[{])\s+")


def _clean_brown(text: str) -> str:
    lines = []
    for line in text.split("\n"):
        words = []
        for raw_token in line.split():
            word = _BROWN_TAG.sub("", raw_token)
            if word == "``":
                word = "\x00OPEN\x00"
            elif word == "''":
                word = "\x00CLOSE\x00"
            if word:
                words.append(word)
        lines.append(" ".join(words))
    joined = "\n".join(lines)
    joined = re.sub(r"\s*\x00OPEN\x00\s*", ' "', joined)
    joined = re.sub(r"\s*\x00CLOSE\x00\s*", '" ', joined)
    joined = _NO_SPACE_BEFORE.sub(r"\1", joined)
    joined = _NO_SPACE_AFTER.sub(r"\1", joined)
    return _clean_generic(joined)


# Every Project Ben-Yehuda text file ends with a volunteer credit paragraph
# ("the text[s] above were produced by Project Ben-Yehuda volunteers online.
# always available at the following address: https://benyehuda.org/read/...").
# It is real Hebrew, but it is the same paragraph on every single file, and a
# link, not prose. Matched on the fixed, distinctive opening rather than on
# "text" alone, since ordinary Hebrew prose can plainly say "the text".
_BENYEHUDA_FOOTER = re.compile(
    re.escape("את הטקסט[ים] לעיל הפיקו מתנדבי פרויקט בן־יהודה") + r".*",
    re.DOTALL,
)

# Ben-Yehuda's texts come out of an HTML pipeline, and some of that survives
# into the plain-text release: a whole-corpus scan after the v0.4.0 rebuild
# found 9,832 literal `&nbsp;` and 9,429 `↩︎` footnote-return arrows in the
# Hebrew side. Only ~0.03% of the corpus by characters, but it is markup, and
# a model that reads `&nbsp;` in its training text learns to write `&nbsp;`.
# unescape() handles the named/numeric entities; the arrow and the resulting
# non-breaking spaces are then normalised to ordinary spaces.
_FOOTNOTE_RETURN = re.compile(r"[↩⏎]︎?️?")


def _clean_benyehuda(text: str) -> str:
    text = _BENYEHUDA_FOOTER.sub("", text)
    text = html.unescape(text)
    text = _FOOTNOTE_RETURN.sub("", text)
    text = text.replace("\xa0", " ")
    return _clean_generic(text)


_CLEANERS = {
    "generic": _clean_generic,
    "gutenberg": _clean_gutenberg,
    "gutenberg_extended": _clean_gutenberg_extended,
    "brown": _clean_brown,
    "europarl": _clean_europarl,
    "benyehuda": _clean_benyehuda,
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
    and no chunk straddles the boundary.

    Returns a manifest describing exactly what was built - it is written next
    to the corpus as ``manifest.json`` and is the provenance record
    `CLAUDE.md` requires.
    """
    import json

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = cache_dir or out_dir / "cache"

    rng = random.Random(seed)
    train_docs: list[str] = []
    val_docs: list[str] = []
    per_source: list[dict[str, object]] = []

    for source in SOURCES:
        raw_texts: Iterator[str]
        if source.kind == "gutenberg_ids":
            raw_texts = _iter_gutenberg_extended_texts(source, cache_dir)
        elif source.kind == "gutenberg_bulk":
            raw_texts = _iter_gutenberg_bulk_texts(source, cache_dir)
        elif source.kind == "benyehuda":
            raw_texts = _iter_benyehuda_texts(source, cache_dir)
        else:
            archive = download_source(source, cache_dir)
            raw_texts = (text for _name, text in _iter_source_texts(archive, source))
        cleaner = _CLEANERS[source.cleaner]

        chunks: list[str] = []
        skipped = 0
        for text in raw_texts:
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
            "movie_reviews": (
                "distributed lowercased and pre-tokenised; casing and spacing destroyed"
            ),
            "oscar-2301_he": "gated web scrape; no per-document quality signal",
            "sefaria_hebrew_library": (
                "GPL-3.0 on text data, and a liturgical/legal register, not prose"
            ),
            "wikipedia_full_en": (
                "6.4M articles, tens of GB; too large to review - used the "
                "'simple' config, sampled and filtered, instead"
            ),
            "wikipedia_simple_en": (
                "removed in v0.3.0: a 9.9M-parameter model cannot reliably "
                "retain encyclopedic facts, only imitate their register while "
                "confabulating the content. web_search gives Swift a way to "
                "look facts up instead of memorizing them wrong."
            ),
            "reuters": (
                "removed in v0.3.0: news content teaches specific, dated "
                "real-world events as if they were stable facts to recite. "
                "web_search is the honest substitute - current events belong "
                "in a lookup, not in frozen weights."
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
