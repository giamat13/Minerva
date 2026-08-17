"""A byte-level BPE tokenizer, trained from scratch on Minerva's own corpus.

This is a real BPE implementation, not a wrapper around someone else's
tokenizer. It learns its merges from the corpus in `data.py`, so Swift's
vocabulary is Swift's own.

Design
------
**Byte-level.** The base vocabulary is the 256 byte values, so every possible
input encodes - there is no unknown token and no way to hand the model text it
cannot represent. Decoding is exact for any round-trip through valid UTF-8.

**Pre-tokenization.** Text is first split by a regex into word-like pieces
(GPT-2's pattern). Merges are only ever learned *within* a piece, which stops
BPE from inventing tokens that straddle a space boundary in ways that hurt
generalisation.

**Training.** The classic merge loop, made fast enough for a 27 MB corpus by
two standard choices: merges are computed over the *unique* pre-token types
weighted by frequency (not over the raw stream), and pair counts are updated
incrementally through a pair -> containing-words index rather than recounted
from scratch after every merge.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
import time
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

__all__ = ["END_OF_TEXT", "BPETokenizer", "train_bpe"]

#: Marks a document boundary. It is a real token in the vocabulary, so the
#: model learns that documents end rather than running one into the next.
END_OF_TEXT = "<|endoftext|>"

# GPT-2's pre-tokenization pattern: contractions, letter runs, digit runs,
# punctuation runs, and whitespace - with a leading space attached to the
# following word, which is why most tokens in the vocabulary begin with " ".
#
# The alternatives must together cover EVERY character, or anything that falls
# through is silently deleted from the corpus. Underscore is the trap: Python's
# `\w` includes it, so `[^\W\d_]` (the standard "letters only" idiom) excludes
# it from the letter run *and* `[^\s\w]` excludes it from the punctuation run.
# It is therefore spliced back into the punctuation alternative explicitly.
# `tests/test_training_tokenizer.py` asserts full byte coverage.
_PRETOKEN_PATTERN = re.compile(
    r"'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+"
)

# Digits are split individually. Grouping them ("1987" as one token) makes
# arithmetic and number handling markedly worse, because the model never sees
# a consistent representation of place value.
_DIGIT_RUN = re.compile(r"\d")


def _pretokenize(text: str) -> list[str]:
    pieces: list[str] = []
    for match in _PRETOKEN_PATTERN.finditer(text):
        piece = match.group()
        if _DIGIT_RUN.search(piece):
            # Emit an optional leading space + each digit separately.
            head = piece[0] if piece[0] == " " else ""
            for index, char in enumerate(piece.lstrip(" ")):
                pieces.append((head if index == 0 else "") + char)
        else:
            pieces.append(piece)
    return pieces


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_bpe(
    text: str,
    vocab_size: int,
    *,
    special_tokens: tuple[str, ...] = (END_OF_TEXT,),
    verbose: bool = True,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Learn a BPE vocabulary from ``text``.

    Returns ``(vocab, merges)`` where ``vocab`` maps token id to its byte
    string and ``merges`` is the ordered list of learned pair merges.

    :param vocab_size: total target size, including the 256 byte tokens and
        the special tokens.
    """
    if vocab_size <= 256 + len(special_tokens):
        raise ValueError(
            f"vocab_size must exceed {256 + len(special_tokens)} "
            f"(256 byte tokens + {len(special_tokens)} special token(s))"
        )

    # Special tokens must not be broken up, so remove them before counting and
    # give them fixed ids at the top of the vocabulary.
    if special_tokens:
        splitter = "|".join(re.escape(token) for token in special_tokens)
        segments = re.split(f"({splitter})", text)
        segments = [seg for seg in segments if seg not in special_tokens]
    else:
        segments = [text]

    # Frequency of each unique pre-token, as a tuple of byte values.
    counts: dict[tuple[int, ...], int] = defaultdict(int)
    for segment in segments:
        for piece in _pretokenize(segment):
            counts[tuple(piece.encode("utf-8"))] += 1

    words: list[list[int]] = [list(word) for word in counts]
    freqs: list[int] = list(counts.values())
    if verbose:
        print(f"  {len(words):,} unique pre-tokens from {sum(freqs):,} total")

    # pair -> total frequency, and pair -> indices of words containing it.
    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    pair_words: dict[tuple[int, int], set[int]] = defaultdict(set)
    for index, word in enumerate(words):
        freq = freqs[index]
        for pair in itertools.pairwise(word):
            pair_counts[pair] += freq
            pair_words[pair].add(index)

    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merges: list[tuple[bytes, bytes]] = []
    next_id = 256
    n_merges = vocab_size - 256 - len(special_tokens)

    started = time.time()
    for step in range(n_merges):
        if not pair_counts:
            if verbose:
                print(f"  corpus exhausted after {step} merges")
            break

        # Ties broken by pair value, so training is fully deterministic.
        best = max(pair_counts, key=lambda p: (pair_counts[p], p))
        if pair_counts[best] < 2:
            if verbose:
                print(f"  no pair occurs twice after {step} merges; stopping")
            break

        merges.append((vocab[best[0]], vocab[best[1]]))
        vocab[next_id] = vocab[best[0]] + vocab[best[1]]

        _apply_merge(best, next_id, words, freqs, pair_counts, pair_words)
        next_id += 1

        if verbose and (step + 1) % 1000 == 0:
            elapsed = time.time() - started
            print(
                f"  merge {step + 1:5d}/{n_merges}  "
                f"{vocab[next_id - 1]!r}  ({elapsed:.0f}s)"
            )

    for offset, token in enumerate(special_tokens):
        vocab[next_id + offset] = token.encode("utf-8")

    return vocab, merges


def _apply_merge(
    pair: tuple[int, int],
    new_id: int,
    words: list[list[int]],
    freqs: list[int],
    pair_counts: dict[tuple[int, int], int],
    pair_words: dict[tuple[int, int], set[int]],
) -> None:
    """Replace every occurrence of ``pair`` with ``new_id``, updating counts.

    Only the words that actually contain the pair are touched, and only their
    own pair statistics are recomputed. That is what turns an O(vocab x corpus)
    training loop into something that finishes in minutes.
    """
    affected = pair_words.pop(pair, set())
    pair_counts.pop(pair, None)

    for index in affected:
        word = words[index]
        freq = freqs[index]

        # Remove this word's old contribution.
        for old in itertools.pairwise(word):
            if old in pair_counts:
                pair_counts[old] -= freq
                if pair_counts[old] <= 0:
                    pair_counts.pop(old, None)
                    pair_words.pop(old, None)
                else:
                    pair_words[old].discard(index)

        merged: list[int] = []
        position = 0
        while position < len(word):
            if (
                position < len(word) - 1
                and word[position] == pair[0]
                and word[position + 1] == pair[1]
            ):
                merged.append(new_id)
                position += 2
            else:
                merged.append(word[position])
                position += 1
        words[index] = merged

        # Add the word's new contribution.
        for new in itertools.pairwise(merged):
            pair_counts[new] += freq
            pair_words[new].add(index)


# ---------------------------------------------------------------------------
# The tokenizer
# ---------------------------------------------------------------------------


class BPETokenizer:
    """Encodes text to token ids and back, using a trained BPE vocabulary."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: tuple[str, ...] = (END_OF_TEXT,),
    ) -> None:
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        self._token_to_id: dict[bytes, int] = {tok: i for i, tok in vocab.items()}
        # Merge rank: lower means "merge earlier", which is how encoding
        # reproduces exactly the order training learned.
        self._ranks: dict[tuple[bytes, bytes], int] = {
            pair: rank for rank, pair in enumerate(merges)
        }
        self._special_ids: dict[str, int] = {
            token: self._token_to_id[token.encode("utf-8")]
            for token in special_tokens
            if token.encode("utf-8") in self._token_to_id
        }
        self._special_pattern = (
            re.compile("(" + "|".join(re.escape(t) for t in special_tokens) + ")")
            if special_tokens
            else None
        )
        self._cache: dict[str, list[int]] = {}

    # -- properties ------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def eot_id(self) -> int:
        """Id of the end-of-document token."""
        return self._special_ids[END_OF_TEXT]

    # -- encoding --------------------------------------------------------

    def _encode_piece(self, piece: str) -> list[int]:
        """BPE-merge one pre-token into ids, greedily by merge rank."""
        cached = self._cache.get(piece)
        if cached is not None:
            return cached

        symbols: list[bytes] = [bytes([b]) for b in piece.encode("utf-8")]
        while len(symbols) > 1:
            best_rank: int | None = None
            best_index = -1
            for index in range(len(symbols) - 1):
                rank = self._ranks.get((symbols[index], symbols[index + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_index = rank, index
            if best_rank is None:
                break
            symbols[best_index : best_index + 2] = [
                symbols[best_index] + symbols[best_index + 1]
            ]

        ids = [self._token_to_id[symbol] for symbol in symbols]
        if len(self._cache) < 500_000:
            self._cache[piece] = ids
        return ids

    def encode(self, text: str, *, allow_special: bool = True) -> list[int]:
        """Encode ``text`` to token ids."""
        if not text:
            return []

        segments: list[str]
        if allow_special and self._special_pattern is not None:
            segments = self._special_pattern.split(text)
        else:
            segments = [text]

        ids: list[int] = []
        for segment in segments:
            if not segment:
                continue
            if allow_special and segment in self._special_ids:
                ids.append(self._special_ids[segment])
                continue
            for piece in _pretokenize(segment):
                ids.extend(self._encode_piece(piece))
        return ids

    def encode_iter(self, chunks: Iterable[str]) -> Iterable[int]:
        """Stream-encode an iterable of text chunks, for corpora that do not fit in memory."""
        for chunk in chunks:
            yield from self.encode(chunk)

    # -- decoding --------------------------------------------------------

    def decode(self, ids: Iterable[int]) -> str:
        """Decode token ids back to text.

        Invalid UTF-8 is replaced rather than raising: a partially generated
        multi-byte character is normal when streaming token by token.
        """
        payload = b"".join(self.vocab.get(i, b"") for i in ids)
        return payload.decode("utf-8", errors="replace")

    # -- extending -------------------------------------------------------

    def add_special_tokens(self, tokens: Iterable[str]) -> list[int]:
        """Append new single-token markers to the vocabulary.

        Used to give a fine-tune its chat markers. Each becomes **one** token
        rather than the five to seven a byte-level BPE would otherwise spend on
        ``"<|assistant|>"`` - which matters a great deal in a 512-token context,
        and makes the format far easier for a small model to learn.

        Existing ids are never renumbered, so a model trained against the old
        vocabulary stays valid; it only needs its embedding matrix extended by
        the number of ids returned here (see
        :meth:`minerva.training.model.SwiftLM.resize_token_embeddings`).

        Returns the ids assigned, in order. Already-known tokens keep their id.
        """
        assigned: list[int] = []
        next_id = max(self.vocab) + 1
        for token in tokens:
            payload = token.encode("utf-8")
            existing = self._token_to_id.get(payload)
            if existing is not None:
                assigned.append(existing)
                continue

            self.vocab[next_id] = payload
            self._token_to_id[payload] = next_id
            self._special_ids[token] = next_id
            self.special_tokens = (*self.special_tokens, token)
            assigned.append(next_id)
            next_id += 1

        # Longest-first, so "<|/think|>" is never matched as "<|" + ...
        ordered = sorted(self.special_tokens, key=len, reverse=True)
        self._special_pattern = re.compile(
            "(" + "|".join(re.escape(t) for t in ordered) + ")"
        )
        self._cache.clear()
        return assigned

    # -- persistence -----------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Write the tokenizer to a single JSON file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "special_tokens": list(self.special_tokens),
            # Bytes are stored as latin-1, which is a lossless 1:1 mapping for
            # every byte value and keeps the file readable.
            "vocab": {str(i): tok.decode("latin-1") for i, tok in self.vocab.items()},
            "merges": [
                [a.decode("latin-1"), b.decode("latin-1")] for a, b in self.merges
            ],
        }
        target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        """Load a tokenizer written by :meth:`save`."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        vocab = {int(i): tok.encode("latin-1") for i, tok in payload["vocab"].items()}
        merges = [(a.encode("latin-1"), b.encode("latin-1")) for a, b in payload["merges"]]
        return cls(vocab, merges, tuple(payload.get("special_tokens", (END_OF_TEXT,))))

    @classmethod
    def train(
        cls, text: str, vocab_size: int, *, verbose: bool = True
    ) -> BPETokenizer:
        """Train a tokenizer on ``text``."""
        vocab, merges = train_bpe(text, vocab_size, verbose=verbose)
        return cls(vocab, merges)

    def __repr__(self) -> str:
        return f"<BPETokenizer vocab_size={self.vocab_size} merges={len(self.merges)}>"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train Swift's BPE tokenizer.")
    parser.add_argument("--data", type=Path, default=Path("data/train.txt"))
    parser.add_argument("--out", type=Path, default=Path("data/tokenizer.json"))
    parser.add_argument("--vocab-size", type=int, default=8192)
    args = parser.parse_args(argv)

    print(f"Training a byte-level BPE tokenizer (vocab {args.vocab_size}) on {args.data}")
    text = args.data.read_text(encoding="utf-8")
    print(f"  corpus: {len(text) / 1e6:.2f} MB")

    started = time.time()
    tokenizer = BPETokenizer.train(text, args.vocab_size)
    tokenizer.save(args.out)
    print(f"\ntrained in {time.time() - started:.0f}s -> {args.out}")

    sample = "The Minerva Swift model, trained from scratch in 1987."
    ids = tokenizer.encode(sample)
    print(f"\nsample: {sample!r}")
    print(f"  {len(ids)} tokens: {[tokenizer.decode([i]) for i in ids]}")
    assert tokenizer.decode(ids) == sample, "round-trip failed"

    ratio = len(text) / max(1, len(tokenizer.encode(text[:2_000_000])) * (len(text) / 2_000_000))
    print(f"  compression: ~{ratio:.2f} characters per token")
    return 0


if __name__ == "__main__":
    sys.exit(main())
