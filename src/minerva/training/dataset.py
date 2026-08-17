"""Turning the corpus into token batches.

Text is tokenized once, ahead of training, into a flat ``uint16`` array on
disk. Swift's vocabulary is 8,192, so a token fits in two bytes and the whole
tokenized corpus is ~15 MB - small enough to memory-map and let the OS page
cache keep it resident, which removes the data loader from the training loop's
critical path entirely.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .tokenizer import BPETokenizer

__all__ = ["TokenDataset", "encode_corpus"]

#: uint16 holds any id below 65,536. Assert rather than assume.
_TOKEN_DTYPE = np.uint16


def encode_corpus(
    text_path: Path, tokenizer: BPETokenizer, out_path: Path, *, verbose: bool = True
) -> int:
    """Tokenize ``text_path`` into a flat binary array of token ids.

    The text is encoded in chunks split on document boundaries, so a document
    separator is never broken across two encode calls.
    """
    if tokenizer.vocab_size > np.iinfo(_TOKEN_DTYPE).max + 1:
        raise ValueError(
            f"vocab_size {tokenizer.vocab_size} does not fit in {_TOKEN_DTYPE.__name__}"
        )

    text = text_path.read_text(encoding="utf-8")
    started = time.time()

    # Encoding 27 MB in one call would hold a very large intermediate list;
    # chunking keeps peak memory flat and lets us report progress.
    ids: list[int] = []
    chunk_size = 1_000_000
    position = 0
    while position < len(text):
        end = min(position + chunk_size, len(text))
        # Extend to the next newline so we never split mid-token-boundary.
        if end < len(text):
            newline = text.find("\n", end)
            end = newline + 1 if newline != -1 else len(text)
        ids.extend(tokenizer.encode(text[position:end]))
        position = end
        if verbose:
            print(
                f"    {position / len(text) * 100:5.1f}%  {len(ids):,} tokens",
                end="\r",
                flush=True,
            )

    array = np.asarray(ids, dtype=_TOKEN_DTYPE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    array.tofile(out_path)

    if verbose:
        print(
            f"    {out_path.name}: {len(array):,} tokens "
            f"({len(text) / max(1, len(array)):.2f} chars/token, {time.time() - started:.0f}s)"
        )
    return len(array)


class TokenDataset:
    """Random fixed-length windows over a tokenized corpus.

    Sampling windows at random offsets - rather than walking the corpus in
    order - decorrelates the examples inside a batch, which matters a lot when
    the corpus is a concatenation of very different registers.
    """

    def __init__(self, path: Path, seq_len: int) -> None:
        self.path = Path(path)
        self.seq_len = seq_len
        # mmap: the array is never fully loaded, and the page cache is shared
        # across the process rather than copied per batch.
        self.tokens = np.memmap(self.path, dtype=_TOKEN_DTYPE, mode="r")
        if len(self.tokens) < seq_len + 1:
            raise ValueError(
                f"{self.path} holds {len(self.tokens)} tokens, too few for "
                f"seq_len {seq_len}"
            )

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def n_windows(self) -> int:
        """How many distinct non-overlapping windows the corpus contains."""
        return len(self.tokens) // self.seq_len

    def batch(
        self,
        batch_size: int,
        generator: np.random.Generator,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample one ``(inputs, targets)`` batch of next-token pairs."""
        high = len(self.tokens) - self.seq_len - 1
        offsets = generator.integers(0, high, size=batch_size)

        # astype(int64) copies out of the memmap, which is required: torch
        # cannot own memory backed by a read-only mapping.
        x = np.stack([self.tokens[o : o + self.seq_len] for o in offsets]).astype(np.int64)
        y = np.stack(
            [self.tokens[o + 1 : o + 1 + self.seq_len] for o in offsets]
        ).astype(np.int64)

        inputs = torch.from_numpy(x)
        targets = torch.from_numpy(y)
        if device is not None:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
        return inputs, targets

    def sequential_batches(
        self, batch_size: int, limit: int | None = None
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Deterministic, non-overlapping batches - used for validation.

        Evaluation must not resample randomly, or the validation number moves
        for reasons that have nothing to do with the model.
        """
        windows = self.n_windows
        if limit is not None:
            windows = min(windows, limit * batch_size)

        batches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for start in range(0, windows - batch_size + 1, batch_size):
            offsets = [(start + i) * self.seq_len for i in range(batch_size)]
            x = np.stack([self.tokens[o : o + self.seq_len] for o in offsets]).astype(np.int64)
            y = np.stack(
                [self.tokens[o + 1 : o + 1 + self.seq_len] for o in offsets]
            ).astype(np.int64)
            batches.append((torch.from_numpy(x), torch.from_numpy(y)))
        return batches


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tokenize the corpus for training.")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer.json"))
    args = parser.parse_args(argv)

    tokenizer = BPETokenizer.load(args.tokenizer)
    print(f"Tokenizing with {tokenizer}")
    total = 0
    for split in ("train", "val"):
        print(f"  {split}:")
        total += encode_corpus(
            args.data / f"{split}.txt", tokenizer, args.data / f"{split}.bin"
        )
    print(f"\n{total:,} tokens total -> {args.data}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
