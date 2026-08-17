#!/usr/bin/env python3
"""Example 2: what Swift is made of.

Opens up the tokenizer and the architecture and measures them. Everything
printed here is computed from the real objects, not quoted from documentation.

Needs only the tokenizer and PyTorch - no trained checkpoint::

    minerva prepare-data
    python examples/02_inside_the_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from minerva.training.model import SWIFT_CONFIG, SwiftLM
from minerva.training.tokenizer import BPETokenizer

TOKENIZER_PATH = Path("data/tokenizer.json")


def show_tokenizer() -> BPETokenizer:
    tokenizer = BPETokenizer.load(TOKENIZER_PATH)
    print("=" * 70)
    print("THE TOKENIZER - byte-level BPE, trained on Minerva's own corpus")
    print("=" * 70)
    print(f"\n  vocabulary: {tokenizer.vocab_size:,}   merges: {len(tokenizer.merges):,}")

    for text in (
        "It is a truth universally acknowledged",
        "The Minerva Swift model, trained in 1987.",
        "שלום, café, 日本語",
    ):
        ids = tokenizer.encode(text)
        pieces = [tokenizer.decode([i]) for i in ids]
        print(f"\n  {text!r}")
        print(f"    {len(ids)} tokens: {pieces}")
        # Decoding tokens one at a time shows U+FFFD for the halves of a
        # multi-byte character - the corpus is English, so Hebrew and Japanese
        # fall back to individual bytes. Decoded together they are exact:
        assert tokenizer.decode(ids) == text, "round-trip must be exact"
        print(f"    round-trip exact: {tokenizer.decode(ids) == text}")

    # Digits are deliberately split one by one so place value stays consistent.
    print("\n  digits are always split individually:")
    print(f"    '1987' -> {[tokenizer.decode([i]) for i in tokenizer.encode('1987')]}")

    longest = sorted(tokenizer.vocab.values(), key=len, reverse=True)[:12]
    print("\n  longest tokens the corpus taught it:")
    print(f"    {[t.decode('utf-8', 'replace') for t in longest]}")
    return tokenizer


def show_architecture(tokenizer: BPETokenizer) -> None:
    print("\n" + "=" * 70)
    print("THE ARCHITECTURE - a decoder-only transformer")
    print("=" * 70)

    config = SWIFT_CONFIG
    model = SwiftLM(config)
    print(f"\n  layers {config.n_layer}   heads {config.n_head}   "
          f"d_model {config.d_model}   head_dim {config.head_dim}")
    print(f"  feed-forward {config.ff_dim} (SwiGLU)   context {config.max_seq_len}")
    print(f"\n  parameters: {model.num_parameters():,} total")
    print(f"              {model.num_parameters(non_embedding=True):,} non-embedding")
    embedding = config.vocab_size * config.d_model
    print(
        f"              {embedding:,} in the token embedding "
        f"({embedding / model.num_parameters():.0%} of the model - which is why "
        f"it is tied to the output layer)"
    )

    print("\n  where the parameters live:")
    by_kind: dict[str, int] = {}
    for name, param in model.named_parameters():
        kind = "embedding" if "embed" in name else name.split(".")[-2]
        by_kind[kind] = by_kind.get(kind, 0) + param.numel()
    for kind, count in sorted(by_kind.items(), key=lambda kv: -kv[1]):
        print(f"    {kind:<12} {count:>10,}")


def show_causality() -> None:
    """Demonstrate the property the whole training objective depends on."""
    print("\n" + "=" * 70)
    print("CAUSALITY - a token cannot see the future")
    print("=" * 70)

    torch.manual_seed(0)
    model = SwiftLM(SWIFT_CONFIG)
    model.eval()

    idx = torch.randint(0, SWIFT_CONFIG.vocab_size, (1, 12))
    with torch.no_grad():
        baseline, _, _ = model(idx)
        altered = idx.clone()
        altered[0, 8] = (altered[0, 8] + 1) % SWIFT_CONFIG.vocab_size
        changed, _, _ = model(altered)

    before = (baseline[0, :8] - changed[0, :8]).abs().max().item()
    at = (baseline[0, 8] - changed[0, 8]).abs().max().item()
    print("\n  changed the token at position 8, then measured logit movement:")
    print(f"    positions 0-7 (the past):   max change {before:.2e}  <- must be ~0")
    print(f"    position 8   (the change):  max change {at:.2e}  <- must not be 0")
    print("\n  If the first number were not ~0, attention would be leaking the")
    print("  future and next-token training would be measuring nothing.")


def main() -> int:
    if not TOKENIZER_PATH.exists():
        print(f"no tokenizer at {TOKENIZER_PATH}. Run: minerva prepare-data", file=sys.stderr)
        return 1

    tokenizer = show_tokenizer()
    show_architecture(tokenizer)
    show_causality()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
