#!/usr/bin/env python3
"""Example 3: train a model from scratch, end to end, in a few minutes.

This is the real pipeline at miniature scale - the same tokenizer, the same
architecture class and the same `Trainer` that produced Swift, just with a
smaller config and a short run so it finishes while you watch.

It trains on Minerva's real corpus and prints real loss and real samples. If
the samples are poor, that is what a 2-minute run produces, and the example
says so rather than cherry-picking.

Run it::

    minerva prepare-data
    python examples/03_train_a_model.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

from minerva.training.dataset import TokenDataset
from minerva.training.model import SwiftConfig, SwiftLM
from minerva.training.tokenizer import BPETokenizer
from minerva.training.trainer import TrainConfig, Trainer

DATA = Path("data")


def main() -> int:
    if not (DATA / "train.bin").exists():
        print(f"no tokenized corpus in {DATA}. Run: minerva prepare-data", file=sys.stderr)
        return 1

    tokenizer = BPETokenizer.load(DATA / "tokenizer.json")

    # A deliberately tiny model: ~1M parameters, 128-token context. Big enough
    # to show the loss falling, small enough to finish in a couple of minutes
    # on a CPU.
    config = SwiftConfig(
        vocab_size=tokenizer.vocab_size,
        n_layer=4,
        n_head=4,
        d_model=128,
        max_seq_len=128,
    )
    model = SwiftLM(config)
    print(f"training a {model.num_parameters():,}-parameter model from scratch")
    print(f"  {config.n_layer} layers, d_model {config.d_model}, context {config.max_seq_len}")

    train_data = TokenDataset(DATA / "train.bin", config.max_seq_len)
    val_data = TokenDataset(DATA / "val.bin", config.max_seq_len)
    print(f"  {len(train_data):,} training tokens, {len(val_data):,} validation tokens")

    train_config = TrainConfig(
        seq_len=config.max_seq_len,
        micro_batch_size=16,
        grad_accum_steps=1,
        max_steps=300,
        warmup_steps=30,
        learning_rate=3e-3,
        eval_interval=100,
        eval_batches=10,
        log_interval=25,
        checkpoint_interval=300,
        sample_interval=150,
        sample_prompt="It is a truth",
        sample_tokens=40,
    )

    with tempfile.TemporaryDirectory() as tmp:
        trainer = Trainer(
            model,
            train_data,
            val_data,
            train_config,
            out_dir=Path(tmp),
            tokenizer=tokenizer,
            device=torch.device("cpu"),
        )
        state = trainer.train()

        print("\n" + "=" * 70)
        print("WHAT 300 STEPS BUYS")
        print("=" * 70)
        for record in state.history:
            print(
                f"  step {record['step']:4.0f}  "
                f"val loss {record['val_loss']:.4f}  "
                f"perplexity {record['val_perplexity']:.1f}"
            )

        print("\n  samples after training:")
        for prompt in ("It is a truth", "The company said", "Mr President"):
            print(f"\n    {prompt!r}")
            print(f"      -> {' '.join(trainer.sample(prompt, 40).split())}")

    print(
        "\n  This model saw ~0.6M tokens. Swift saw ~37M and is 10x larger.\n"
        "  Expect word-shaped output with weak grammar here - that is what this\n"
        "  much compute produces, and pretending otherwise would be dishonest."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
