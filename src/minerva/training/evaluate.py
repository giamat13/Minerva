"""Measuring a trained Minerva model.

Reports what a language model can honestly be judged on without an instruction
benchmark it was never trained for:

* **Held-out loss and perplexity** on the validation split.
* **Bits per byte**, which — unlike perplexity — is comparable across models
  with different tokenizers, because it normalises by the raw text rather than
  by however many tokens that tokenizer happened to use.
* **The training curve**, read back from the run's own JSONL log.
* **Real samples** from each register in the training mixture.

    python -m minerva.training.evaluate --checkpoint checkpoints/swift/best.pt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

from .dataset import TokenDataset
from .model import SwiftConfig, SwiftLM
from .tokenizer import BPETokenizer

__all__ = ["EvaluationReport", "evaluate_checkpoint", "held_out_loss"]

# Prompts drawn from the registers the corpus actually contains. A base model
# continues the distribution it saw, so these are what it can be judged on.
SAMPLE_PROMPTS: tuple[str, ...] = (
    "It is a truth universally acknowledged, that",
    "Showers continued throughout the week in the",
    "Mr President, I should like to thank the",
    "My fellow citizens, we gather today to",
    "The company said it expects",
)


class EvaluationReport(dict):
    """The measured results, as a plain dict so it can be written to JSON."""


@torch.no_grad()
def held_out_loss(
    model: SwiftLM, dataset: TokenDataset, *, batch_size: int = 8, max_batches: int | None = None
) -> tuple[float, int]:
    """Mean cross-entropy over deterministic, non-overlapping validation windows.

    Returns ``(mean_loss, tokens_scored)``. Deterministic on purpose: a
    resampled validation number moves for reasons unrelated to the model.
    """
    model.eval()
    batches = dataset.sequential_batches(batch_size, limit=max_batches)
    if not batches:
        raise ValueError("validation split is too small to form a single batch")

    total = 0.0
    tokens = 0
    for inputs, targets in batches:
        _, loss, _ = model(inputs, targets=targets)
        assert loss is not None
        total += loss.item()
        tokens += int(targets.numel())
    return total / len(batches), tokens


def load_checkpoint(path: Path) -> tuple[SwiftLM, dict]:
    """Load a checkpoint written by :mod:`minerva.training.trainer`."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = SwiftConfig.from_dict(payload["model_config"])
    model = SwiftLM(config)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload


def summarise_curve(log_path: Path) -> list[dict[str, float]]:
    """Read the validation points out of a run's JSONL log."""
    if not log_path.is_file():
        return []
    points: list[dict[str, float]] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "val_loss" in record:
            points.append(record)
    return points


def evaluate_checkpoint(
    checkpoint: Path,
    data_dir: Path,
    *,
    batch_size: int = 8,
    max_batches: int | None = None,
    sample_tokens: int = 60,
    seed: int = 1729,
) -> EvaluationReport:
    """Measure a checkpoint and return the report."""
    model, payload = load_checkpoint(checkpoint)
    tokenizer = BPETokenizer.load(data_dir / "tokenizer.json")

    val = TokenDataset(data_dir / "val.bin", model.config.max_seq_len)
    loss, tokens_scored = held_out_loss(
        model, val, batch_size=batch_size, max_batches=max_batches
    )

    # Bits per byte normalises by the raw text, so it is comparable across
    # tokenizers in a way perplexity is not.
    val_text = (data_dir / "val.txt").read_text(encoding="utf-8")
    chars_per_token = len(val_text.encode("utf-8")) / max(1, len(val.tokens))
    bits_per_byte = loss / math.log(2) / chars_per_token

    samples = {}
    for prompt in SAMPLE_PROMPTS:
        ids = tokenizer.encode(prompt) or [tokenizer.eot_id]
        out = model.generate(
            torch.tensor([ids], dtype=torch.long),
            sample_tokens,
            temperature=0.8,
            top_k=40,
            repetition_penalty=1.1,
            eos_id=tokenizer.eot_id,
            seed=seed,
        )
        samples[prompt] = tokenizer.decode(out[0].tolist())

    state = payload.get("state", {})
    return EvaluationReport(
        {
            "checkpoint": str(checkpoint),
            "parameters": model.num_parameters(),
            "parameters_non_embedding": model.num_parameters(non_embedding=True),
            "vocab_size": tokenizer.vocab_size,
            "context_length": model.config.max_seq_len,
            "steps_trained": state.get("step"),
            "tokens_seen": state.get("tokens_seen"),
            "val_loss": round(loss, 5),
            "val_perplexity": round(math.exp(loss), 3),
            "bits_per_byte": round(bits_per_byte, 4),
            "val_tokens_scored": tokens_scored,
            "chance_perplexity": tokenizer.vocab_size,
            "curve": summarise_curve(checkpoint.parent / "training_log.jsonl"),
            "samples": samples,
        }
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained Minerva model.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/swift/best.pt"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--sample-tokens", type=int, default=60)
    parser.add_argument("--json", type=Path, default=None, help="also write the report as JSON")
    args = parser.parse_args(argv)

    if not args.checkpoint.is_file():
        print(
            f"no checkpoint at {args.checkpoint}. Train one with: minerva train",
            file=sys.stderr,
        )
        return 1

    report = evaluate_checkpoint(
        args.checkpoint,
        args.data,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        sample_tokens=args.sample_tokens,
    )

    print(f"Minerva evaluation - {report['checkpoint']}")
    print(f"  parameters      {report['parameters']:,} "
          f"({report['parameters_non_embedding']:,} non-embedding)")
    print(f"  vocabulary      {report['vocab_size']:,}   context {report['context_length']}")
    if report["steps_trained"]:
        print(f"  trained         {report['steps_trained']:,} steps, "
              f"{(report['tokens_seen'] or 0) / 1e6:.1f}M tokens")

    print("\n  held-out validation")
    print(f"    loss            {report['val_loss']:.4f}")
    print(f"    perplexity      {report['val_perplexity']:.2f}   "
          f"(chance = {report['chance_perplexity']:,})")
    print(f"    bits per byte   {report['bits_per_byte']:.4f}")
    print(f"    tokens scored   {report['val_tokens_scored']:,}")

    curve = report["curve"]
    if curve:
        print("\n  training curve")
        for point in curve:
            print(f"    step {point['step']:>6,}  "
                  f"val loss {point['val_loss']:.4f}  "
                  f"ppl {point['val_perplexity']:>8.2f}")

    print("\n  samples (temperature 0.8, top-k 40)")
    for prompt, text in report["samples"].items():
        print(f"\n    prompt: {prompt!r}")
        print(f"    {' '.join(text.split())}")

    if args.json:
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n  report written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
