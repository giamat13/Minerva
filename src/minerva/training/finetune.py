"""Instruction tuning: turning Swift into Swift-Instruct.

Stage two. Pretraining (`trainer.py`) taught Swift English; this stage teaches
it a conversation, when to call a tool, and when to admit it does not know.

What makes this a fine-tune rather than a second pretraining run:

* It **starts from the pretrained checkpoint**, not from random weights.
* The learning rate is an order of magnitude lower, so the language the model
  already learned is not overwritten.
* Loss is **masked to the assistant's own tokens**. The model is not trained to
  predict the user's questions or the tool's output - only what it must itself
  produce. See :func:`minerva.training.chat.supervised_segments`.
* The tokenizer gains the chat markers as single tokens and the embedding
  matrix is extended to match, keeping every pretrained row.

    python -m minerva.training.finetune

The result is a separate model in the catalogue - `swift-instruct` - so the
base model stays available and honest about being a base model.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from ..messages import Message, Role
from .chat import CHAT_SPECIAL_TOKENS, format_conversation, supervised_segments
from .instruct_data import INSTRUCT_EXAMPLES, build_examples
from .model import SwiftConfig, SwiftLM
from .tokenizer import BPETokenizer

__all__ = ["FinetuneConfig", "encode_supervised", "finetune"]

#: Masked positions. Matches the ``ignore_index`` the model's loss uses.
IGNORE = -100


@dataclass
class FinetuneConfig:
    """Instruction-tuning hyper-parameters."""

    epochs: int = 12
    """The set is ~120 conversations, so an epoch is tiny. Enough passes are
    needed for the format to stick; too many and the model recites the answers
    verbatim. Chosen by watching held-out loss - see docs/TRAINING.md."""

    batch_size: int = 8
    learning_rate: float = 2e-4
    """~10x below pretraining. Higher, and the model forgets how to write
    English before it learns how to hold a conversation."""

    warmup_steps: int = 20
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95

    val_fraction: float = 0.12
    """Held out to detect the point where it stops learning and starts reciting."""

    max_seq_len: int = 512
    seed: int = 1729
    log_interval: int = 10

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def encode_supervised(
    messages: list[Message], tokenizer: BPETokenizer, max_seq_len: int
) -> tuple[list[int], list[int]]:
    """Encode one conversation into ``(input_ids, labels)``.

    Labels are :data:`IGNORE` everywhere the model is not being asked to
    produce the token. Over-long conversations are dropped by the caller rather
    than truncated: cutting a conversation mid-turn would teach the model to
    stop in the middle of an answer.
    """
    input_ids: list[int] = []
    labels: list[int] = []

    for text, is_target in supervised_segments(messages):
        ids = tokenizer.encode(text)
        input_ids.extend(ids)
        labels.extend(ids if is_target else [IGNORE] * len(ids))

    # Next-token objective: predict position i+1 from position i.
    return input_ids[:-1][:max_seq_len], labels[1:][:max_seq_len]


def _collate(
    batch: list[tuple[list[int], list[int]]], pad_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad a batch to a common length, masking the padding out of the loss."""
    width = max(len(ids) for ids, _ in batch)
    inputs = torch.full((len(batch), width), pad_id, dtype=torch.long)
    targets = torch.full((len(batch), width), IGNORE, dtype=torch.long)
    for row, (ids, labels) in enumerate(batch):
        inputs[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        targets[row, : len(labels)] = torch.tensor(labels, dtype=torch.long)
    return inputs, targets


@torch.no_grad()
def routing_accuracy(
    model: SwiftLM,
    tokenizer: BPETokenizer,
    conversations: list[list[Message]],
) -> float:
    """Fraction of conversations where the model opens its turn correctly.

    This is the capability that matters, and validation loss cannot see it.
    Loss is dominated by the dozens of content tokens in each answer, while the
    decision "call a tool, think first, or answer directly" is a **single
    token** immediately after ``<|assistant|>`` - so a model can memorise every
    answer, drive loss right down, and still route every question wrong.

    Measured by teacher-forcing the prompt and comparing the argmax at that one
    position against what the example says should come next.
    """
    model.eval()
    correct = 0
    counted = 0

    for messages in conversations:
        ids: list[int] = []
        expected: int | None = None
        for text, is_target in supervised_segments(messages):
            encoded = tokenizer.encode(text)
            if is_target and expected is None:
                expected = encoded[0]
                break
            ids.extend(encoded)
        if expected is None or not ids:
            continue

        logits, _, _ = model(torch.tensor([ids], dtype=torch.long))
        counted += 1
        correct += int(logits[0, -1].argmax()) == expected

    model.train()
    return correct / max(1, counted)


@torch.no_grad()
def _evaluate(
    model: SwiftLM, batches: list[tuple[torch.Tensor, torch.Tensor]]
) -> float:
    model.eval()
    total = 0.0
    for inputs, targets in batches:
        _, loss, _ = model(inputs, targets=targets)
        assert loss is not None
        total += loss.item()
    model.train()
    return total / max(1, len(batches))


def finetune(
    base_checkpoint: Path,
    tokenizer_path: Path,
    out_dir: Path,
    config: FinetuneConfig | None = None,
    drop_thinking: bool = True,
) -> dict[str, object]:
    """Run instruction tuning and write the new model. Returns a report."""
    config = config or FinetuneConfig()
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)

    # -- tokenizer: add the chat markers as single tokens -----------------
    tokenizer = BPETokenizer.load(tokenizer_path)
    before = tokenizer.vocab_size
    tokenizer.add_special_tokens(CHAT_SPECIAL_TOKENS)
    print(
        f"  vocabulary {before} -> {tokenizer.vocab_size} "
        f"(+{len(CHAT_SPECIAL_TOKENS)} chat markers)"
    )

    # -- model: continue from the pretrained weights ----------------------
    payload = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
    model = SwiftLM(SwiftConfig.from_dict(payload["model_config"]))
    model.load_state_dict(payload["model_state"])
    model.resize_token_embeddings(tokenizer.vocab_size)
    model.train()
    print(f"  starting from {base_checkpoint} ({model.num_parameters():,} parameters)")

    # -- data --------------------------------------------------------------
    # Swift's spec sets supports_thinking=False, so at inference the prompt
    # always stops at <|assistant|> and the model must answer immediately.
    # Training on examples that put reasoning prose in exactly that position
    # teaches it to emit the reasoning AS the answer: the v0.4.0 run replied
    # to "Who is the president of Brazil?" with "Brazil is a fact I was not
    # trained on, so I should l..." - the verbatim `think` string from its
    # training example. Dropping the thinking blocks removes that
    # train/inference mismatch; `--keep-thinking` restores them for a model
    # that will actually run above DO.
    conversations = build_examples()
    if drop_thinking:
        for messages in conversations:
            for message in messages:
                if message.role is Role.ASSISTANT and message.thinking:
                    message.thinking = None
    encoded: list[tuple[list[int], list[int]]] = []
    kept_conversations: list[list[Message]] = []
    dropped = 0
    for messages in conversations:
        ids, labels = encode_supervised(messages, tokenizer, config.max_seq_len)
        if not any(label != IGNORE for label in labels):
            dropped += 1
            continue
        encoded.append((ids, labels))
        kept_conversations.append(messages)

    rng = random.Random(config.seed)
    order = list(range(len(encoded)))
    rng.shuffle(order)
    n_val = max(1, int(len(encoded) * config.val_fraction))
    val_idx, train_idx = order[:n_val], order[n_val:]
    val_data = [encoded[i] for i in val_idx]
    train_data = [encoded[i] for i in train_idx]
    val_conversations = [kept_conversations[i] for i in val_idx]
    train_conversations = [kept_conversations[i] for i in train_idx]

    supervised_tokens = sum(
        sum(1 for label in labels if label != IGNORE) for _, labels in encoded
    )
    print(
        f"  {len(encoded)} conversations ({len(train_data)} train / {len(val_data)} val), "
        f"{dropped} dropped"
    )
    total_tokens = sum(len(ids) for ids, _ in encoded)
    print(
        f"  {total_tokens:,} tokens, of which {supervised_tokens:,} are "
        f"supervised ({supervised_tokens / max(1, total_tokens):.0%})"
    )

    val_batches = [
        _collate(val_data[i : i + config.batch_size], tokenizer.eot_id)
        for i in range(0, len(val_data), config.batch_size)
    ]

    # -- optimiser ---------------------------------------------------------
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
    )

    steps_per_epoch = max(1, math.ceil(len(train_data) / config.batch_size))
    total_steps = steps_per_epoch * config.epochs
    print(f"  {config.epochs} epochs x {steps_per_epoch} steps = {total_steps} steps\n")

    history: list[dict[str, float]] = []
    best_val = float("inf")
    best_route = -1.0
    step = 0
    started = time.time()

    for epoch in range(config.epochs):
        rng.shuffle(train_data)
        for start in range(0, len(train_data), config.batch_size):
            inputs, targets = _collate(
                train_data[start : start + config.batch_size], tokenizer.eot_id
            )

            if step < config.warmup_steps:
                lr = config.learning_rate * (step + 1) / config.warmup_steps
            else:
                progress = (step - config.warmup_steps) / max(
                    1, total_steps - config.warmup_steps
                )
                cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
                lr = config.learning_rate * (
                    config.min_lr_ratio + (1 - config.min_lr_ratio) * cosine
                )
            for group in optimizer.param_groups:
                group["lr"] = lr

            _, loss, _ = model(inputs, targets=targets)
            assert loss is not None
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            step += 1

            if step % config.log_interval == 0:
                print(
                    f"  step {step:4d}/{total_steps}  epoch {epoch + 1:2d}  "
                    f"loss {loss.item():6.4f}  lr {lr:.2e}"
                )

        val_loss = _evaluate(model, val_batches)
        route_val = routing_accuracy(model, tokenizer, val_conversations)
        route_train = routing_accuracy(model, tokenizer, train_conversations)
        record = {
            "epoch": epoch + 1,
            "step": step,
            "val_loss": round(val_loss, 5),
            "routing_val": round(route_val, 4),
            "routing_train": round(route_train, 4),
        }
        history.append(record)

        # Selected on held-out ROUTING, not on loss. Loss is dominated by
        # answer text; routing is the capability being trained.
        marker = ""
        if route_val > best_route or (route_val == best_route and val_loss < best_val):
            best_route, best_val = route_val, val_loss
            marker = "  <- best"
            _save(model, tokenizer, config, out_dir, "best", val_loss, history)
        print(
            f"  ---- epoch {epoch + 1}: val loss {val_loss:.4f}  "
            f"routing train {route_train:.0%} / val {route_val:.0%}{marker}"
        )

    _save(model, tokenizer, config, out_dir, "last", best_val, history)
    tokenizer.save(out_dir / "tokenizer.json")

    report = {
        "base_checkpoint": str(base_checkpoint),
        "examples": len(INSTRUCT_EXAMPLES),
        "conversations": len(encoded),
        "supervised_tokens": supervised_tokens,
        "vocab_size": tokenizer.vocab_size,
        "epochs": config.epochs,
        "best_val_loss": round(best_val, 5),
        "best_routing_val": round(best_route, 4),
        "history": history,
        "minutes": round((time.time() - started) / 60, 2),
    }
    (out_dir / "finetune_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"  finished in {report['minutes']} minutes, best val loss {best_val:.4f}")
    return report


def _save(
    model: SwiftLM,
    tokenizer: BPETokenizer,
    config: FinetuneConfig,
    out_dir: Path,
    name: str,
    val_loss: float,
    history: list[dict[str, float]],
) -> None:
    path = out_dir / f"{name}.pt"
    tmp = path.with_suffix(".pt.tmp")
    torch.save(
        {
            "format_version": 1,
            "model_config": model.config.to_dict(),
            "model_state": model.state_dict(),
            "finetune_config": config.to_dict(),
            "state": {"step": history[-1]["step"] if history else 0},
            "val_loss": val_loss,
            "chat_format": True,
        },
        tmp,
    )
    tmp.replace(path)
    # The tokenizer travels with the weights: it has markers the base one lacks.
    tokenizer.save(out_dir / "tokenizer.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Instruction-tune Swift.")
    parser.add_argument("--base", type=Path, default=Path("checkpoints/swift/best.pt"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer.json"))
    parser.add_argument("--out", type=Path, default=Path("checkpoints/swift-instruct"))
    parser.add_argument("--epochs", type=int, default=FinetuneConfig.epochs)
    parser.add_argument("--lr", type=float, default=FinetuneConfig.learning_rate)
    parser.add_argument(
        "--keep-thinking",
        action="store_true",
        help="keep <|think|> blocks in the training data; only useful for a model "
             "that will actually run above DO (Swift does not - see finetune()).",
    )
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args(argv)

    if args.threads:
        torch.set_num_threads(args.threads)
    if not args.base.is_file():
        print(f"no base checkpoint at {args.base}. Run: minerva train", file=sys.stderr)
        return 1

    print("Instruction-tuning Minerva Swift")
    finetune(
        args.base,
        args.tokenizer,
        args.out,
        FinetuneConfig(epochs=args.epochs, learning_rate=args.lr),
        drop_thinking=not args.keep_thinking,
    )

    # Show what it actually does now, rather than asserting that it works.
    from .chat import parse_response

    tokenizer = BPETokenizer.load(args.out / "tokenizer.json")
    payload = torch.load(args.out / "best.pt", map_location="cpu", weights_only=False)
    model = SwiftLM(SwiftConfig.from_dict(payload["model_config"]))
    model.load_state_dict(payload["model_state"])
    model.eval()

    print("\n  samples:")
    from ..messages import user as user_message

    for prompt in ("What is 23 times 19?", "Hello.", "Who won the World Cup in 2022?"):
        text = format_conversation([user_message(prompt)], thinking=False)
        ids = torch.tensor([tokenizer.encode(text)], dtype=torch.long)
        out = model.generate(
            ids, 60, temperature=0.0, eos_id=tokenizer._special_ids.get("<|end|>")
        )
        generated = tokenizer.decode(out[0, ids.shape[1] :].tolist())
        parsed = parse_response(generated)
        print(f"    {prompt!r}")
        print(f"      calls:  {[str(c) for c in parsed.tool_calls]}")
        print(f"      answer: {parsed.content!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
