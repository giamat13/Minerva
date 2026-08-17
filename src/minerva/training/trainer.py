"""Pretraining Swift from scratch.

A complete, honest training loop: AdamW with decoupled weight decay, a cosine
schedule with linear warmup, gradient accumulation, gradient clipping,
deterministic seeding, periodic validation on a held-out split, checkpointing
with resume, and a JSONL log of every measurement taken.

Nothing here is simulated. Running this produces real weights from real text.

    python -m minerva.training.trainer --data data --out checkpoints/swift
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .dataset import TokenDataset
from .model import SWIFT_CONFIG, SwiftConfig, SwiftLM
from .tokenizer import BPETokenizer

__all__ = ["TrainConfig", "Trainer"]


@dataclass
class TrainConfig:
    """Optimisation hyper-parameters."""

    # -- batch -----------------------------------------------------------
    seq_len: int = 512
    micro_batch_size: int = 8
    grad_accum_steps: int = 2
    """Tokens per optimiser step = seq_len x micro_batch x grad_accum."""

    # -- schedule --------------------------------------------------------
    max_steps: int = 4500
    warmup_steps: int = 200
    learning_rate: float = 2e-3
    """High by large-model standards and correct for a small one. Chosen by a
    real 70-step probe over {1,2,3,5}e-3 before the run: 1e-3 and 2e-3 were
    tied on early loss, 3e-3 and 5e-3 were clearly worse."""
    min_lr_ratio: float = 0.1
    """Cosine decays to ``learning_rate * min_lr_ratio``, not to zero - the
    last steps still need to learn something."""

    # -- optimiser -------------------------------------------------------
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # -- evaluation and checkpointing ------------------------------------
    eval_interval: int = 250
    eval_batches: int = 20
    log_interval: int = 10
    checkpoint_interval: int = 500
    sample_interval: int = 500
    sample_prompt: str = "The"
    sample_tokens: int = 80

    # -- housekeeping ----------------------------------------------------
    seed: int = 1729
    compile_model: bool = False
    """torch.compile helps on GPU; on CPU it usually costs more than it saves."""
    max_hours: float | None = None
    """Wall-clock budget. Training stops cleanly at the limit with a full
    checkpoint, rather than being killed halfway through a write."""

    def tokens_per_step(self) -> int:
        return self.seq_len * self.micro_batch_size * self.grad_accum_steps

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainState:
    """Everything needed to resume training exactly where it stopped."""

    step: int = 0
    tokens_seen: int = 0
    best_val_loss: float = float("inf")
    history: list[dict[str, float]] = field(default_factory=list)


class Trainer:
    """Drives pretraining end to end."""

    def __init__(
        self,
        model: SwiftLM,
        train_data: TokenDataset,
        val_data: TokenDataset,
        config: TrainConfig,
        *,
        out_dir: Path,
        tokenizer: BPETokenizer | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.config = config
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)
        self.train_data = train_data
        self.val_data = val_data
        self.tokenizer = tokenizer
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.state = TrainState()
        self.rng = np.random.default_rng(config.seed)
        self.optimizer = self._build_optimizer()
        self.log_path = self.out_dir / "training_log.jsonl"

    # -- optimiser -------------------------------------------------------

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """AdamW, with weight decay applied only to matrices.

        Decaying gains and biases (RMSNorm weights here) shrinks them toward
        zero for no benefit - they are scales, not features.
        """
        decay: list[torch.nn.Parameter] = []
        no_decay: list[torch.nn.Parameter] = []
        for _name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            (decay if param.dim() >= 2 else no_decay).append(param)

        groups = [
            {"params": decay, "weight_decay": self.config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        print(
            f"  optimiser: {sum(p.numel() for p in decay):,} decayed, "
            f"{sum(p.numel() for p in no_decay):,} not decayed"
        )
        return torch.optim.AdamW(
            groups,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            eps=1e-8,
        )

    def learning_rate_at(self, step: int) -> float:
        """Linear warmup, then cosine decay to ``min_lr_ratio`` of the peak."""
        cfg = self.config
        if step < cfg.warmup_steps:
            return cfg.learning_rate * (step + 1) / max(1, cfg.warmup_steps)
        progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
        progress = min(1.0, progress)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return cfg.learning_rate * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * cosine)

    # -- evaluation ------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> float:
        """Mean loss over a fixed, deterministic slice of the validation split."""
        self.model.eval()
        batches = self.val_data.sequential_batches(
            self.config.micro_batch_size, limit=self.config.eval_batches
        )
        if not batches:
            raise RuntimeError("validation split is too small to form a single batch")

        total = 0.0
        for inputs, targets in batches:
            _, loss, _ = self.model(
                inputs.to(self.device), targets=targets.to(self.device)
            )
            assert loss is not None
            total += loss.item()
        self.model.train()
        return total / len(batches)

    @torch.no_grad()
    def sample(self, prompt: str, max_new_tokens: int) -> str:
        """Generate a continuation, so progress is visible as text, not just a number."""
        if self.tokenizer is None:
            return "(no tokenizer attached)"
        ids = self.tokenizer.encode(prompt) or [self.tokenizer.eot_id]
        idx = torch.tensor([ids], dtype=torch.long, device=self.device)
        out = self.model.generate(
            idx,
            max_new_tokens,
            temperature=0.8,
            top_k=40,
            eos_id=self.tokenizer.eot_id,
            seed=self.config.seed,
        )
        return self.tokenizer.decode(out[0].tolist())

    # -- checkpointing ---------------------------------------------------

    def save_checkpoint(self, name: str, *, val_loss: float | None = None) -> Path:
        """Write a self-describing checkpoint.

        The architecture config travels with the weights, so a checkpoint can
        always be loaded without guessing how it was built.
        """
        path = self.out_dir / f"{name}.pt"
        payload = {
            "format_version": 1,
            "model_config": self.model.config.to_dict(),
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "train_config": self.config.to_dict(),
            "state": asdict(self.state),
            "val_loss": val_loss,
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        # Write to a temporary file and rename: a crash mid-write must never
        # leave a corrupt checkpoint where a good one used to be.
        tmp = path.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        tmp.replace(path)
        return path

    def load_checkpoint(self, path: Path) -> None:
        """Resume from a checkpoint written by :meth:`save_checkpoint`."""
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model_state"])
        self.optimizer.load_state_dict(payload["optimizer_state"])
        saved = payload.get("state", {})
        self.state = TrainState(
            step=saved.get("step", 0),
            tokens_seen=saved.get("tokens_seen", 0),
            best_val_loss=saved.get("best_val_loss", float("inf")),
            history=saved.get("history", []),
        )
        # Re-derive the sampling stream from the resumed step so a resumed run
        # does not replay the same batches it already saw.
        self.rng = np.random.default_rng(self.config.seed + self.state.step)
        print(f"  resumed from {path} at step {self.state.step}")

    def _log(self, record: dict[str, Any]) -> None:
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    # -- the loop --------------------------------------------------------

    def train(self) -> TrainState:
        """Run training to ``max_steps`` or the wall-clock budget."""
        cfg = self.config
        self.model.train()
        torch.manual_seed(cfg.seed)

        tokens_per_step = cfg.tokens_per_step()
        started = time.time()
        deadline = started + cfg.max_hours * 3600 if cfg.max_hours else None

        print(
            f"\n  {cfg.max_steps} steps x {tokens_per_step:,} tokens/step = "
            f"{cfg.max_steps * tokens_per_step / 1e6:.1f}M tokens "
            f"({cfg.max_steps * tokens_per_step / len(self.train_data):.2f} epochs)"
        )
        if deadline:
            print(f"  wall-clock budget: {cfg.max_hours:.1f}h")
        print()

        window_loss = 0.0
        window_steps = 0
        step_started = time.time()

        while self.state.step < cfg.max_steps:
            step = self.state.step
            lr = self.learning_rate_at(step)
            for group in self.optimizer.param_groups:
                group["lr"] = lr

            self.optimizer.zero_grad(set_to_none=True)
            accumulated = 0.0
            for _ in range(cfg.grad_accum_steps):
                inputs, targets = self.train_data.batch(
                    cfg.micro_batch_size, self.rng, self.device
                )
                _, loss, _ = self.model(inputs, targets=targets)
                assert loss is not None
                # Scale so the accumulated gradient is the mean over the whole
                # effective batch, not the sum of the micro-batch means.
                (loss / cfg.grad_accum_steps).backward()
                accumulated += loss.item() / cfg.grad_accum_steps

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), cfg.grad_clip
            ).item()
            self.optimizer.step()

            self.state.step += 1
            self.state.tokens_seen += tokens_per_step
            window_loss += accumulated
            window_steps += 1

            if self.state.step % cfg.log_interval == 0:
                elapsed = time.time() - step_started
                mean_loss = window_loss / window_steps
                throughput = tokens_per_step * window_steps / elapsed
                print(
                    f"  step {self.state.step:5d}/{cfg.max_steps}  "
                    f"loss {mean_loss:6.4f}  ppl {math.exp(min(20, mean_loss)):8.2f}  "
                    f"lr {lr:.2e}  |g| {grad_norm:5.2f}  "
                    f"{throughput:6.0f} tok/s  "
                    f"{(time.time() - started) / 60:5.1f}m"
                )
                self._log(
                    {
                        "step": self.state.step,
                        "train_loss": round(mean_loss, 5),
                        "lr": lr,
                        "grad_norm": round(grad_norm, 4),
                        "tokens": self.state.tokens_seen,
                        "tokens_per_second": round(throughput, 1),
                        "elapsed_seconds": round(time.time() - started, 1),
                    }
                )
                window_loss, window_steps = 0.0, 0
                step_started = time.time()

            if self.state.step % cfg.eval_interval == 0:
                val_loss = self.evaluate()
                record = {
                    "step": self.state.step,
                    "val_loss": round(val_loss, 5),
                    "val_perplexity": round(math.exp(min(20, val_loss)), 3),
                    "tokens": self.state.tokens_seen,
                    "elapsed_seconds": round(time.time() - started, 1),
                }
                self.state.history.append(record)
                self._log(record)

                marker = ""
                if val_loss < self.state.best_val_loss:
                    self.state.best_val_loss = val_loss
                    self.save_checkpoint("best", val_loss=val_loss)
                    marker = "  <- best"
                print(
                    f"  ---- step {self.state.step}: val loss {val_loss:.4f}  "
                    f"ppl {math.exp(min(20, val_loss)):.2f}{marker}"
                )
                step_started = time.time()

            if cfg.sample_interval and self.state.step % cfg.sample_interval == 0:
                text = self.sample(cfg.sample_prompt, cfg.sample_tokens)
                print(f"  ---- sample: {text[:400]!r}\n")
                step_started = time.time()

            if self.state.step % cfg.checkpoint_interval == 0:
                self.save_checkpoint("last")

            if deadline and time.time() > deadline:
                print(f"\n  wall-clock budget reached at step {self.state.step}")
                break

        val_loss = self.evaluate()
        if val_loss < self.state.best_val_loss:
            self.state.best_val_loss = val_loss
            self.save_checkpoint("best", val_loss=val_loss)
        self.save_checkpoint("last", val_loss=val_loss)

        total = time.time() - started
        print(
            f"\n  finished {self.state.step} steps in {total / 60:.1f}m  "
            f"({self.state.tokens_seen / 1e6:.2f}M tokens)"
        )
        print(f"  final val loss {val_loss:.4f}  best {self.state.best_val_loss:.4f}")
        return self.state


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pretrain Swift from scratch.")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("checkpoints/swift"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=TrainConfig.max_steps)
    parser.add_argument("--lr", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.micro_batch_size)
    parser.add_argument("--grad-accum", type=int, default=TrainConfig.grad_accum_steps)
    parser.add_argument("--seq-len", type=int, default=TrainConfig.seq_len)
    parser.add_argument("--max-hours", type=float, default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    args = parser.parse_args(argv)

    if args.threads:
        torch.set_num_threads(args.threads)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BPETokenizer.load(args.data / "tokenizer.json")

    model_config = SwiftConfig(
        **{
            **SWIFT_CONFIG.to_dict(),
            "vocab_size": tokenizer.vocab_size,
            "max_seq_len": args.seq_len,
        }
    )
    train_config = TrainConfig(
        seq_len=args.seq_len,
        micro_batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        max_steps=args.steps,
        learning_rate=args.lr,
        max_hours=args.max_hours,
        seed=args.seed,
    )

    model = SwiftLM(model_config)
    train_data = TokenDataset(args.data / "train.bin", args.seq_len)
    val_data = TokenDataset(args.data / "val.bin", args.seq_len)

    print("Pretraining Minerva Swift from scratch")
    print(f"  device       {device}  ({platform.processor() or platform.machine()})")
    print(f"  threads      {torch.get_num_threads()}")
    print(f"  parameters   {model.num_parameters():,} "
          f"({model.num_parameters(non_embedding=True):,} non-embedding)")
    print(f"  vocabulary   {tokenizer.vocab_size:,}")
    print(f"  train tokens {len(train_data):,}")
    print(f"  val tokens   {len(val_data):,}")

    trainer = Trainer(
        model,
        train_data,
        val_data,
        train_config,
        out_dir=args.out,
        tokenizer=tokenizer,
        device=device,
    )
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Record the exact configuration this run used, next to its checkpoints.
    (args.out / "run_config.json").parent.mkdir(parents=True, exist_ok=True)
    (args.out / "run_config.json").write_text(
        json.dumps(
            {
                "model": model_config.to_dict(),
                "training": train_config.to_dict(),
                "parameters": model.num_parameters(),
                "train_tokens": len(train_data),
                "device": str(device),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    trainer.train()
    return 0


if __name__ == "__main__":
    sys.exit(main())
