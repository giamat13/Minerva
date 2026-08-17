"""Training Minerva's own models from scratch.

This package is what makes Minerva a model company rather than a client for
someone else's weights. It contains, in the order you use them:

``data``       Downloads and cleans the pretraining corpus from real sources,
               with provenance and licensing recorded for every one.
``tokenizer``  A byte-level BPE implementation, trained on that corpus.
``dataset``    Packs tokenized text into memory-mapped batches.
``model``      :class:`~minerva.training.model.SwiftLM` - the architecture:
               a decoder-only transformer with RMSNorm, RoPE, SwiGLU,
               grouped-query attention and tied embeddings.
``trainer``    The pretraining loop: AdamW, cosine schedule with warmup,
               gradient accumulation and clipping, validation, checkpointing.

The whole pipeline::

    minerva prepare-data          # corpus -> tokenizer -> token bins
    minerva train                 # pretrain from scratch
    minerva ask "The"             # run the weights you just trained

PyTorch is an optional dependency (``pip install 'minerva[training]'``), so
importing :mod:`minerva` never pulls it in. This module imports lazily for the
same reason.

See ``docs/TRAINING.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = [
    "SOURCES",
    "SWIFT_CONFIG",
    "BPETokenizer",
    "SwiftConfig",
    "SwiftLM",
    "TokenDataset",
    "TrainConfig",
    "Trainer",
    "build_corpus",
    "train_bpe",
]

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from .data import SOURCES, build_corpus
    from .dataset import TokenDataset
    from .model import SWIFT_CONFIG, SwiftConfig, SwiftLM
    from .tokenizer import BPETokenizer, train_bpe
    from .trainer import TrainConfig, Trainer


# Only `data` and `tokenizer` are pure-stdlib; everything else needs torch.
# Resolving names lazily means `from minerva.training import build_corpus`
# works without torch installed, and only the torch-backed names raise.
_LAZY: dict[str, str] = {
    "SOURCES": ".data",
    "build_corpus": ".data",
    "BPETokenizer": ".tokenizer",
    "train_bpe": ".tokenizer",
    "TokenDataset": ".dataset",
    "SWIFT_CONFIG": ".model",
    "SwiftConfig": ".model",
    "SwiftLM": ".model",
    "TrainConfig": ".trainer",
    "Trainer": ".trainer",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    try:
        module = import_module(module_name, __name__)
    except ImportError as exc:
        raise ImportError(
            f"{name} needs PyTorch, which is not installed. "
            f"Install it with: pip install 'minerva[training]'"
        ) from exc
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(__all__)
