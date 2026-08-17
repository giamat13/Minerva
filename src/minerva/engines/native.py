"""The native Minerva engine - runs Minerva's own trained weights.

This is what makes Minerva a model company rather than a client: it loads a
checkpoint produced by :mod:`minerva.training.trainer` and runs it in-process.
No daemon, no external service, no third-party weights.

What Swift actually is
----------------------
Swift is a **base language model**: it was pretrained on raw text to predict
the next token, and nothing else. It has had no instruction tuning, no chat
fine-tuning and no RLHF. That has direct, honest consequences, and this engine
declares them rather than papering over them:

* It **continues text**; it does not answer questions or follow instructions.
  Give it ``"The Bahia cocoa zone"`` and it writes plausible newswire. Give it
  ``"What is the capital of France?"`` and it will most likely write more
  questions, because that is what follows a question in its training data.
* It **cannot call tools**. Tool calling is a trained behaviour, and Swift was
  never trained for it, so :attr:`capabilities` reports ``tools=False`` and
  asking for tools raises rather than silently producing nothing.
* It **has no reasoning phase**, so every thinking level resolves to ``DO``.
  The solfege scale still exists platform-wide for models that can use it.

Making Swift chat or call tools is a *training* problem, not an engine
problem - see ``docs/TRAINING.md``.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from ..errors import (
    CapabilityError,
    EngineRequestError,
    EngineUnavailableError,
    ModelNotInstalledError,
)
from ..messages import Message, Role
from .base import (
    Engine,
    EngineCapabilities,
    EngineHealth,
    GenerationRequest,
    GenerationResult,
    StreamChunk,
    Usage,
)

__all__ = ["DEFAULT_CHECKPOINT_DIR", "NativeEngine"]

DEFAULT_CHECKPOINT_DIR = "checkpoints"

#: Checkpoint file names, in preference order. "best" is chosen by held-out
#: validation loss, which is the one you almost always want.
_CHECKPOINT_NAMES = ("best.pt", "last.pt")


class NativeEngine(Engine):
    """Runs a Minerva checkpoint in-process with PyTorch."""

    name = "minerva"
    capabilities = EngineCapabilities(
        streaming=True,
        # Both of these are false because Swift is a base model, not because
        # the engine cannot express them. When a Minerva model is trained for
        # tool use, this changes with the model, not with the plumbing.
        tools=False,
        thinking=False,
        thinking_trace=False,
        thinking_budget=False,
        vision=False,
    )

    def __init__(
        self,
        checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
        *,
        tokenizer_path: str | Path | None = None,
        device: str | None = None,
        threads: int | None = None,
    ) -> None:
        """
        :param checkpoint_dir: directory holding one sub-directory per trained
            model, e.g. ``checkpoints/swift/best.pt``.
        :param tokenizer_path: tokenizer JSON. Defaults to the one beside the
            checkpoint, then ``data/tokenizer.json``.
        :param device: ``"cpu"``, ``"cuda"``, or ``None`` to pick automatically.
        :param threads: torch CPU thread count.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.tokenizer_path = Path(tokenizer_path) if tokenizer_path else None
        self._device_name = device
        self._threads = threads
        # Loaded lazily and cached per model, so constructing an engine costs
        # nothing and importing minerva never imports torch.
        self._loaded: dict[str, tuple[Any, Any]] = {}

    # -- lazy torch ------------------------------------------------------

    def _torch(self) -> Any:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise EngineUnavailableError(
                "PyTorch is not installed, so the native Minerva engine cannot run. "
                "Install it with: pip install 'minerva[training]'"
            ) from exc
        if self._threads:
            torch.set_num_threads(self._threads)
        return torch

    def _device(self) -> Any:
        torch = self._torch()
        if self._device_name:
            return torch.device(self._device_name)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- checkpoints -----------------------------------------------------

    def _checkpoint_path(self, model: str) -> Path:
        """Locate the checkpoint file for ``model``."""
        directory = self.checkpoint_dir / model
        for name in _CHECKPOINT_NAMES:
            candidate = directory / name
            if candidate.is_file():
                return candidate
        raise ModelNotInstalledError(
            f"no trained weights for {model!r}: expected one of "
            f"{', '.join(str(directory / n) for n in _CHECKPOINT_NAMES)}. "
            f"Train it with: minerva train --model {model}"
        )

    def _find_tokenizer(self, model: str) -> Path:
        candidates = []
        if self.tokenizer_path:
            candidates.append(self.tokenizer_path)
        candidates.append(self.checkpoint_dir / model / "tokenizer.json")
        candidates.append(Path("data") / "tokenizer.json")
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise ModelNotInstalledError(
            f"no tokenizer found for {model!r}; looked in: "
            + ", ".join(str(c) for c in candidates)
        )

    def load(self, model: str) -> tuple[Any, Any]:
        """Load (and cache) the model and tokenizer for ``model``."""
        if model in self._loaded:
            return self._loaded[model]

        torch = self._torch()
        from ..training.model import SwiftConfig, SwiftLM
        from ..training.tokenizer import BPETokenizer

        path = self._checkpoint_path(model)
        payload = torch.load(path, map_location="cpu", weights_only=False)

        # The architecture travels with the weights, so a checkpoint is always
        # self-describing and never needs a matching config file.
        config = SwiftConfig.from_dict(payload["model_config"])
        network = SwiftLM(config)
        network.load_state_dict(payload["model_state"])
        network.eval()
        network.to(self._device())

        tokenizer = BPETokenizer.load(self._find_tokenizer(model))
        if tokenizer.vocab_size != config.vocab_size:
            raise EngineRequestError(
                f"tokenizer vocabulary ({tokenizer.vocab_size}) does not match the "
                f"checkpoint ({config.vocab_size}); they are from different runs"
            )

        self._loaded[model] = (network, tokenizer)
        return network, tokenizer

    # -- Engine interface ------------------------------------------------

    def health(self) -> EngineHealth:
        """Report whether torch is present and any weights exist. Never raises."""
        endpoint = str(self.checkpoint_dir.resolve())
        try:
            import torch
        except ImportError:
            return EngineHealth(
                available=False,
                endpoint=endpoint,
                detail="PyTorch is not installed (pip install 'minerva[training]')",
            )

        models = tuple(self.list_available_models())
        if not models:
            return EngineHealth(
                available=False,
                endpoint=endpoint,
                version=torch.__version__,
                detail=(
                    f"no trained checkpoints under {endpoint}. "
                    f"Train one with: minerva train"
                ),
            )
        return EngineHealth(
            available=True, endpoint=endpoint, version=torch.__version__, models=models
        )

    def list_available_models(self) -> list[str]:
        """Names of every directory under the checkpoint root holding weights."""
        if not self.checkpoint_dir.is_dir():
            return []
        found = []
        for entry in sorted(self.checkpoint_dir.iterdir()):
            if entry.is_dir() and any((entry / n).is_file() for n in _CHECKPOINT_NAMES):
                found.append(entry.name)
        return found

    def has_model(self, model: str) -> bool:
        return model in self.list_available_models()

    def chat(self, request: GenerationRequest) -> GenerationResult:
        """Continue the prompt. Collects :meth:`stream`."""
        final: GenerationResult | None = None
        for chunk in self.stream(request):
            if chunk.kind == "done":
                final = chunk.result
        if final is None:  # pragma: no cover - stream always ends with "done"
            raise EngineRequestError("native engine produced no result")
        return final

    def stream(self, request: GenerationRequest) -> Iterator[StreamChunk]:
        """Continue the prompt, yielding text as it is generated."""
        if request.tools:
            raise CapabilityError(
                "Swift is a base language model and was never trained to call tools. "
                "Run without tools, or use an engine serving a tool-trained model."
            )

        torch = self._torch()
        network, tokenizer = self.load(request.model)

        prompt = flatten_messages(request.messages)
        prompt_ids = tokenizer.encode(prompt)
        if not prompt_ids:
            prompt_ids = [tokenizer.eot_id]

        sampling = request.sampling
        max_new = sampling.max_tokens or 128
        # Leave room for the answer: a base model with a full context window
        # has nowhere to put its continuation.
        budget = network.config.max_seq_len - max_new - 1
        if budget < 1:
            raise EngineRequestError(
                f"max_tokens {max_new} leaves no room in a "
                f"{network.config.max_seq_len}-token context"
            )
        prompt_ids = prompt_ids[-budget:]

        idx = torch.tensor([prompt_ids], dtype=torch.long, device=self._device())
        started = time.monotonic()
        pieces: list[int] = []

        # Bytes are emitted as they arrive, but a multi-byte character may
        # straddle two tokens, so text is flushed only when it decodes cleanly.
        pending: list[int] = []
        for token in network.generate_stream(
            idx,
            max_new,
            temperature=1.0 if sampling.temperature is None else sampling.temperature,
            top_k=sampling.top_k,
            top_p=sampling.top_p,
            repetition_penalty=sampling.repeat_penalty or 1.0,
            eos_id=tokenizer.eot_id,
            seed=sampling.seed,
        ):
            token_id = int(token.item())
            pieces.append(token_id)

            # The document separator is a control token telling us to stop.
            # Decoding it would print a literal "<|endoftext|>" to the user.
            if token_id == tokenizer.eot_id:
                continue

            pending.append(token_id)
            text = tokenizer.decode(pending)
            if "�" in text:
                continue  # incomplete UTF-8; wait for the next token
            pending = []
            yield StreamChunk(kind="content", text=text)

        if pending:
            yield StreamChunk(kind="content", text=tokenizer.decode(pending))

        completion = tokenizer.decode([t for t in pieces if t != tokenizer.eot_id])
        message = Message(role=Role.ASSISTANT, content=completion)
        result = GenerationResult(
            message=message,
            model=request.model,
            finish_reason="stop" if tokenizer.eot_id in pieces else "length",
            usage=Usage(prompt_tokens=len(prompt_ids), completion_tokens=len(pieces)),
            duration_seconds=time.monotonic() - started,
            raw={"prompt": prompt},
        )
        yield StreamChunk(kind="done", result=result)

    def close(self) -> None:
        self._loaded.clear()


def flatten_messages(messages: Sequence[Message]) -> str:
    """Render a conversation as the plain text a base model continues.

    A base model has never seen a chat template, so wrapping the conversation
    in role markers would only feed it tokens it cannot interpret. The honest
    rendering is the one closest to its training distribution: the text of the
    messages, in order. The system prompt is dropped entirely - Swift cannot
    follow instructions, and prepending them would just corrupt the prompt.
    """
    parts = [
        message.content
        for message in messages
        if message.role is not Role.SYSTEM and message.content
    ]
    return "\n\n".join(parts)
