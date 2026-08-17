"""Tests for the native Minerva engine.

These run real weights. Rather than depend on the ~2.5-hour production run,
most tests train a tiny model for a handful of steps and save a real
checkpoint - the same save/load/generate path the shipped model uses.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="needs the 'training' extra")

from minerva.engines.base import GenerationRequest, SamplingParams  # noqa: E402
from minerva.engines.native import NativeEngine, flatten_messages  # noqa: E402
from minerva.errors import CapabilityError, ModelNotInstalledError  # noqa: E402
from minerva.messages import Role, assistant, system, user  # noqa: E402
from minerva.tools.registry import default_registry  # noqa: E402
from minerva.training.model import SwiftConfig, SwiftLM  # noqa: E402
from minerva.training.tokenizer import BPETokenizer  # noqa: E402

pytestmark = pytest.mark.torch

SAMPLE = (
    "It is a truth universally acknowledged, that a single man in possession "
    "of a good fortune, must be in want of a wife."
) * 8


@pytest.fixture(scope="module")
def trained_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A real, tiny, actually-trained checkpoint on disk."""
    root = tmp_path_factory.mktemp("checkpoints")
    model_dir = root / "tiny"
    model_dir.mkdir()

    tokenizer = BPETokenizer.train(SAMPLE, 320, verbose=False)
    tokenizer.save(model_dir / "tokenizer.json")

    config = SwiftConfig(
        vocab_size=tokenizer.vocab_size, n_layer=2, n_head=4, d_model=64, max_seq_len=64
    )
    torch.manual_seed(0)
    model = SwiftLM(config)

    # A few real optimisation steps, so the weights are trained rather than
    # random - the engine must work on something that actually learned.
    ids = torch.tensor([tokenizer.encode(SAMPLE)[:512]], dtype=torch.long)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(15):
        window = ids[:, :64]
        _, loss, _ = model(window[:, :-1], targets=window[:, 1:])
        assert loss is not None
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(
        {
            "format_version": 1,
            "model_config": config.to_dict(),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "state": {"step": 15},
        },
        model_dir / "best.pt",
    )
    return root


@pytest.fixture
def engine(trained_checkpoint: Path) -> NativeEngine:
    return NativeEngine(checkpoint_dir=trained_checkpoint, device="cpu")


class TestDiscovery:
    def test_it_finds_trained_checkpoints(self, engine: NativeEngine) -> None:
        assert engine.list_available_models() == ["tiny"]
        assert engine.has_model("tiny")
        assert not engine.has_model("nonexistent")

    def test_health_reports_a_usable_engine(self, engine: NativeEngine) -> None:
        health = engine.health()
        assert health.available is True
        assert health.models == ("tiny",)
        assert health.version == torch.__version__

    def test_an_empty_checkpoint_directory_is_reported_not_raised(
        self, tmp_path: Path
    ) -> None:
        health = NativeEngine(checkpoint_dir=tmp_path / "nothing").health()
        assert health.available is False
        assert "minerva train" in (health.detail or ""), "the fix must be actionable"

    def test_a_missing_model_names_the_command_that_creates_it(
        self, engine: NativeEngine
    ) -> None:
        with pytest.raises(ModelNotInstalledError, match="minerva train"):
            engine.load("ghost")


class TestLoading:
    def test_the_checkpoint_is_self_describing(self, engine: NativeEngine) -> None:
        # The architecture travels with the weights; no config file needed.
        model, tokenizer = engine.load("tiny")
        assert model.config.n_layer == 2
        assert model.config.vocab_size == tokenizer.vocab_size

    def test_the_model_is_cached_across_calls(self, engine: NativeEngine) -> None:
        first, _ = engine.load("tiny")
        second, _ = engine.load("tiny")
        assert first is second

    def test_a_mismatched_tokenizer_is_caught(
        self, trained_checkpoint: Path, tmp_path: Path
    ) -> None:
        # A tokenizer from a different run would silently produce garbage.
        wrong = BPETokenizer.train(SAMPLE, 300, verbose=False)
        wrong_path = tmp_path / "wrong.json"
        wrong.save(wrong_path)

        engine = NativeEngine(checkpoint_dir=trained_checkpoint, tokenizer_path=wrong_path)
        with pytest.raises(Exception, match="different runs"):
            engine.load("tiny")

    def test_closing_releases_the_cache(self, engine: NativeEngine) -> None:
        engine.load("tiny")
        engine.close()
        assert engine._loaded == {}


class TestGeneration:
    def _request(self, prompt: str, **sampling: object) -> GenerationRequest:
        return GenerationRequest(
            model="tiny",
            messages=[user(prompt)],
            sampling=SamplingParams(max_tokens=16, seed=0, **sampling),  # type: ignore[arg-type]
        )

    def test_it_produces_real_text(self, engine: NativeEngine) -> None:
        result = engine.chat(self._request("It is a truth"))
        assert result.message.role is Role.ASSISTANT
        assert isinstance(result.content, str)
        assert result.usage.completion_tokens is not None
        assert 0 < result.usage.completion_tokens <= 16

    def test_it_reports_prompt_and_completion_tokens(self, engine: NativeEngine) -> None:
        result = engine.chat(self._request("It is a truth universally"))
        assert result.usage.prompt_tokens and result.usage.prompt_tokens > 0
        assert result.duration_seconds is not None and result.duration_seconds >= 0

    def test_streaming_reassembles_into_the_final_text(self, engine: NativeEngine) -> None:
        chunks = list(engine.stream(self._request("It is a truth")))
        assert chunks[-1].kind == "done"
        final = chunks[-1].result
        assert final is not None
        streamed = "".join(c.text for c in chunks if c.kind == "content")
        assert streamed == final.content

    def test_the_same_seed_reproduces_the_same_text(self, engine: NativeEngine) -> None:
        first = engine.chat(self._request("It is a truth")).content
        second = engine.chat(self._request("It is a truth")).content
        assert first == second

    def test_greedy_decoding_works(self, engine: NativeEngine) -> None:
        result = engine.chat(self._request("It is a truth", temperature=0.0))
        assert isinstance(result.content, str)

    def test_an_empty_prompt_still_generates(self, engine: NativeEngine) -> None:
        result = engine.chat(
            GenerationRequest(
                model="tiny", messages=[user("")], sampling=SamplingParams(max_tokens=8, seed=0)
            )
        )
        assert result.usage.completion_tokens

    def test_a_max_tokens_larger_than_the_context_is_refused_clearly(
        self, engine: NativeEngine
    ) -> None:
        request = GenerationRequest(
            model="tiny",
            messages=[user("hello")],
            sampling=SamplingParams(max_tokens=10_000),
        )
        with pytest.raises(Exception, match="leaves no room"):
            engine.chat(request)

    def test_an_over_long_prompt_is_truncated_not_crashed(
        self, engine: NativeEngine
    ) -> None:
        result = engine.chat(self._request(SAMPLE))
        assert result.usage.prompt_tokens is not None
        assert result.usage.prompt_tokens < 64


class TestHonestCapabilities:
    """Swift is a base model, and the engine must say so rather than pretend."""

    def test_it_declares_no_tool_support(self, engine: NativeEngine) -> None:
        assert engine.capabilities.tools is False
        assert engine.capabilities.thinking is False
        assert engine.capabilities.streaming is True

    def test_asking_for_tools_raises_instead_of_silently_ignoring_them(
        self, engine: NativeEngine
    ) -> None:
        request = GenerationRequest(
            model="tiny",
            messages=[user("hello")],
            tools=list(default_registry().specs()),
            sampling=SamplingParams(max_tokens=8),
        )
        with pytest.raises(CapabilityError, match="never trained to call tools"):
            list(engine.stream(request))


class TestPromptFlattening:
    def test_messages_are_joined_as_plain_text(self) -> None:
        assert flatten_messages([user("first"), assistant("second")]) == "first\n\nsecond"

    def test_the_system_prompt_is_dropped(self) -> None:
        # A base model cannot follow instructions; including them would only
        # corrupt the continuation.
        assert flatten_messages([system("be helpful"), user("hello")]) == "hello"

    def test_empty_messages_are_skipped(self) -> None:
        assert flatten_messages([user("a"), assistant(""), user("b")]) == "a\n\nb"

    def test_an_empty_conversation_flattens_to_nothing(self) -> None:
        assert flatten_messages([]) == ""
