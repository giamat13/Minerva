"""Shared test fixtures.

A note on how this suite is built, because it is a deliberate choice:

**There are no mock engines.** Nothing in this repository fakes a model
response. The suite is split in two:

* Unit tests exercise real, pure logic - the thinking scale, JSON Schema
  inference, the calculator's evaluator, message encoding/decoding, the
  registries. They need no engine because none of that code needs one.

* Integration tests (``@pytest.mark.integration``) talk to a **live** engine.
  If no engine is reachable they are skipped, loudly and by name. They are
  never quietly replaced by a stub - a green test run that proved nothing is
  worse than a skipped one.

Run everything::

    ollama serve &
    ollama pull qwen3:1.7b
    pytest

Run only the tests that need no engine::

    pytest -m "not integration"
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from minerva.config import MinervaConfig
from minerva.engines.base import Engine, SamplingParams
from minerva.engines.ollama import OllamaEngine
from minerva.models.base import MinervaModel, ModelSpec
from minerva.models.registry import get_spec
from minerva.thinking import ThinkingLevel
from minerva.tools.registry import ToolRegistry, default_registry

# A spec for a hypothetical Minerva model that *does* support tools, thinking
# and a system prompt.
#
# Platform behaviour - system-prompt handling, tool attachment, thinking
# resolution - is the same for every model, so testing it against Swift would
# couple those tests to whatever Swift happens to be today. Swift is currently
# a base model with none of these capabilities, which is a fact about its
# training, not about the platform. Tests that are really about `MinervaModel`
# use this spec; tests that are about Swift use `SWIFT`.
CAPABLE_SPEC = ModelSpec(
    name="capable-test-model",
    display_name="Capable Test Model",
    version="1.0.0",
    description="A hypothetical tool- and thinking-capable Minerva model, for tests.",
    tier="medium",
    engine="ollama",
    engine_model="test-model:1b",
    engine_model_fallbacks=("test-model:0.5b",),
    system_prompt="You are a capable test model.",
    sampling=SamplingParams(temperature=0.7, top_p=0.8),
    default_thinking=ThinkingLevel.FA,
    max_thinking=ThinkingLevel.SOL,
    supports_tools=True,
    supports_thinking=True,
    context_length=8192,
    parameter_count="1B",
)


def _host() -> str:
    return os.environ.get("MINERVA_OLLAMA_HOST", "http://127.0.0.1:11434")


@pytest.fixture(scope="session")
def config() -> MinervaConfig:
    """Configuration pointed at whatever engine the environment provides."""
    return MinervaConfig(ollama_host=_host())


@pytest.fixture(scope="session")
def live_engine(config: MinervaConfig) -> Iterator[Engine]:
    """A real, reachable engine - or a skip.

    The probe is a genuine HTTP call to ``/api/version``; there is no way for
    this fixture to hand back something that is not a live daemon.
    """
    engine = OllamaEngine(host=config.ollama_host, timeout=config.request_timeout)
    health = engine.health()
    if not health.available:
        engine.close()
        pytest.skip(
            f"no live engine: {health}. Start one with `ollama serve` "
            f"(or set MINERVA_OLLAMA_HOST) to run integration tests."
        )
    yield engine
    engine.close()


# Ollama tags that support both tool calling and a reasoning phase, most
# preferred first. The integration suite needs a model with those capabilities;
# which one is installed varies by machine.
_PREFERRED_OLLAMA_TAGS = ("qwen3:1.7b", "qwen3:0.6b", "qwen3:4b", "qwen3:8b")


@pytest.fixture(scope="session")
def ollama_model(live_engine: Engine) -> MinervaModel:
    """A tool- and thinking-capable model on the live Ollama daemon.

    Note this is deliberately *not* Swift. Swift is Minerva's own trained base
    model and runs on the native engine; these tests exercise the Ollama engine
    against whatever capable third-party weights the machine happens to have.
    """
    installed = live_engine.list_available_models()
    tag = next((t for t in _PREFERRED_OLLAMA_TAGS if t in installed), None)
    if tag is None:
        pytest.skip(
            f"none of {_PREFERRED_OLLAMA_TAGS} is installed on {live_engine.name} "
            f"(found: {installed or 'nothing'}); run `ollama pull qwen3:1.7b`"
        )
    spec = CAPABLE_SPEC.with_overrides(
        name="ollama-integration", engine_model=tag, engine_model_fallbacks=()
    )
    return MinervaModel(spec, live_engine, tools=default_registry())


@pytest.fixture(scope="session")
def trained_swift() -> MinervaModel:
    """Swift, loaded from a real checkpoint - or a skip if it has not been trained."""
    pytest.importorskip("torch", reason="needs the 'training' extra")
    from minerva.engines.native import NativeEngine

    checkpoint_dir = Path(os.environ.get("MINERVA_CHECKPOINT_DIR", "checkpoints"))
    engine = NativeEngine(checkpoint_dir=checkpoint_dir, device="cpu")
    health = engine.health()
    if not health.available:
        pytest.skip(f"no trained Swift: {health}. Run `minerva prepare-data && minerva train`.")

    spec = get_spec("swift")
    model = MinervaModel(spec, engine)
    if not model.is_installed():
        pytest.skip(
            f"no checkpoint for {spec.name!r} under {checkpoint_dir} "
            f"(found: {engine.list_available_models()})"
        )
    return model


@pytest.fixture
def empty_registry() -> ToolRegistry:
    return ToolRegistry()
