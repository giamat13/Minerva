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

import pytest

from minerva.config import MinervaConfig
from minerva.engines.base import Engine
from minerva.engines.ollama import OllamaEngine
from minerva.models.base import MinervaModel
from minerva.models.registry import get_spec
from minerva.tools.registry import ToolRegistry, default_registry


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


@pytest.fixture(scope="session")
def swift(live_engine: Engine) -> MinervaModel:
    """Swift, bound to the live engine - or a skip if its weights are missing."""
    spec = get_spec("swift")
    model = MinervaModel(spec, live_engine, tools=default_registry())
    if not model.is_installed():
        pytest.skip(
            f"none of {spec.candidate_engine_models()} is installed on "
            f"{live_engine.name}; run `minerva pull swift`"
        )
    return model


@pytest.fixture
def empty_registry() -> ToolRegistry:
    return ToolRegistry()
