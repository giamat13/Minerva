"""Inference engines - the layer that actually runs a model.

Minerva separates *what a model is* (see :mod:`minerva.models`) from *what
runs it*. Today the only engine is Ollama; the abstraction exists so adding
another one is a new module plus one line in the registry, with no changes
anywhere else.
"""

from __future__ import annotations

from .base import (
    Engine,
    EngineCapabilities,
    EngineHealth,
    GenerationRequest,
    GenerationResult,
    SamplingParams,
    StreamChunk,
    Usage,
)
from .ollama import OllamaEngine
from .registry import (
    available_engines,
    build_engine,
    clear_engine_cache,
    get_engine,
    register_engine_factory,
)

__all__ = [
    "Engine",
    "EngineCapabilities",
    "EngineHealth",
    "GenerationRequest",
    "GenerationResult",
    "OllamaEngine",
    "SamplingParams",
    "StreamChunk",
    "Usage",
    "available_engines",
    "build_engine",
    "clear_engine_cache",
    "get_engine",
    "register_engine_factory",
]
