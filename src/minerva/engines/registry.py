"""The engine registry.

Engines are registered by name and built lazily, so importing Minerva never
opens a socket. ``get_engine("ollama")`` returns a shared instance;
``build_engine("ollama", host=...)`` builds a fresh one you own.

ADDING AN ENGINE
----------------
Write the class (see :mod:`minerva.engines.base`), then add one line to
:data:`_FACTORIES` below. Nothing else in the codebase needs to change.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any

from ..config import MinervaConfig, get_config
from ..errors import EngineNotFoundError
from .base import Engine

__all__ = [
    "available_engines",
    "build_engine",
    "clear_engine_cache",
    "get_engine",
    "register_engine_factory",
]

# An engine factory receives the active configuration plus any keyword
# overrides and returns a ready-to-use Engine.
EngineFactory = Callable[..., Engine]


def _build_ollama(config: MinervaConfig, **overrides: Any) -> Engine:
    from .ollama import OllamaEngine

    return OllamaEngine(
        host=overrides.pop("host", config.ollama_host),
        timeout=overrides.pop("timeout", config.request_timeout),
        **overrides,
    )


# --- REGISTERED ENGINES ----------------------------------------------------
# Add new engines here. Keep the key short and lowercase - it is what users
# type in config files, env vars and on the CLI.
_FACTORIES: dict[str, EngineFactory] = {
    "ollama": _build_ollama,
}

# Shared instances, keyed by engine name. Engines are safe to reuse: they hold
# a pooled HTTP client and no per-request state.
_CACHE: dict[str, Engine] = {}


def register_engine_factory(name: str, factory: EngineFactory, *, replace: bool = False) -> None:
    """Register an engine factory at runtime (useful for plugins and tests)."""
    key = name.strip().lower()
    if key in _FACTORIES and not replace:
        raise ValueError(f"engine {key!r} is already registered; pass replace=True to override")
    _FACTORIES[key] = factory
    _CACHE.pop(key, None)


def available_engines() -> list[str]:
    """Names of every registered engine."""
    return sorted(_FACTORIES)


def build_engine(
    name: str | None = None, *, config: MinervaConfig | None = None, **overrides: Any
) -> Engine:
    """Build a **new** engine instance. The caller owns it and should close it."""
    config = config or get_config()
    key = (name or config.engine).strip().lower()
    try:
        factory = _FACTORIES[key]
    except KeyError:
        raise EngineNotFoundError(
            f"no engine named {key!r}; available engines: {', '.join(available_engines())}"
        ) from None
    return factory(config, **overrides)


def get_engine(name: str | None = None, *, config: MinervaConfig | None = None) -> Engine:
    """Return the shared engine instance for ``name``, building it on first use."""
    config = config or get_config()
    key = (name or config.engine).strip().lower()
    if key not in _CACHE:
        _CACHE[key] = build_engine(key, config=config)
    return _CACHE[key]


def clear_engine_cache() -> None:
    """Close and drop every cached engine (call after changing configuration)."""
    while _CACHE:
        _, engine = _CACHE.popitem()
        # A failure while closing must never mask the caller's real error.
        with contextlib.suppress(Exception):
            engine.close()
