"""Configuration.

Settings are resolved from three sources, in increasing order of precedence:

1. Built-in defaults (below).
2. A TOML file - ``./minerva.toml``, ``./minerva.local.toml`` or
   ``~/.config/minerva/config.toml``, whichever is found first.
3. ``MINERVA_*`` environment variables.

Explicit arguments passed in code always beat all three.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from .errors import ConfigurationError
from .thinking import ThinkingLevel, parse_thinking_level

__all__ = [
    "CONFIG_FILENAMES",
    "MinervaConfig",
    "get_config",
    "load_config",
    "set_config",
]

ENV_PREFIX = "MINERVA_"

#: Searched in order; the first one that exists wins.
CONFIG_FILENAMES: tuple[Path, ...] = (
    Path("minerva.local.toml"),
    Path("minerva.toml"),
    Path.home() / ".config" / "minerva" / "config.toml",
)


@dataclass(slots=True)
class MinervaConfig:
    """Process-wide settings."""

    # -- engine ----------------------------------------------------------
    engine: str = "minerva"
    """Which engine to use by default: ``"minerva"`` runs our own trained
    weights in-process; ``"ollama"`` serves third-party weights."""

    checkpoint_dir: str = "checkpoints"
    """Where trained Minerva checkpoints live (``MINERVA_CHECKPOINT_DIR``)."""

    torch_threads: int | None = None
    """CPU threads for the native engine. ``None`` keeps torch's own default."""

    ollama_host: str = "http://127.0.0.1:11434"
    """Base URL of the Ollama daemon (``MINERVA_OLLAMA_HOST``)."""

    request_timeout: float = 300.0
    """Read timeout in seconds. A cold model load on CPU can take minutes."""

    keep_alive: str | None = None
    """How long the engine keeps the model resident, e.g. ``"5m"``, ``"0"``."""

    # -- defaults for a session -----------------------------------------
    default_model: str = "swift-instruct"
    """Minerva model name used when none is given (``MINERVA_DEFAULT_MODEL``)."""

    thinking_level: ThinkingLevel = ThinkingLevel.FA
    """Default note on the solfege scale (``MINERVA_THINKING_LEVEL``)."""

    stream: bool = True
    """Stream output by default in the CLI and in :class:`~minerva.runtime.agent.Agent`."""

    show_thinking: bool = False
    """Render the reasoning trace to the user. Off by default: it is working
    material, not the answer."""

    max_tool_iterations: int | None = None
    """Override the agent's tool-calling ceiling. ``None`` means "derive it from
    the thinking level" (see :attr:`~minerva.thinking.ThinkingProfile.max_tool_iterations`)."""

    enable_tools: bool = True
    """Whether tools are offered to the model at all."""

    # -- provenance ------------------------------------------------------
    source_file: Path | None = field(default=None, compare=False)
    """Which TOML file this configuration came from, if any."""

    def __post_init__(self) -> None:
        self.thinking_level = parse_thinking_level(self.thinking_level)
        if self.request_timeout <= 0:
            raise ConfigurationError("request_timeout must be positive")
        if self.max_tool_iterations is not None and self.max_tool_iterations < 1:
            raise ConfigurationError("max_tool_iterations must be at least 1")

    def replace(self, **changes: Any) -> MinervaConfig:
        """A copy with the given fields changed."""
        current = {f.name: getattr(self, f.name) for f in fields(self)}
        current.update(changes)
        return MinervaConfig(**current)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _coerce(name: str, value: Any, target: Any) -> Any:
    """Coerce a value read from TOML or the environment into the field's type."""
    if target is bool:
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in _TRUE:
            return True
        if text in _FALSE:
            return False
        raise ConfigurationError(f"{name}: expected a boolean, got {value!r}")
    if target is float:
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"{name}: expected a number, got {value!r}") from exc
    if target is int:
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"{name}: expected an integer, got {value!r}") from exc
    return value


#: field name -> (python type, is_optional)
_FIELD_TYPES: dict[str, type] = {
    "engine": str,
    "checkpoint_dir": str,
    "torch_threads": int,
    "ollama_host": str,
    "request_timeout": float,
    "keep_alive": str,
    "default_model": str,
    "stream": bool,
    "show_thinking": bool,
    "max_tool_iterations": int,
    "enable_tools": bool,
}


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except OSError as exc:
        raise ConfigurationError(f"cannot read config file {path}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigurationError(f"{path} is not valid TOML: {exc}") from exc

    # Both a flat file and a [minerva] table are accepted.
    section = data.get("minerva")
    return dict(section) if isinstance(section, dict) else data


def _find_config_file(explicit: Path | None = None) -> Path | None:
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise ConfigurationError(f"config file not found: {path}")
        return path
    if env_path := os.environ.get(f"{ENV_PREFIX}CONFIG"):
        path = Path(env_path).expanduser()
        if not path.is_file():
            raise ConfigurationError(f"{ENV_PREFIX}CONFIG points at a missing file: {path}")
        return path
    for candidate in CONFIG_FILENAMES:
        if candidate.is_file():
            return candidate
    return None


def load_config(path: Path | None = None, *, use_env: bool = True) -> MinervaConfig:
    """Build a :class:`MinervaConfig` from file and environment."""
    values: dict[str, Any] = {}

    config_file = _find_config_file(path)
    if config_file is not None:
        for key, raw in _read_toml(config_file).items():
            name = key.replace("-", "_")
            if name == "thinking_level":
                values[name] = parse_thinking_level(raw)
            elif name in _FIELD_TYPES:
                values[name] = _coerce(f"{config_file}:{key}", raw, _FIELD_TYPES[name])
            # Unknown keys are ignored on purpose: a config file may legitimately
            # carry settings for a newer or older version of Minerva.
        values["source_file"] = config_file

    if use_env:
        for name, target in _FIELD_TYPES.items():
            env_name = f"{ENV_PREFIX}{name.upper()}"
            if (raw := os.environ.get(env_name)) is not None:
                values[name] = _coerce(env_name, raw, target)
        if (raw := os.environ.get(f"{ENV_PREFIX}THINKING_LEVEL")) is not None:
            values["thinking_level"] = parse_thinking_level(raw)

    return MinervaConfig(**values)


_ACTIVE: MinervaConfig | None = None


def get_config() -> MinervaConfig:
    """The active process-wide configuration, loaded on first use."""
    global _ACTIVE
    if _ACTIVE is None:
        _ACTIVE = load_config()
    return _ACTIVE


def set_config(config: MinervaConfig | None) -> None:
    """Replace the active configuration (``None`` forces a reload next time).

    Cached engines are dropped, since they were built from the old settings.
    """
    global _ACTIVE
    _ACTIVE = config

    from .engines.registry import clear_engine_cache

    clear_engine_cache()
