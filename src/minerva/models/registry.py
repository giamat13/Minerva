"""The model registry - Minerva's catalogue.

Every model the platform knows about is listed in :data:`_SPECS` below.
:func:`load_model` binds one to a live engine and hands you something you can
talk to.

------------------------------------------------------------------------------
ADDING A NEW MODEL
------------------------------------------------------------------------------
1. Create ``src/minerva/models/<name>.py`` with a :class:`~minerva.models.base.ModelSpec`.
   Copy ``swift.py`` - it is commented as a template.
2. Import the spec below and add it to :data:`_SPECS`.
3. Add a case to ``tests/test_models_registry.py`` (the catalogue tests are
   parametrised over every registered spec, so most of it is automatic).

Nothing else in the codebase needs to change: the CLI, the agent loop, the
tool layer and the thinking scale all read from this registry.
See ``docs/ADDING_A_MODEL.md``.
"""

from __future__ import annotations

from collections.abc import Iterable

from ..config import MinervaConfig, get_config
from ..engines.base import Engine
from ..engines.registry import get_engine
from ..errors import ModelNotFoundError
from ..tools.registry import ToolRegistry, default_registry
from .base import MinervaModel, ModelSpec
from .swift import SWIFT

__all__ = [
    "get_spec",
    "list_specs",
    "load_model",
    "model_names",
    "register_spec",
]


# --- THE MINERVA CATALOGUE -------------------------------------------------
# Ordered smallest to largest. Add new models here.
_SPECS: dict[str, ModelSpec] = {
    SWIFT.name: SWIFT,
    # Future models go here, e.g.:
    #   FALCON.name: FALCON,   # medium tier
    #   OWL.name: OWL,         # large tier - Minerva's own bird
}

# Alternative names users may type. Keep this small and obvious.
_ALIASES: dict[str, str] = {
    "minerva-swift": "swift",
    "small": "swift",
}


def register_spec(spec: ModelSpec, *, aliases: Iterable[str] = (), replace: bool = False) -> None:
    """Register a model at runtime (for plugins, experiments and tests)."""
    key = spec.name.strip().lower()
    if key in _SPECS and not replace:
        raise ValueError(f"model {key!r} is already registered; pass replace=True to override")
    _SPECS[key] = spec
    for alias in aliases:
        _ALIASES[alias.strip().lower()] = key


def model_names() -> list[str]:
    """Every registered model name, smallest tier first."""
    order = {"small": 0, "medium": 1, "large": 2}
    return sorted(_SPECS, key=lambda name: (order.get(_SPECS[name].tier, 99), name))


def list_specs() -> list[ModelSpec]:
    """Every registered :class:`~minerva.models.base.ModelSpec`, in catalogue order."""
    return [_SPECS[name] for name in model_names()]


def get_spec(name: str) -> ModelSpec:
    """Look up a spec by name or alias.

    :raises ModelNotFoundError: listing what *is* available.
    """
    key = name.strip().lower()
    key = _ALIASES.get(key, key)
    try:
        return _SPECS[key]
    except KeyError:
        raise ModelNotFoundError(
            f"no Minerva model named {name!r}; available models: {', '.join(model_names())}"
        ) from None


def load_model(
    name: str | None = None,
    *,
    engine: Engine | None = None,
    tools: ToolRegistry | None = None,
    config: MinervaConfig | None = None,
) -> MinervaModel:
    """Load a model and bind it to an engine.

    :param name: Minerva model name. Defaults to
        :attr:`~minerva.config.MinervaConfig.default_model`.
    :param engine: Engine to run on. Defaults to the one the spec asks for.
    :param tools: Tools to expose. Defaults to the built-in registry when tools
        are enabled in configuration, otherwise none.
    """
    config = config or get_config()
    spec = get_spec(name or config.default_model)

    if engine is None:
        engine = get_engine(spec.engine, config=config)

    if tools is None and config.enable_tools and spec.supports_tools:
        tools = default_registry()

    return MinervaModel(spec, engine, tools=tools, keep_alive=config.keep_alive)
