"""Exception hierarchy for Minerva.

Every failure mode in the platform raises a subclass of :class:`MinervaError`,
so callers can catch the whole family with one `except` clause while still
being able to distinguish, say, "the engine is down" from "the tool blew up".
"""

from __future__ import annotations


class MinervaError(Exception):
    """Base class for every error raised by Minerva."""


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


class ConfigurationError(MinervaError):
    """The requested configuration is invalid or incomplete."""


# --------------------------------------------------------------------------
# Engines (the things that actually run inference)
# --------------------------------------------------------------------------


class EngineError(MinervaError):
    """Base class for engine-level failures."""


class EngineUnavailableError(EngineError):
    """The engine could not be reached at all (daemon down, wrong host, ...)."""


class EngineRequestError(EngineError):
    """The engine was reached but rejected or failed the request."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class EngineResponseError(EngineError):
    """The engine replied with something we could not parse or understand."""


class EngineNotFoundError(EngineError):
    """No engine is registered under the requested name."""


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


class ModelError(MinervaError):
    """Base class for model-level failures."""


class ModelNotFoundError(ModelError):
    """No Minerva model is registered under the requested name."""


class ModelNotInstalledError(ModelError):
    """The model is registered in Minerva but its weights are not on the engine."""


class CapabilityError(ModelError):
    """The model (or its engine) does not support a requested capability."""


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------


class ToolError(MinervaError):
    """Base class for tool-level failures."""


class ToolNotFoundError(ToolError):
    """The model asked for a tool that is not in the registry."""


class ToolValidationError(ToolError):
    """The arguments the model produced do not match the tool's schema."""


class ToolExecutionError(ToolError):
    """The tool itself raised while running.

    The original exception is kept on ``__cause__`` and on :attr:`original`.
    """

    def __init__(self, tool_name: str, original: BaseException) -> None:
        super().__init__(f"tool {tool_name!r} failed: {original.__class__.__name__}: {original}")
        self.tool_name = tool_name
        self.original = original


# --------------------------------------------------------------------------
# Thinking
# --------------------------------------------------------------------------


class ThinkingLevelError(MinervaError, ValueError):
    """An unknown thinking level was requested."""


# --------------------------------------------------------------------------
# Runtime / agent loop
# --------------------------------------------------------------------------


class AgentError(MinervaError):
    """Base class for failures in the agent loop."""


class MaxIterationsExceeded(AgentError):
    """The tool-calling loop hit its iteration ceiling without settling."""

    def __init__(self, max_iterations: int) -> None:
        super().__init__(
            f"agent did not produce a final answer within {max_iterations} tool-calling "
            f"iterations; raise max_iterations or simplify the task"
        )
        self.max_iterations = max_iterations
