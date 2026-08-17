"""Minerva - an AI model platform.

Named after the Roman goddess of wisdom, craft and strategy. The first model
in the family is **Swift**, the smallest and fastest.

Quick start::

    from minerva import Agent, load_model

    model = load_model("swift")
    agent = Agent(model)
    print(agent.run("What is 17 * 43?", thinking="sol").answer)

Everything here runs against a real inference engine (Ollama by default).
There are no simulated responses anywhere in this package: if no engine is
reachable, calls raise :class:`~minerva.errors.EngineUnavailableError`.

Layout
------
``minerva.thinking``  The solfege thinking scale: DO, RE, MI, FA, SOL, LA, SI.
``minerva.messages``  Conversation primitives shared by every layer.
``minerva.engines``   Inference backends (Ollama today, more later).
``minerva.models``    The model catalogue - Swift, and future models.
``minerva.tools``     Tool/function calling.
``minerva.runtime``   The agent loop and multi-turn sessions.
``minerva.cli``       The ``minerva`` command-line interface.
"""

from __future__ import annotations

from .config import MinervaConfig, get_config, load_config, set_config
from .engines import Engine, EngineHealth, GenerationResult, SamplingParams, get_engine
from .errors import MinervaError
from .messages import Message, Role, ToolCall, ToolResult, assistant, system, user
from .models import MinervaModel, ModelSpec, list_specs, load_model, model_names
from .runtime import Agent, AgentRun, Session
from .thinking import ThinkingLevel, ThinkingProfile, parse_thinking_level
from .tools import Tool, ToolRegistry, default_registry, tool

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "AgentRun",
    "Engine",
    "EngineHealth",
    "GenerationResult",
    "Message",
    "MinervaConfig",
    "MinervaError",
    "MinervaModel",
    "ModelSpec",
    "Role",
    "SamplingParams",
    "Session",
    "ThinkingLevel",
    "ThinkingProfile",
    "Tool",
    "ToolCall",
    "ToolRegistry",
    "ToolResult",
    "__version__",
    "assistant",
    "default_registry",
    "get_config",
    "get_engine",
    "list_specs",
    "load_config",
    "load_model",
    "model_names",
    "parse_thinking_level",
    "set_config",
    "system",
    "tool",
    "user",
]
