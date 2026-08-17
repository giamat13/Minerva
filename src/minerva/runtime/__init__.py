"""Runtime: the loop that turns a model into an agent.

* :class:`~minerva.runtime.agent.Agent` - one request, driven to a final answer,
  executing any tools the model calls along the way.
* :class:`~minerva.runtime.session.Session` - a multi-turn conversation with
  memory and an Extended Thinking policy.
"""

from __future__ import annotations

from .agent import Agent, AgentRun, AgentStep
from .session import Session

__all__ = ["Agent", "AgentRun", "AgentStep", "Session"]
