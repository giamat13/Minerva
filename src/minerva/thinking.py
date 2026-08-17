"""Thinking levels on the solfege scale.

Minerva controls how hard a model thinks before it answers using the seven
notes of the solfege scale, in ascending order:

    DO (דו)  ->  RE (רה)  ->  MI (מי)  ->  FA (פה)  ->  SOL (סול)  ->  LA (לה)  ->  SI (סי)
     0            1            2            3             4             5           6
    silent      whisper       quiet      moderate       strong        loud       full voice

The metaphor is deliberate: a higher note is a higher pitch of deliberation.
``DO`` is silence - the model answers immediately with no reasoning phase.
``SI`` is the top of the scale - the longest, most thorough reasoning the
engine can produce.

Extended Thinking
-----------------
Levels ``LA`` and ``SI`` enable *Extended Thinking*: the reasoning trace is
not just longer, it is also **preserved** across turns, so the model can build
on its earlier reasoning instead of starting from scratch every message. Lower
levels still produce a reasoning trace (from ``RE`` upward) but it is treated
as scratch work and dropped from the transcript once the turn is over.

Engine mapping
--------------
Different engines expose thinking differently, so a level is not a raw
parameter - it is an intent. :class:`ThinkingProfile` carries three encodings
of that intent and each engine picks whichever one it understands:

* ``enabled``      - a plain on/off switch (Ollama's ``think: true|false``).
* ``effort``       - a coarse "low"/"medium"/"high" knob (gpt-oss, o-series).
* ``budget_tokens``- an explicit reasoning-token budget (Anthropic-style).

Adding a new engine therefore never requires touching this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Literal

from .errors import ThinkingLevelError

__all__ = [
    "ThinkingEffort",
    "ThinkingLevel",
    "ThinkingProfile",
    "parse_thinking_level",
]

ThinkingEffort = Literal["low", "medium", "high"]


class ThinkingLevel(IntEnum):
    """The seven notes of the solfege scale, ordered by thinking intensity.

    Because this is an :class:`~enum.IntEnum`, levels compare naturally::

        >>> ThinkingLevel.SOL > ThinkingLevel.MI
        True
        >>> ThinkingLevel.LA.is_extended
        True
    """

    DO = 0
    RE = 1
    MI = 2
    FA = 3
    SOL = 4
    LA = 5
    SI = 6

    # -- naming ----------------------------------------------------------

    @property
    def hebrew_name(self) -> str:
        """The note's Hebrew name, e.g. ``'סול'`` for :attr:`SOL`."""
        return _HEBREW_NAMES[self]

    @property
    def latin_name(self) -> str:
        """The note's Latin name in lowercase, e.g. ``'sol'``."""
        return self.name.lower()

    @property
    def description(self) -> str:
        """A one-line, human-readable description of this level."""
        return _DESCRIPTIONS[self]

    # -- semantics -------------------------------------------------------

    @property
    def is_silent(self) -> bool:
        """True when no reasoning phase should run at all (only :attr:`DO`)."""
        return self is ThinkingLevel.DO

    @property
    def is_extended(self) -> bool:
        """True for the levels that enable Extended Thinking (``LA`` and ``SI``)."""
        return self >= ThinkingLevel.LA

    @property
    def profile(self) -> ThinkingProfile:
        """The engine-agnostic :class:`ThinkingProfile` for this level."""
        return _PROFILES[self]

    # -- parsing / laddering ---------------------------------------------

    @classmethod
    def parse(cls, value: ThinkingLevel | str | int) -> ThinkingLevel:
        """Parse a level from a name (Latin or Hebrew), an index, or itself."""
        return parse_thinking_level(value)

    def louder(self, steps: int = 1) -> ThinkingLevel:
        """Move up the scale, clamped at :attr:`SI`."""
        return ThinkingLevel(min(int(self) + steps, int(ThinkingLevel.SI)))

    def quieter(self, steps: int = 1) -> ThinkingLevel:
        """Move down the scale, clamped at :attr:`DO`."""
        return ThinkingLevel(max(int(self) - steps, int(ThinkingLevel.DO)))

    def __str__(self) -> str:
        return self.latin_name


@dataclass(frozen=True, slots=True)
class ThinkingProfile:
    """The concrete, engine-facing meaning of a :class:`ThinkingLevel`.

    Engines translate whichever field they support and ignore the rest, so a
    single level works across engines with very different thinking APIs.
    """

    level: ThinkingLevel
    """The solfege level this profile was derived from."""

    enabled: bool
    """Whether a reasoning phase should run at all."""

    effort: ThinkingEffort | None
    """Coarse effort knob for engines that expose ``low``/``medium``/``high``."""

    budget_tokens: int
    """Suggested reasoning-token budget for engines that take an explicit budget."""

    extended: bool
    """Whether Extended Thinking is on: reasoning is preserved across turns."""

    max_tool_iterations: int
    """How many tool-calling rounds the agent loop should allow by default.

    Deeper thinking usually means a longer plan, so the ceiling scales with
    the note. Callers can always override it explicitly on the agent.
    """

    @property
    def preserve_thinking(self) -> bool:
        """Alias for :attr:`extended`, named from the transcript's point of view."""
        return self.extended


# ---------------------------------------------------------------------------
# The scale itself. Everything below is data, not logic - tuning Minerva's
# thinking behaviour means editing these tables and nothing else.
# ---------------------------------------------------------------------------

_HEBREW_NAMES: dict[ThinkingLevel, str] = {
    ThinkingLevel.DO: "דו",
    ThinkingLevel.RE: "רה",
    ThinkingLevel.MI: "מי",
    ThinkingLevel.FA: "פה",
    ThinkingLevel.SOL: "סול",
    ThinkingLevel.LA: "לה",
    ThinkingLevel.SI: "סי",
}

_DESCRIPTIONS: dict[ThinkingLevel, str] = {
    ThinkingLevel.DO: "Silent - answer immediately, no reasoning phase.",
    ThinkingLevel.RE: "Whisper - a brief sanity check before answering.",
    ThinkingLevel.MI: "Quiet - short reasoning for simple multi-step questions.",
    ThinkingLevel.FA: "Moderate - the balanced default for everyday work.",
    ThinkingLevel.SOL: "Strong - sustained reasoning for genuinely hard problems.",
    ThinkingLevel.LA: "Loud - Extended Thinking; reasoning is preserved across turns.",
    ThinkingLevel.SI: "Full voice - Extended Thinking at maximum depth.",
}

# Token budgets grow roughly geometrically up the scale. They are *suggestions*:
# an engine that has no notion of a reasoning budget simply ignores them.
_PROFILES: dict[ThinkingLevel, ThinkingProfile] = {
    ThinkingLevel.DO: ThinkingProfile(
        level=ThinkingLevel.DO,
        enabled=False,
        effort=None,
        budget_tokens=0,
        extended=False,
        max_tool_iterations=4,
    ),
    ThinkingLevel.RE: ThinkingProfile(
        level=ThinkingLevel.RE,
        enabled=True,
        effort="low",
        budget_tokens=256,
        extended=False,
        max_tool_iterations=4,
    ),
    ThinkingLevel.MI: ThinkingProfile(
        level=ThinkingLevel.MI,
        enabled=True,
        effort="low",
        budget_tokens=1_024,
        extended=False,
        max_tool_iterations=6,
    ),
    ThinkingLevel.FA: ThinkingProfile(
        level=ThinkingLevel.FA,
        enabled=True,
        effort="medium",
        budget_tokens=4_096,
        extended=False,
        max_tool_iterations=8,
    ),
    ThinkingLevel.SOL: ThinkingProfile(
        level=ThinkingLevel.SOL,
        enabled=True,
        effort="medium",
        budget_tokens=8_192,
        extended=False,
        max_tool_iterations=10,
    ),
    ThinkingLevel.LA: ThinkingProfile(
        level=ThinkingLevel.LA,
        enabled=True,
        effort="high",
        budget_tokens=16_384,
        extended=True,
        max_tool_iterations=12,
    ),
    ThinkingLevel.SI: ThinkingProfile(
        level=ThinkingLevel.SI,
        enabled=True,
        effort="high",
        budget_tokens=32_768,
        extended=True,
        max_tool_iterations=16,
    ),
}

# Accepted spellings. Users type these on the CLI, in config files and in env
# vars, so we are generous: Latin names, Hebrew names, common variants and the
# numeric index all resolve to the same level.
_ALIASES: dict[str, ThinkingLevel] = {}


def _register_aliases() -> None:
    extra: dict[ThinkingLevel, tuple[str, ...]] = {
        ThinkingLevel.DO: ("doh", "ut", "off", "none", "silent"),
        ThinkingLevel.RE: ("ray", "minimal", "whisper"),
        ThinkingLevel.MI: ("me", "low", "quiet"),
        ThinkingLevel.FA: ("fah", "medium", "default", "moderate"),
        ThinkingLevel.SOL: ("so", "soh", "high", "strong"),
        ThinkingLevel.LA: ("lah", "very-high", "very_high", "loud"),
        ThinkingLevel.SI: ("ti", "te", "max", "maximum", "full"),
    }
    for level in ThinkingLevel:
        for name in (level.latin_name, level.hebrew_name, str(int(level)), *extra[level]):
            _ALIASES[name] = level


_register_aliases()


def parse_thinking_level(value: ThinkingLevel | str | int) -> ThinkingLevel:
    """Resolve ``value`` to a :class:`ThinkingLevel`.

    Accepts the enum itself, an index ``0..6``, a Latin note name (``"sol"``),
    a Hebrew note name (``"סול"``) or one of the documented aliases
    (``"high"``, ``"max"``, ``"off"``, ...). Matching is case-insensitive and
    tolerant of surrounding whitespace.

    :raises ThinkingLevelError: if the value does not name a level.
    """
    if isinstance(value, ThinkingLevel):
        return value

    if isinstance(value, bool):  # bool is an int subclass - reject it explicitly.
        raise ThinkingLevelError(f"invalid thinking level: {value!r}")

    if isinstance(value, int):
        try:
            return ThinkingLevel(value)
        except ValueError as exc:
            raise ThinkingLevelError(
                f"thinking level index {value} is out of range (expected 0..6)"
            ) from exc

    if isinstance(value, str):
        key = value.strip().lower()
        if key in _ALIASES:
            return _ALIASES[key]
        raise ThinkingLevelError(
            f"unknown thinking level {value!r}; expected one of: "
            + ", ".join(f"{lvl.latin_name} ({lvl.hebrew_name})" for lvl in ThinkingLevel)
        )

    raise ThinkingLevelError(f"invalid thinking level: {value!r}")
