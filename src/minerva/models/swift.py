"""Swift - the first and smallest model in the Minerva family.

Named after the bird. A swift weighs about forty grams and spends almost its
whole life in the air; it is not the largest bird in the sky, it is the one
that never has to stop. That is the design brief for this model: small enough
to stay resident on a laptop, fast enough that you never wait for it, and
capable enough to reason and call tools when the question deserves it.

------------------------------------------------------------------------------
THIS FILE IS THE TEMPLATE FOR EVERY FUTURE MINERVA MODEL.
------------------------------------------------------------------------------
To add a model (say, a mid-sized one), copy this file, change the values, and
register the new spec in ``registry.py``. Read the comments below - they
explain *why* each field is set the way it is, which is the part you actually
need in order to make good choices for a different model.
"""

from __future__ import annotations

from ..engines.base import SamplingParams
from ..thinking import ThinkingLevel
from .base import ModelSpec

__all__ = ["SWIFT", "SWIFT_SYSTEM_PROMPT"]


# The system prompt is part of the model's identity, so it lives with the spec
# rather than being scattered across call sites. Keep it short: every token
# here is a token the model does not get to spend on the user's actual problem,
# and that pressure is real on a model this size.
SWIFT_SYSTEM_PROMPT = """\
You are Swift, the smallest and fastest model in the Minerva family.

Answer directly and concisely. Prefer a short, correct answer over a long,
hedged one. When you do not know something, say so plainly instead of guessing.

You may be given tools. Use a tool whenever it gives a more reliable answer
than reasoning alone - arithmetic, the current date and time, and anything
else that depends on facts you cannot see. Never invent a tool result.\
"""


SWIFT = ModelSpec(
    # -- identity ------------------------------------------------------------
    # `name` is Minerva's stable, user-facing handle. It must not change when
    # the underlying weights are upgraded - that is the whole point of having
    # a Minerva name separate from an engine tag.
    name="swift",
    display_name="Minerva Swift",
    # This versions the *specification* (prompt, sampling, defaults), not the
    # weights. Bump it whenever you change anything below.
    version="0.1.0",
    description=(
        "The smallest, fastest Minerva model. Built for quick answers, tool "
        "calling and interactive use on local hardware."
    ),
    tier="small",
    # -- execution -----------------------------------------------------------
    engine="ollama",
    # The engine-side tag. Qwen3 1.7B is the primary target: it is genuinely
    # small (~1.4 GB quantised), and it supports both tool calling and a
    # reasoning phase, which is what lets Swift honour the full solfege scale.
    engine_model="qwen3:1.7b",
    # If the preferred tag is not installed we fall back rather than failing.
    # Order matters: smaller and thinking-capable first, then anything that at
    # least keeps Swift usable. Note that llama3.2:1b supports tools but not a
    # reasoning phase - on that fallback, thinking levels degrade gracefully to
    # a normal answer instead of erroring.
    engine_model_fallbacks=(
        "qwen3:0.6b",
        "qwen3:4b",
        "llama3.2:1b",
    ),
    system_prompt=SWIFT_SYSTEM_PROMPT,
    # Decoding defaults. These are the values Qwen3 documents for its
    # thinking-capable models: enough temperature to avoid the repetition loops
    # small models fall into, low enough to stay on task. Anything left as
    # None keeps the engine's own default rather than silently overriding the
    # value baked into the model file.
    sampling=SamplingParams(
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        repeat_penalty=1.05,
        context_length=8_192,
    ),
    # -- behaviour -----------------------------------------------------------
    # FA (פה) is the balanced middle of the scale: enough deliberation for
    # everyday questions without paying for it on trivial ones.
    default_thinking=ThinkingLevel.FA,
    # Swift's ceiling is SOL (סול). Asking a 1.7B model to think at LA or SI
    # buys drift, not insight - past a point it talks itself out of correct
    # answers. Requests above the ceiling are clamped, never rejected, so the
    # same code works unchanged against a larger Minerva model later, where
    # the ceiling will be SI and Extended Thinking will actually engage.
    max_thinking=ThinkingLevel.SOL,
    supports_tools=True,
    supports_thinking=True,
    supports_vision=False,
    context_length=8_192,
    parameter_count="1.7B",
    tags=("small", "fast", "local", "tools", "thinking"),
)
