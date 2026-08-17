"""Swift-Instruct - Swift, taught to hold a conversation and use tools.

Stage two of the same model. `swift` is the pretrained base; this is that
checkpoint fine-tuned on ~120 hand-written conversations
(:mod:`minerva.training.instruct_data`) onto the chat format in
:mod:`minerva.training.chat`.

Both stay in the catalogue on purpose. The base model is the honest artefact of
pretraining and remains the right thing for text continuation; this one is the
right thing for a conversation. Neither pretends to be the other.

What the fine-tune actually bought
----------------------------------
Measured on held-out conversations (`minerva evaluate-instruct`), not asserted:

* It **holds the conversation format** and stops where it should.
* It **routes correctly** - deciding to call a tool, think first, or answer
  directly - most of the time.
* It **declines to invent** when it does not know something, which was the
  single most important thing to train into a 9.9M model pretrained on
  19th-century novels and 1987 newswire.

What it did **not** buy, and this file will not claim otherwise: reliable tool
*arguments*. The model frequently copies the wrong number out of the question.
Routing is a one-token decision it can learn from ~120 examples; copying
arbitrary operands is a general skill that needs far more data than a set this
size can carry. The numbers are in ``docs/TRAINING.md``.

`supports_tools=True` is therefore a statement that it genuinely emits and uses
tool calls - which it does, and which the agent loop really executes - not a
claim that it gets them right. Read the measured reliability before relying on
it.
"""

from __future__ import annotations

from ..engines.base import SamplingParams
from ..thinking import ThinkingLevel
from .base import ModelSpec

__all__ = ["SWIFT_INSTRUCT"]


SWIFT_INSTRUCT = ModelSpec(
    # -- identity ------------------------------------------------------------
    name="swift-instruct",
    display_name="Minerva Swift-Instruct",
    version="0.1.0",
    description=(
        "Swift, instruction-tuned on hand-written conversations. Holds a chat "
        "turn, calls tools and admits what it does not know. Tool arguments "
        "are unreliable - see docs/TRAINING.md."
    ),
    tier="small",
    # -- execution -----------------------------------------------------------
    engine="minerva",
    engine_model="swift-instruct",
    engine_model_fallbacks=(),
    # The chat format carries no system-prompt slot: none of the fine-tuning
    # conversations had one, so sending one would put the model off the
    # distribution it was actually trained on. `format_conversation` drops it.
    system_prompt=None,
    # Lower temperature than the base model. This model is being asked for
    # structure - a well-formed tool call, a clean stop - and structure is
    # exactly what sampling noise destroys first.
    sampling=SamplingParams(
        temperature=0.6,
        top_k=40,
        top_p=0.95,
        repeat_penalty=1.05,
        max_tokens=96,
        context_length=512,
    ),
    # -- behaviour -----------------------------------------------------------
    # THINKING IS OFF, AND THAT IS A MEASUREMENT, NOT AN OVERSIGHT.
    #
    # The machinery is complete and tested: above DO the engine opens
    # `<|think|>` in the prompt, the model really does produce a reasoning
    # trace, the parser reads it back, and there is a recovery path for a turn
    # that never leaves the block. The fine-tuning set contains 29 reasoning
    # examples.
    #
    # It makes the model worse. On the same held-out set, greedy decoding
    # (`minerva evaluate-instruct`):
    #
    #                       thinking off (DO)   thinking on (MI)
    #     routing accuracy        94.1%              61.8%
    #     tool name accuracy      94.4%              33.3%
    #     argument accuracy       16.7%               0.0%
    #     final answer correct    25.0%               0.0%
    #     honest refusal          66.7%              33.3%
    #
    # Forcing a 9.9M model to reason first does not buy deliberation, it buys
    # 30 more tokens of drift before the answer - and the answer is then built
    # on the drift. CLAUDE.md forbids advertising a capability the weights do
    # not have, so the ceiling is DO and the scale stays clamped.
    #
    # To change this, train it: more and better reasoning traces, on a larger
    # model. The line to edit is this one. See docs/TRAINING.md.
    default_thinking=ThinkingLevel.DO,
    max_thinking=ThinkingLevel.DO,
    supports_thinking=False,
    supports_tools=True,
    supports_vision=False,
    context_length=512,
    parameter_count="9.9M",
    tags=("small", "instruct", "tools", "from-scratch", "local", "cpu"),
)
