"""Swift - the first model in the Minerva family, trained from scratch.

Named after the bird. A swift weighs about forty grams and spends almost its
whole life in the air; it is not the largest bird in the sky, it is the one
that never has to stop. That is the design brief: small enough to stay
resident anywhere, fast enough that you never wait for it.

What Swift is
-------------
Swift is **Minerva's own model**. Its architecture is
:class:`minerva.training.model.SwiftLM`, its tokenizer was trained on
Minerva's corpus, and every one of its ~9.9M parameters was learned by
:mod:`minerva.training.trainer` from the real text described in
:mod:`minerva.training.data`. Nothing here is a fine-tune of someone else's
weights.

Swift is a **base** language model. It was pretrained to predict the next
token and has had no instruction tuning, no chat tuning and no RLHF, so:

* it continues text rather than answering questions,
* it cannot call tools, and
* it has no reasoning phase, so every thinking level resolves to ``DO``.

Those are training-stage facts, not missing plumbing, and the spec below
states them honestly rather than advertising capabilities the weights do not
have. See ``docs/TRAINING.md`` for what it would take to change each one.

------------------------------------------------------------------------------
THIS FILE IS THE TEMPLATE FOR EVERY FUTURE MINERVA MODEL.
------------------------------------------------------------------------------
To add a model, copy this file, change the values, and register the new spec
in ``registry.py``. The comments explain *why* each field is what it is, which
is the part you need in order to make good choices for a different model.
"""

from __future__ import annotations

from ..engines.base import SamplingParams
from ..thinking import ThinkingLevel
from .base import ModelSpec

__all__ = ["SWIFT"]


SWIFT = ModelSpec(
    # -- identity ------------------------------------------------------------
    # `name` is Minerva's stable, user-facing handle, and it is also the
    # checkpoint directory name the native engine looks for
    # (checkpoints/swift/best.pt). It must not change when the weights are
    # retrained - that is the whole point of a Minerva name.
    name="swift",
    display_name="Minerva Swift",
    # This versions the *specification*. Retraining the weights bumps the
    # checkpoint, not necessarily this.
    version="0.1.0",
    description=(
        "Minerva's first model: a 9.9M-parameter decoder-only transformer "
        "pretrained from scratch on 27 MB of real English prose. A base "
        "language model - it continues text, it does not follow instructions."
    ),
    tier="small",
    # -- execution -----------------------------------------------------------
    # The native engine runs our own checkpoint in-process. No daemon, no
    # third-party weights, no network.
    engine="minerva",
    # For the native engine this is the checkpoint directory name, not an
    # external model tag.
    engine_model="swift",
    engine_model_fallbacks=(),
    # A base model has never seen a system prompt, so giving it one would just
    # feed it tokens it cannot interpret and corrupt the continuation.
    system_prompt=None,
    # Sampling defaults for a small base model. Temperature is kept below 1.0
    # and top-k is fairly tight because a 9.9M model's tail is mostly noise;
    # a mild repetition penalty holds off the loops small models fall into.
    # context_length matches the trained max_seq_len exactly - RoPE would
    # extrapolate beyond it, but the model has never been trained there.
    sampling=SamplingParams(
        temperature=0.8,
        top_k=40,
        top_p=0.95,
        repeat_penalty=1.1,
        max_tokens=128,
        context_length=512,
    ),
    # -- behaviour -----------------------------------------------------------
    # Swift has no reasoning phase to switch on: it was never trained to
    # produce one. `supports_thinking=False` makes `resolve_thinking` return
    # DO for every request, so the solfege scale degrades cleanly instead of
    # pretending. A future Minerva model trained with reasoning traces flips
    # these two lines and inherits the whole scale unchanged.
    default_thinking=ThinkingLevel.DO,
    max_thinking=ThinkingLevel.DO,
    supports_thinking=False,
    # Tool calling is a trained behaviour. Swift never saw a tool call in
    # pretraining, so advertising it would be a lie the runtime would then
    # have to cover for.
    supports_tools=False,
    supports_vision=False,
    context_length=512,
    parameter_count="9.9M",
    tags=("small", "base", "from-scratch", "local", "cpu"),
)
