# Adding a model to Minerva

The platform was built so that a new model is **one new file plus one line in
the registry**. Nothing in the CLI, the agent loop, the tool layer or the
thinking scale needs to change.

Read `src/minerva/models/swift.py` first — it is written as a commented
template, and it explains *why* each field is set the way it is.

---

## 1. Write the spec

Create `src/minerva/models/<name>.py`:

```python
"""Kestrel - the mid-sized model in the Minerva family.

Named after the falcon that hovers: it holds position over a problem for
longer than Swift can, and sees more of it at once.
"""

from __future__ import annotations

from ..engines.base import SamplingParams
from ..thinking import ThinkingLevel
from .base import ModelSpec

KESTREL_SYSTEM_PROMPT = """\
You are Kestrel, the mid-sized model in the Minerva family.
...
"""

KESTREL = ModelSpec(
    name="kestrel",                    # stable, user-facing, never changes
    display_name="Minerva Kestrel",
    version="0.1.0",                   # versions the SPEC, not the weights
    description="...",                 # one or two real sentences
    tier="medium",                     # small | medium | large

    engine="ollama",
    engine_model="qwen3:8b",           # the engine-side tag to run
    engine_model_fallbacks=("qwen3:4b",),

    system_prompt=KESTREL_SYSTEM_PROMPT,
    sampling=SamplingParams(temperature=0.7, top_p=0.8, context_length=32_768),

    default_thinking=ThinkingLevel.FA,
    max_thinking=ThinkingLevel.SI,     # a bigger model earns a higher ceiling
    supports_tools=True,
    supports_thinking=True,
    context_length=32_768,
    parameter_count="8B",
    tags=("medium", "local", "tools", "thinking"),
)
```

### Choosing the fields

| Field | How to decide |
|---|---|
| `name` | Short, lowercase, memorable. It is the public handle and must survive weight upgrades. |
| `engine_model` | The tag the engine actually runs. Changing this later is fine — that is the point of having a separate Minerva name. |
| `engine_model_fallbacks` | Tags to accept when the preferred one is not installed, most preferred first. Keeps a spec usable across machines. |
| `system_prompt` | Part of the model's identity, so it lives with the spec. Keep it short: every token here is a token the model does not spend on the user's problem. |
| `sampling` | Only set what you mean. A field left as `None` keeps whatever is baked into the model file — setting an "obvious default" silently overrides the model author. |
| `default_thinking` | Where the model sits when nothing is specified. `FA` is the balanced middle. |
| `max_thinking` | The highest note that still *helps*. Small models drift when pushed too far; requests above the ceiling are clamped, never rejected. |

---

## 2. Register it

In `src/minerva/models/registry.py`:

```python
from .kestrel import KESTREL

_SPECS: dict[str, ModelSpec] = {
    SWIFT.name: SWIFT,
    KESTREL.name: KESTREL,      # <- add this
}
```

Add an alias if there is an obvious one:

```python
_ALIASES: dict[str, str] = {
    "minerva-swift": "swift",
    "small": "swift",
    "medium": "kestrel",        # <- and this
}
```

---

## 3. Test it

The catalogue tests in `tests/test_models.py` are parametrised over every
registered spec, so most coverage is automatic: completeness, thinking-ceiling
sanity, unique engine-model candidates. Add a small class for what is specific
to your model:

```python
class TestKestrelSpec:
    def test_it_reaches_extended_thinking(self) -> None:
        assert KESTREL.max_thinking is ThinkingLevel.SI
```

Then verify against real hardware:

```bash
ollama pull qwen3:8b
minerva doctor                       # should list kestrel as installed
minerva ask -m kestrel -t sol "..."  # a real answer from real weights
pytest -m integration
```

---

## 4. If the model needs a *different engine*

Only if the weights are not served by Ollama. See
[`ADDING_AN_ENGINE.md`](ADDING_AN_ENGINE.md) — write the engine, register it,
then point your spec at it with `engine="<name>"`. The model layer does not
change.

---

## Checklist

- [ ] `src/minerva/models/<name>.py` with a complete `ModelSpec`
- [ ] Spec imported and listed in `_SPECS` in `registry.py`
- [ ] Alias added if there is a natural one
- [ ] Model-specific tests in `tests/test_models.py`
- [ ] `minerva doctor` reports it as installed
- [ ] `pytest -m integration` passes against real weights
- [ ] README's model table updated
