# Adding a model to Minerva

There are two kinds of Minerva model, and the procedure differs:

* **A model we train** — the Minerva way. Weights come from
  `minerva.training`, run on the native engine. Swift is one.
* **A model someone else trained** — served through an external engine such as
  Ollama. Useful for comparison and for capabilities we have not trained yet
  (tool calling, reasoning).

Both end as a `ModelSpec` in `models/registry.py`, and in both cases adding one
is **a new file plus one line in the registry**. Nothing in the CLI, the agent
loop, the tool layer or the thinking scale changes.

Read `src/minerva/models/swift.py` first — it is written as a commented
template and explains *why* each field is set the way it is.

---

## Track A — a model we train ourselves

### 1. Choose a size and train it

Sizing is a compute decision, not a taste decision. Measure your hardware, then
use `C ≈ 6ND` with `D ≈ 20N` as the starting point. The arithmetic for Swift is
worked through in [`TRAINING.md`](TRAINING.md).

```python
# src/minerva/training/model.py — add alongside SWIFT_CONFIG
KESTREL_CONFIG = SwiftConfig(
    vocab_size=16384,
    n_layer=12,
    n_head=12,
    n_kv_head=4,      # grouped-query attention starts paying off at this size
    d_model=768,
    max_seq_len=2048,
)                     # ~90M parameters
```

```bash
minerva prepare-data --vocab-size 16384
minerva train --out checkpoints/kestrel --steps 50000 --batch-size 64
minerva evaluate --checkpoint checkpoints/kestrel/best.pt
```

`SwiftLM` is size-agnostic; only the config changes.

### 2. Write the spec

`src/minerva/models/kestrel.py`:

```python
KESTREL = ModelSpec(
    name="kestrel",
    display_name="Minerva Kestrel",
    version="0.1.0",
    description="...",              # two real sentences
    tier="medium",

    engine="minerva",               # our own engine
    engine_model="kestrel",         # the CHECKPOINT DIRECTORY name
    engine_model_fallbacks=(),

    system_prompt=None,             # unless it was trained with one
    sampling=SamplingParams(temperature=0.8, top_k=40, context_length=2048),

    default_thinking=ThinkingLevel.DO,
    max_thinking=ThinkingLevel.DO,
    supports_thinking=False,        # unless trained with reasoning traces
    supports_tools=False,           # unless trained on tool calls
    context_length=2048,
    parameter_count="90M",
    tags=("medium", "base", "from-scratch"),
)
```

### 3. Declare capabilities honestly

This is the field set people get wrong, and `CLAUDE.md` is explicit about it:
**`supports_tools` and `supports_thinking` describe what the weights were
trained to do, not what you wish they did.**

| If the model was… | then… |
|---|---|
| pretrained only (a base model) | `supports_tools=False`, `supports_thinking=False`, `max_thinking=DO` |
| fine-tuned on instructions | it can follow instructions; tools still need tool training |
| trained on tool-call traces | `supports_tools=True` |
| trained with reasoning traces | `supports_thinking=True`, and set `max_thinking` to the highest note that still *helps* |

Advertising a capability the weights lack does not add the capability — it just
moves the failure somewhere confusing.

`max_thinking` deserves thought. Requests above it are **clamped, not
rejected**, so the same caller code works across the whole family. Pick the
highest note where more deliberation still improves answers; past that point a
small model talks itself out of correct ones.

---

## Track B — a model served by an external engine

Same file, different execution fields:

```python
KESTREL = ModelSpec(
    name="kestrel",
    ...
    engine="ollama",
    engine_model="qwen3:8b",                    # an engine-side tag
    engine_model_fallbacks=("qwen3:4b",),       # tried in order if absent
    system_prompt=KESTREL_SYSTEM_PROMPT,        # instruct models take one
    supports_tools=True,                        # if the weights support it
    supports_thinking=True,
    max_thinking=ThinkingLevel.SI,
)
```

Verify with `minerva doctor`, which reports whether the tag is installed, and
`ollama pull qwen3:8b` if it is not.

If the weights are not served by any registered engine, write one first —
[`ADDING_AN_ENGINE.md`](ADDING_AN_ENGINE.md).

---

## Registering (both tracks)

`src/minerva/models/registry.py`:

```python
from .kestrel import KESTREL

_SPECS: dict[str, ModelSpec] = {
    SWIFT.name: SWIFT,
    KESTREL.name: KESTREL,      # <- add this
}

_ALIASES: dict[str, str] = {
    "minerva-swift": "swift",
    "small": "swift",
    "medium": "kestrel",        # <- and this, if there is a natural alias
}
```

---

## Field reference

| Field | How to decide |
|---|---|
| `name` | Short, lowercase, memorable. The public handle; must survive retraining. For the native engine it is also the checkpoint directory name. |
| `version` | Versions the **spec** (prompt, sampling, defaults), not the weights. |
| `engine_model` | Native engine: the checkpoint directory. External engine: the tag to run. |
| `engine_model_fallbacks` | External engines only — tags to accept when the preferred one is absent. There is nothing to fall back to for our own checkpoints. |
| `system_prompt` | Only if the model was trained to use one. A base model has never seen one and prepending it corrupts the prompt. |
| `sampling` | Set only what you mean. A field left `None` keeps the engine's default; setting an "obvious default" silently overrides the model author. |
| `context_length` | What it was **trained** on. RoPE will extrapolate further, but the model has never seen those positions. |

---

## Testing

The catalogue tests in `tests/test_models.py` are parametrised over every
registered spec, so completeness, thinking-ceiling sanity and unique engine
candidates come for free. Add a class for what is specific to your model:

```python
class TestKestrelSpec:
    def test_it_runs_on_our_own_engine(self) -> None:
        assert KESTREL.engine == "minerva"
```

Note that tests about *platform* behaviour (system prompts, tool attachment,
thinking resolution) use `CAPABLE_SPEC` from `conftest.py`, not a real model —
so they do not break when a real model's capabilities change.

Then verify against reality:

```bash
minerva doctor                       # is it installed / trained?
minerva ask -m kestrel "The"         # real output from real weights
pytest -m integration
```

---

## Checklist

- [ ] `src/minerva/models/<name>.py` with a complete `ModelSpec`
- [ ] Capabilities describe the **weights**, not the aspiration
- [ ] Spec imported and listed in `_SPECS` in `registry.py`
- [ ] Model-specific tests in `tests/test_models.py`
- [ ] `minerva doctor` reports it as available
- [ ] `minerva evaluate` numbers recorded (for a model we trained)
- [ ] README's model table updated
