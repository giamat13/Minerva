# Adding an engine

An **engine** is the thing that actually runs inference. A **model** is a
specification. Keeping them apart is what lets one Minerva model run through
Ollama on a laptop today and through a served backend tomorrow without touching
the model definition, the tools or the agent loop.

There are two reference implementations, and they sit at opposite ends of the
contract - read whichever is closer to what you are building:

* `src/minerva/engines/native.py` - **in-process**. Loads a Minerva checkpoint
  and runs it with PyTorch. No network, no daemon, lazy torch import.
* `src/minerva/engines/ollama.py` - **over HTTP**. Talks to a local daemon,
  with streaming NDJSON, tool schemas on the wire and health probes.

---

## 1. Subclass `Engine`

Create `src/minerva/engines/<name>.py`:

```python
from ..engines.base import (
    Engine, EngineCapabilities, EngineHealth,
    GenerationRequest, GenerationResult, StreamChunk,
)

class MyEngine(Engine):
    name = "myengine"
    capabilities = EngineCapabilities(
        streaming=True,
        tools=True,
        thinking=True,
        thinking_trace=True,     # reasoning comes back as its own field
        thinking_budget=True,    # accepts a numeric reasoning-token budget
    )

    def health(self) -> EngineHealth: ...
    def list_available_models(self) -> list[str]: ...
    def chat(self, request: GenerationRequest) -> GenerationResult: ...
    def stream(self, request: GenerationRequest) -> Iterator[StreamChunk]: ...
```

### The four methods

**`health()`** — a real probe of the backend. **It must never raise.** Report
failure by returning `EngineHealth(available=False, detail=...)`; the CLI's
`doctor` command and the test fixtures depend on that contract.

**`list_available_models()`** — the engine-side identifiers currently
installed or served. May raise `EngineUnavailableError`.

**`chat(request)`** — one non-streaming generation. Translate the request,
send it, decode the reply into a `GenerationResult` whose `message` is an
assistant `Message` carrying content, any reasoning trace, and any tool calls.

**`stream(request)`** — the same, yielding `StreamChunk`s of kind
`"thinking"`, `"content"` and `"tool_call"`. **The final chunk must have
`kind="done"` and carry the assembled `GenerationResult`** — callers that only
want the end state rely on it, and the agent loop raises if it is missing.

---

## 2. Translate thinking

Do **not** add a new thinking level. The solfege scale is engine-agnostic; a
`ThinkingProfile` carries three encodings of the same intent and your engine
picks whichever it understands:

| Field | For engines that expose… | Ollama uses |
|---|---|---|
| `enabled` | an on/off switch | ✅ `think: false` for DO |
| `effort` | `"low"`/`"medium"`/`"high"` | ✅ `think: "medium"` |
| `budget_tokens` | a numeric reasoning budget | ❌ ignored |

If the models your engine serves cannot reason at all, say so —
`NativeEngine` sets `thinking=False` and every level then resolves to `DO`.
That is the honest reporting `CLAUDE.md` requires: never accept a thinking
request you cannot honour and quietly return an ordinary answer.

```python
def _encode_thinking(profile):
    if profile is None:
        return None
    if not profile.enabled:
        return {"type": "disabled"}
    return {"type": "enabled", "budget_tokens": profile.budget_tokens}
```

Declare which of these you honour in `capabilities` so callers get a clear
`CapabilityError` instead of a confusing wire error.

---

## 3. Translate tools

`request.tools` is already a list of OpenAI-style function specs:

```json
{"type": "function", "function": {"name": ..., "description": ..., "parameters": {...}}}
```

Most backends take that shape verbatim. When you decode tool calls back,
`ToolCall.arguments` must be a **decoded mapping** — if your backend returns a
JSON string, decode it in the engine so nothing above ever has to care.

---

## 4. Register it

In `src/minerva/engines/registry.py`:

```python
def _build_myengine(config: MinervaConfig, **overrides: Any) -> Engine:
    from .myengine import MyEngine
    return MyEngine(host=overrides.pop("host", config.myengine_host), **overrides)

_FACTORIES: dict[str, EngineFactory] = {
    "ollama": _build_ollama,
    "myengine": _build_myengine,     # <- add this
}
```

Add any new settings to `MinervaConfig` in `src/minerva/config.py` and to
`_FIELD_TYPES` so they can be set from a TOML file or a `MINERVA_*` variable.
The import stays inside the factory so importing Minerva never opens a socket
or pulls in an optional dependency.

---

## 5. Test it

Two layers, mirroring the Ollama engine:

**Codec tests** (`tests/test_engine_<name>.py`) — pure translation, no HTTP.
Assert on the real wire shapes your backend produces.

**Integration tests** (`tests/integration/`) — against a live backend, marked
`@pytest.mark.integration`, skipping with a clear reason when it is not
reachable.

**Do not write a mock engine.** A test that passes against a simulated backend
proves nothing about the backend, and a green suite that proved nothing is
worse than a skipped one. See `tests/conftest.py`.

For an in-process engine there is a third, better option, which
`tests/test_engine_native.py` uses: **train a real, tiny model in the test
fixture**. Fifteen optimisation steps on a 64-dimension model takes under a
second and gives you genuinely trained weights on disk to load and generate
from - no simulation anywhere.

---

## Checklist

- [ ] `health()` reports failure instead of raising
- [ ] `stream()` always ends with a `"done"` chunk carrying the result
- [ ] `capabilities` matches what the backend genuinely supports
- [ ] Tool-call arguments decoded to a mapping inside the engine
- [ ] Unset sampling fields are **not** sent (never override the model file's defaults)
- [ ] `EngineUnavailableError` message says how to fix it
- [ ] Registered in `_FACTORIES`; new settings added to `MinervaConfig`
- [ ] Codec tests + integration tests, no mocks
