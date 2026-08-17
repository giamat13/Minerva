# Minerva

**An AI model platform. The first model in the family is Swift.**

Minerva runs language models locally against a real inference engine, with tool
calling, a seven-level thinking scale, and a structure built from day one to
hold more than one model.

There is no simulation anywhere in this repository. Every code path talks to a
real engine; when no engine is reachable, Minerva says so and stops.

---

## Why these names

### Minerva — the platform

Minerva is the Roman goddess of **wisdom, craft and strategy** — and, notably,
not of raw force. She is the goddess of the *skilled* act: the weaver, the
artisan, the strategist who wins by thinking rather than by overwhelming.

That is the argument this project is making. An AI platform does not have to be
the largest to be the most useful; it has to be the most deliberate. Minerva's
symbol is the owl, the bird that sees in the dark — an appropriate emblem for
software whose job is to make sense of things with incomplete light.

The name also sets the standard the project holds itself to, written down in
[`CLAUDE.md`](CLAUDE.md): craft over volume. Real code, real training data,
no shortcuts.

### Swift — the first model

The swift is a small bird — roughly forty grams — that spends almost its entire
life airborne. It eats, sleeps and mates on the wing, and can stay aloft for
months without landing. It is not the largest bird in the sky. It is the one
that never has to stop.

That is the design brief for this model exactly:

| The bird | The model |
|---|---|
| ~40 grams | ~1.7B parameters — small enough to stay resident on a laptop |
| Fastest bird in level flight | Fast enough that you never wait for it |
| Lives in the air, rarely lands | Runs locally, always on, no round trip to a datacentre |
| Small but far from simple — it navigates continents | Small but capable — it reasons and calls tools |

Swift is the smallest model in the family. Later models will be named for
larger, slower, more far-seeing birds — the family grows upward from here.

---

## What the system is for

Three things, in order of importance:

1. **Run real models locally, properly.** Not a demo, not a wrapper with a
   `sleep()` in it. A production-shaped client for a real engine, with
   streaming, tool calling, timeouts, health checks and honest error messages.

2. **Make deliberation a dial, not an accident.** Most systems treat "how hard
   should the model think?" as an implicit property of the prompt. Minerva
   makes it an explicit, named parameter — seven levels on the solfège scale —
   so you can spend reasoning where it pays and skip it where it does not.

3. **Be ready for the models that come next.** Swift is the first, not the
   only. Adding a model is one new file and one line in a registry. The
   architecture is the point; Swift is the proof it works.

---

## The thinking scale

Minerva controls deliberation with the seven notes of the solfège scale, in
ascending order. A higher note is a higher pitch of thinking.

| # | Note | Hebrew | Budget | Effort | Extended | Meaning |
|---|------|--------|--------|--------|----------|---------|
| 0 | `do`  | דו  | –      | off    | no  | Silent — answer immediately, no reasoning phase |
| 1 | `re`  | רה  | 256    | low    | no  | Whisper — a brief sanity check |
| 2 | `mi`  | מי  | 1,024  | low    | no  | Quiet — short reasoning for simple multi-step questions |
| 3 | `fa`  | פה  | 4,096  | medium | no  | Moderate — the balanced default |
| 4 | `sol` | סול | 8,192  | medium | no  | Strong — sustained reasoning for hard problems |
| 5 | `la`  | לה  | 16,384 | high   | **yes** | Loud — **Extended Thinking** |
| 6 | `si`  | סי  | 32,768 | high   | **yes** | Full voice — Extended Thinking at maximum depth |

**Extended Thinking** (`la` and `si`) does more than lengthen the reasoning: it
**preserves** the reasoning trace across turns, so the model builds on its
earlier thinking instead of re-deriving it. Below `la` the trace is treated as
scratch work and dropped once the turn ends.

Every level is accepted by Latin name, Hebrew name, index, or a plain-language
alias:

```bash
minerva ask -t sol  "..."      # Latin
minerva ask -t סול  "..."      # Hebrew
minerva ask -t 4    "..."      # index
minerva ask -t high "..."      # alias
```

Each model declares its own ceiling. Swift tops out at `sol` — pushing a 1.7B
model to `si` buys drift, not insight — and requests above the ceiling are
**clamped, never rejected**, so the same code works unchanged against a larger
Minerva model later.

---

## Install

Requires Python 3.11+ and a running [Ollama](https://ollama.com).

```bash
git clone https://github.com/giamat13/minerva
cd minerva
pip install -e ".[dev]"

ollama serve &          # start the engine
minerva pull swift      # fetch Swift's weights (~1.4 GB)
minerva doctor          # verify: engine reachable, weights installed
```

---

## Use it

### Command line

```bash
minerva ask "What is 17 * 43?"                  # one question
minerva ask -t sol --show-thinking "..."        # crank the thinking up, show it
minerva chat                                    # interactive, with memory
minerva models -v                               # the catalogue
minerva tools                                   # what the model can call
minerva thinking                                # the scale
minerva doctor                                  # is anything actually working?
```

Inside `minerva chat`, `/thinking sol` changes the level mid-conversation and
`/reset` clears the transcript.

### Python

```python
from minerva import Agent, load_model

model = load_model("swift")

# Simplest form.
print(model.ask("Why do swifts sleep in flight?", thinking="mi"))

# With tools: the agent loop runs them for real and feeds results back.
agent = Agent(model, thinking="sol")
run = agent.run("What is 4871 * 293, and what day of the week is it today?")
print(run.answer)
print(f"{len(run.tool_calls)} tool call(s), {run.completion_tokens} tokens")
```

### A custom tool

The schema the model sees is derived from the type hints and the docstring, so
there is no second copy of the signature to drift:

```python
from minerva import Agent, ToolRegistry, load_model, tool

@tool
def check_stock(product: str) -> dict:
    """Check how many units of a product are in the warehouse.

    Use this whenever you are asked about availability. Never guess.

    Args:
        product: The product name, e.g. "swift".
    """
    return {"product": product, "units": inventory[product]}

agent = Agent(load_model("swift"), tools=ToolRegistry([check_stock]))
print(agent.run("How many swifts do we have?").answer)
```

### Conversations

```python
from minerva import Session, load_model

session = Session(load_model("swift"), thinking="fa")
session.send("My name is Dana and I keep bees.")
print(session.send("What is my name?"))      # remembers
session.save("conversation.json")
```

Runnable versions of all of this are in [`examples/`](examples/).

---

## The model family

| Model | Tier | Size | Engine model | Thinking | Tools |
|-------|------|------|--------------|----------|-------|
| **Swift** | small | 1.7B | `qwen3:1.7b` | default `fa`, max `sol` | ✅ |

More models are coming; the registry is built to take them.
See [`docs/ADDING_A_MODEL.md`](docs/ADDING_A_MODEL.md).

---

## Layout

```
src/minerva/
├── thinking.py         The solfège scale — engine-agnostic, seven notes
├── messages.py         Conversation primitives shared by every layer
├── config.py           Settings: defaults < TOML file < MINERVA_* env vars
├── errors.py           One exception hierarchy, actionable messages
├── engines/            WHAT RUNS A MODEL
│   ├── base.py           The Engine contract
│   ├── ollama.py         Real Ollama HTTP client — streaming, tools, thinking
│   └── registry.py       ← add new engines here
├── models/             WHAT A MODEL IS
│   ├── base.py           ModelSpec (inert data) + MinervaModel (spec + engine)
│   ├── swift.py          Swift — written as the template for future models
│   └── registry.py       ← add new models here
├── tools/              WHAT A MODEL CAN CALL
│   ├── base.py           @tool decorator; JSON Schema from type hints
│   ├── registry.py       Tool sets, real execution, error feedback
│   └── builtin/          calculate, current_time, days_between
├── runtime/            THE LOOP
│   ├── agent.py          Generation + real tool execution, to a final answer
│   └── session.py        Multi-turn memory + the Extended Thinking policy
└── cli.py              The `minerva` command
```

The separation between `models/` and `engines/` is the load-bearing decision: a
model is a specification you can print and diff, an engine is execution. That
is what makes a new model a single file.

---

## Configuration

Defaults < `minerva.toml` < `MINERVA_*` environment variables < explicit
arguments in code.

```toml
# minerva.toml
ollama_host    = "http://127.0.0.1:11434"
default_model  = "swift"
thinking_level = "fa"        # or "פה", or 3
show_thinking  = false
enable_tools   = true
```

```bash
export MINERVA_OLLAMA_HOST=http://gpu-box:11434
export MINERVA_THINKING_LEVEL=סול
```

---

## Development

```bash
pytest -m "not integration"   # 284 tests, no engine required
pytest -m integration         # real inference; needs `ollama serve` + weights
ruff check . && mypy          # both clean
```

**On testing philosophy:** there are no mock engines in this repository, and
there never will be. Unit tests cover real logic that genuinely needs no
engine; integration tests talk to a live daemon and **skip with a stated
reason** when one is not reachable. A green test run that proved nothing is
worse than a skipped one.

Contributor rules — including the standard for training data — are in
[`CLAUDE.md`](CLAUDE.md).

---

## תקציר בעברית

**Minerva** היא פלטפורמת מודלי AI. השם הוא על שם האלה הרומית של החוכמה והמלאכה
— לא של הכוח הגס. זו בדיוק הטענה של הפרויקט: מערכת טובה נמדדת בדיוק ובמלאכה,
לא בגודל.

**Swift** הוא המודל הראשון והקטן במשפחה, על שם הסיס — ציפור של כארבעים גרם
שמבלה כמעט את כל חייה באוויר ואינה נאלצת לנחות. קטן, מהיר, תמיד זמין, ובכל
זאת יודע לחשוב ולהפעיל כלים.

**סולם החשיבה** בנוי משבעת צלילי הסולם — דו, רה, מי, פה, סול, לה, סי — מהשקט
לחזק. `דו` הוא ללא חשיבה כלל, ו־`לה` ו־`סי` מפעילים Extended Thinking שבו עקבות
החשיבה נשמרים בין תורות. אפשר לבחור צליל בשם לטיני, בשם עברי או במספר.

**המערכת בנויה להתרחבות**: הוספת מודל חדש = קובץ אחד ושורה אחת ברג'יסטרי;
הוספת כלי = פונקציה אחת עם type hints ו-docstring. הכל קוד אמיתי שרץ מול מנוע
אמיתי — בלי Mocks ובלי קיצורי דרך.

---

## Licence

MIT. See [`LICENSE`](LICENSE).
