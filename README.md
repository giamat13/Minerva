# Minerva

**An AI model company. Our first model is Swift, and we trained it ourselves.**

Swift is not a fine-tune and not a wrapper around someone else's weights. Its
architecture, its tokenizer, its training corpus and every one of its 9,875,520
parameters come from this repository. `minerva train` reproduces it from
scratch.

Minerva is also the platform around the model: a real inference engine, tool
calling, a seven-level thinking scale, and a structure built to hold the models
that come after Swift.

Nothing here is simulated. There are no mock engines, and where something could
not be verified, this README says so.

---

## Why these names

### Minerva — the company

Minerva is the Roman goddess of **wisdom, craft and strategy** — and, notably,
not of raw force. She is the goddess of the *skilled* act: the weaver, the
artisan, the strategist who wins by thinking rather than by overwhelming.

That is the argument this project makes. We cannot outspend anyone on compute.
What we can do is be deliberate: real data chosen example by example, an
architecture understood component by component, hyper-parameters picked by
experiment rather than assumption, and honest reporting of what came out.

Her symbol is the owl — the bird that sees in the dark.

### Swift — the first model

The swift is a small bird, roughly forty grams, that spends almost its entire
life airborne. It eats, sleeps and mates on the wing and can stay aloft for
months without landing. It is not the largest bird in the sky. It is the one
that never has to stop.

| The bird | The model |
|---|---|
| ~40 grams | 9.9M parameters — runs on a laptop CPU |
| Fastest bird in level flight | Generates without a GPU |
| Lives in the air, rarely lands | Runs in-process; no daemon, no datacentre |
| Small but far from simple — it navigates continents | Small but genuinely trained: a full modern transformer stack |

Later models will be named for larger, slower, further-seeing birds. The family
grows upward from here.

---

## How Swift was made

The whole pipeline is in this repository and reproducible in three commands.

```bash
minerva prepare-data     # real corpus -> BPE tokenizer -> token bins
minerva train            # stage 1: pretrain from scratch      (155 min, CPU)
minerva finetune         # stage 2: teach it to hold a chat    (2 min, CPU)
minerva ask "What is 17 times 43?"
```

**1. The corpus** — ~27 MB of real, human-written English prose across six
sources and four registers: Project Gutenberg literature, Reuters newswire,
European Parliament debate, US political oratory, and informal web text. Every
source is downloaded from its original distributor with its licence and origin
recorded in a generated `manifest.json`. Nothing is templated, generated or
augmented.

Two available corpora were **rejected on quality grounds**: the Brown corpus
(distributed POS-tagged) and Pang & Lee's movie reviews (distributed lowercased
and pre-tokenised). Together they would have added 17 MB. Volume was not a good
enough reason to teach the model damaged typography.

**2. The tokenizer** — a byte-level BPE, implemented here and trained on that
corpus in 95 seconds. 8,192 tokens, all 256 byte values in the base vocabulary
(so there is no unknown token), digits always split individually. Measured
compression: 3.66 characters per token → 7,377,901 training tokens.

**3. The architecture** — `SwiftLM`, a decoder-only transformer: pre-norm
RMSNorm, RoPE, SwiGLU feed-forward, grouped-query-capable attention, tied
embeddings, scaled residual init.

```python
SwiftConfig(vocab_size=8192, n_layer=6, n_head=8, d_model=320, max_seq_len=512)
# 9,875,520 parameters (7,254,080 non-embedding)
```

**4. The run** — sized from measured hardware, not guessed. The machine does
430 GFLOPS of fp32 matmul and ~3–4k training tokens/second on 4 CPU cores; at
`6ND` FLOPs that put the compute-optimal point near 9.9M parameters × 37M
tokens ≈ 5 epochs. The learning rate came from a real 70-step probe over
{1,2,3,5}×10⁻³ before committing to the run.

**5. The result** — 155 minutes on 4 CPU cores, 36.86M tokens, 5 epochs:

| | measured |
|---|---|
| held-out loss | **3.4579** |
| held-out perplexity | **31.75** (chance = 8,192) |
| bits per byte | **1.3571** |

The curve fell monotonically from perplexity 126.9 at step 250 to 31.8 at step
4,500, and **was still falling when the run ended** — Swift is data-limited,
not compute-limited. More corpus is the highest-value next change.

Full detail, including the sizing arithmetic, the curve, real samples and a bug
the tests caught, is in [`docs/TRAINING.md`](docs/TRAINING.md).

### What it actually learned

Real, unedited output at temperature 0.8:

> **"The company said it expects"** → *to provide a 5-1/2 pct stock dividend of
> 10 cts a share. It said it will continue to operate in the New York and Exco
> Health Commission to pay off the total value of the common shares on the
> outstanding common stock.*

It reproduces Reuters house style down to the abbreviations — `pct`, `cts`,
`dlrs` — and picked up `(Applause.)` from the State of the Union transcripts.
Nobody coded those; they are corpus conventions it inferred.

It also gets things wrong, and this README will say so: **register conditioning
is unreliable** — an Austen opening can produce political oratory — and
semantics drift within a couple of sentences. Local coherence is good, global
coherence is not.

---

## Stage two: Swift-Instruct

The base model continues text; it does not answer questions. So there is a
second stage, and a second model in the catalogue.

`minerva finetune` trains `swift` on **185 hand-written conversations** onto a
chat format with nine single-token markers. No script, no templates, no
permuted slot values. Tool results in the training data are never invented: an
example declares the *call*, and the build runs the real tool to get the result.

**It genuinely calls tools.** The model decides, the real tool executes, the
result is fed back:

```
Q: What is 17 times 43?
   -> model asked for: calculate({"expression": "17 * 43"})
   <- real tool returned: '731'
A: '17 times 43 is 731.'

Q: Who won the World Cup in 2022?
A: 'I do not know. I am a small model and I was not trained on recent events.'
```

Measured on 34 **held-out** hand-written prompts, greedy decoding
(`minerva evaluate-instruct`):

| | measured |
|---|---|
| format valid | **100.0%** |
| routing accuracy (call a tool or not) | **94.1%** |
| tool name accuracy | **88.9%** |
| **argument accuracy** | **27.8%** |
| final answer correct | **31.2%** |
| honest refusal | **66.7%** |

It reliably decides **whether** and **which**. It is bad at **arguments** — 
asked for `23 times 19` it will call `23 * 13`. Routing is a one-token decision
learnable from 185 examples; copying an arbitrary second operand is a general
skill that needs far more data. That limitation is in the model's spec, in the
docs, and in `examples/04_tools_and_thinking.py`, which prints a failure next
to every success.

**Thinking was trained, measured, and switched off.** The scale is fully wired
for this model — the engine opens `<|think|>`, the model produces a trace, the
parser reads it back — and 29 training conversations contain reasoning. It
makes the model *worse*: routing falls 94.1% → 61.8%, tool accuracy 88.9% →
33.3%, arguments 27.8% → 0%. So `swift-instruct` ships `max_thinking=do`.
Forcing a 9.9M model to reason first buys drift, not deliberation.

---

## What Swift is, and what it is not

**`swift` is a base language model.** It was pretrained to predict the next
token and has had no instruction tuning, no chat tuning and no RLHF. So:

* It **continues text**. Give it `"The company said it expects"` and it writes
  plausible Reuters copy. Give it `"What is the capital of France?"` and it
  will most likely write *more questions* — because that is what follows a
  question in its training data. It also picks the wrong register a fair
  fraction of the time; see the measured samples above.
* It **cannot call tools.** Tool calling is a trained behaviour.
* It **has no reasoning phase**, so every thinking level resolves to `do`.

These are facts about its training stage, not gaps in the platform, and the
code states them rather than papering over them: `NativeEngine.capabilities`
reports `tools=False`, and asking for tools raises instead of silently
returning nothing.

At 9.9M parameters on 27 MB, Swift is the scale of a small research baseline.
It learns grammar, vocabulary, register and local coherence. **It is not a
chatbot and this project will not describe it as one.**

---

## The thinking scale

Minerva controls deliberation with the seven notes of the solfège scale.

| # | Note | Hebrew | Budget | Effort | Extended |
|---|------|--------|--------|--------|----------|
| 0 | `do`  | דו  | –      | off    | no |
| 1 | `re`  | רה  | 256    | low    | no |
| 2 | `mi`  | מי  | 1,024  | low    | no |
| 3 | `fa`  | פה  | 4,096  | medium | no |
| 4 | `sol` | סול | 8,192  | medium | no |
| 5 | `la`  | לה  | 16,384 | high   | **yes** |
| 6 | `si`  | סי  | 32,768 | high   | **yes** |

`la` and `si` enable **Extended Thinking**: the reasoning trace is preserved
across turns rather than dropped as scratch work.

Levels are accepted by Latin name, Hebrew name, index or alias:

```bash
minerva ask -t sol "..."     minerva ask -t סול "..."
minerva ask -t 4   "..."     minerva ask -t high "..."
```

The scale is **engine-agnostic**: a `ThinkingProfile` carries the same intent in
three encodings (on/off, coarse effort, token budget) and each engine picks the
one it speaks. Each model declares its own ceiling, and requests above it are
**clamped, never rejected**.

Both Minerva models ceiling at `do`, and that is a **measurement**. Swift is a
base model with no reasoning phase at all. Swift-Instruct *was* trained with
reasoning traces and the whole path works — but it measurably degrades the
answers, so the ceiling stays at `do`. The scale is real, tested machinery
waiting for a model that benefits from it.

---

## Install

Python 3.11+.

```bash
git clone https://github.com/giamat13/minerva
cd minerva
pip install -e ".[training]"      # torch + numpy, needed to train or run Swift

minerva prepare-data              # downloads ~28 MB, trains the tokenizer
minerva train                     # pretrains from scratch
minerva doctor                    # verify: engine ready, checkpoint present
```

The core platform has exactly **one** runtime dependency (`httpx`). PyTorch is
an optional extra, so using Minerva against an external engine stays a
one-dependency install.

---

## Use it

### Command line

```bash
minerva ask "It is a truth universally"      # continue a prompt
minerva ask --no-stream -m swift "The"       # wait for the full continuation
minerva models -v                            # the catalogue
minerva thinking                             # the scale
minerva doctor                               # is anything actually working?
minerva train --resume checkpoints/swift/last.pt
```

### Python

```python
from minerva import load_model

model = load_model("swift")          # loads our checkpoint in-process
print(model.ask("It is a truth universally"))
```

### Training from Python

```python
from minerva.training import SWIFT_CONFIG, SwiftLM, TokenDataset, TrainConfig, Trainer

model = SwiftLM(SWIFT_CONFIG)
trainer = Trainer(
    model,
    TokenDataset("data/train.bin", 512),
    TokenDataset("data/val.bin", 512),
    TrainConfig(max_steps=4500, learning_rate=2e-3),
    out_dir="checkpoints/swift",
)
trainer.train()
```

Runnable examples are in [`examples/`](examples/).

---

## The model family

| Model | Tier | Params | Trained on | Engine | Tools | Thinking |
|-------|------|--------|------------|--------|-------|----------|
| **Swift** | small | 9.9M | 27 MB, from scratch | `minerva` (in-process) | ✗ base model | ✗ base model |
| **Swift-Instruct** | small | 9.9M | + 185 hand-written conversations | `minerva` (in-process) | ✅ 94% routing | ✗ measured to hurt |

Swift v0.1.0: held-out perplexity **31.75**, bits/byte **1.3571**, trained in
155 minutes on 4 CPU cores. Swift-Instruct v0.1.0: **94.1%** routing accuracy,
**27.8%** argument accuracy on held-out prompts.

More models are coming; the registry is built to take them.
See [`docs/ADDING_A_MODEL.md`](docs/ADDING_A_MODEL.md).

---

## Layout

```
src/minerva/
├── training/           HOW A MINERVA MODEL IS MADE
│   ├── data.py           Corpus: real sources, licences, cleaning, split
│   ├── tokenizer.py      Byte-level BPE, implemented and trained here
│   ├── dataset.py        Memory-mapped token batches
│   ├── model.py          SwiftLM: RMSNorm, RoPE, SwiGLU, GQA, tied embeddings
│   ├── trainer.py        Stage 1: pretraining from scratch
│   ├── chat.py           The chat format - one definition, used by all three
│   ├── instruct_data.py  185 hand-written conversations, real tool results
│   ├── finetune.py       Stage 2: instruction tuning, assistant-only loss
│   └── instruct_eval.py  Held-out evaluation of the tuned model
├── engines/            WHAT RUNS A MODEL
│   ├── native.py         Runs OUR checkpoints in-process
│   ├── ollama.py         Runs third-party weights via a local daemon
│   └── registry.py       ← add new engines here
├── models/             WHAT A MODEL IS
│   ├── swift.py          The base model's spec — the template for the next
│   ├── swift_instruct.py The chat model's spec
│   └── registry.py       ← add new models here
├── tools/              WHAT A MODEL CAN CALL
├── runtime/            The agent loop and multi-turn sessions
├── thinking.py         The solfège scale — engine-agnostic, seven notes
└── cli.py              The `minerva` command
```

The separation between `models/` and `engines/` is the load-bearing decision: a
model is a specification you can print and diff, an engine is execution. Swift
being trained here rather than downloaded changed one line of its spec —
`engine="minerva"` — and nothing else in the platform.

---

## Development

```bash
pytest -m "not integration"   # no engine or GPU required
pytest -m integration         # real inference against a live engine
ruff check . && mypy          # both clean
```

**On testing philosophy:** there are no mock engines in this repository, and
there never will be. Unit tests cover real logic; integration tests talk to a
live engine and **skip with a stated reason** when one is not reachable. A green
test run that proved nothing is worse than a skipped one.

That policy pays: the tokenizer's byte-coverage test found a real bug in which
every underscore was silently deleted from the corpus. It is documented,
including why the training run was not restarted, in
[`docs/TRAINING.md`](docs/TRAINING.md).

Contributor rules — including the standard for training data — are in
[`CLAUDE.md`](CLAUDE.md).

---

## תקציר בעברית

**Minerva** היא חברת מודלי AI. השם על שם האלה הרומית של החוכמה והמלאכה — לא של
הכוח הגס. זו בדיוק הטענה: מערכת טובה נמדדת בדיוק ובמלאכה, לא בגודל.

**Swift-Instruct** הוא שלב שני: אותו מודל, שאומן על 185 שיחות שנכתבו ביד, ולמד
פורמט שיחה והפעלת כלים. הוא **באמת מפעיל כלים** — 94.1% דיוק בהחלטה אם להפעיל
כלי ו-88.9% בבחירת הכלי — אבל רק 27.8% בארגומנטים. חשיבה אומנה, נמדדה, **וכובתה**:
היא מורידה את דיוק הניתוב מ-94.1% ל-61.8%. כל המספרים בתיעוד.

**Swift** הוא המודל הראשון, על שם הסיס — ציפור של כארבעים גרם שמבלה כמעט את כל
חייה באוויר ואינה נאלצת לנחות. הוא **אומן מאפס בתוך הפרויקט הזה**: הארכיטקטורה,
הטוקנייזר, קורפוס האימון וכל 9,875,520 הפרמטרים נוצרו כאן. לא fine-tune ולא
עטיפה למשקולות של מישהו אחר.

**מה Swift כן ומה לא.** הוא מודל בסיס (base model): אומן לחזות את הטוקן הבא
בלבד, בלי instruction tuning ובלי RLHF. לכן הוא **ממשיך טקסט** ולא עונה על
שאלות, **לא יודע להפעיל כלים**, ו**אין לו שלב חשיבה**. אלה עובדות על שלב האימון
שלו, לא חוסרים בפלטפורמה, והקוד מצהיר עליהן במפורש במקום להעמיד פנים.

**סולם החשיבה** בנוי משבעת הצלילים — דו, רה, מי, פה, סול, לה, סי. `לה` ו־`סי`
מפעילים Extended Thinking. הסולם בלתי תלוי במנוע וממתין למודל שיאומן להשתמש בו.

הכל קוד אמיתי שרץ — בלי Mocks ובלי קיצורי דרך.

---

## Licence

MIT. See [`LICENSE`](LICENSE).
