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
minerva train            # stage 1: pretrain from scratch      (~90 min, CPU)
minerva finetune         # stage 2: teach it to hold a chat    (~1.5 min, CPU)
minerva ask "What is 17 times 43?"
```

**1. The corpus** — 42.33 MB of real, human-written prose across seven
sources: Project Gutenberg literature, the Brown corpus (short-declarative
press and fiction), European Parliament debate, US political oratory,
informal web text, and **curated Hebrew literature** from Project Ben-Yehuda
(added in v0.2.0). Every source is downloaded from its original distributor
with its licence and origin recorded in a generated `manifest.json`. Nothing
is templated, generated or augmented.

**v0.3.0 removed Simple English Wikipedia and Reuters newswire** — not for
quality, for capability: Swift is 9.9M parameters, with nowhere to reliably
store "the capital of X" or "what happened on date Y". Training on dense
factual text doesn't give a model that size the capacity to recall facts, it
gives it fluent confabulation — a wrong answer stated exactly as confidently
as a right one. [`web_search`](#tools) is the honest replacement: the model
looks a fact up in a real, live search instead of guessing at one it was
never big enough to remember correctly. Full reasoning in
[`src/minerva/training/data.py`](src/minerva/training/data.py)'s module
docstring.

Removing those two sources cost more than their facts, though, and this
project measured the cost rather than guessing at it: they were also the
corpus's only short, declarative, factual-*register* prose. A retrain without
them produced a base model whose finetune scored far below every earlier
round (routing accuracy fell from the 90s into the 60–70% range). The fix was
the **Brown corpus**, added back for exactly that register — rejected in
v0.1.0/v0.2.0 only for a fixable formatting reason (it is distributed
POS-tagged), never for its content, which is 1961 American press and fiction:
old and general enough in subject to teach sentence structure without being a
database of facts worth memorising. Full story, numbers included, in
[`docs/TRAINING.md`](docs/TRAINING.md#5c-the-run-that-produced-swift-v030--removing-facts-restoring-register).

The Hebrew source is Project Ben-Yehuda's public-domain library — the Hebrew
analogue of Project Gutenberg — curated to seven canonical authors (Bialik,
Rachel Bluwstein, Brenner, Ahad Ha'am, Mendele Mocher Sforim, Tchernichovsky,
Frishman) and filtered to their *original*, non-translated work. Both it and
Simple English Wikipedia (while it was still in the corpus) were found via
[Hugging Face Datasets](https://huggingface.co/datasets), reviewed by hand,
and pinned to a specific revision, per `CLAUDE.md`'s rules for using it as a
source. Other real candidates (a Hebrew web scrape, a GPL-3.0 religious-text
library, full English Wikipedia, and Pang & Lee's lowercased/pre-tokenised
movie reviews) were reviewed and rejected — the reasons are in
[`docs/TRAINING.md`](docs/TRAINING.md).

**2. The tokenizer** — a byte-level BPE, implemented here and trained on that
corpus. 8,192 tokens, all 256 byte values in the base vocabulary (so there is
no unknown token — true for Hebrew as much as English), digits always split
individually. Measured compression: 2.886 characters/token overall (3.338 for
the English portion, 2.276 for Hebrew — already a 4.5× improvement over
unmerged byte-level Hebrew). A vocab increase to 12,000 was measured and
**rejected**: it bought ~5% better compression for +12.3% more parameters,
without closing the Hebrew/English compression gap. See
[`docs/TRAINING.md`](docs/TRAINING.md) for the full comparison.

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
tokens. The learning rate came from a real 70-step probe over {1,2,3,5}×10⁻³
before committing to the run. **v0.2.0 kept the same 36.86M-token budget** —
per `CLAUDE.md`, growing the training time needs a strong measured reason, and
there wasn't one — so the bigger corpus means 2.1 epochs instead of 5.0,
strictly closer to compute-optimal at the same parameter count.

**5. The result** — 98.9 minutes on 4 CPU cores (faster hardware this run),
36.86M tokens, 2.1 epochs:

| | v0.1.0 (English only) | v0.2.0 (+ Hebrew) |
|---|---|---|
| held-out loss | 3.4579 | **3.8542** |
| held-out perplexity | 31.75 | **47.19** (chance = 8,192) |
| bits per byte | 1.3571 | **1.4516** |

The blended number is worse, and that is reported here rather than hidden.
Scored separately by language, the real story is more specific: English's
bits/byte rose a real but modest 1.3571 → 1.4678 (fewer epochs, an
8,192-token vocabulary now shared with a second script), while Hebrew comes in
at 1.4270 bits/byte — Hebrew's *perplexity* looks much worse (77.32 vs 35.57)
but that is mostly a token-accounting artifact: Hebrew tokens carry more bytes
each, so on the fair, tokenizer-agnostic bits-per-byte measure the model is
doing about as well on Hebrew, its first exposure to the language from 17.3 MB,
as on English. Full breakdown, including why, in
[`docs/TRAINING.md`](docs/TRAINING.md#5b-the-run-that-produced-swift-v020--adding-hebrew).

### What it actually learned

Real, unedited output at temperature 0.8:

> **"The company said it expects"** → *to report net earnings growth from its
> yearly earnings. It said it expects the year results to show sales and
> earnings growth in 1987/87. It added that it expects to push on sales of
> 22 mln dlrs.*

> **"שלום רב"** → *לנו כי אין אנו רשאים לדברים: יודעים אתם, כי גם אלה הם הם
> חורבים על אחרים, הם באים ואומרים, כי הם אינם נוצרים, כי הם הם עצמם אינם
> רוצים להניח להם את הלבבות...*

It reproduces Reuters house style down to the abbreviations — `pct`, `cts`,
`dlrs` — and picked up `(Applause.)` from the State of the Union transcripts.
In Hebrew it produces genuinely grammatical morphology (correct verb
conjugation, correct ה-/ו-/ל-/ש- prefix attachment) and — without being told
the rule — learned the corpus's own convention of niqqud on poetry but not
prose. Nobody coded any of that; it is corpus conventions the model inferred,
in two languages now instead of one.

It also gets things wrong, and this README will say so: **register conditioning
is unreliable** — an Austen opening can produce political oratory — and
semantics drift within a couple of sentences. Local coherence is good, global
coherence is not.

---

## Stage two: Swift-Instruct

The base model continues text; it does not answer questions. So there is a
second stage, and a second model in the catalogue.

`minerva finetune` trains `swift` on **219 hand-written conversations** (185
English + 34 Hebrew) onto a chat format with nine single-token markers. No
script, no templates, no permuted slot values. Tool results in the training
data are never invented: an example declares the *call*, and the build runs
the real tool to get the result.

**It genuinely calls tools, in either language.** The model decides, the real
tool executes, the result is fed back:

```
Q: What is 17 times 43?
   -> model asked for: calculate({"expression": "17 * 43"})
   <- real tool returned: '731'
A: '17 times 43 is 731.'

Q: כמה זה 1999 ועוד 2024?
   -> model asked for: calculate({"expression": "1999 + 2024"})
   <- real tool returned: '4023'
A: 'יוצא 4023.'

Q: Who won the World Cup in 2022?
A: 'I do not know. I am a small model and I was not trained on recent events.'
```

`swift-instruct` was retrained twice for v0.2.0. Round one kept the same 185
English examples on the new bilingual base and measured a real regression
(English competence itself weakened during pretraining, §5b). Round two added
34 hand-written Hebrew conversations — same rules, written one at a time, no
templates — plus 10 held-out Hebrew evaluation prompts to actually measure it.
Full history in
[`docs/TRAINING.md`](docs/TRAINING.md#v020-round-two-34-hand-written-hebrew-examples-added).
Scored separately by language, greedy decoding (`minerva evaluate-instruct`):

| | v0.1.0, 185 English | v0.2.0, 185 English only | v0.2.0 +Hebrew — English cases | v0.2.0 +Hebrew — Hebrew cases |
|---|---|---|---|---|
| held-out cases | 34 | 34 | 34 | 10 |
| routing accuracy | 94.1% | 91.2% | **97.1%** | 80.0% |
| tool name accuracy | 88.9% | 94.4% | 94.4% | 83.3% |
| argument accuracy | 27.8% | 16.7% | **22.2%** | 16.7% |
| final answer correct | 31.2% | 18.8% | 18.8% | 20.0% |
| honest refusal | 66.7% | 50.0% | **66.7%** | **0.0%** |

A genuine surprise: adding Hebrew examples did not dilute English further — it
**recovered** most of round one's regression (routing 91.2% → 97.1%, refusal
back to v0.1.0's 66.7%). Most likely, 34 more examples in either language is
simply more signal for the one-token routing decision than 185 alone carried,
and the habit transfers across the language boundary.

Hebrew itself is real but weaker, reported plainly: routing (80.0%) and tool
name accuracy (83.3%) are decent for a first round on 34 examples, but
**honest refusal on Hebrew is 0%** — both held-out Hebrew refusal cases got a
wrong or garbled answer instead of a decline, one of them in English despite
being asked in Hebrew. Five Hebrew refusal examples was too few to learn the
habit from. It is still bad at **arguments** in both languages — asked
`Add 314 and 159` it answers `'4710'` (correct: 473) — copying arbitrary
operands correctly is a general skill 219 hand-written examples cannot fully
carry. That limitation is in the model's spec, in the docs, and in
`examples/04_tools_and_thinking.py`, which prints a failure next to every
success.

**Thinking was trained, measured, and switched off.** The scale is fully wired
for this model — the engine opens `<|think|>`, the model produces a trace, the
parser reads it back — and 29 training conversations contain reasoning. On the
v0.1.0 base it made the model *worse*: routing falls 94.1% → 61.8%, tool
accuracy 88.9% → 33.3%, arguments 27.8% → 0% (this comparison was not re-run on
the v0.2.0 bilingual base — there is no reason to expect it reverses, but it
has not been measured there). So `swift-instruct` ships `max_thinking=do`.
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

At 9.9M parameters on 50.8 MB, Swift is the scale of a small research
baseline. It learns grammar, vocabulary, register and local coherence in two
languages. **It is not a chatbot and this project will not describe it as
one.**

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
pip install -e ".[training]"      # torch + numpy + pyarrow, needed to train or run Swift

minerva prepare-data              # ~450 MB raw downloads, curated down to a 50.8 MB corpus
minerva train                     # pretrains from scratch
minerva doctor                    # verify: engine ready, checkpoint present
```

That 450 MB isn't the corpus size — two sources (Simple English Wikipedia,
Project Ben-Yehuda) are distributed as one full dump each, and `data.py`
downloads the whole dump once, caches it, and curates a much smaller slice out
of it. See [`docs/TRAINING.md`](docs/TRAINING.md) for exactly what is kept and
why.

The core platform has exactly **one** runtime dependency (`httpx`). PyTorch is
an optional extra, so using Minerva against an external engine stays a
one-dependency install.

---

## Use it

### Command line

```bash
minerva ask "It is a truth universally"      # continue a prompt
minerva ask --no-stream -m swift "The"       # wait for the full continuation
minerva chat                                 # interactive conversation
minerva serve                                # local web chat UI on :8420
minerva models -v                            # the catalogue
minerva thinking                             # the scale
minerva doctor                               # is anything actually working?
minerva train --resume checkpoints/swift/last.pt
```

### Web chat UI

`minerva serve` runs a local chat page on `http://127.0.0.1:8420/`, built on
the standard library's `http.server` — a browser tab is friendlier than a
terminal, but it does not earn Minerva a web framework. Multi-turn memory,
a thinking-level selector, a reveal-reasoning toggle, RTL-aware bubbles for
Hebrew, and visible tool calls when the model reaches for one.

Type **`DEVDEBUG`** into the chat box for a capability report: held-out loss
and perplexity, routing accuracy, tool-name accuracy, argument accuracy,
final-answer correctness and honest-refusal rate — read straight from
`data/eval_report.json` and `data/instruct_eval_report.json`, the files
`minerva evaluate` and `minerva evaluate-instruct` write. Real measured
numbers, including the unflattering ones.

### Training on GitHub Actions

Training runs on GitHub's free runners without touching a local machine:

```bash
gh workflow run train.yml                      # start (fresh)
gh workflow run train.yml -f resume=true       # continue where it stopped
gh run watch                                   # follow the live log
gh run download --name checkpoints             # fetch the trained weights
gh run download --name eval-reports            # fetch the measured numbers
```

A hosted runner is 2 cores with a hard 6-hour job limit, so training is
**resumable across runs** rather than one long job: `--max-hours` stops
cleanly before the limit and leaves a real checkpoint, every run uploads its
progress as an artifact (`if: always()`, so a crash keeps its work), and
`resume=true` picks up from the newest one. Re-run with `resume=true` until
the log prints `finished N steps`. See
[`.github/workflows/train.yml`](.github/workflows/train.yml).

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
| **Swift** | small | 9.9M | 50.8 MB (English + Hebrew), from scratch | `minerva` (in-process) | ✗ base model | ✗ base model |
| **Swift-Instruct** | small | 9.9M | + 219 hand-written conversations (185 EN + 34 HE) | `minerva` (in-process) | ✅ 97% EN / 80% HE routing | ✗ measured to hurt |

Swift v0.2.0: held-out perplexity **47.19**, bits/byte **1.4516**, trained in
98.9 minutes on 4 CPU cores — see
[`docs/TRAINING.md`](docs/TRAINING.md#5b-the-run-that-produced-swift-v020--adding-hebrew)
for the real, measured cost of adding Hebrew at this size. Swift-Instruct
v0.2.0: **97.1%** routing accuracy on English, **80.0%** on Hebrew (a first,
smaller round — honest refusal in Hebrew is still 0%, not hidden).

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
│   ├── instruct_data.py  219 hand-written conversations, real tool results
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

**Swift-Instruct** הוא שלב שני: אותו מודל, שאומן על 219 שיחות שנכתבו ביד —
185 באנגלית ו-**34 בעברית**, שנוספו בסבב נפרד, כל אחת נכתבה בנפרד, בלי תבנית
ובלי סקריפט. הוא **באמת מפעיל כלים בשתי השפות** — 97.1% דיוק בניתוב באנגלית,
80.0% בעברית. הפתעה אמיתית: הוספת עברית לא פגעה באנגלית — היא **שיפרה** אותה
בחזרה לרמת v0.1.0 (ניתוב 91.2%→97.1%, סירוב כן 50%→66.7%). אבל העברית עצמה
עדיין חלשה יותר: **0% סירוב כן בעברית** — שני מקרי הבדיקה שדרשו "אני לא יודע"
קיבלו תשובה שגויה במקום סירוב. זה סבב ראשון ומדוד, לא יכולת גמורה — המספרים
המלאים, כולל החולשה, בתיעוד. חשיבה אומנה, נמדדה, **וכובתה** (על הבסיס הישן;
לא נמדד מחדש על הבסיס הדו-לשוני).

**Swift** הוא המודל הראשון, על שם הסיס — ציפור של כארבעים גרם שמבלה כמעט את כל
חייה באוויר ואינה נאלצת לנחות. הוא **אומן מאפס בתוך הפרויקט הזה**: הארכיטקטורה,
הטוקנייזר, קורפוס האימון וכל 9,875,520 הפרמטרים נוצרו כאן. לא fine-tune ולא
עטיפה למשקולות של מישהו אחר.

**v0.2.0 הוסיפה עברית.** הקורפוס גדל מ-27MB ל-50.8MB — נוספו ספרות עברית
אמיתית מפרויקט בן־יהודה (ביאליק, רחל, ברנר, אחד העם, מנדלי מוכר ספרים,
טשרניחובסקי, פרישמן, מסוננת ליצירות מקוריות בלבד) וערכי ויקיפדיה אנגלית
פשוטה. גודל אוצר המילים של הטוקנייזר נמדד — לא נוחש — והוחלט להשאיר על 8,192,
כי הגדלה ל-12,000 קנתה רק כ-5% שיפור בדחיסה תמורת 12.3% יותר פרמטרים. התוצאה:
**המודל כותב עברית תקנית מבחינה דקדוקית** — נטיות פועל נכונות, צירופי אותיות
יחס נכונים — אבל האנגלית נפגעה מדידה קלה (כ-8% ב-bits/byte), כי אותו תקציב
אימון קבוע עכשיו מחולק בין שתי שפות. המספרים המלאים, כולל הסיבה, בתיעוד — לא
הוסתר שום מספר שהתקבל גרוע יותר.

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
