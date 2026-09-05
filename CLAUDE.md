# CLAUDE.md — working rules for this repository

Instructions for Claude (and any other agent or contributor) working on
Minerva. Read this before writing code.

---

## 1. Training: quality only, never generated in bulk

**This is the most important rule in this file.**

Training data for a Minerva model must be **genuinely good, individually
considered examples**. Every example must be worth the tokens it costs.

**What "quality" means here:** an example that was not produced by an
algorithm, a template, or a permutation of slot values — and is genuinely
good on its own merits. That is the actual bar. It does not require a human
to have personally read and signed off on every individual line before it
ships; it requires that nothing in the dataset was mechanically generated to
hit a row count, and that what is there is real.

### Forbidden

- ❌ **A script that generates training examples from an algorithm.** Loops
  over templates, permutations of slot values, `f"What is {a} + {b}?"` across a
  thousand number pairs, combinatorial expansion of a phrase list — none of
  this is training data. It is padding with the shape of training data.
- ❌ **Synthetic data produced only to hit a row count.** "We need 10,000
  examples" is never a reason to make 10,000 examples.
- ❌ **Duplicating or lightly paraphrasing examples to inflate a dataset.**
- ❌ **Copying a public dataset in wholesale** without checking its licence
  and deciding example by example that it belongs here.

### Required

- ✅ Every example teaches something a real user actually needs — drawn from
  real tasks, real questions, real failures, real transcripts.
- ✅ Diversity comes from **genuinely different problems**, not from varying
  the numbers in one problem.
- ✅ Every dataset has a written provenance note: where the examples came from,
  what they are meant to teach, and what was rejected.
- ✅ Wrong, sloppy or ambiguous examples are **deleted**, not "cleaned up
  later". One bad example teaches a bad habit that a hundred good ones must
  then unteach.
- ✅ **A model must always be able to say "I don't know."** This is the most
  important habit any Minerva model is taught, more important than any single
  fact. Confabulation — stating a wrong answer with the same confidence as a
  right one — is a worse failure than refusing, because a refusal is visibly
  honest and a wrong answer is not. Every instruct set needs real, hand-written
  examples of admitting ignorance, and — for anything a tool could actually
  resolve (a current fact, a live number, an event) — real examples of
  reaching for that tool instead of guessing or flatly refusing. See
  `src/minerva/training/instruct_data.py`'s `_UNKNOWN` and `_WEB_SEARCH`
  sections for the pattern: a flat "I don't know" only for what no tool can
  fix (the model's own memory of this conversation, a capability it genuinely
  lacks, a request it should decline regardless of the facts); a tool call for
  everything else.
- ✅ **Train for fluency in the model's supported languages, not for facts.**
  A Minerva model's job is to hold a clear, natural conversation in the
  languages its pretraining corpus actually covers (English and Hebrew, as of
  v0.3.0) — not to memorise the world. Pretraining data should be prose that
  teaches grammar, register and natural phrasing (literature, oratory,
  transcribed speech, everyday writing); encyclopedic and news content is
  deliberately excluded (see `src/minerva/training/data.py`), because a small
  model asked to be a fact database learns to confabulate instead of learning
  to speak well. Any live fact belongs in a tool call (`web_search` and
  friends), never in the weights. Instruct data should widen ordinary
  conversational range — greetings, opinions, clarifying questions,
  self-description, everyday reasoning — in every language the base model was
  actually pretrained on. Never add instruct examples in a language the
  pretraining corpus has no real exposure to: a handful of fine-tuning rows
  cannot teach a language the base model never saw, only make it memorise a
  few fixed phrases and confabulate on anything else — the same failure this
  rule exists to prevent, in a new language instead of a new fact.

### Algorithmic examples: generate *training data*, never runtime replies

First, the distinction that matters most, because getting it wrong turns the
model into a puppet:

- ✅ **What "algorithm" means here:** code that *generates training examples*,
  which are then used to really train the weights. The model learns from
  them and afterwards stands on its own. The generator is build-time only and
  never ships in the inference path.
- ❌ **What it never means:** code that looks at an incoming message at
  runtime and decides what to answer. No keyword matching, no "if the message
  is a greeting, reply with a greeting", no regex dispatch, no canned reply
  table, no post-processing that repairs the model's output. That is not a
  model answering, it is a lookup table wearing a model's name, and §2
  already forbids it as a fake response. **If a metric in this file improves
  because of something in the request path rather than in the weights, the
  improvement is fake and the metric is now lying.**

With that settled: generated examples are allowed for exactly one purpose —
teaching the model how a sentence is **put together** — and are forbidden for
teaching it *what to put in one*.

The failure this rule exists to prevent is a real one from an earlier attempt
at building a model here: it was trained entirely on algorithmic data, so
asked for "a creative name for a shop" it answered "shop" plus a random
adjective drawn from the five it had ever seen. It had learned a template,
not a language, and the template was visible in every answer.

- ✅ **Allowed:** generated examples that vary *structure* - clause order,
  question versus statement, tense, agreement, how a reply attaches to a
  question. The thing being learned is the shape.
- ❌ **Forbidden:** generated examples that vary *content* from a fixed pool -
  slotting nouns, adjectives, names, places or numbers out of a list. That
  teaches a closed vocabulary and produces exactly the "shop + random
  adjective" behaviour above.
- ✅ **Keep the balance explicit.** Hand-written examples must stay the
  substantial share of any instruct set, and any algorithmic block has to be
  a named, separately-counted section, so the ratio can be read off the file
  rather than guessed. If generated examples ever outnumber hand-written
  ones, that is a decision to argue for in writing, not a drift to discover
  later.
- ✅ **Vocabulary comes from the pretraining corpus**, which is real
  human-written prose, and from nowhere else. That is the whole reason the
  corpus is 185 MB of literature rather than a phrase list.

### The standard

> A hundred examples someone thought hard about beat a hundred thousand a
> script produced. If you cannot explain why a specific example is in the
> dataset, it does not belong in the dataset.

The same standard applies to **evaluation** sets, and applies double: an eval
built from generated permutations measures nothing except whether the model
learned the template.

### How this project already applies the rule

`src/minerva/training/data.py` is the worked example, and new corpora must meet
the same bar:

* Every source carries its **origin and licence** in code, and the build writes
  a `manifest.json` with per-source counts, character totals and SHA-256 hashes.
* The corpus is **not vendored** — it is downloaded from the original
  distributor, so provenance stays verifiable.
* Real corpora were **rejected on quality grounds** and the reasons are
  recorded in the module docstring: Pang & Lee's movie reviews (distributed
  lowercased and pre-tokenised) among them. Volume was not a good enough
  reason to teach the model damaged typography. The Brown corpus was
  rejected the same way in v0.1.0/v0.2.0 (distributed POS-tagged) and later
  *un*-rejected in v0.3.0 once a real cleaner made that a solved formatting
  problem rather than a reason to exclude the text — see data.py's "Added
  back in v0.3.0" section for why reversing a documented rejection still
  counts as writing down the judgement call, not skipping it.
* Validation is held out **per source, by character count**, not by document
  count across a concatenation, so the number measures the real mixture.

When you add data, add it that way. When you reject data, write down why.

### Approved sources to draw from

**Hugging Face Datasets — <https://huggingface.co/datasets>** is an approved
place to look for training and evaluation data, alongside the corpora already
in `data.py`.

It does not get an exemption from anything above. It makes the rules *harder*
to follow, because a dataset there is one line of code away from being in your
training set:

- ❌ **Never `load_dataset(...)` a corpus straight into training.** A dataset
  being popular, large, or benchmark-standard is not a reason to trust it.
  Plenty of them are machine-translated, model-generated, deduplicated badly,
  or lowercased and pre-tokenised like the two corpora this project already
  rejected.
- ✅ **Read a real sample first** — a few hundred rows, by eye — and decide
  whether the text is something Minerva should learn to imitate.
- ✅ **Check the licence on the dataset card** and record it in `SOURCES`, with
  the dataset id and revision, exactly as the existing sources do. "It was on
  Hugging Face" is not a licence.
- ✅ **Take the subset you can justify**, not the whole thing because it
  downloads in one call. If you cannot say what a split teaches Swift, leave it
  out.
- ✅ **Prefer human-written text.** Many instruction datasets there are
  model-generated. Using one to *draft* candidates is allowed; shipping it
  unread is not, and it must be labelled as model-generated in the provenance
  note.
- ✅ **Pin the revision.** Datasets are edited in place; an unpinned build is
  not reproducible.

---

## 2. No shortcuts, anywhere

Minerva contains **real, working code only**. If it is in this repository, it
runs.

### Forbidden

- ❌ **Mock engines, fake model responses, canned answers, stubbed clients.**
  There is not one in this repository and there must never be one. A test that
  passes against a simulated backend proves nothing about the backend.
- ❌ **`TODO`, `NotImplementedError` or `pass` standing in for work that was
  supposed to be done in this change.** Either implement it or do not claim it.
- ❌ **A function that returns a plausible value without doing the work.**
- ❌ **Silencing an error to make something pass** — bare `except: pass`,
  broadening an exception clause, deleting an assertion, `# type: ignore` on a
  real type error, marking a failing test `xfail` because it is failing.
- ❌ **Sample/demo/placeholder data presented as if it were real output.**
- ❌ **Reporting something as working when it was not run.** If you did not
  execute it, say so.

### Required

- ✅ Every feature works against a **real engine** (Ollama, or another real
  backend added through `docs/ADDING_AN_ENGINE.md`).
- ✅ When something cannot be verified here — no daemon, no weights, no network
  — **say so explicitly** and make the test skip loudly with the reason. A
  skipped test that says why is honest; a mock that turns it green is not.
- ✅ Failures are **loud and actionable**. Compare:
  `cannot reach Ollama at http://127.0.0.1:11434 (Connection refused). Start it
  with 'ollama serve', or point Minerva elsewhere with MINERVA_OLLAMA_HOST.`
  against a silent fallback to a canned reply. Always the first.
- ✅ If a task turns out to be bigger than expected, **do the whole thing or
  report exactly what is missing**. Never quietly deliver a narrower version
  and describe it as complete.
- ✅ **Report what a model actually is.** Swift is a 23.2M-parameter base model:
  it continues text, it cannot call tools, it has no reasoning phase. The
  engine declares those limits in `capabilities`, the model spec declares them
  in `supports_tools` / `supports_thinking`, and the README states them in
  plain words. Never advertise a capability the weights do not have and let the
  runtime cover for it.
- ✅ **Keep the capability reports current.** `minerva evaluate` and `minerva
  evaluate-instruct` write real, measured numbers to `data/eval_report.json`
  and `data/instruct_eval_report.json`. The web UI's `DEVDEBUG` panel
  (type `DEVDEBUG` into the chat box — see `webui.py`'s `_stats_payload` and
  `webui_chat.html`'s `renderStats`) reads those files fresh on every
  request and shows them verbatim: held-out loss/perplexity, routing
  accuracy, tool-name accuracy, argument accuracy, final-answer accuracy,
  honest-refusal rate. **Regenerate both reports every time a checkpoint they
  describe is retrained or refinetuned.** A stale report shown as current is
  exactly the confident-but-wrong claim this whole project exists to avoid —
  the panel has no way to know its source files are stale, so that discipline
  has to hold on the writing side.
- ✅ **Write down the bugs and the judgement calls.** When the byte-coverage
  test found that the pre-tokenizer was silently deleting every underscore, the
  fix, the measured impact (1,249 characters out of 27M — 0.0046%) and the
  decision not to restart a 2.5-hour run all went into `docs/TRAINING.md`. A
  quietly-fixed bug is a bug nobody can audit.

---

## 3. Architecture: stay ready for what comes next

Minerva is a model *family*. Swift is the first member, not the only one. New
models, engines and tools are added regularly, so the structure must keep
absorbing them without rewrites.

- **Adding a model** = one new file + one line in `models/registry.py`.
  Never scatter model-specific behaviour through the codebase.
  → `docs/ADDING_A_MODEL.md`
- **Adding a tool** = one function with type hints and a docstring.
  → `docs/ADDING_A_TOOL.md`
- **Adding an engine** = one new class + one line in `engines/registry.py`.
  → `docs/ADDING_AN_ENGINE.md`
- **Training a model** = a corpus in `training/data.py`, a `SwiftConfig`, a run.
  → `docs/TRAINING.md`
- **The thinking scale is engine-agnostic.** Never add an eighth level and
  never put engine-specific parameters in `thinking.py`. A `ThinkingProfile`
  carries the intent in three encodings; each engine picks the one it speaks.

Before adding a layer, an abstraction or a dependency, ask whether it earns its
place. Minerva has exactly one runtime dependency (`httpx`). Keep it that way
unless there is a real reason not to.

---

## 4. Comments and documentation

Future sessions of this project will add models and tools, so **write for the
person who arrives next**.

- Comment **why**, not what. `swift.py` is the model of this: it explains why a
  base model ships no system prompt, why its thinking ceiling is `DO`, and why
  unset sampling fields stay unset.
- Registries and extension points carry a block comment naming the procedure
  for extending them.
- Update the relevant `docs/*.md` in the same change as the code. Documentation
  written "afterwards" is documentation written never.

---

## 5. Testing

- Unit tests cover real logic. Integration tests (`-m integration`) cover real
  inference against a live engine, and skip with a clear reason otherwise.
- Test the **failure paths**. A tool's error message is prompt text the model
  has to act on; it deserves a test.
- Never weaken a test to make it pass. Fix the code, or fix the test because
  the *expectation* was wrong — and say which.

```bash
pytest -m "not integration"   # everything that needs no engine
pytest -m integration         # real inference against a live engine
pytest -m torch               # the training stack; needs the `training` extra
ruff check . && mypy          # both must be clean before committing
```

---

## 6. Style

- Python 3.11+, full type annotations, `from __future__ import annotations`.
- `ruff` and `mypy` clean. No new `# type: ignore` without a comment saying why.
- Errors subclass `MinervaError`, and every message tells the reader how to fix
  the problem.

---

## 7. Git workflow

- **Intermediate commits (no push) are fine on large or multi-part work** —
  a task with several distinct pieces, or one that spans a long session.
  Committing partway through, to checkpoint real, working progress, is
  encouraged there, not something to ask permission for each time — it is a
  local, reversible action. **A small task does not need a commit at every
  step** — one commit at the end is enough; do not fragment a single small
  change into several commits just to "checkpoint" it. Pushing,
  force-pushing, and rewriting published history still need the user's
  explicit go-ahead, per the general safety rules this project otherwise
  follows.

---

## 8. Model size: no arbitrary cap, sized to the requirement

**There is no fixed parameter budget, and there is exactly one model.**
Swift's size is not a rule, it is what its corpus and this machine support:
9.9M in v0.3.0 on a 164M-token corpus, 23.2M in v0.5.0 on a 518M-token one.
Size the model to the job - a model that has to hold a sensible, quick
conversation gets whatever it needs - and let the two real constraints decide
the number, not habit:

**One model at a time.** Minerva is built to become a family, and it will get
stronger models later - but a named architecture with no trained weights is a
promise, not a model, and this file forbids claiming capability that the
weights do not have. So growing Swift means *changing Swift's config*, not
adding a second entry beside it that nobody has trained. When a genuinely
stronger model is actually going to be trained (a GPU appears, or a longer
budget), add it then, as its own config plus a `--arch` flag, so the shipped
checkpoint never stops matching the code that describes it. Until then, one
config, one checkpoint directory per architecture, and a new directory
whenever the shape changes - resuming a 23M run from 9.9M weights either
fails or silently trains the wrong thing.

1. **Data.** Chinchilla's ~20 tokens per parameter. Below that the extra
   parameters memorise instead of generalising, which is the "shop + random
   adjective" failure in §1 wearing a bigger coat. **This is no longer the
   binding constraint**: Project Gutenberg's own catalogue lists 57,136
   English texts, so the corpus can grow by orders of magnitude on demand.
2. **Compute.** This is now what binds, and it must be *measured on the
   machine in hand* before a size is chosen, never assumed.

### The hardware, measured

| | value |
|---|---|
| local cores | **14 logical** (an earlier version of this file said 4 - that was wrong) |
| local GPU | **none usable** - Intel integrated only, and torch is a CPU build |
| measured throughput | ~5,200 tok/s at 9.9M params, seq 512 |
| hosted runner | 2 cores, ~1,790 tok/s - slower than local, so it never rescues a long run |

Throughput falls roughly with parameter count, so the training cost of a
size is knowable in advance and should be worked out before committing:

| params | tokens for Chinchilla | one pass, 518M-token corpus |
|---|---|---|
| 9.9M | 198M | ~28 h local / ~80 h CI |
| 23.2M (v0.5.0) | 464M | ~57 h local / ~188 h CI |
| 91M | 1.8B | weeks |
| 211M | 4.2B | months |

Two measurements that decide how a run is scheduled, both taken on this
machine at 23.2M, batch 4 x seq 512:

* **14 threads gives ~2,520 tok/s; 10 threads gives ~2,365** - about 6%. The
  14 "cores" are 7 physical with hyperthreading, so the last four buy very
  little. Leaving them for the desktop is nearly free, and a machine that
  stays usable while it trains is worth 6%.
* **CI is ~3x slower than local**, not faster (2 cores against 14). At 23.2M a
  full pass is 31 sequential 6-hour jobs, which is not a plan. CI is a free
  parallel *sample* of the same config, not the primary run - see section 9.

### The honest ceiling

On **CPU only**, the mid-20M range is the largest size that trains in days
rather than weeks. v0.5.0 sits at 23.2M for a reason that is arithmetic, not
taste: the 518M-token corpus supports 23M under Chinchilla (464M needed) and
does not support 29M (582M needed). More data is available on demand - 52,610
of Gutenberg's 57,136 English texts are still undownloaded - but every book
added lengthens the run, so data and compute have to be chosen together.

Above that range the arithmetic stops being about willingness and starts
being about months of wall clock. **A CUDA GPU is the actual unlock** - it
moves a 91M-200M model from months to hours - and if the goal is a model that
genuinely converses, say so plainly rather than quietly training something too
small and reporting the metrics as if the target had been met.

Never present a size limit as a preference. State the measurement, state
what it costs, and let the person decide.

---

## 9. Running training: start both, keep whichever is faster

A real training run goes to **both** places at once — this machine and the
GitHub Actions workflow (`.github/workflows/train.yml`) — and whichever gets
there first is the one that counts. Neither is reliably faster, so the
answer is measured per run rather than assumed:

| | local | hosted runner |
|---|---|---|
| cores | 14 logical | 2 |
| measured throughput (9.9M, seq 512) | ~5,200 tok/s | ~1,790 tok/s |
| job limit | none | 6 h hard, so the run must resume across jobs |
| costs the user's machine | yes | no |

Local is roughly three times as fast per token, and the runner is usually
better at grinding through a long budget unattended. Start both, watch the
step counters, and take the checkpoint that is further along.

Rules that keep the two from corrupting each other:

- ✅ **Never let both write the same directory.** Local training owns
  `checkpoints/swift`. A CI artifact is downloaded somewhere else and only
  promoted over the local one after comparing real progress.
- ✅ **Compare by step count from `training_log.jsonl`**, not by file
  timestamp and not by `best.pt` — see the note in the workflow about why
  `best.pt`'s step can lag actual progress.
- ✅ **Stop the loser once one side finishes.** A run that can no longer win
  is just burning a machine or a free-tier minute; cancel it and say so.
### Merging the two runs

Averaging two checkpoints' weights — a "model soup" — is a real technique, not
a trick, and a merged model can genuinely beat both inputs. It is allowed
here. What it is not is free, or automatic, and it never simply adds the two
step counts together.

Merging is only defined inside **one loss basin**. Check all three before
attempting it:

1. **Identical architecture, tensor for tensor** — `vocab_size` above all,
   since it sets the embedding shape. Different shapes cannot be averaged at
   all.
2. **The same initialisation.** Both runs must start from the same seed. Two
   models trained from *different* random inits average to noise: a network's
   hidden units can be permuted freely, so averaging lines up features that
   have nothing to do with each other.
3. **A measured eval of the merged checkpoint against both inputs.** Merging
   is an experiment whose result is a number, never an assumption. If the soup
   is worse, keep the better input and write down that it was worse.

And know what merging cannot do. Two runs that share a seed *and* a data order
are the same trajectory at different points, not independent explorations —
averaging an earlier point into a later one drags the result backwards. A soup
pays off when the runs actually diverged (different data order, different
seed, different hyper-parameters), which is a thing to arrange deliberately if
a soup is the goal.

- ✅ **Taking the further-along checkpoint is always legitimate.** It is the
  default, and it needs no justification beyond the step count.
- ✅ **Merging is legitimate too, once the three checks above pass and the
  merged model has been evaluated.** Report the measured numbers for the soup
  and for both inputs, and keep whichever actually won.
- ❌ **Never merge on the assumption that more runs means a better model.**
  A soup of mismatched or same-trajectory checkpoints is not a stronger model,
  it is a worse one with a more impressive story.

> Learned the hard way in v0.5.0: CI trained a 23.2M model on a 8,192-token
> vocabulary while local trained 26.8M on 16,384, because the workflow left
> `--vocab-size` at its default. The two were never comparable, let alone
> mergeable, and nothing said so until someone asked whether they could be
> combined. **If both sides are meant to be mergeable, every corpus and
> config flag has to match explicitly, not by default.**

---

## הנחיות בעברית (תקציר)

**אימונים — איכות בלבד.**
ההגדרה של "איכותי": דוגמה שלא נוצרה מאלגוריתם, מתבנית או מפרמוטציה של ערכים —
ושהיא טובה באמת לגופה. זה הקריטריון בפועל; אין דרישה שבן־אדם יקרא ויאשר ידנית
כל שורה בנפרד לפני שהיא נכנסת. **אסור** סקריפט שמייצר אימונים לפי אלגוריתם או
תבנית, ואסור לנפח דאטהסט בשכפולים או בפרמוטציות של אותה שאלה. מאה דוגמאות
שחשבו עליהן שוות יותר ממאה אלף שסקריפט ייצר. אם אי אפשר להסביר למה דוגמה
מסוימת נמצאת בדאטהסט — היא לא צריכה להיות שם. אותו כלל חל גם על סטים של הערכה
(eval).

**אלגוריתם = ייצור דאטה לאימון, לא זיהוי הודעות בזמן ריצה.**
"אלגוריתם" כאן פירושו קוד שמייצר *דוגמאות אימון*, שבהן מאמנים באמת את
המשקולות — והמודל אחר כך עומד בזכות עצמו. המחולל רץ רק בזמן בנייה ולעולם לא
בנתיב ההסקה. **אסור** קוד שמסתכל על ההודעה הנכנסת בזמן ריצה ומחליט מה לענות:
בלי התאמת מילות מפתח, בלי "אם זו ברכה תחזיר ברכה", בלי טבלת תשובות מוכנות
ובלי תיקון של הפלט אחרי המודל. זה לא מודל שעונה, זו טבלה שמתחזה למודל. אם
מדד בקובץ הזה משתפר בגלל משהו בנתיב הבקשה ולא במשקולות — השיפור מזויף והמדד
משקר.

**ובנוסף: רק למבנה משפט, אף פעם לא לאוצר מילים.**
מותר לייצר דוגמאות אלגוריתמית למטרה אחת בלבד: ללמד איך *בונים* משפט — סדר
הפסוקיות, שאלה מול קביעה, זמן, התאמה, איך תשובה נקשרת לשאלה. **אסור** לייצר
דוגמאות שממלאות תוכן מתוך רשימה סגורה (שמות עצם, שמות תואר, מקומות, מספרים):
זה מלמד אוצר מילים סגור. זה בדיוק הכישלון מניסיון קודם לבנות מודל כאן —
הוא אומן רק על אלגוריתם, וכשביקשו ממנו "שם יצירתי לחנות" הוא ענה "חנות" ועוד
שם תואר אקראי מתוך החמישה שהכיר. הוא למד תבנית, לא שפה. **חובה לשמור על
איזון מפורש**: הדוגמאות הידניות נשארות החלק המשמעותי, וכל בלוק אלגוריתמי
יושב בסקשן נפרד וסָפוּר, כך שאפשר לקרוא את היחס מהקובץ ולא לנחש אותו. אוצר
המילים מגיע מקורפוס האימון המקדים — פרוזה אנושית אמיתית — ומשום מקום אחר.

**הכי חשוב: המודל תמיד צריך לדעת להגיד "אני לא יודע".** זו ההרגל הכי חשוב
שמודל של Minerva לומד — יותר חשוב מכל עובדה בודדת. תשובה שגויה שנאמרת בביטחון
כאילו היא נכונה גרועה יותר מסירוב, כי סירוב הוא כן באופן גלוי ותשובה שגויה
לא. כל סט אימונים חייב דוגמאות אמיתיות וכתובות ביד של הודאה באי־ידיעה, וגם —
לכל דבר שכלי יכול בפועל לפתור (עובדה עדכנית, מספר חי, אירוע) — דוגמאות של
פנייה לאותו כלי במקום ניחוש או סירוב סתמי. ראו את `_UNKNOWN` ו-`_WEB_SEARCH`
ב-`src/minerva/training/instruct_data.py`.

**מאמנים לשטף, לא לעובדות.** התפקיד של מודל Minerva הוא לנהל שיחה ברורה
וטבעית בשפות שהאימון שלו באמת מכסה (אנגלית ועברית, נכון ל-v0.3.0) — לא לשנן
את העולם. דאטה של אימון מקדים צריך ללמד דקדוק, רגיסטר וניסוח טבעי (ספרות,
נאומים, כתיבה יומיומית); תוכן אנציקלופדי וחדשותי מוצא בכוונה (ראו
`src/minerva/training/data.py`). כל עובדה חיה שייכת לקריאה לכלי (`web_search`
וכדומה), לא למשקולות. אסור להוסיף דוגמאות אימון בשפה שדאטת האימון המקדים לא
באמת חשפה אליה — כמה שורות של fine-tuning לא יכולות ללמד שפה שהמודל הבסיסי
מעולם לא ראה, רק לגרום לו לשנן כמה משפטים קבועים ולהמציא על כל השאר.

**בלי קיצורי דרך.**
רק קוד אמיתי שרץ. אין Mocks, אין תשובות מזויפות, אין דמה, אין `TODO` במקום
עבודה שהייתה אמורה להיעשות, ואין השתקה של שגיאות כדי ש"יעבור". כשמשהו לא ניתן
לבדיקה בסביבה הנוכחית — אומרים את זה במפורש והטסט מדלג עם סיבה ברורה, לא
מחליפים אותו במוק. אם משימה יצאה גדולה מהצפוי — מבצעים אותה במלואה או מדווחים
בדיוק מה חסר.

**מבנה שמוכן להמשך.**
הוספת מודל = קובץ אחד ושורה אחת ברג'יסטרי. הוספת כלי = פונקציה אחת. הוספת מנוע
= מחלקה אחת ושורה אחת. סולם החשיבה נשאר בלתי תלוי במנוע — שבעה צלילים, בלי
הוספות ובלי פרמטרים ספציפיים למנוע.

**דיווח כן על מה שהמודל באמת יודע.**
Swift הוא מודל בסיס בן 23.2M פרמטרים: הוא ממשיך טקסט, לא מפעיל כלים ואין לו שלב
חשיבה. המגבלות האלה מוצהרות בקוד (`capabilities`, `supports_tools`,
`supports_thinking`) ובתיעוד. אסור לפרסם יכולת שהמשקולות לא באמת מספקות ולתת
לקוד לכסות על זה. כשמתגלה באג — כותבים אותו, את ההשפעה המדודה ואת ההחלטה
שהתקבלה, ולא מתקנים בשקט.

**מודל אחד בכל רגע נתון.** Minerva בנויה להיות משפחה, ויהיו מודלים חזקים
יותר בהמשך — אבל ארכיטקטורה עם שם ובלי משקולות מאומנות היא הבטחה, לא מודל,
והקובץ הזה אוסר להציג יכולת שאין למשקולות. לכן הגדלה של Swift נעשית **בשינוי
הקונפיג של Swift**, ולא בהוספת רשומה שנייה לידו שאף אחד לא אימן. כשבאמת הולכים
לאמן מודל חזק יותר (יש GPU, או תקציב זמן ארוך) — מוסיפים אותו אז, עם קונפיג
משלו ודגל `--arch`. עד אז: קונפיג אחד, ותיקיית checkpoint חדשה בכל שינוי צורה —
להמשיך ריצה של 23M ממשקולות של 9.9M או ייכשל או יאמן בשקט את הדבר הלא נכון.

**אין תקרת פרמטרים שרירותית.** גודל המודל נגזר מהמשימה, לא מהרגל. שני אילוצים אמיתיים קובעים: דאטה (בערך 20 טוקנים לפרמטר — וזה כבר לא הצוואר, בקטלוג של גוטנברג יש 57,136 טקסטים באנגלית) ומחשוב, שנמדד על המכונה שבפועל ולא מנוחש. במכונה הזו: 14 ליבות, **אין GPU** (אינטל משולב, torch בגרסת CPU), כ-2,520 טוקנים לשנייה ב-23.2M. לכן על CPU בלבד טווח ה-20M הוא הגודל הגדול ביותר שמתאמן בימים ולא בשבועות. v0.5.0 עומד על 23.2M כי זה מה שהקורפוס מחזיק: 518M טוקנים תומכים ב-23M (צריך 464M) ולא ב-29M (צריך 582M). ועוד מדידה שקובעת איך מתזמנים ריצה: 10 threads נותן 94% מהמהירות של 14 — כלומר להשאיר ארבע ליבות פנויות למשתמש עולה כ-6% בלבד. **GPU הוא הפתרון האמיתי** אם המטרה היא מודל שבאמת משוחח. אסור להציג מגבלת גודל כהעדפה — מציגים את המדידה, את המחיר, ונותנים למשתמש להחליט.

**מריצים אימון בשני המקומות, ולוקחים את המתקדם.**
כל ריצת אימון אמיתית יוצאת לדרך גם במחשב המקומי וגם ב-GitHub Actions. מקומית
יש 14 ליבות מול 2 ב-runner, כלומר המקומי מהיר בערך פי שלושה, אבל ל-runner אין
מגבלת זמן על המכונה של המשתמש ויש לו מגבלת 6 שעות לכל job, ולכן הוא ממשיך
מ-checkpoint בין ריצות. **אסור** ששתי הריצות יכתבו לאותה תיקייה — האימון
המקומי הוא הבעלים של תיקיית ה-checkpoint שלו, וארטיפקט מ-CI יורד לתיקייה אחרת
ומקודם רק אחרי השוואת התקדמות אמיתית לפי מספר הצעד מ-`training_log.jsonl`
(לא לפי תאריך הקובץ ולא לפי `best.pt`). כשצד אחד מסיים — עוצרים את השני
ומדווחים על כך.

**מיזוג שתי הריצות — מותר, אבל רק כשהן באמת ניתנות למיזוג.**
מיצוע משקולות של שני checkpoints ("model soup") הוא טכניקה אמיתית, ומודל
ממוזג באמת יכול להיות טוב משני המקורות. זה מותר כאן. מה שזה **לא**: זה לא
מחבר את מספרי הצעדים, וזה לא עובד תמיד. מיזוג מוגדר רק בתוך אותו אגן הפסד,
ולכן בודקים שלושה דברים לפני:

1. **ארכיטקטורה זהה לחלוטין** — במיוחד `vocab_size`, שקובע את צורת טבלת
   ה-embedding. צורות שונות פשוט אי אפשר למצע.
2. **אותה אתחול (seed).** שני מודלים שאומנו מאתחולים אקראיים שונים ממוצעים
   לרעש: אפשר להחליף יחידות נסתרות ברשת בסדר כלשהו, ולכן המיצוע מיישר פיצ'רים
   שאין ביניהם שום קשר.
3. **הערכה מדודה של המודל הממוזג מול שני המקורות.** מיזוג הוא ניסוי שהתוצאה
   שלו היא מספר, לא הנחה. אם הממוזג יצא גרוע — שומרים את הטוב מבין המקורות
   וכותבים שהוא יצא גרוע.

וצריך לדעת מה מיזוג לא עושה: שתי ריצות עם אותו seed ואותו סדר דאטה הן **אותו
מסלול בשתי נקודות**, לא שתי חקירות עצמאיות — ומיצוע נקודה מוקדמת לתוך מאוחרת
מושך את התוצאה אחורה. soup משתלם כשהריצות באמת התפצלו (סדר דאטה שונה, seed
שונה, היפר-פרמטרים שונים), וזה משהו שמסדרים מראש בכוונה אם רוצים soup.

> נלמד בדרך הקשה ב-v0.5.0: ה-CI אימן מודל 23.2M על אוצר מילים של 8,192 בעוד
> שהמקומי אימן 26.8M על 16,384, כי ה-workflow השאיר את `--vocab-size` בברירת
> המחדל. השניים מעולם לא היו ברי-השוואה, בוודאי לא ברי-מיזוג, ושום דבר לא אמר
> את זה עד ששאלו אם אפשר לחבר אותם. **אם שני הצדדים אמורים להיות ברי-מיזוג —
> כל דגל של קורפוס וקונפיג חייב להיות זהה במפורש, לא בברירת מחדל.**

**קומיטים באמצע העבודה.**
מותר וכדאי לעשות קומיט (בלי push) כדי לשמור התקדמות אמיתית ועובדת באמצע
משימה גדולה או מרובת חלקים, בלי לבקש אישור בכל פעם — זו פעולה מקומית והפיכה.
במשימה קטנה אין צורך בקומיט על כל שלב — קומיט אחד בסוף מספיק. Push,
force-push ושינוי היסטוריה שפורסמה עדיין דורשים אישור מפורש מהמשתמש.
