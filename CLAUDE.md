# CLAUDE.md — working rules for this repository

Instructions for Claude (and any other agent or contributor) working on
Minerva. Read this before writing code.

---

## 1. Training: quality only, never generated in bulk

**This is the most important rule in this file.**

Training data for a Minerva model must be **genuinely good, individually
considered examples**. Every example must be worth the tokens it costs.

### Forbidden

- ❌ **A script that generates training examples from an algorithm.** Loops
  over templates, permutations of slot values, `f"What is {a} + {b}?"` across a
  thousand number pairs, combinatorial expansion of a phrase list — none of
  this is training data. It is padding with the shape of training data.
- ❌ **Synthetic data produced only to hit a row count.** "We need 10,000
  examples" is never a reason to make 10,000 examples.
- ❌ **Duplicating or lightly paraphrasing examples to inflate a dataset.**
- ❌ **Copying a public dataset in wholesale** without reading it, checking its
  licence, and deciding example by example that it belongs here.
- ❌ **Model-generated examples accepted without review.** Using a model to
  *draft* candidates is fine; shipping them unread is not.

### Required

- ✅ Every example is **read by a human or reviewed with real care** before it
  enters the dataset.
- ✅ Every example teaches something a real user actually needs — drawn from
  real tasks, real questions, real failures, real transcripts.
- ✅ Diversity comes from **genuinely different problems**, not from varying
  the numbers in one problem.
- ✅ Every dataset has a written provenance note: where the examples came from,
  who reviewed them, what they are meant to teach, and what was rejected.
- ✅ Wrong, sloppy or ambiguous examples are **deleted**, not "cleaned up
  later". One bad example teaches a bad habit that a hundred good ones must
  then unteach.

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
* Two real corpora were **rejected on quality grounds** and the reasons are
  recorded in the module docstring: the Brown corpus (distributed POS-tagged)
  and Pang & Lee's movie reviews (distributed lowercased and pre-tokenised).
  Together they were 17 MB — more than half again the literary corpus. Volume
  was not a good enough reason to teach the model damaged typography.
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
- ✅ **Report what a model actually is.** Swift is a 9.9M-parameter base model:
  it continues text, it cannot call tools, it has no reasoning phase. The
  engine declares those limits in `capabilities`, the model spec declares them
  in `supports_tools` / `supports_thinking`, and the README states them in
  plain words. Never advertise a capability the weights do not have and let the
  runtime cover for it.
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

## הנחיות בעברית (תקציר)

**אימונים — איכות בלבד.**
מותר רק דאטה איכותי שנבדק אחד־אחד. **אסור** סקריפט שמייצר אימונים לפי אלגוריתם
או תבנית, אסור לנפח דאטהסט בשכפולים או בפרמוטציות של אותה שאלה, ואסור להכניס
דוגמאות שלא נקראו. מאה דוגמאות שחשבו עליהן שוות יותר ממאה אלף שסקריפט ייצר. אם
אי אפשר להסביר למה דוגמה מסוימת נמצאת בדאטהסט — היא לא צריכה להיות שם. אותו כלל
חל גם על סטים של הערכה (eval).

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
Swift הוא מודל בסיס בן 9.9M פרמטרים: הוא ממשיך טקסט, לא מפעיל כלים ואין לו שלב
חשיבה. המגבלות האלה מוצהרות בקוד (`capabilities`, `supports_tools`,
`supports_thinking`) ובתיעוד. אסור לפרסם יכולת שהמשקולות לא באמת מספקות ולתת
לקוד לכסות על זה. כשמתגלה באג — כותבים אותו, את ההשפעה המדודה ואת ההחלטה
שהתקבלה, ולא מתקנים בשקט.
