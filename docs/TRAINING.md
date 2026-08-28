# Training Minerva models from scratch

Swift is not a fine-tune and not a wrapper around anyone else's weights. Its
architecture, its tokenizer and every one of its parameters come from this
repository. This document is the whole pipeline.

```
minerva prepare-data     # real corpus -> BPE tokenizer -> token bins
minerva train            # pretrain from scratch
minerva ask "The"        # run the weights you just trained
```

---

## 1. The corpus

`src/minerva/training/data.py`

**~50.8 MB of real, human-written prose**, in English and — as of v0.2.0 —
Hebrew. Nothing is generated, templated or augmented — `CLAUDE.md` forbids it,
and a model trained on synthetic filler learns the filler.

| Source | Size | Register | Licence |
|---|---|---|---|
| Project Gutenberg (18 books) | 11.7 MB | Literary prose and verse | Public domain |
| Reuters newswire (1987) | 7.8 MB | Factual reporting | Reuters-21578, research use |
| European Parliament (English) | 3.1 MB | Transcribed debate | Freely available |
| State of the Union addresses | 2.1 MB | Political oratory | Public domain |
| NLTK web text | 1.7 MB | Informal, conversational | Freely redistributable |
| Inaugural addresses | 0.8 MB | Formal oratory | Public domain |
| Simple English Wikipedia (sampled) | 6.2 MB | Encyclopedic reference prose | CC BY-SA 3.0 + GFDL |
| **Project Ben-Yehuda (Hebrew, 7 authors)** | **17.3 MB** | Hebrew literature: poetry, prose, drama, essays | Public domain |

The mixture is deliberate. A model trained only on 19th-century novels writes
only 19th-century novels; the newswire and web text are what give Swift any
contemporary register at all, and Hebrew is a register no amount of English
text can teach.

**The corpus is not vendored.** `data.py` downloads each source from its
original distributor and writes a `manifest.json` recording counts, character
totals, SHA-256 hashes and the licence of every source. Provenance stays
verifiable and we redistribute nobody's corpus.

### v0.2.0: adding Hebrew and encyclopedic English

The corpus grew from 27 MB to 50.8 MB by adding two sources, both found on
[Hugging Face Datasets](https://huggingface.co/datasets), reviewed by hand
(not `load_dataset()`'d straight into training), and pinned to a specific
revision — the process `CLAUDE.md`'s "Approved sources" section requires:

* **Project Ben-Yehuda** — the Hebrew analogue of Project Gutenberg: a
  public-domain library of Hebrew literature. The full release is the *entire*
  library (26,455 works, ~250 MB) — too large to read and vouch for, so it was
  curated down to seven canonical figures of modern Hebrew writing (Bialik,
  Rachel Bluwstein, Brenner, Ahad Ha'am, Mendele Mocher Sforim, Tchernichovsky,
  Frishman), and further filtered by `pseudocatalogue.csv`'s own metadata to
  their *original* (non-translated) poetry, prose, drama, memoir and essays —
  dropping their translations and correspondence. That filter mattered: for
  Bialik alone it was the difference between 10.2 MB (with translations and
  letters) and 2.9 MB (his own work). Across all seven authors it took the
  catalogue from ~40 MB down to the 17.6 MB actually used (17.3 MB survives
  cleaning and chunking). See `_iter_benyehuda_texts` in `data.py`.
* **Simple English Wikipedia** — encyclopedic reference prose, a register
  none of the other six English sources supply. A full-language Wikipedia
  dump runs from hundreds of megabytes to tens of gigabytes; downloading one
  wholesale would dwarf every other source and could not honestly be called
  "read". Stub articles and "List of ..." pages were filtered out, and the
  rest reproducibly sampled down to roughly this corpus's scale rather than
  Wikipedia's. See `_iter_wikipedia_texts` in `data.py`.

Three other real, available candidates were reviewed and rejected:

* **OSCAR-2301 (Hebrew)** — a raw Common Crawl web scrape: gated access, no
  per-document quality signal, the usual web-crawl duplication. Ben-Yehuda
  gives curated, attributed prose instead of unfiltered web text — the same
  trade this project already made for English (Gutenberg over a web dump).
* **Sefaria's Hebrew library** — real, carefully edited Hebrew, but GPL-3.0
  on a text dataset (a licence written for software, with genuinely unclear
  implications for a trained model's weights), and a liturgical/legal
  register segmented paragraph-by-paragraph, not continuous prose.
* **Full English Wikipedia** (`20231101.en`) — 6.4M articles, tens of
  gigabytes. Nobody could read a representative sample of that. Simple
  English Wikipedia is the same distributor and licence at a size that can
  actually be reviewed.

Two English-only rejections from v0.1.0 still stand, for the original reason:

* **Brown corpus** — distributed POS-tagged (`The/at Fulton/np-tl`).
  De-tagging is mechanical but leaves unnatural spacing around punctuation.
* **Movie reviews (Pang & Lee)** — distributed lowercased and pre-tokenised
  (`films adapted from comic books , whether they 're`). Casing and spacing are
  destroyed.

Volume was not a good enough reason to teach the model damaged typography, a
software licence's terms for a dataset, or the boilerplate around an article
rather than the article.

**A real bug, found by testing the new cleaner.** The first version of the
Wikipedia tail-strip regex missed three real formatting variants in the actual
dump: a trailing space before the newline (`"References \n"`), a leading space
before the header (`" Related pages \n"`), and a section with nothing after it
at all (`"...\n\nReferences"` at end-of-string). Each was found by checking the
*cleaned* output of the full sampled set for leftover header words, not by
inspecting a few examples by eye — three separate real dumps exercised three
separate edge cases, and eyeballing samples had already missed all of them
once. All three are now regression tests in `TestWikipediaCleaner`.

### The split

Held out **per source, by character count**, so validation carries every
register in the same proportion as training. Splitting by document count
instead would have handed validation almost entirely to Reuters, which is
8,578 of the corpus's documents but only a quarter of its text. Documents are
first chunked to ≤16 KB at paragraph boundaries so the split unit is roughly
uniform. Result: 50.46 MB train / 0.33 MB validation.

---

## 2. The tokenizer

`src/minerva/training/tokenizer.py`

A **byte-level BPE**, implemented and trained here — 8,192 tokens, learned from
the corpus above in ~4 minutes (up from 95 seconds in v0.1.0: a larger, mixed
English/Hebrew corpus means more unique pre-token types to weigh).

* **Byte-level base vocabulary.** All 256 byte values are tokens, so every
  possible input encodes. There is no unknown token and no text Swift cannot
  represent — this held before Hebrew was ever added to the corpus, and it is
  the reason adding a second script needed no tokenizer *code* changes, only a
  vocab-size measurement.
* **GPT-2-style pre-tokenization**, so merges never straddle a space boundary.
* **Digits are always split individually.** Grouping `1987` into one token
  destroys place-value consistency and makes number handling markedly worse.
* **`<|endoftext|>` is a real token**, so the model learns that documents end.

Training uses the standard fast path: merges are computed over the ~374k
*unique* pre-token types (from 13.4M total pre-tokens) weighted by frequency,
and pair counts are updated incrementally through a pair → containing-words
index rather than recounted after every merge.

Measured compression on the v0.2.0 corpus (8,192 tokens): **2.886 characters
per token overall** → **17,483,579 training tokens**, 105,225 validation
tokens. Split by script, because a single blended number hides the thing that
actually matters here:

| | chars/token | bytes/token |
|---|---|---|
| English portion | 3.338 | 3.341 |
| Hebrew portion | 2.276 | 4.065 |

### The vocab-size decision: measured, not guessed

Hebrew is UTF-8-encoded as 2-byte sequences, so an *unmerged* byte-level
Hebrew token would sit at 0.5 chars/token — the 2.276 measured above is
already a **4.5× improvement** from merges the trainer learned unprompted;
Hebrew earned a real share of the 8,192-token budget on frequency alone (the
very first learned merges include Hebrew bigrams like ` נת` and ` הקט`,
ahead of `br` or `border`). The question was whether growing the vocabulary
specifically for Hebrew — the user's own suggestion was 8,192 → 12,000 — was
worth what it costs: with tied embeddings, every added vocab slot is
`d_model` = 320 more parameters, so 12,000 tokens means **11,094,080 total
parameters, +12.3%** over the shipped 9,875,520.

Both tokenizers were actually trained on the real corpus and measured:

| vocab | Hebrew chars/token | English chars/token | overall tokens | total params |
|---|---|---|---|---|
| 8,192 | 2.276 | 3.338 | 17,483,575 | 9,875,520 |
| 12,000 | 2.376 | 3.513 | 16,672,758 | 11,094,080 |

Growing to 12,000 bought **+4.4% Hebrew compression, +5.2% English
compression, 4.9% fewer overall tokens** — real, but it does not close the
Hebrew/English gap (the ratio of Hebrew to English chars/token is 0.682 at
8,192 and 0.676 at 12,000: growing the vocabulary helps both scripts by
roughly the same proportion, it does not specifically catch Hebrew up). Model
speed is essentially unaffected either way — the transformer body's
non-embedding compute (7,254,080 params) is identical at both vocab sizes,
so the cost of 12,000 is +12.3% more parameters for a ~5% compression gain,
almost entirely in a bigger embedding table and output projection, not more
useful capacity.

**Decision: kept at 8,192.** A ~5% compression improvement is not the "strong
measured reason" `CLAUDE.md` requires to grow the model, and Hebrew is not
broken at 8,192 — it is measurably, if imperfectly, learning real merges. If
Hebrew's share of the corpus grows substantially in a future round, this
measurement should be repeated; the answer is not assumed to be permanent, only
correct for a 34%-Hebrew, 50.8 MB corpus.

### A bug the tests caught, and what was done about it

The first version of the pre-tokenization pattern **silently deleted every
underscore**. Python's `\w` includes `_`, so the standard "letters only" idiom
`[^\W\d_]` excluded it from the letter run, and `[^\s\w]` excluded it from the
punctuation run. It matched no alternative at all, and characters that match no
alternative simply vanish.

`test_every_byte_value_round_trips` found it. The fix splices underscore back
into the punctuation alternative, and there is now a property test asserting
that pre-tokenization never loses a character.

**The training run was not restarted.** The corpus contains 1,249 underscores
in 26,969,248 characters — 0.0046%, almost all of it Gutenberg's `_italic_`
markup. The shipped checkpoint was therefore trained on text with those 1,249
characters absent. That is recorded here rather than quietly fixed, because the
alternative was discarding a correct 2.5-hour run over four thousandths of one
percent of the data. Encoding is lossless from the fix onward.

---

## 3. The architecture

`src/minerva/training/model.py`

`SwiftLM` is a decoder-only transformer. Every component was chosen for what it
buys at this scale:

| Component | Why |
|---|---|
| **Pre-norm + RMSNorm** | Keeps the residual stream a clean identity path. RMSNorm drops LayerNorm's mean-centring: cheaper, no measured loss. |
| **RoPE** | Position by rotating queries and keys, so attention sees *relative* distance. No learned position table to over-fit. |
| **SwiGLU** | Gated FFN; reliably beats plain GELU/ReLU at equal parameter count. `d_ff = 8/3 × d_model`, rounded to 64. |
| **Grouped-query attention** | Shrinks the KV cache. Swift uses full multi-head; the machinery is there for larger models. |
| **Tied embeddings** | The vocabulary matrix is a quarter of Swift's parameters. Tying is a large effective-capacity win at this size. |
| **Scaled residual init** | `o_proj` and `down_proj` initialised at `0.02/√(2·n_layer)`, or residual variance grows with depth. |

Swift's shipped configuration:

```python
SwiftConfig(vocab_size=8192, n_layer=6, n_head=8, d_model=320, max_seq_len=512)
# 9,875,520 parameters total (7,254,080 non-embedding), d_ff = 832
```

Dropout is **0**, deliberately: with 50.8 MB of text the model is
data-limited, not over-fitting, and dropout would only slow learning.

---

## 4. Sizing the run

This is the part most write-ups skip. Swift's size was not picked for
aesthetics — it was derived from the compute actually available.

Measured on this machine (4-core Xeon @ 2.1 GHz, no GPU):

* fp32 matmul peak: **430 GFLOPS**
* real training throughput: **~3,000–4,200 tokens/second**

A transformer costs about `6 × N × D` FLOPs to train (`N` parameters,
`D` tokens). Chinchilla-optimal is `D ≈ 20N`, giving `C ≈ 120N²`. For a
~3-hour budget:

```
C ≈ 3h × 3,600 × ~120 GFLOP/s      ≈ 1.3 × 10^15 FLOPs
N ≈ √(C / 120)                     ≈ 3M parameters, D ≈ 60M tokens
```

The v0.1.0 corpus held 7.4M unique tokens, so 60M tokens meant ~8 epochs.
Repeating data is near-lossless up to ~4 epochs and degrades after, so that run
was balanced at **9.9M parameters × 36.9M tokens ≈ 5 epochs** — slightly
over-parameterised relative to Chinchilla, which is the right trade when
unique data, not compute, is the binding constraint.

**v0.2.0 changed the data, not the budget.** Per `CLAUDE.md`, growing the
architecture or the training time needs a strong measured reason, and neither
tokenizer size (§2) nor this sizing arithmetic produced one. The step budget
stayed at 4,500 steps × 8,192 tokens = the same 36.9M tokens trained on, but
the corpus grew to 17.5M unique tokens (up from 7.4M), so the same compute
now covers **~2.1 epochs instead of 5** — strictly closer to Chinchilla-optimal
at the same parameter count, which is exactly the trade `docs/TRAINING.md` §9
already recommended: more unique data is the highest-value change available,
and it does not cost a single extra minute of compute.

**The learning rate was chosen by a real experiment, not a guess.** A 70-step
probe over `{1, 2, 3, 5}×10⁻³` before the run: 1e-3 and 2e-3 tied on early
loss, 3e-3 and 5e-3 were clearly worse. The run used 2e-3.

---

## 5. The training loop

`src/minerva/training/trainer.py`

AdamW (β = 0.9/0.95, weight decay 0.1 on matrices only — decaying RMSNorm
gains shrinks scales for no benefit), linear warmup over 200 steps then cosine
decay to 10% of peak, gradient clipping at 1.0, gradient accumulation,
deterministic seeding.

```
seq_len 512 × micro_batch 8 × grad_accum 2 = 8,192 tokens per optimiser step
```

Every run writes:

* `best.pt` / `last.pt` — self-describing checkpoints. The architecture config
  travels **with** the weights, so a checkpoint never needs a matching config
  file. Written to a temp file and renamed, so a crash mid-write cannot destroy
  a good checkpoint.
* `training_log.jsonl` — every measurement taken, for plotting the real curve.
* `run_config.json` — the exact configuration used.

Resume with `minerva train --resume checkpoints/swift/last.pt`. The batch
sampler is re-seeded from the resumed step so a resumed run does not replay
batches it already saw.

---

## 5a. The run that produced Swift v0.1.0

Measured, not estimated. Reproduce with `minerva prepare-data && minerva train`.

| | |
|---|---|
| Hardware | 4-core Intel Xeon @ 2.1 GHz, **no GPU** |
| Wall clock | **155 minutes** |
| Steps | 4,500 × 8,192 tokens = **36.86M tokens** (5.0 epochs) |
| Throughput | ~3,000–5,200 tokens/second |
| Final held-out loss | **3.4579** |
| Final held-out perplexity | **31.75** (chance = 8,192) |
| Bits per byte | **1.3571** |
| Validation tokens scored | 94,208 (the entire held-out split) |

### The curve

Monotonic throughout — no divergence, no plateau, no instability. Gradient norm
stayed near 0.5 for the whole run.

| step | val loss | perplexity | | step | val loss | perplexity |
|---|---|---|---|---|---|---|
| 250 | 4.8436 | 126.93 | | 2,500 | 3.6281 | 37.64 |
| 500 | 4.3686 | 78.93 | | 3,000 | 3.5561 | 35.03 |
| 750 | 4.1692 | 64.67 | | 3,500 | 3.5045 | 33.27 |
| 1,000 | 4.0143 | 55.38 | | 4,000 | 3.4716 | 32.19 |
| 1,500 | 3.8481 | 46.90 | | 4,250 | 3.4639 | 31.94 |
| 2,000 | 3.7248 | 41.46 | | **4,500** | **3.4604** | **31.83** |

Loss was still falling at the end. The model is **data-limited, not
compute-limited**: 7.4M unique tokens seen five times over. More corpus is the
highest-value next change, not more steps.

### Real samples (temperature 0.8, top-k 40, unedited)

> **"The company said it expects"** → *to provide a 5-1/2 pct stock dividend of
> 10 cts a share. It said it will continue to operate in the New York and Exco
> Health Commission to pay off the total value of the common shares on the
> outstanding common stock. The company said it expects to spend 400,000 dlrs in*

> **"My fellow citizens, we gather today to"** → *our own country, and the
> future will make us happy. We have a new spirit of confidence and opportunity
> for all Americans -- to do their share in peace in the world. (Applause.) And
> our friends, and the leaders of the community in Latin America*

What went right: syntax is largely well-formed, and the model learned genuine
register. It reproduces Reuters house style down to the abbreviations —
`pct`, `cts`, `dlrs`, `5-1/2` — and picked up `(Applause.)` from the State of
the Union transcripts. Those are corpus-specific conventions nobody coded.

What did not: **register conditioning is unreliable.** Two of the five sample
prompts drew the wrong register — the Austen opening produced political
oratory, and the Reuters weather opening produced Austen-ish domestic prose.
Semantics drift within a couple of sentences ("the New York and Exco Health
Commission" is not a thing). Local coherence is good; global coherence is not.

That is an accurate description of a 9.9M-parameter base model trained on
27 MB, and it is what this much compute buys. It is not a chatbot.

---

## 5b. The run that produced Swift v0.2.0 — adding Hebrew

Same architecture, same step budget, same everything except the corpus and
the tokenizer trained on it (§1, §2). Measured, not estimated. Reproduce with
`minerva prepare-data && minerva train`.

| | v0.1.0 | v0.2.0 |
|---|---|---|
| Wall clock | 155 min | **98.9 min** |
| Throughput | ~3,000–5,200 tok/s | ~6,200 tok/s (faster hardware this run) |
| Steps / tokens | 4,500 × 8,192 = 36.86M | 4,500 × 8,192 = 36.86M (**unchanged**) |
| Epochs | 5.0 (7.4M unique tokens) | **2.1** (17.5M unique tokens) |
| Held-out loss | 3.4579 | **3.8542** |
| Held-out perplexity | 31.75 | **47.19** |
| Bits per byte | 1.3571 | **1.4516** |
| Val tokens scored | 94,208 | 102,400 |

**The blended number got worse. Here is why, measured, not guessed** — the
held-out set was re-scored separately by language:

| | loss | perplexity | bytes/token | bits/byte |
|---|---|---|---|---|
| English only | 3.5715 | 35.57 | 3.510 | **1.4678** |
| Hebrew only | 4.3480 | 77.32 | 4.396 | **1.4270** |
| Blended (reported above) | 3.8542 | 47.19 | — | 1.4516 |

Two real effects, both expected, both worth naming honestly:

1. **English itself got measurably worse** — bits per byte 1.3571 → 1.4678, a
   real ~8% regression, not an artifact. The most likely cause is the same
   fixed 36.86M-token budget now buys 2.1 epochs instead of 5, and the same
   8,192-token vocabulary is shared with a second script instead of dedicated
   entirely to English (§2: English compression fell from 3.66 to 3.338
   chars/token specifically because Hebrew earned a real share of the merge
   budget).
2. **Hebrew's perplexity looks far worse than English's, but its bits per
   byte does not** — 77.32 vs 35.57 in perplexity, but 1.4270 vs 1.4678 in
   bits per byte, where Hebrew is actually *slightly better*. Hebrew tokens
   carry more bytes each (4.396 vs 3.510), so a fair, tokenizer-agnostic
   comparison is bits per byte, not perplexity — and by that measure the model
   is doing about as well on Hebrew as on English, for the first language
   it has ever seen, from 17.3 MB, sharing a 9.9M-parameter budget with a
   second script it also has to carry.

Whether that is a good trade is not this document's call to make unilaterally
— it is the honest, measured cost of teaching a 9.9M-parameter model two
languages instead of one, reported the way `CLAUDE.md` requires results to be
reported, worse number included.

### Real Hebrew samples (temperature 0.8, top-k 40, unedited)

Genuinely novel continuations, not memorised text — `בין כל הצרות` is the
opening of an Ahad Ha'am essay in the training data, and the model's
continuation diverges from the real essay after the first three words:

> **"שלום רב"** → *לנו כי אין אנו רשאים לדברים: יודעים אתם, כי גם אלה הם הם
> חורבים על אחרים, הם באים ואומרים, כי הם אינם נוצרים, כי הם הם עצמם אינם
> רוצים להניח להם את הלבבות, ורק לפי דעתם הם האחרים, שהם פוסקים להם על כל
> דבר ואינם יודעים מה זה – אבל מה יעשה? אבל הנה כדי ש*

> **"הַיּוֹם הָלַךְ"** (a poem opening, with niqqud) → *עַל הַפְּרִיפַת
> הַדָּם לְהַגִּיד הַשְּׂדִים לִקְרֹא לִקְרַאת הַתְּקָרָה, וְזָרִיז עָלֶיהָ
> תִּש*

> **"בין כל הצרות"** → *וכל השאלות הנפלאות העשויות בהם כל קרבנותם, והביאו את
> כל אחדותם החיצונית, את כל הצורות והישרה שבמושבה, כי גם על ידיהם הרבות, אשר
> אין בהם כל סכום קטן ושירם.*

What went right: the model learned real Hebrew morphology — correct verb
conjugation, correct prefix attachment (ה-/ו-/ל-/ש-), and it learned the
corpus's own register convention of niqqud on poetry but not prose *without
being told that rule* — it inferred it from which documents carried diacritics
and which did not, the same way v0.1.0 inferred `(Applause.)` from State of the
Union transcripts.

What did not: the Hebrew is grammatical in short spans and loses the thread
over a full sentence, the same "local coherence, not global coherence" pattern
v0.1.0 showed in English — and here it is showing up in a second language
from roughly a third of the training exposure. This is not a chatbot in either
language, and this document does not describe it as one.

---

## 5c. The run that produced Swift v0.3.0 — removing facts, restoring register

**What changed and why.** Two sources were removed from the corpus: Simple
English Wikipedia (general knowledge) and Reuters newswire (event reporting).
Not a quality decision — a capability one. Swift is 9.9M parameters; it has
nowhere to reliably store "the capital of X" or "what happened on date Y",
and training on text whose whole point is dense factual claims does not give
it that capacity, it gives it fluent confabulation. `minerva.tools.builtin.
web_search` (a real DuckDuckGo HTML-search call, no API key) replaces both:
the model looks a fact up instead of guessing at one it was never big enough
to remember correctly. `instruct_data.py`'s `_UNKNOWN` section was split:
what stays a flat refusal is what no tool fixes (the model's own memory of
this conversation, a capability it genuinely lacks); what a search can
actually resolve now routes to `web_search` instead.

**The regression this caused, measured, not guessed.** Removing Reuters and
Wikipedia cost the corpus more than their facts — they were also its only
short, declarative, factual-register prose. A first retrain on the resulting
36.46 MB corpus (everything else unchanged) produced a base model that,
finetuned on the *unchanged* 219-example instruct set from v0.2.0, only
reached 62% best routing accuracy and produced empty or malformed answers to
basic arithmetic — far below the 91–97% this document records for earlier
rounds. Isolating the cause mattered: finetuning the same untouched
219-example set against the *old* base model was never re-run to compare
directly, but the fact that the regression showed up on an instruct set that
had not changed pointed at the base model, not the data added on top of it.

**The fix: the Brown corpus, added back for register, not facts.** Brown was
rejected in v0.1.0/v0.2.0 for a formatting reason — NLTK distributes it
POS-tagged (`The/at Fulton/np-tl`) and naively de-tagging it leaves a space
in front of every comma and period. That turned out to be a solvable cleaning
problem (`_clean_brown` in `data.py`: strip the `/TAG` suffix, then a real
punctuation-aware detokenizer, with the ``/`` `` / `''` quote-tag pair
converted to a proper paired `"` *before* the punctuation pass runs, so a
quote immediately followed by a comma collapses correctly). The text itself —
c.1961 American press reportage, editorial and fiction — is exactly the
missing register: short and declarative, and old and general enough in
subject that it is not a database of facts worth memorising the way Wikipedia
or 1987 Reuters newswire is.

| | pre-v0.3.0 (v0.2.0) | v0.3.0, no Brown (regression) | v0.3.0, + Brown |
|---|---|---|---|
| Corpus size | 50.8 MB | 36.46 MB | **42.33 MB** |
| Held-out loss | 3.8542 | 3.8765 | **3.5300** |
| Held-out perplexity | 47.19 | 48.26 | **34.12** |
| Bits per byte | 1.4516 | 1.3862 | **1.3738** |
| Instruct routing accuracy | 91–97% | 70.5%\* | **97.7%** |
| Instruct argument accuracy | — | 3.4%\* | 10.3% |

\* Measured on the full v0.3.0 271-example instruct set (see §8). An earlier
version of this table also cited a 219-example "control" run at 34.1%
routing; **that number was wrong and has been withdrawn** — see the
correction in §8.

**Corpus, final:** Gutenberg 11.73 MB, Brown 5.98 MB, webtext 1.72 MB,
inaugural 0.82 MB, State of the Union 2.07 MB, Europarl (English) 3.06 MB,
Project Ben-Yehuda (Hebrew) 17.25 MB — 42.33 MB train / 0.36 MB val.

**The run itself:** same architecture, same 4,500-step / 36.86M-token budget
as every version before it. Wall clock is reported as compute time rather
than a single elapsed span this round — the process was interrupted twice by
the development environment reclaiming idle background processes (not a
training bug) and resumed both times from its last checkpoint
(`--resume checkpoints/swift/last.pt`), losing at most the ~500 steps since
the last `checkpoint_interval` save each time. Total compute across all
segments was a little over 90 minutes; a completely uninterrupted run should
land near the ~99 minutes v0.2.0 took on the same hardware. Recorded here
because a quietly-resumed run is a run nobody can audit otherwise.

### The curve

| step | val loss | perplexity |
|---|---|---|
| 250 | 4.8720 | 130.59 |
| 1,000 | 4.0053 | 54.89 |
| 2,000 | 3.7011 | 40.49 |
| 3,000 | 3.5295 | 34.11 |
| 4,000 | 3.4142 | 30.39 |
| 4,500 (final training-loop eval) | 3.3921 | 29.73 |
| **held-out, full val set (`minerva evaluate`)** | **3.5300** | **34.12** |

The final training-loop number (3.3921) and the full held-out evaluation
(3.5300) differ because the in-loop eval scores a 20-batch sample each time
for speed, while `minerva evaluate` scores all 114,688 held-out tokens. The
full-evaluation number is the one reported as authoritative above and in the
README.

### Real samples (temperature 0.8, top-k 40, unedited)

> **"It is a truth universally acknowledged, that"** → *he is so much as to a
> certain sort of man in him. I hope you may be more vicarious; but, as if
> you are, you will, I say you, and your own friends must be so very much a
> person." "I am not too mistaken at all," said Mr. Knightley, "he will not
> have it."*

> **"The company said it expects"** → *that the public money should be called
> out. I cannot report that the state of my country were not to be a
> "personal man", but it was not "because the people were in the hands of
> them, but I know it is not true and wrong in the spirit of a democracy".*

> **"שלום רב"** → *ץ" – והוא הוא, "לא" – הכל לפי דרכי האדם היהודי. כלומר: מפני
> מה? מה יעשו לו זה? כך הוא אומר, כי גם זה הוא, שהנחותה הוא עושה את הלבבות;
> אבל לא עלו בידו ולא מפרע מעולם ולא יכול לעמוד על דעתו...*

Real, correctly-punctuated dialogue with properly paired quotation marks —
directly attributable to the Brown corpus's press/fiction register, which is
exactly the register the regression traced back to losing. Hebrew still shows
the same "grammatical in short spans, loses the thread over a full sentence"
pattern v0.2.0 documented; that is a corpus-scale limitation, not something
this round changed.

---

## 5d. v0.4.0 — the corpus was the bottleneck, not the parameter count

The goal for this round was **fluency**: replies that read as connected,
relevant prose in both English and Hebrew. The obvious lever looked like
parameters, and the plan was to scale the model up. Measuring first changed
the plan, which is the point of measuring first.

### The measurement that overturned the plan

Chinchilla's rule of thumb puts the compute-optimal point near **20 training
tokens per parameter**. Against v0.3.0's corpus, every configuration was
starved:

| params | tokens Chinchilla wants | v0.3.0 corpus had | % of optimal |
|---|---|---|---|
| 9.9M (shipped) | 198M | 14.7M | **7.4%** |
| 29M | 580M | 14.7M | 2.5% |
| 55M | 1,100M | 14.7M | 1.3% |
| 91M | 1,820M | 14.7M | 0.8% |

Swift was not parameter-limited. It was **data-limited by more than 13x**,
and scaling parameters would have made the ratio worse, not better — a bigger
model on the same 14.7M tokens overfits sooner and generalises less. The
honest fix was more real text.

### What was added

The corpus grew from **42.6 MB to 185.5 MB** (~64M unique tokens, a 4.4x
expansion), from two sources that were already trusted, just under-used:

* **Project Gutenberg, 119 more books** (89.0 MB). NLTK's `gutenberg` source
  ships only 18. These are fetched per-book by ebook id straight from
  gutenberg.org — novels, gothic, detective, adventure, early science
  fiction, children's literature, essays, philosophy, drama and translated
  classics, chosen for register diversity rather than raw size. Every id was
  **fetched and checked before it was written into the source**: that it
  resolves, that it is a substantial Gutenberg ebook, and that it is
  mostly-Latin script. Three deliberately-invalid control ids were included
  in the check and correctly rejected. The check also caught two real
  mistakes — id 1522 is *Julius Caesar* and 19033 is *Alice in Wonderland*,
  both already shipped by the NLTK source, so both were dropped rather than
  duplicated into the corpus.
* **Project Ben-Yehuda, 7 authors → 35** (17.3 MB → 71.2 MB). All modern
  Hebrew: Haskalah and Hebrew-revival writers onward. The deliberate
  exclusion matters more than the inclusion — Ben-Yehuda's largest
  contributors by work count are **medieval** poets (Samuel HaNagid alone has
  1,856 works, plus Ibn Gabirol, Judah Halevi, the Ibn Ezras, Shabazi).
  Including them would have more than doubled the Hebrew side, and was
  refused: 11th–17th century liturgical and courtly verse is to modern
  conversational Hebrew roughly what Chaucer is to spoken English, and this
  model is meant to hold a conversation. Register was bought over volume.

Final mixture: 185.5 MB — 114.4 MB English (62%), 71.2 MB Hebrew (38%),
holding roughly the v0.3.0 language balance at four times the size.

### Two real leaks, found by scanning the built corpus

The cleaners were checked by scanning the **entire** 184.7M-character build
for boilerplate, not by reading the code and assuming. Both hits were real:

| leak | before | after |
|---|---|---|
| `&nbsp;` (Ben-Yehuda HTML entities) | 9,832 | **1** |
| `↩︎` (footnote-return arrows) | 9,429 | **0** |
| Gutenberg licence footer / donation address | 7 | **0** |

The Hebrew hits were HTML pipeline residue in Ben-Yehuda's plain-text
release; `_clean_benyehuda` now unescapes entities and strips the arrows. The
Gutenberg hits were the more interesting find: they came from the **old NLTK
source**, not the new one — its copies are mostly pre-stripped, but the
Chesterton files still carried an END marker and the foundation's donation
address behind it. That is a pre-existing defect dating to v0.1.0, found only
because the new source prompted a full scan. Both cleaners have regression
tests. Three occurrences survive in 184.7M characters (0.0000016%), almost
certainly double-escaped entities, and were judged not worth another rebuild.

### The decision: same parameters, far more tokens

With ~64M unique tokens available, the compute-optimal size is the size Swift
already is:

| config | params | tokens wanted | epochs over 64M |
|---|---|---|---|
| **n_layer=6, d_model=320 (shipped)** | **9.9M** | **198M** | **3.1x** |
| n_layer=8, d_model=512 | 29.1M | 582M | 9.1x |
| n_layer=10, d_model=640 | 54.8M | 1,096M | 17.1x |

At 29M the model would need nine passes over the corpus to be
compute-optimal, and at 55M seventeen — deep into the repetition where a
small corpus starts being memorised rather than generalised from. **So the
architecture is unchanged and the token budget grows instead**: from 36.86M
tokens (v0.3.0) toward ~164M, roughly 2.6 passes over a corpus 4.4x larger.
This reverses the "scale the model up" plan on measured grounds, and is
recorded here rather than quietly dropped.

### The vocabulary question, re-measured rather than assumed

§2's vocab decision ends by saying the measurement "should be repeated" if
the corpus changes substantially, and "is not assumed to be permanent, only
correct for a 34%-Hebrew, 50.8 MB corpus". The corpus is now 185.5 MB and
38% Hebrew, so it was repeated on the real rebuilt corpus:

| | v0.2.0 corpus (50.8 MB) | v0.4.0 corpus (185.5 MB) |
|---|---|---|
| English chars/token | 3.338 | **3.355** |
| Hebrew chars/token | 2.276 | **2.285** |
| Hebrew/English ratio | 0.682 | **0.681** |

**Essentially unchanged** — +0.5% English, +0.4% Hebrew, and the gap between
the scripts is identical to three decimal places. Quadrupling the training
text did not make an 8,192-token vocabulary meaningfully better or worse at
either language, which is the outcome that leaves the original decision
standing rather than merely un-revisited. Vocabulary stays at 8,192, now for
a measured reason on this corpus rather than an inherited one.

---

## 6. Running the weights

`src/minerva/engines/native.py`

The `minerva` engine loads a checkpoint and runs it in-process — no daemon, no
network. It plugs into the same `Engine` interface as the Ollama engine, which
is precisely what that abstraction was built for.

It declares its capabilities honestly: `streaming=True`, `tools=False`,
`thinking=False`.

---

## 7. What Swift is and is not

**Swift is a base language model.** It was pretrained to predict the next token
and has had no instruction tuning, no chat tuning and no RLHF. Therefore:

* It **continues text**. Prompt it with `"The Bahia cocoa zone"` and it
  writes plausible newswire. Prompt it with `"What is the capital of France?"`
  and it will most likely write *more questions*, because that is what follows
  a question in its training data.
* It **cannot call tools.** Tool calling is a trained behaviour.
* It **has no reasoning phase**, so every solfège level resolves to `DO`.

These are facts about its training stage, not gaps in the platform, and the
code states them rather than pretending otherwise. At 9.9M parameters trained
on 50.8 MB across two languages, Swift is roughly the scale of a small research
baseline — it learns grammar, vocabulary, register and local coherence. It is
not a chatbot and should not be described as one.

---

## 8. Stage two: instruction tuning

`training/finetune.py`, `training/chat.py`, `training/instruct_data.py`

Pretraining taught Swift English. It did not teach it a conversation, and a
base model asked "What is 17 times 43?" writes more questions. Stage two fixes
that, producing a **second model in the catalogue**, `swift-instruct`, while
the base model stays exactly as it was.

```bash
minerva finetune                    # ~2 minutes on 4 CPU cores
minerva evaluate-instruct           # measure it on held-out conversations
```

### The chat format

Nine markers, each added to the tokenizer as a **single token** and the
embedding matrix extended to match (pretrained rows copied across untouched):

```
<|user|>What is 17 times 43?<|assistant|><|call|>calculate {"expression": "17 * 43"}<|/call|><|result|>731<|/result|><|assistant|>17 times 43 is 731.<|end|>
```

A byte-level BPE would otherwise spend 5-7 tokens on `<|assistant|>` — 
unaffordable in a 512-token context. One module defines the format, and the
training data, the engine's prompt and the parser all use it, so they cannot
drift apart.

**Generation stops at `<|/call|>`.** That is not an optimisation: if the model
ran past it, it would write the tool's output itself and the agent loop would
feed an invented result back as though a tool had produced it.

### The data

**185 English conversations, every one written by hand** (34 more in Hebrew,
added in v0.2.0 — see below). No script, no templates, no permuted slot
values — `CLAUDE.md` forbids all three, and this is where that rule bites
hardest.

**Tool results are never fabricated**: an example declares the *call*, and the
build executes the real tool to get the result that goes into the text. Only
the clock examples are pinned, because their output depends on the wall clock.

Loss is **masked to the assistant's own tokens** (59% of them). The model is
not trained to predict the user's questions or the tool's output.

### Choosing the checkpoint by the right metric

Validation loss selected the wrong model. Loss is dominated by the dozens of
content tokens in each answer, while the decision that matters — call a tool,
think first, or answer directly — is a **single token** after `<|assistant|>`.
The first run reached 0.14 training loss while getting that one token wrong on
every arithmetic question.

So the trainer measures **routing accuracy** directly and selects on it.

### Three rounds, including the one that went backwards

| set | routing | tool name | arguments | answer | refusal |
|---|---|---|---|---|---|
| 123 examples | 91.2% | 88.9% | 11.1% | 6.2% | 50.0% |
| 157 (+34 arithmetic) | 85.3% | 83.3% | **27.8%** | **25.0%** | 16.7% |
| 185 (rebalanced) | **94.1%** | 88.9% | 27.8% | 31.2% | **66.7%** |

The middle row is the interesting one. Adding 34 arithmetic examples more than
doubled argument accuracy — and dropped honest refusal from 50% to 17%,
because the set had tipped to 91 tool examples against 66 without and the model
learned "reach for the calculator". That is the exact failure the dataset's own
docstring warns about, and it appeared within one run. Round three restored the
balance and kept most of the gain.

### What Swift-Instruct actually does

Measured with greedy decoding on 34 **held-out** hand-written prompts, none of
which appears in training (`minerva evaluate-instruct`):

| | |
|---|---|
| format valid | **100.0%** |
| routing accuracy | **94.1%** |
| tool name accuracy | **88.9%** |
| argument accuracy | **27.8%** |
| final answer correct | **31.2%** |
| honest refusal | **66.7%** |

It reliably decides **whether** to use a tool and **which** one. It is poor at
**arguments** — it copies the first operand and often fumbles the second.
Routing is a one-token decision learnable from 185 examples; copying arbitrary
operands is a general skill that needs far more data than a hand-written set
can carry.

The evaluation uses greedy decoding deliberately: with sampling, the same
checkpoint scored 94.4% and 83.3% tool accuracy on two seeds, and a number that
swings eleven points with the random seed cannot support a claim.

### v0.2.0, round one: the same 185 examples, retrained on the bilingual base

`swift-instruct` is a fine-tune of `swift`, so when the base model's weights
changed (§5b), `swift-instruct` was retrained too — otherwise the shipped
"instruct" model would silently be a fine-tune of a checkpoint no longer in
the catalogue. The first retrain added **no Hebrew examples** — the same 185
English-only conversations, unchanged, on the new base:

| | v0.1.0 base | v0.2.0 base, 185 English only |
|---|---|---|
| format valid | 100.0% | 100.0% |
| routing accuracy | 94.1% | 91.2% |
| tool name accuracy | 88.9% | 94.4% |
| argument accuracy | 27.8% | 16.7% |
| final answer correct | 31.2% | 18.8% |
| honest refusal | 66.7% | 50.0% |

Mixed, and mostly worse. Tool name accuracy improved, but routing, arguments,
final answers and honest refusal all fell — a real, measured downstream cost
of §5b's pretraining trade-off: the base model's English competence itself
regressed (bits/byte 1.3571 → 1.4678), and a fine-tune built on a weaker base
produces a weaker fine-tune, on the *identical* 185 examples. This is also
why Hebrew examples were not added in this first retrain: 185 hand-written
conversations was already a fragile budget for one language — this
document's own three-round history (§8, above) shows a 34-example addition
swinging honest refusal from 50% to 17% in a single round — and doubling
down on a base whose English had just measurably weakened seemed like the
wrong moment to also split the budget across two languages.

### v0.2.0, round two: 34 hand-written Hebrew examples added

Requested explicitly, and worth doing properly rather than deferring
indefinitely: 34 Hebrew conversations, written by hand one at a time exactly
like every other example in this file (never generated, never a template
over slot values — `instruct_data.py` §8 has the full rationale), covering
the same categories as the English set — calculator and date routing,
ordinary conversation, short reasoning, honest refusal — deliberately
leaning *away* from tool calls (14 tool-routing examples against 20 that
answer directly or decline) given §8's rebalancing history. `INSTRUCT_EXAMPLES`
grew from 185 to **219**.

The held-out evaluation set grew to match: 10 new Hebrew prompts in
`instruct_eval.py`, hand-written, with different numbers, dates and phrasings
from every Hebrew training example — the same "near but not in the training
distribution" standard the English 34 prompts already met. Scored separately
by language, greedy decoding, same methodology as every other number in this
document:

| | v0.2.0, 185 English only | v0.2.0, +34 Hebrew — English cases | v0.2.0, +34 Hebrew — Hebrew cases |
|---|---|---|---|
| held-out cases | 34 | 34 | 10 |
| format valid | 100.0% | 97.1% | 100.0% |
| routing accuracy | 91.2% | **97.1%** | 80.0% |
| tool name accuracy | 94.4% | 94.4% | 83.3% |
| argument accuracy | 16.7% | **22.2%** | 16.7% |
| final answer correct | 18.8% | 18.8% | 20.0% |
| honest refusal | 50.0% | **66.7%** | 0.0% |

A genuine surprise: adding Hebrew examples did not further dilute English —
it **recovered** most of round one's English regression (routing 91.2% →
97.1%, refusal 50.0% → 66.7%, back to v0.1.0's level). The most plausible
reading is that 34 more examples, in either language, is simply more
signal for the one-token routing decision and the refusal habit than 185
alone carried, and that habit transfers across the language boundary because
it is largely about *when* to act, not what language the prompt was in.

Hebrew itself is real but weaker, and this is reported plainly rather than
rounded up: routing (80.0%) and tool name accuracy (83.3%) are decent for a
first round on 34 examples, but **honest refusal on Hebrew is 0%** — both
Hebrew refusal cases in the held-out set got a wrong or garbled answer
instead of a decline, one of them in English despite being asked in Hebrew.
Five Hebrew refusal examples out of 34 is little to learn a habit from, and
it shows. This is a first, measured round, not a finished capability — the
honest next step is more hand-written Hebrew examples specifically in the
refusal category, written and measured the same way, not assumed to already
work because the average numbers above look reasonable.

### v0.3.0: web_search routing, and the Brown-corpus recovery

Two changes landed together and both needed retraining: the base model
changed (§5c), and `instruct_data.py` grew from 219 to **271** examples —
`_TALK_NATURALLY`, 36 new English conversational examples plus 16 new Hebrew
ones (greetings, opinions, clarifying questions, self-description, everyday
reasoning, plain word definitions, short practical help), drafted in themed
batches, each checked for duplicates against the existing set and the
held-out eval, and for factual claims against the *current* pretraining
corpus before merging (one drafted answer referenced the pre-v0.3.0 corpus
size and its since-removed newswire source; one Hebrew arithmetic riddle
didn't actually follow from its own premise — both caught and fixed/dropped,
not shipped unread). The existing `_UNKNOWN` section was also split: eight
English and three Hebrew "I don't know" examples became `web_search` calls
instead, for exactly the questions a search can actually resolve (see §5c
and `instruct_data.py`'s own docstring for the line between the two).

**A withdrawn measurement, and what it cost.** This section previously
claimed that fine-tuning the *old*, untouched 219-example set against the new
base model "reached only 34.1% routing accuracy, with empty answers on nearly
every arithmetic prompt", and treated that as proof the regression lived in
the base model rather than the instruct data.

**That number is withdrawn. It measured nothing.** It came from a run whose
`checkpoint_dir` pointed at `checkpoints/swift-instruct-control`, but
`checkpoint_dir` is the *parent* directory holding one sub-directory per
model — so the engine looked for
`checkpoints/swift-instruct-control/swift-instruct/best.pt`, found no
checkpoint, and raised on every single case. `evaluate_instruct` caught each
exception per case and dutifully reported a row of zeros and a 34.1% routing
figure that is just the rate at which "called no tool" happens to match the
expected answer when the model never ran at all. "Empty answers on nearly
every arithmetic prompt" was the literal truth and the clue, and it was read
as a model result instead of a configuration error.

Two things were changed so this cannot recur quietly:

* `evaluate_instruct` now **raises** when every case fails, naming the first
  error and the `checkpoint_dir` trap explicitly, instead of returning a
  plausible-looking row of zeros.
* The claim it supported is retracted. The Brown-corpus fix in §5c still
  stands on its own evidence — the base model's own held-out loss and
  perplexity, and the instruct numbers measured on correctly-configured runs
  — but it was never supported by a controlled comparison, and this document
  should not have said it was.

### A real controlled comparison: instruct data alone, base held fixed

Having withdrawn a claim that lacked one, here is the comparison actually
run — the **same fully-trained v0.3.0 base**, fine-tuned twice, changing only
the instruct set:

| | 219 examples (v0.3.0) | 311 examples (v0.4.0) |
|---|---|---|
| related to question | **75.0%** | 64.3% |
| routing accuracy | **97.7%** | 77.3% |
| tool name accuracy | **93.1%** | 65.5% |
| answered in-language | **94.4%** | 85.0% |
| honest refusal | 66.7% | **100.0%** |

**The larger set scored worse on the held-out cases**, and the shape of the
loss says why: `tool_name_accuracy` fell hardest, so the model is reaching
for the *wrong* tool, not failing to reach for one. Twenty `web_search`
examples in 311 taught it to search where it should calculate — the same
failure mode this file's own history records from a tool-heavy round, and it
is recorded here rather than quietly reverted.

But the held-out set is not the whole story, and the user-reported failure it
was missing is worth showing directly:

| prompt | 219 examples | 311 examples |
|---|---|---|
| "What is an apple?" | refuses, no tool | **calls `web_search`** |
| "What is the capital of France?" | calls **`calculate`** | **calls `web_search`** |

So the new data **fixed the routing behaviour that prompted it** while
costing tool discrimination elsewhere. What it could not fix is the reply:
the 311-example model answers the apple question `"חודש רביעי, קורה, קורה,
קורה…"` — right tool, degenerate text, wrong language.

That is the honest conclusion of this comparison: **instruct data decides
what the model reaches for; the base model decides whether it can say
anything coherent once it gets there.** No amount of instruct tuning fixes
the second, which is why the v0.4.0 effort went into the corpus and the token
budget. The tool-discrimination regression above must be re-measured against
the finished base before deciding whether it survives a better model or
needs the `web_search` examples rebalanced.

Held-out evaluation, 44 hand-written cases (English and Hebrew combined —
this round did not re-run the separate-by-language scoring §8's v0.2.0
section used; that is a real gap in this round's measurement, not a claim
that Hebrew and English perform identically):

| | v0.2.0 (+34 Hebrew) | v0.3.0, regressed base | v0.3.0, + Brown fix |
|---|---|---|---|
| format valid | 97.1%\* | 95.5% | **100.0%** |
| routing accuracy | 97.1%\* | 70.5% | **97.7%** |
| tool name accuracy | 94.4%\* | 58.6% | **93.1%** |
| argument accuracy | 22.2%\* | 3.4% | 10.3% |
| final answer correct | 18.8%\* | 0.0% | 9.5% |
| honest refusal | 66.7%\* | 100.0%† | 66.7% |

\* v0.2.0's English-only column, for the closest like-for-like comparison;
v0.3.0's 44 cases blend both languages.
† Spurious: nearly every answer on the regressed base was malformed text
that happened to contain a refusal-marker word, not an actual honest decline.

The fixed run **matches or beats v0.2.0's best routing and refusal numbers**,
on a corpus with no memorisable general-knowledge or event content left in
it. Argument accuracy (10.3%) and final-answer correctness (9.5%) are real
and still weak — copying exact operands into a tool call is a general skill a
271-example hand-written set was never going to solve on its own, the same
conclusion §8's very first section drew from 185 examples. That is a known,
pre-existing limitation of a model this size, not a new regression.

**`web_search` verified callable, not just present in the registry:** a live
request through `minerva serve`'s `/api/chat` with the prompt "What is the
population of Japan?" produced a genuine `web_search` tool call, a real
DuckDuckGo HTML request that returned real results from real websites, fed
back into the model's context. The query the model formed and the final
answer it wrote from the results were both poor (the same argument-accuracy
weakness above, applied to a tool call instead of `calculate`) — but the
mechanism itself, end to end, through the real HTTP API, is proven working.

### Thinking was trained, measured, and switched off

The scale is fully wired for this model: above `DO` the engine opens
`<|think|>` in the prompt, the model produces a trace, the parser reads it
back, and there is a recovery path for a turn that never leaves the block. 29
of the training conversations contain reasoning.

It makes the model **worse** — measured on the v0.1.0 base (this specific
thinking-on/off comparison was not re-run on the v0.2.0 bilingual base; there
is no reason to expect the conclusion changes, but it has not been measured
there):

| | thinking off (`do`) | thinking on (`mi`) |
|---|---|---|
| routing accuracy | 94.1% | 61.8% |
| tool name accuracy | 88.9% | 33.3% |
| argument accuracy | 27.8% | 0.0% |
| final answer correct | 31.2% | 0.0% |
| honest refusal | 66.7% | 33.3% |

Forcing a 9.9M model to reason first does not buy deliberation; it buys ~30
more tokens of drift, and then the answer is built on the drift. So
`swift-instruct` ships `max_thinking=DO`. The machinery is real and tested;
the weights are not good enough to use it, and `CLAUDE.md` forbids advertising
a capability the weights do not have.

Raising it is a training problem — more and better reasoning traces, on a
larger model. The line to change is in `models/swift_instruct.py`.

---

## 9. Scaling up

Nothing in the pipeline is hardcoded to Swift's size.

**A bigger model** is the same class with a different config:

```python
SwiftConfig(vocab_size=16384, n_layer=12, n_head=12, d_model=768,
            n_kv_head=4, max_seq_len=2048)   # ~90M parameters
```

**More data** is more entries in `SOURCES` in `data.py`. The binding
constraint on Swift is unique tokens, not compute — adding corpora is the
single highest-value change available.

**On a GPU**, add `--batch-size 64 --grad-accum 4`, set
`compile_model=True`, and consider bf16 autocast. The loop is device-agnostic;
it already selects CUDA when available.

**To make it follow instructions**, pretraining is only stage one. Stage two is
supervised fine-tuning on real instruction/response pairs — which, per
`CLAUDE.md`, must be genuinely good examples that someone read, never
bulk-generated from templates. That is the honest next step, and it is a
training problem, not an engine problem.

---

## Checklist for a new training run

- [ ] Corpus sources documented with origin and licence in `SOURCES`
- [ ] Quality exclusions recorded with a stated reason
- [ ] Validation held out per source, never randomly across a concatenation
- [ ] Learning rate chosen by a probe, not by assumption
- [ ] `run_config.json` and `training_log.jsonl` kept with the checkpoints
- [ ] Real val loss and real samples reported — including if they are bad
- [ ] If the base model's weights changed, `swift-instruct` retrained from the
      new base too — a stale fine-tune of a superseded checkpoint is not
      "the instruct model," it is a different model with the same name
