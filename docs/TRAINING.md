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

**~27 MB of real, human-written English prose.** Nothing is generated,
templated or augmented — `CLAUDE.md` forbids it, and a model trained on
synthetic filler learns the filler.

| Source | Size | Register | Licence |
|---|---|---|---|
| Project Gutenberg (18 books) | 11.7 MB | Literary prose and verse | Public domain |
| Reuters newswire (1987) | 7.8 MB | Factual reporting | Reuters-21578, research use |
| European Parliament (English) | 3.1 MB | Transcribed debate | Freely available |
| State of the Union addresses | 2.1 MB | Political oratory | Public domain |
| NLTK web text | 1.7 MB | Informal, conversational | Freely redistributable |
| Inaugural addresses | 0.8 MB | Formal oratory | Public domain |

The mixture is deliberate. A model trained only on 19th-century novels writes
only 19th-century novels; the newswire and web text are what give Swift any
contemporary register at all.

**The corpus is not vendored.** `data.py` downloads each source from its
original distributor and writes a `manifest.json` recording counts, character
totals, SHA-256 hashes and the licence of every source. Provenance stays
verifiable and we redistribute nobody's corpus.

### Two corpora were rejected

Both were available, real, and would have added ~17 MB. Both were excluded on
quality grounds, which is exactly the judgement `CLAUDE.md` asks for:

* **Brown corpus** — distributed POS-tagged (`The/at Fulton/np-tl`).
  De-tagging is mechanical but leaves unnatural spacing around punctuation.
* **Movie reviews (Pang & Lee)** — distributed lowercased and pre-tokenised
  (`films adapted from comic books , whether they 're`). Casing and spacing are
  destroyed.

Volume was not a good enough reason to teach the model damaged typography.

### The split

Held out **per source, by character count**, so validation carries every
register in the same proportion as training. Splitting by document count
instead would have handed validation almost entirely to Reuters, which is
8,578 of the corpus's documents but only a quarter of its text. Documents are
first chunked to ≤16 KB at paragraph boundaries so the split unit is roughly
uniform. Result: 26.97 MB train / 0.35 MB validation.

---

## 2. The tokenizer

`src/minerva/training/tokenizer.py`

A **byte-level BPE**, implemented and trained here — 8,192 tokens, learned from
the corpus above in 95 seconds.

* **Byte-level base vocabulary.** All 256 byte values are tokens, so every
  possible input encodes. There is no unknown token and no text Swift cannot
  represent.
* **GPT-2-style pre-tokenization**, so merges never straddle a space boundary.
* **Digits are always split individually.** Grouping `1987` into one token
  destroys place-value consistency and makes number handling markedly worse.
* **`<|endoftext|>` is a real token**, so the model learns that documents end.

Training uses the standard fast path: merges are computed over the ~113k
*unique* pre-token types weighted by frequency, and pair counts are updated
incrementally through a pair → containing-words index rather than recounted
after every merge.

Measured compression on the corpus: **3.66 characters per token** →
**7,377,901 training tokens**, 94,856 validation tokens.

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

Dropout is **0**, deliberately: with 27 MB of text the model is data-limited,
not over-fitting, and dropout would only slow learning.

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

The corpus holds 7.4M unique tokens, so 60M tokens means ~8 epochs. Repeating
data is near-lossless up to ~4 epochs and degrades after, so the run was
balanced at **9.9M parameters × 36.9M tokens ≈ 5 epochs** — slightly
over-parameterised relative to Chinchilla, which is the right trade when
unique data, not compute, is the binding constraint.

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
on 27 MB, Swift is roughly the scale of a small research baseline — it learns
grammar, vocabulary, register and local coherence. It is not a chatbot and
should not be described as one.

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

**185 conversations, every one written by hand.** No script, no templates, no
permuted slot values — `CLAUDE.md` forbids all three, and this is where that
rule bites hardest.

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

### Thinking was trained, measured, and switched off

The scale is fully wired for this model: above `DO` the engine opens
`<|think|>` in the prompt, the model produces a trace, the parser reads it
back, and there is a recovery path for a turn that never leaves the block. 29
of the training conversations contain reasoning.

It makes the model **worse**:

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
