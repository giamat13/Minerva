"""Swift's architecture: a decoder-only transformer, written from scratch.

This is Minerva's own model definition - not a fine-tune, not a wrapper around
someone else's weights. Every parameter Swift has is learned by
`minerva.training.trainer` from the corpus in `minerva.training.data`.

The architecture is the modern decoder-only stack, chosen component by
component for what it actually buys at this scale:

* **Pre-norm with RMSNorm** - normalising the input to each sub-layer rather
  than its output keeps the residual stream a clean identity path, which is
  what makes deep stacks trainable without warmup tricks. RMSNorm drops
  LayerNorm's mean-centring: it costs less and, empirically, loses nothing.
* **RoPE** (rotary position embeddings) - position is injected by rotating
  queries and keys, so attention sees *relative* distance. No learned position
  table means nothing to over-fit and no hard length ceiling.
* **SwiGLU** feed-forward - a gated activation that reliably beats plain
  GELU/ReLU MLPs at equal parameter count.
* **Grouped-query attention** - fewer key/value heads than query heads,
  shrinking the KV cache during generation. At Swift's size it defaults to
  full multi-head (``n_kv_head == n_head``); the machinery is here so larger
  Minerva models can use it without an architecture change.
* **Tied embeddings** - the input embedding and the output projection share
  one matrix. On a model this small the vocabulary embedding is a third of all
  parameters, so tying is a large effective-capacity win.

Scaling up
----------
Nothing here is hardcoded to Swift's size. A larger Minerva model is the same
class with a different :class:`SwiftConfig` - see ``docs/TRAINING.md``.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import asdict, dataclass, replace
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SWIFT_CONFIG", "SwiftConfig", "SwiftLM"]


@dataclass(frozen=True)
class SwiftConfig:
    """Architecture hyper-parameters."""

    vocab_size: int = 8192
    n_layer: int = 6
    n_head: int = 8
    n_kv_head: int | None = None
    """Key/value heads for grouped-query attention. ``None`` = full multi-head."""
    d_model: int = 320
    d_ff: int | None = None
    """Feed-forward inner width. ``None`` derives ~8/3 x d_model, rounded to 64."""
    max_seq_len: int = 512
    rope_theta: float = 10_000.0
    dropout: float = 0.0
    """Zero is correct here: with 27 MB of text and ~4 epochs the model is
    data-limited, not over-fitting, and dropout would only slow learning."""
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.d_model % self.n_head != 0:
            raise ValueError(f"d_model {self.d_model} must divide by n_head {self.n_head}")
        if self.n_head % self.kv_heads != 0:
            raise ValueError(
                f"n_head {self.n_head} must be a multiple of n_kv_head {self.kv_heads}"
            )

    @property
    def kv_heads(self) -> int:
        return self.n_kv_head or self.n_head

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    @property
    def ff_dim(self) -> int:
        if self.d_ff is not None:
            return self.d_ff
        # SwiGLU has three matrices instead of two, so the usual 4x is scaled
        # by 2/3 to keep the parameter count comparable. Rounded to a multiple
        # of 64 to keep the GEMMs well-shaped.
        return 64 * round(self.d_model * 8 / 3 / 64)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> SwiftConfig:
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in fields})  # type: ignore[arg-type]


#: Swift's shipped architecture. Sized for the compute this project actually
#: has: ~7M non-embedding-heavy parameters trained on ~6.3M tokens for a few
#: epochs is close to compute-optimal on a 4-core CPU. See docs/TRAINING.md.
SWIFT_CONFIG = SwiftConfig(
    vocab_size=8192,
    n_layer=6,
    n_head=8,
    d_model=320,
    max_seq_len=512,
)


class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Computed in fp32 so the reciprocal square root stays accurate even
        # when the model runs in a lower precision.
        dtype = x.dtype
        x32 = x.float()
        normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (normed.to(dtype)) * self.weight


def build_rope_cache(
    seq_len: int, head_dim: int, theta: float, device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute the cos/sin tables for rotary position embeddings."""
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )
    positions = torch.arange(seq_len, dtype=torch.float32, device=device)
    angles = torch.outer(positions, inv_freq)  # (seq_len, head_dim/2)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate the pairs of a query/key tensor by their positional angle.

    ``x`` is ``(batch, heads, seq, head_dim)``; ``cos``/``sin`` are
    ``(seq, head_dim/2)`` slices covering exactly this window of positions.
    """
    x1, x2 = x.float().chunk(2, dim=-1)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated.to(x.dtype)


class Attention(nn.Module):
    """Causal self-attention with RoPE and optional grouped-query heads."""

    def __init__(self, config: SwiftConfig) -> None:
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_kv_head = config.kv_heads
        self.head_dim = config.head_dim

        self.q_proj = nn.Linear(config.d_model, self.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, self.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_head * self.head_dim, config.d_model, bias=False)
        self.dropout = config.dropout

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        batch, seq, _ = x.shape

        q = self.q_proj(x).view(batch, seq, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, self.n_kv_head, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if cache is not None:
            past_k, past_v = cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_cache = (k, v)

        # Grouped-query attention: repeat each kv head to cover its query group.
        if self.n_kv_head != self.n_head:
            repeat = self.n_head // self.n_kv_head
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        # With a KV cache, a single new query attends to the whole past, so the
        # causal mask is only needed when processing more than one position.
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=seq > 1,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, seq, -1)
        return self.o_proj(attended), new_cache


class SwiGLU(nn.Module):
    """Gated feed-forward network: ``down(silu(gate(x)) * up(x))``."""

    def __init__(self, config: SwiftConfig) -> None:
        super().__init__()
        hidden = config.ff_dim
        self.gate_proj = nn.Linear(config.d_model, hidden, bias=False)
        self.up_proj = nn.Linear(config.d_model, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    """One transformer block: pre-normed attention, then pre-normed SwiGLU."""

    def __init__(self, config: SwiftConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = Attention(config)
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = SwiGLU(config)
        self.dropout = nn.Dropout(config.dropout) if config.dropout else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        attended, new_cache = self.attn(self.attn_norm(x), cos, sin, cache)
        x = x + self.dropout(attended)
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x, new_cache


class SwiftLM(nn.Module):
    """The Swift language model."""

    def __init__(self, config: SwiftConfig) -> None:
        super().__init__()
        self.config = config

        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        cos, sin = build_rope_cache(config.max_seq_len, config.head_dim, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Scaled init for the residual projections: without it the residual
        # stream's variance grows with depth and early training is unstable.
        for name, param in self.named_parameters():
            if name.endswith(("o_proj.weight", "down_proj.weight")):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # -- parameter accounting --------------------------------------------

    def num_parameters(self, *, non_embedding: bool = False) -> int:
        """Total parameters; optionally excluding the (tied) token embedding."""
        total = sum(p.numel() for p in self.parameters())
        if non_embedding:
            total -= self.embed.weight.numel()
        return total

    # -- extending -------------------------------------------------------

    def resize_token_embeddings(self, new_vocab_size: int) -> None:
        """Grow the vocabulary, keeping every existing token's embedding.

        Fine-tuning adds chat markers to the tokenizer, so the model needs rows
        for them. The learned rows are copied across unchanged and only the new
        ones are initialised, which is what keeps a fine-tune a continuation of
        pretraining rather than a restart.

        New rows are initialised small rather than at the usual 0.02: an
        untrained row that starts as large as a trained one produces wild logits
        for the new tokens on the very first step.
        """
        old_size = self.config.vocab_size
        if new_vocab_size == old_size:
            return
        if new_vocab_size < old_size:
            raise ValueError(
                f"refusing to shrink the vocabulary from {old_size} to {new_vocab_size}; "
                f"that would discard learned embeddings"
            )

        embedding = nn.Embedding(new_vocab_size, self.config.d_model)
        nn.init.normal_(embedding.weight, mean=0.0, std=0.002)
        with torch.no_grad():
            embedding.weight[:old_size] = self.embed.weight
        self.embed = embedding

        if self.config.tie_embeddings:
            self.lm_head = nn.Linear(self.config.d_model, new_vocab_size, bias=False)
            self.lm_head.weight = self.embed.weight
        else:
            head = nn.Linear(self.config.d_model, new_vocab_size, bias=False)
            nn.init.normal_(head.weight, mean=0.0, std=0.002)
            with torch.no_grad():
                head.weight[:old_size] = self.lm_head.weight
            self.lm_head = head

        # The config is frozen, so replace it rather than mutating it.
        object.__setattr__(self, "config", replace(self.config, vocab_size=new_vocab_size))

    # -- forward ---------------------------------------------------------

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Run the model.

        :param idx: ``(batch, seq)`` token ids.
        :param targets: ``(batch, seq)`` next-token targets; when given, the
            cross-entropy loss is returned alongside the logits.
        :param caches: per-layer KV caches from a previous step, for generation.
        :param position_offset: where this window starts, so RoPE angles
            continue correctly when generating with a cache.
        """
        _batch, seq = idx.shape
        end = position_offset + seq
        if end > self.config.max_seq_len:
            raise ValueError(
                f"sequence position {end} exceeds max_seq_len {self.config.max_seq_len}"
            )

        # register_buffer types these as Tensor | Module to mypy; they are
        # always Tensors here.
        rope_cos = cast(torch.Tensor, self.rope_cos)
        rope_sin = cast(torch.Tensor, self.rope_sin)
        cos = rope_cos[position_offset:end]
        sin = rope_sin[position_offset:end]

        x = self.embed(idx)
        new_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for index, block in enumerate(self.blocks):
            layer_cache = caches[index] if caches is not None else None
            x, updated = block(x, cos, sin, layer_cache)
            if updated is not None:
                new_caches.append(updated)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
            )
        return logits, loss, new_caches

    # -- generation ------------------------------------------------------

    @torch.no_grad()
    def generate_stream(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 0.8,
        top_k: int | None = 40,
        top_p: float | None = 0.95,
        repetition_penalty: float = 1.0,
        eos_id: int | None = None,
        stop_ids: frozenset[int] | None = None,
        seed: int | None = None,
    ) -> Iterator[torch.Tensor]:
        """Sample a continuation, yielding one ``(batch, 1)`` token at a time.

        Uses a KV cache, so generation is linear in length rather than
        quadratic. This is the streaming core; :meth:`generate` collects it.

        ``stop_ids`` halts generation on any of several tokens, which is what a
        tool-calling turn needs: the model must stop at the close of a tool
        call so the tool can actually run, and only then continue with the real
        result appended. Generating past it would make the model invent the
        tool's output - the exact failure this project refuses to ship.
        """
        stops = set(stop_ids or ())
        if eos_id is not None:
            stops.add(eos_id)
        self.eval()
        generator = None
        if seed is not None:
            generator = torch.Generator(device=idx.device).manual_seed(seed)

        idx = self._truncate_prompt(idx)
        logits, _, caches = self(idx)
        offset = idx.size(1)
        produced = idx

        for _ in range(max_new_tokens):
            step_logits = logits[:, -1, :].float()

            if repetition_penalty != 1.0:
                step_logits = _apply_repetition_penalty(
                    step_logits, produced, repetition_penalty
                )

            if temperature <= 0:
                next_token = step_logits.argmax(dim=-1, keepdim=True)
            else:
                step_logits = step_logits / temperature
                step_logits = _filter_logits(step_logits, top_k=top_k, top_p=top_p)
                probs = F.softmax(step_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1, generator=generator)

            yield next_token
            produced = torch.cat([produced, next_token], dim=1)

            if stops and int(next_token[0, 0]) in stops:
                return
            if offset + 1 >= self.config.max_seq_len:
                return

            logits, _, caches = self(next_token, caches=caches, position_offset=offset)
            offset += 1

    def _truncate_prompt(self, idx: torch.Tensor) -> torch.Tensor:
        """Trim an over-long prompt from the left.

        The most recent context is what matters for the next token. Both
        :meth:`generate` and :meth:`generate_stream` apply this, so the
        sequence returned always matches what the model actually conditioned
        on - concatenating onto the untruncated prompt would silently return
        text the model never saw.
        """
        if idx.size(1) >= self.config.max_seq_len:
            return idx[:, -(self.config.max_seq_len - 1) :]
        return idx

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, **kwargs: object) -> torch.Tensor:
        """Sample a continuation. Returns the prompt plus the generated tokens."""
        idx = self._truncate_prompt(idx)
        pieces = list(self.generate_stream(idx, max_new_tokens, **kwargs))  # type: ignore[arg-type]
        if not pieces:
            return idx
        return torch.cat([idx, *pieces], dim=1)


def _apply_repetition_penalty(
    logits: torch.Tensor, generated: torch.Tensor, penalty: float
) -> torch.Tensor:
    """Divide the logits of already-generated tokens, discouraging loops."""
    for batch_index in range(generated.size(0)):
        unique = torch.unique(generated[batch_index])
        scores = logits[batch_index, unique]
        # Negative logits must be multiplied, not divided, to be pushed down.
        logits[batch_index, unique] = torch.where(scores > 0, scores / penalty, scores * penalty)
    return logits


def _filter_logits(
    logits: torch.Tensor, *, top_k: int | None, top_p: float | None
) -> torch.Tensor:
    """Apply top-k then nucleus (top-p) filtering."""
    if top_k is not None and 0 < top_k < logits.size(-1):
        threshold = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    if top_p is not None and 0.0 < top_p < 1.0:
        ordered, indices = torch.sort(logits, descending=True, dim=-1)
        cumulative = torch.cumsum(F.softmax(ordered, dim=-1), dim=-1)
        remove = cumulative - F.softmax(ordered, dim=-1) > top_p
        remove[..., 0] = False  # always keep the most likely token
        ordered = ordered.masked_fill(remove, float("-inf"))
        logits = torch.empty_like(logits).scatter_(-1, indices, ordered)

    return logits
