"""Tests for Swift's architecture.

These build real (tiny) models and run real forward and backward passes. The
properties asserted here are the ones that silently break a training run if
they regress: causality, the KV cache, RoPE, initialisation scale, and whether
gradients actually reach every parameter.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch", reason="needs the 'training' extra")

from minerva.training.model import (  # noqa: E402
    SWIFT_CONFIG,
    SwiftConfig,
    SwiftLM,
    apply_rope,
    build_rope_cache,
)

pytestmark = pytest.mark.torch

# Small enough to be fast, large enough to exercise every code path.
TINY = SwiftConfig(vocab_size=256, n_layer=2, n_head=4, d_model=64, max_seq_len=32)


@pytest.fixture
def model() -> SwiftLM:
    torch.manual_seed(0)
    return SwiftLM(TINY)


class TestConfig:
    def test_head_dim_divides_the_model_width(self) -> None:
        assert TINY.head_dim * TINY.n_head == TINY.d_model

    def test_an_indivisible_width_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must divide"):
            SwiftConfig(d_model=100, n_head=8)

    def test_ff_dim_defaults_to_two_thirds_of_four_x(self) -> None:
        # SwiGLU has three matrices, so 4x is scaled by 2/3 to keep the
        # parameter count comparable to a plain MLP.
        assert SwiftConfig(d_model=320).ff_dim == 832

    def test_grouped_query_attention_requires_divisible_heads(self) -> None:
        # 8 query heads cannot be split evenly into 3 kv groups.
        with pytest.raises(ValueError, match="multiple of"):
            SwiftConfig(d_model=64, n_head=8, n_kv_head=3)

    def test_config_round_trips_through_a_dict(self) -> None:
        assert SwiftConfig.from_dict(TINY.to_dict()) == TINY

    def test_from_dict_ignores_unknown_keys(self) -> None:
        # Checkpoints from a newer version must still load.
        data = {**TINY.to_dict(), "future_option": 3}
        assert SwiftConfig.from_dict(data) == TINY


class TestForward:
    def test_output_shape_is_batch_seq_vocab(self, model: SwiftLM) -> None:
        idx = torch.randint(0, TINY.vocab_size, (3, 16))
        logits, loss, _ = model(idx)
        assert logits.shape == (3, 16, TINY.vocab_size)
        assert loss is None

    def test_loss_is_returned_when_targets_are_given(self, model: SwiftLM) -> None:
        idx = torch.randint(0, TINY.vocab_size, (2, 16))
        _, loss, _ = model(idx, targets=idx)
        assert loss is not None and loss.ndim == 0

    def test_initial_loss_matches_a_uniform_distribution(self) -> None:
        # A correctly initialised model knows nothing, so its loss must be
        # ln(vocab_size). A materially lower value means the output layer is
        # mis-scaled and training will start from a broken place.
        torch.manual_seed(0)
        big = SwiftLM(SwiftConfig(vocab_size=2048, n_layer=2, n_head=4, d_model=64))
        idx = torch.randint(0, 2048, (8, 64))
        targets = torch.randint(0, 2048, (8, 64))
        _, loss, _ = big(idx, targets=targets)
        assert loss is not None
        assert abs(loss.item() - math.log(2048)) < 0.3

    def test_a_sequence_past_the_context_limit_is_refused(self, model: SwiftLM) -> None:
        idx = torch.randint(0, TINY.vocab_size, (1, TINY.max_seq_len + 1))
        with pytest.raises(ValueError, match="exceeds max_seq_len"):
            model(idx)


class TestCausality:
    def test_a_token_cannot_see_the_future(self, model: SwiftLM) -> None:
        """The property the whole objective depends on.

        If attention leaks future tokens, training loss collapses and the model
        is worthless at generation. Change a later token and the logits at an
        earlier position must not move at all.
        """
        model.eval()
        idx = torch.randint(0, TINY.vocab_size, (1, 12))
        with torch.no_grad():
            baseline, _, _ = model(idx)

            altered = idx.clone()
            altered[0, 8] = (altered[0, 8] + 1) % TINY.vocab_size
            changed, _, _ = model(altered)

        assert torch.allclose(baseline[0, :8], changed[0, :8], atol=1e-5)
        assert not torch.allclose(baseline[0, 8], changed[0, 8], atol=1e-5)


class TestRoPE:
    def test_the_cache_has_one_angle_per_position_pair(self) -> None:
        cos, sin = build_rope_cache(16, 8, 10_000.0)
        assert cos.shape == (16, 4)
        assert sin.shape == (16, 4)

    def test_position_zero_is_the_identity_rotation(self) -> None:
        cos, sin = build_rope_cache(4, 8, 10_000.0)
        x = torch.randn(1, 1, 1, 8)
        assert torch.allclose(apply_rope(x, cos[:1], sin[:1]), x, atol=1e-6)

    def test_rotation_preserves_vector_norm(self) -> None:
        # RoPE is a rotation, so it must not change magnitudes.
        cos, sin = build_rope_cache(8, 16, 10_000.0)
        x = torch.randn(2, 3, 8, 16)
        rotated = apply_rope(x, cos, sin)
        assert torch.allclose(x.norm(dim=-1), rotated.norm(dim=-1), atol=1e-5)

    def test_attention_depends_on_relative_distance(self) -> None:
        """The point of RoPE: the same pair of tokens, shifted together,
        should produce a similar attention score."""
        cos, sin = build_rope_cache(32, 8, 10_000.0)
        q = torch.randn(1, 1, 1, 8)
        k = torch.randn(1, 1, 1, 8)

        def score(qi: int, ki: int) -> float:
            qr = apply_rope(q, cos[qi : qi + 1], sin[qi : qi + 1])
            kr = apply_rope(k, cos[ki : ki + 1], sin[ki : ki + 1])
            return float((qr * kr).sum())

        assert abs(score(5, 3) - score(15, 13)) < 1e-4


class TestKVCache:
    def test_cached_decoding_matches_a_full_forward_pass(self, model: SwiftLM) -> None:
        """If this drifts, generation quietly diverges from training."""
        model.eval()
        idx = torch.randint(0, TINY.vocab_size, (1, 10))
        with torch.no_grad():
            full, _, _ = model(idx)

            prefix, _, caches = model(idx[:, :-1])
            last, _, _ = model(idx[:, -1:], caches=caches, position_offset=9)

        assert torch.allclose(full[:, -1], last[:, 0], atol=1e-4)
        assert torch.allclose(full[:, :-1], prefix, atol=1e-4)

    def test_the_cache_grows_by_one_position_per_step(self, model: SwiftLM) -> None:
        model.eval()
        idx = torch.randint(0, TINY.vocab_size, (1, 4))
        with torch.no_grad():
            _, _, caches = model(idx)
            assert caches[0][0].shape[2] == 4
            _, _, caches = model(idx[:, :1], caches=caches, position_offset=4)
            assert caches[0][0].shape[2] == 5

    def test_there_is_one_cache_per_layer(self, model: SwiftLM) -> None:
        _, _, caches = model(torch.randint(0, TINY.vocab_size, (1, 4)))
        assert len(caches) == TINY.n_layer


class TestGroupedQueryAttention:
    def test_fewer_kv_heads_still_produces_the_right_shape(self) -> None:
        config = SwiftConfig(
            vocab_size=128, n_layer=1, n_head=8, n_kv_head=2, d_model=64, max_seq_len=16
        )
        model = SwiftLM(config)
        logits, _, caches = model(torch.randint(0, 128, (2, 8)))
        assert logits.shape == (2, 8, 128)
        # The cache holds kv heads, not query heads - that is the saving.
        assert caches[0][0].shape[1] == 2

    def test_it_uses_fewer_parameters_than_full_attention(self) -> None:
        base = dict(vocab_size=128, n_layer=2, n_head=8, d_model=64, max_seq_len=16)
        full = SwiftLM(SwiftConfig(**base))
        grouped = SwiftLM(SwiftConfig(**base, n_kv_head=2))
        assert grouped.num_parameters() < full.num_parameters()


class TestParametersAndGradients:
    def test_embeddings_are_tied_by_default(self, model: SwiftLM) -> None:
        assert model.lm_head.weight is model.embed.weight

    def test_untying_adds_a_second_matrix(self) -> None:
        untied = SwiftLM(SwiftConfig(**{**TINY.to_dict(), "tie_embeddings": False}))
        assert untied.lm_head.weight is not untied.embed.weight
        assert untied.num_parameters() > SwiftLM(TINY).num_parameters()

    def test_non_embedding_count_excludes_the_vocabulary_matrix(
        self, model: SwiftLM
    ) -> None:
        difference = model.num_parameters() - model.num_parameters(non_embedding=True)
        assert difference == TINY.vocab_size * TINY.d_model

    def test_gradients_reach_every_parameter(self, model: SwiftLM) -> None:
        """A parameter with no gradient is dead weight that never learns."""
        idx = torch.randint(0, TINY.vocab_size, (2, 8))
        _, loss, _ = model(idx, targets=idx)
        assert loss is not None
        loss.backward()

        missing = [
            name
            for name, param in model.named_parameters()
            if param.requires_grad and (param.grad is None or param.grad.abs().sum() == 0)
        ]
        assert not missing, f"parameters received no gradient: {missing}"

    def test_a_single_step_reduces_the_loss_on_one_batch(self, model: SwiftLM) -> None:
        """The end-to-end sanity check: the thing can actually learn."""
        idx = torch.randint(0, TINY.vocab_size, (4, 16))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

        _, first, _ = model(idx, targets=idx)
        assert first is not None
        for _ in range(20):
            _, loss, _ = model(idx, targets=idx)
            assert loss is not None
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        _, final, _ = model(idx, targets=idx)
        assert final is not None
        assert final.item() < first.item() * 0.8


class TestGeneration:
    def test_it_produces_the_requested_number_of_tokens(self, model: SwiftLM) -> None:
        prompt = torch.randint(0, TINY.vocab_size, (1, 4))
        out = model.generate(prompt, 10, temperature=0.8, seed=0)
        assert out.shape == (1, 14)
        assert torch.equal(out[:, :4], prompt), "the prompt must be preserved"

    def test_streaming_and_batch_generation_agree(self, model: SwiftLM) -> None:
        prompt = torch.randint(0, TINY.vocab_size, (1, 4))
        batched = model.generate(prompt, 8, temperature=0.8, seed=42)
        streamed = list(model.generate_stream(prompt, 8, temperature=0.8, seed=42))
        assert torch.equal(batched[:, 4:], torch.cat(streamed, dim=1))

    def test_greedy_decoding_is_deterministic(self, model: SwiftLM) -> None:
        prompt = torch.randint(0, TINY.vocab_size, (1, 4))
        first = model.generate(prompt, 8, temperature=0.0)
        second = model.generate(prompt, 8, temperature=0.0)
        assert torch.equal(first, second)

    def test_the_same_seed_reproduces_a_sample(self, model: SwiftLM) -> None:
        prompt = torch.randint(0, TINY.vocab_size, (1, 4))
        first = model.generate(prompt, 8, temperature=1.0, seed=7)
        second = model.generate(prompt, 8, temperature=1.0, seed=7)
        assert torch.equal(first, second)

    def test_it_stops_at_the_end_of_text_token(self, model: SwiftLM) -> None:
        prompt = torch.randint(0, TINY.vocab_size, (1, 4))
        # Force token 5 to always win, then ask it to stop on token 5.
        with torch.no_grad():
            model.lm_head.weight[5] += 100.0
        out = model.generate(prompt, 20, temperature=0.0, eos_id=5)
        assert out.shape[1] == 5, "generation must halt on the first EOS"

    def test_top_k_restricts_the_sampled_set(self, model: SwiftLM) -> None:
        prompt = torch.randint(0, TINY.vocab_size, (1, 4))
        out = model.generate(prompt, 30, temperature=2.0, top_k=1, seed=0)
        greedy = model.generate(prompt, 30, temperature=0.0)
        # top_k=1 is greedy regardless of temperature.
        assert torch.equal(out, greedy)

    def test_it_never_runs_past_the_context_window(self, model: SwiftLM) -> None:
        prompt = torch.randint(0, TINY.vocab_size, (1, TINY.max_seq_len - 4))
        out = model.generate(prompt, 100, temperature=0.8, seed=0)
        assert out.shape[1] <= TINY.max_seq_len

    def test_an_over_long_prompt_is_truncated_rather_than_crashing(
        self, model: SwiftLM
    ) -> None:
        """The returned sequence must be what the model actually conditioned on.

        Regression: `generate` used to concatenate onto the untruncated prompt
        while the streaming core truncated internally, returning text the model
        had never seen.
        """
        prompt = torch.randint(0, TINY.vocab_size, (1, TINY.max_seq_len + 20))
        out = model.generate(prompt, 2, temperature=0.8, seed=0)
        assert out.shape[1] <= TINY.max_seq_len
        # The kept context is the tail of the prompt, not the head.
        assert torch.equal(out[0, : TINY.max_seq_len - 1], prompt[0, -(TINY.max_seq_len - 1) :])


class TestShippedConfig:
    def test_swifts_config_is_internally_consistent(self) -> None:
        assert SWIFT_CONFIG.d_model % SWIFT_CONFIG.n_head == 0
        assert SWIFT_CONFIG.max_seq_len == 512
        assert SWIFT_CONFIG.vocab_size == 8192

    def test_it_is_the_size_the_documentation_claims(self) -> None:
        model = SwiftLM(SWIFT_CONFIG)
        total = model.num_parameters()
        assert 9.5e6 < total < 10.5e6, f"Swift is {total:,} parameters"
