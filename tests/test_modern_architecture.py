"""P8 modern architecture variants remain shape- and cache-compatible."""

import pytest
import torch

from minigpt.attention import (
    FusedQKVProjection,
    apply_rotary_position_embedding,
    repeat_kv_heads,
    rms_normalize_heads,
)
from minigpt.model import FeedForward, MiniGPT, RMSNorm


def test_rms_norm_produces_unit_root_mean_square() -> None:
    norm = RMSNorm(8)
    values = torch.randn(2, 3, 8)
    output = norm(values)
    rms = output.float().pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)


def test_swiglu_feed_forward_preserves_shape() -> None:
    layer = FeedForward(embedding_dim=16, expansion_factor=2, activation="swiglu")
    inputs = torch.randn(2, 5, 16)
    assert layer(inputs).shape == inputs.shape


def test_fused_qkv_projection_returns_three_equal_shapes() -> None:
    projection = FusedQKVProjection(16)
    outputs = projection(torch.randn(2, 5, 16))
    assert [item.shape for item in outputs] == [(2, 5, 16)] * 3


def test_fused_gqa_projection_uses_compact_kv_dimensions() -> None:
    projection = FusedQKVProjection(16, kv_dim=8)
    query, key, value = projection(torch.randn(2, 5, 16))
    assert query.shape == (2, 5, 16)
    assert key.shape == value.shape == (2, 5, 8)


def test_qk_norm_produces_unit_rms_per_head() -> None:
    values = torch.randn(2, 4, 5, 8)
    normalized = rms_normalize_heads(values)
    rms = normalized.float().pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)


def test_repeat_kv_heads_preserves_group_mapping() -> None:
    values = torch.tensor([[[[1.0]], [[2.0]]]])
    expanded = repeat_kv_heads(values, num_query_heads=4)
    assert expanded[:, :, 0, 0].tolist() == [[1.0, 1.0, 2.0, 2.0]]


def test_rope_preserves_shape_and_position_zero_first_pair() -> None:
    query = torch.randn(2, 4, 3, 8)
    key = torch.randn(2, 4, 3, 8)
    rotated_query, rotated_key = apply_rotary_position_embedding(query, key)
    assert rotated_query.shape == query.shape
    assert rotated_key.shape == key.shape
    assert torch.allclose(rotated_query[:, :, 0], query[:, :, 0])


def test_all_modern_options_support_forward_and_kv_cache() -> None:
    torch.manual_seed(7)
    model = MiniGPT(
        vocab_size=32,
        block_size=16,
        embedding_dim=32,
        num_heads=4,
        num_layers=2,
        expansion_factor=2,
        position_encoding="rope",
        norm="rmsnorm",
        activation="swiglu",
        attention_backend="sdpa",
        qkv_projection="fused",
    ).eval()
    token_ids = torch.randint(0, 32, (2, 7))
    full_logits = model(token_ids)
    prefix_logits, cache = model(token_ids[:, :6], use_cache=True)
    next_logits, next_cache = model(
        token_ids[:, 6:], past_key_values=cache, use_cache=True
    )
    assert full_logits.shape == (2, 7, 32)
    assert prefix_logits.shape == (2, 6, 32)
    assert next_logits.shape == (2, 1, 32)
    assert all(item[0].shape[2] == 7 for item in next_cache)
    assert torch.allclose(full_logits[:, -1:], next_logits, atol=1e-5)


@pytest.mark.parametrize("attention_backend", ["manual", "sdpa"])
def test_qk_norm_and_gqa_support_attention_weights_and_cache(
    attention_backend: str,
) -> None:
    torch.manual_seed(11)
    model = MiniGPT(
        vocab_size=32,
        block_size=16,
        embedding_dim=32,
        num_heads=4,
        num_kv_heads=2,
        num_layers=2,
        expansion_factor=2,
        position_encoding="rope",
        norm="rmsnorm",
        activation="swiglu",
        attention_backend=attention_backend,
        qkv_projection="fused",
        qk_norm=True,
    ).eval()
    token_ids = torch.randint(0, 32, (2, 7))
    full_logits, weights = model(token_ids, output_attentions=True)
    _, cache = model(token_ids[:, :6], use_cache=True)
    next_logits, next_cache = model(
        token_ids[:, 6:], past_key_values=cache, use_cache=True
    )

    assert weights[0].shape == (2, 4, 7, 7)
    assert cache[0][0].shape == (2, 2, 6, 8)
    assert next_cache[0][0].shape == (2, 2, 7, 8)
    assert torch.allclose(full_logits[:, -1:], next_logits, atol=1e-5)


def test_model_rejects_incompatible_gqa_head_count() -> None:
    with pytest.raises(ValueError, match="num_kv_heads"):
        MiniGPT(16, 8, 24, 6, 1, num_kv_heads=4)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("position_encoding", "bad"),
        ("norm", "bad"),
        ("activation", "bad"),
        ("attention_backend", "bad"),
        ("qkv_projection", "bad"),
    ],
)
def test_model_rejects_unknown_architecture_options(name: str, value: str) -> None:
    arguments = {name: value}
    with pytest.raises(ValueError):
        MiniGPT(16, 8, 16, 4, 1, **arguments)
