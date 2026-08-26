"""P8 modern architecture variants remain shape- and cache-compatible."""

import pytest
import torch

from minigpt.attention import (
    FusedQKVProjection,
    apply_rotary_position_embedding,
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
