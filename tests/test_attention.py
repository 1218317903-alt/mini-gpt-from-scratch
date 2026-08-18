"""Attention 单元测试。"""

import torch

from minigpt.attention import (
    MultiHeadCausalSelfAttention,
    QKVProjection,
    apply_causal_mask,
    merge_heads,
    scaled_dot_product_attention,
    scaled_dot_product_scores,
    split_heads,
)

B, T, C, H, D = 2, 4, 8, 2, 4


def test_qkv_shapes() -> None:
    projection = QKVProjection(C)
    hidden_states = torch.randn(B, T, C)
    query, key, value = projection(hidden_states)

    assert query.shape == (B, T, C)
    assert key.shape == (B, T, C)
    assert value.shape == (B, T, C)


def test_split_and_merge_heads() -> None:
    hidden_states = torch.randn(B, T, C)
    split = split_heads(hidden_states, H)
    merged = merge_heads(split)

    assert split.shape == (B, H, T, D)
    assert merged.shape == (B, T, C)
    assert torch.allclose(hidden_states, merged)


def test_attention_scores_shape() -> None:
    query = torch.randn(B, H, T, D)
    key = torch.randn(B, H, T, D)
    scores = scaled_dot_product_scores(query, key)

    assert scores.shape == (B, H, T, T)


def test_causal_mask_blocks_future_positions() -> None:
    scores = torch.zeros(B, H, T, T)
    masked_scores = apply_causal_mask(scores)
    _, weights = scaled_dot_product_attention(
        torch.randn(B, H, T, D),
        torch.randn(B, H, T, D),
        torch.randn(B, H, T, D),
    )

    assert torch.isneginf(masked_scores[..., 0, 1:]).all()
    assert (weights[..., 0, 1:] == 0).all()
    assert (weights[..., 1, 2:] == 0).all()
    assert torch.allclose(
        weights.sum(dim=-1),
        torch.ones(B, H, T),
    )


def test_attention_output_shape() -> None:
    query = torch.randn(B, H, T, D)
    key = torch.randn(B, H, T, D)
    value = torch.randn(B, H, T, D)
    output, weights = scaled_dot_product_attention(
        query,
        key,
        value,
    )

    assert output.shape == (B, H, T, D)
    assert weights.shape == (B, H, T, T)


def test_multi_head_attention_shape() -> None:
    attention = MultiHeadCausalSelfAttention(
        embedding_dim=C,
        num_heads=H,
        block_size=T,
    )
    hidden_states = torch.randn(B, T, C)
    output = attention(hidden_states)

    assert output.shape == (B, T, C)


def test_attention_does_not_mix_batches() -> None:
    attention = MultiHeadCausalSelfAttention(C, H, T)
    attention.eval()

    first_input = torch.randn(B, T, C)
    second_input = first_input.clone()
    second_input[0] = torch.randn(T, C)

    first_output = attention(first_input)
    second_output = attention(second_input)

    assert torch.allclose(
        first_output[1],
        second_output[1],
    )
