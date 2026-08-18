"""MiniGPT 模型单元测试。"""

import pytest
import torch

from minigpt.model import (
    FeedForward,
    MiniGPT,
    PositionEmbedding,
    TokenEmbedding,
    TransformerBlock,
    count_parameters,
)


B, T, C, H, V = 2, 4, 8, 2, 20


def create_tiny_model() -> MiniGPT:
    return MiniGPT(
        vocab_size=V,
        block_size=T,
        embedding_dim=C,
        num_heads=H,
        num_layers=2,
    )


def test_token_embedding_shape() -> None:
    embedding = TokenEmbedding(V, C)
    token_ids = torch.randint(0, V, (B, T), dtype=torch.long)

    output = embedding(token_ids)

    assert output.shape == (B, T, C)


def test_position_embedding_shape() -> None:
    position_embedding = PositionEmbedding(T, C)
    hidden_states = torch.randn(B, T, C)

    output = position_embedding(hidden_states)

    assert output.shape == (B, T, C)


def test_feed_forward_shape() -> None:
    feed_forward = FeedForward(C)
    hidden_states = torch.randn(B, T, C)

    output = feed_forward(hidden_states)

    assert output.shape == (B, T, C)


def test_transformer_block_shape() -> None:
    block = TransformerBlock(C, H, T)
    hidden_states = torch.randn(B, T, C)

    output = block(hidden_states)

    assert output.shape == (B, T, C)


def test_minigpt_logits_shape() -> None:
    model = create_tiny_model()
    token_ids = torch.randint(0, V, (B, T), dtype=torch.long)

    logits = model(token_ids)

    assert logits.shape == (B, T, V)


def test_minigpt_backward() -> None:
    model = create_tiny_model()
    token_ids = torch.randint(0, V, (B, T), dtype=torch.long)

    logits = model(token_ids)
    loss = logits.square().mean()
    loss.backward()

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    assert trainable_parameters
    assert all(
        parameter.grad is not None
        for parameter in trainable_parameters
    )


def test_minigpt_rejects_long_input() -> None:
    model = create_tiny_model()
    token_ids = torch.ones(B, T + 1, dtype=torch.long)

    with pytest.raises(ValueError, match="超过 block_size"):
        model(token_ids)


def test_minigpt_does_not_mix_batches() -> None:
    model = create_tiny_model()
    model.eval()

    first_input = torch.randint(0, V, (B, T), dtype=torch.long)
    second_input = first_input.clone()
    second_input[0] = torch.randint(0, V, (T,), dtype=torch.long)

    first_logits = model(first_input)
    second_logits = model(second_input)

    assert torch.allclose(
        first_logits[1],
        second_logits[1],
    )


def test_weight_tying_is_optional() -> None:
    untied_model = MiniGPT(V, T, C, H, 1, tie_weights=False)
    tied_model = MiniGPT(V, T, C, H, 1, tie_weights=True)

    assert untied_model.lm_head.weight is not (
        untied_model.token_embedding.embedding.weight
    )
    assert tied_model.lm_head.weight is (
        tied_model.token_embedding.embedding.weight
    )
    assert count_parameters(tied_model) < count_parameters(untied_model)
