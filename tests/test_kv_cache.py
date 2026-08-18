"""KV Cache correctness tests for stage five."""

import pytest
import torch

from minigpt.generation import generate
from minigpt.model import MiniGPT
from minigpt.utils import compile_model


def make_model(*, block_size: int = 12) -> MiniGPT:
    torch.manual_seed(7)
    model = MiniGPT(
        vocab_size=17,
        block_size=block_size,
        embedding_dim=16,
        num_heads=4,
        num_layers=2,
        expansion_factor=2,
        dropout=0.0,
    )
    model.eval()
    return model


def test_prefill_cache_has_one_entry_per_layer() -> None:
    model = make_model()
    prompt = torch.tensor([[1, 2, 3], [4, 5, 6]])

    logits, cache = model(prompt, use_cache=True)

    assert logits.shape == (2, 3, 17)
    assert len(cache) == model.num_layers
    for key, value in cache:
        assert key.shape == (2, 4, 3, 4)
        assert value.shape == key.shape


def test_decode_appends_one_cache_position() -> None:
    model = make_model()
    _, cache = model(torch.tensor([[1, 2, 3]]), use_cache=True)

    logits, updated_cache = model(
        torch.tensor([[4]]),
        past_key_values=cache,
        use_cache=True,
    )

    assert logits.shape == (1, 1, 17)
    for old_layer, new_layer in zip(cache, updated_cache, strict=True):
        old_key, old_value = old_layer
        new_key, new_value = new_layer
        assert new_key.size(2) == old_key.size(2) + 1
        assert torch.equal(new_key[:, :, :-1], old_key)
        assert torch.equal(new_value[:, :, :-1], old_value)


def test_cached_decode_logits_match_full_forward() -> None:
    model = make_model()
    prompt = torch.tensor([[1, 2, 3, 4]])
    continuation = torch.tensor([[5, 6]])
    full_logits = model(torch.cat((prompt, continuation), dim=1))

    _, cache = model(prompt, use_cache=True)
    cached_logits, _ = model(
        continuation,
        past_key_values=cache,
        use_cache=True,
    )

    torch.testing.assert_close(
        cached_logits,
        full_logits[:, -continuation.size(1) :],
        atol=1e-5,
        rtol=1e-5,
    )


def test_cached_and_baseline_greedy_generation_match() -> None:
    model = make_model()
    prompt = torch.tensor([[1, 2, 3]])

    baseline = generate(
        model,
        prompt,
        max_new_tokens=5,
        block_size=model.block_size,
        do_sample=False,
        use_kv_cache=False,
    )
    cached = generate(
        model,
        prompt,
        max_new_tokens=5,
        block_size=model.block_size,
        do_sample=False,
        use_kv_cache=True,
    )

    assert torch.equal(cached, baseline)


def test_cache_and_attention_weights_can_be_returned_together() -> None:
    model = make_model()
    logits, cache, weights = model(
        torch.tensor([[1, 2, 3]]),
        use_cache=True,
        output_attentions=True,
    )

    assert logits.shape == (1, 3, 17)
    assert len(cache) == len(weights) == model.num_layers
    assert weights[0].shape == (1, model.num_heads, 3, 3)


def test_cache_rejects_context_overflow() -> None:
    model = make_model(block_size=4)
    prompt = torch.tensor([[1, 2, 3]])

    with pytest.raises(ValueError, match="prompt_length"):
        generate(
            model,
            prompt,
            max_new_tokens=2,
            block_size=model.block_size,
            use_kv_cache=True,
        )


def test_compile_model_disabled_preserves_original_module() -> None:
    model = make_model()

    assert compile_model(model, enabled=False) is model


def test_compile_model_uses_requested_mode(monkeypatch) -> None:
    model = make_model()
    calls: list[tuple[torch.nn.Module, str]] = []

    def fake_compile(module, *, mode):
        calls.append((module, mode))
        return module

    monkeypatch.setattr(torch, "compile", fake_compile)
    execution_model = compile_model(
        model,
        enabled=True,
        mode="reduce-overhead",
    )

    assert execution_model is model
    assert calls == [(model, "reduce-overhead")]
