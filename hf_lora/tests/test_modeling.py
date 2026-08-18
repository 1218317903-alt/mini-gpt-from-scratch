from __future__ import annotations

from hf_lora.modeling import estimate_lora_parameters, summarize_parameters
from torch import nn


class TinyTargets(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(6, 4, bias=False)
        self.v_proj = nn.Linear(6, 2, bias=False)
        self.other = nn.Linear(6, 3, bias=False)


def test_lora_parameter_estimate_uses_real_linear_shapes() -> None:
    matched, parameters = estimate_lora_parameters(
        TinyTargets(), ("q_proj", "v_proj"), rank=2
    )
    assert matched == 2
    assert parameters == 2 * (6 + 4) + 2 * (6 + 2)


def test_parameter_summary_distinguishes_frozen_parameters() -> None:
    model = TinyTargets()
    model.other.weight.requires_grad = False
    summary = summarize_parameters(model)
    assert summary.total == 24 + 12 + 18
    assert summary.trainable == 24 + 12
    assert summary.frozen == 18
