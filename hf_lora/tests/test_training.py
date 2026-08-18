from __future__ import annotations

import pytest
import torch
from hf_lora.training import build_scheduler


def test_warmup_cosine_scheduler_reaches_minimum_ratio() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = build_scheduler(
        optimizer, warmup_steps=2, total_steps=6, minimum_ratio=0.1
    )
    used = []
    for _ in range(6):
        used.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert used[0] == pytest.approx(0.5)
    assert max(used) == pytest.approx(1.0)
    assert used[-1] == pytest.approx(0.1)
