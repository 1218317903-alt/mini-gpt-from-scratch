from __future__ import annotations

import pytest

from minigpt.distributed import (
    DistributedEnvironment,
    DistributedTrainingPlan,
    distributed_hardware_advice,
    shard_indices,
)


def test_distributed_environment_reads_torchrun_contract(monkeypatch) -> None:
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")

    environment = DistributedEnvironment.from_environment()

    assert environment == DistributedEnvironment(rank=1, local_rank=1, world_size=2)


def test_effective_global_batch_includes_every_data_parallel_rank() -> None:
    plan = DistributedTrainingPlan(
        strategy="ddp",
        world_size=4,
        micro_batch_size=2,
        gradient_accumulation_steps=8,
        backend="nccl",
    )

    assert plan.effective_global_batch_size == 64


def test_round_robin_shards_do_not_overlap() -> None:
    first = set(shard_indices(10, rank=0, world_size=2))
    second = set(shard_indices(10, rank=1, world_size=2))

    assert not first & second
    assert first | second == set(range(10))


def test_single_gpu_advice_forbids_multi_gpu_performance_claims() -> None:
    advice = distributed_hardware_advice(1)

    assert any("Do not report multi-GPU speedup" in item for item in advice)


def test_invalid_rank_is_rejected() -> None:
    with pytest.raises(ValueError, match="inside world_size"):
        DistributedEnvironment(rank=2, local_rank=0, world_size=2).validate()
