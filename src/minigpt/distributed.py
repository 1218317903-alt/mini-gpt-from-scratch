"""Hardware-aware distributed-training concepts used by tutorials and tests."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

DistributedStrategy = Literal["ddp", "fsdp"]


@dataclass(frozen=True)
class DistributedEnvironment:
    """The process identity convention populated by ``torchrun``."""

    rank: int
    local_rank: int
    world_size: int

    @classmethod
    def from_environment(cls) -> DistributedEnvironment:
        """Read and validate RANK, LOCAL_RANK, and WORLD_SIZE."""
        try:
            rank = int(os.getenv("RANK", "0"))
            local_rank = int(os.getenv("LOCAL_RANK", "0"))
            world_size = int(os.getenv("WORLD_SIZE", "1"))
        except ValueError as error:
            raise ValueError("distributed rank variables must be integers.") from error
        result = cls(rank=rank, local_rank=local_rank, world_size=world_size)
        result.validate()
        return result

    def validate(self) -> None:
        if self.world_size <= 0:
            raise ValueError("world_size must be positive.")
        if not 0 <= self.rank < self.world_size:
            raise ValueError("rank must be inside world_size.")
        if self.local_rank < 0:
            raise ValueError("local_rank cannot be negative.")


@dataclass(frozen=True)
class DistributedTrainingPlan:
    """Explicit batch and hardware assumptions for a teaching experiment."""

    strategy: DistributedStrategy
    world_size: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    backend: str

    def __post_init__(self) -> None:
        if self.strategy not in {"ddp", "fsdp"}:
            raise ValueError("strategy must be ddp or fsdp.")
        if (
            min(
                self.world_size,
                self.micro_batch_size,
                self.gradient_accumulation_steps,
            )
            <= 0
        ):
            raise ValueError("distributed batch values must be positive.")
        if self.backend not in {"gloo", "nccl"}:
            raise ValueError("backend must be gloo or nccl.")

    @property
    def effective_global_batch_size(self) -> int:
        """Return micro batch × accumulation × data-parallel processes."""
        return (
            self.micro_batch_size * self.gradient_accumulation_steps * self.world_size
        )


def shard_indices(
    dataset_size: int,
    *,
    rank: int,
    world_size: int,
) -> tuple[int, ...]:
    """Illustrate non-overlapping round-robin data-parallel shards."""
    if dataset_size < 0:
        raise ValueError("dataset_size cannot be negative.")
    environment = DistributedEnvironment(
        rank=rank,
        local_rank=rank,
        world_size=world_size,
    )
    environment.validate()
    return tuple(range(rank, dataset_size, world_size))


def distributed_hardware_advice(cuda_device_count: int) -> tuple[str, ...]:
    """State what can be honestly tested on the detected CUDA hardware."""
    if cuda_device_count < 0:
        raise ValueError("cuda_device_count cannot be negative.")
    if cuda_device_count >= 2:
        return (
            "Run single-node multi-GPU DDP correctness and throughput tests.",
            "Run FSDP memory and sharded-checkpoint experiments.",
            "Record NCCL scaling efficiency instead of assuming linear speedup.",
        )
    return (
        "Run the CPU/Gloo DDP smoke test to learn process and gradient semantics.",
        "Do not report multi-GPU speedup, NCCL behavior, or FSDP memory savings.",
        "Use the DDP/FSDP code walkthrough as teaching-only evidence.",
    )
