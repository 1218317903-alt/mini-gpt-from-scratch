"""Two-process CPU/Gloo DDP smoke test for teaching on a single-GPU machine."""

from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        return int(server.getsockname()[1])


def _worker(rank: int, world_size: int, port: int, output_directory: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(42)
        model = DistributedDataParallel(torch.nn.Linear(2, 1, bias=False))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        inputs = torch.tensor([[float(rank + 1), 1.0]])
        targets = torch.tensor([[float(2 * rank + 1)]])
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(inputs), targets)
        loss.backward()
        gradient = next(model.parameters()).grad
        assert gradient is not None
        gathered_gradients = [torch.zeros_like(gradient) for _ in range(world_size)]
        dist.all_gather(gathered_gradients, gradient)
        optimizer.step()
        parameter = next(model.parameters()).detach()
        gathered_parameters = [torch.zeros_like(parameter) for _ in range(world_size)]
        dist.all_gather(gathered_parameters, parameter)
        maximum_gradient_difference = max(
            float((item - gathered_gradients[0]).abs().max())
            for item in gathered_gradients
        )
        maximum_parameter_difference = max(
            float((item - gathered_parameters[0]).abs().max())
            for item in gathered_parameters
        )
        payload = {
            "rank": rank,
            "local_loss_before_all_reduce": float(loss.detach()),
            "synchronized_gradient": gradient.tolist(),
            "updated_parameter": parameter.tolist(),
            "maximum_gradient_difference_across_ranks": maximum_gradient_difference,
            "maximum_parameter_difference_across_ranks": maximum_parameter_difference,
        }
        path = Path(output_directory) / f"rank-{rank}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    finally:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("output/ddp_cpu_smoke"),
    )
    args = parser.parse_args()
    if args.world_size < 2:
        raise ValueError("world_size must be at least two for the DDP smoke test.")
    args.output_directory.mkdir(parents=True, exist_ok=True)
    port = _free_port()
    mp.spawn(
        _worker,
        args=(args.world_size, port, str(args.output_directory)),
        nprocs=args.world_size,
        join=True,
    )
    records = [
        json.loads((args.output_directory / f"rank-{rank}.json").read_text())
        for rank in range(args.world_size)
    ]
    summary = {
        "backend": "gloo",
        "device": "cpu",
        "world_size": args.world_size,
        "gradient_synchronized": all(
            record["maximum_gradient_difference_across_ranks"] == 0
            for record in records
        ),
        "parameters_equal_after_step": all(
            record["maximum_parameter_difference_across_ranks"] == 0
            for record in records
        ),
        "claim_boundary": (
            "This validates DDP mechanics only; it is not a GPU/NCCL speed benchmark."
        ),
        "ranks": records,
    }
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
