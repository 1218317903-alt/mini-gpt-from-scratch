"""Report which post-training and distributed experiments this machine can run."""

from __future__ import annotations

import json
import platform
from typing import Any

import torch

from minigpt.distributed import distributed_hardware_advice


def main() -> None:
    devices = []
    for index in range(torch.cuda.device_count()):
        properties: Any = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "memory_mib": int(properties.total_memory) // 1024**2,
            }
        )
    maximum_memory = max(
        (int(device["memory_mib"]) for device in devices),
        default=0,
    )
    payload = {
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "devices": devices,
        "recommended_experiments": {
            "dpo_0_5b_4bit_lora": maximum_memory >= 6144,
            "dpo_3b_4bit_lora": maximum_memory >= 10240,
            "gpu_ddp": torch.cuda.device_count() >= 2,
            "gpu_fsdp": torch.cuda.device_count() >= 2,
            "cpu_gloo_ddp_smoke": True,
        },
        "distributed_claim_boundary": distributed_hardware_advice(
            torch.cuda.device_count()
        ),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
