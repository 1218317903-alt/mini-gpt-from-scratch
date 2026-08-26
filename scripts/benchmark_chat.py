"""Record chat-backend latency and optional CUDA memory for local baselines."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from agent.llm import HuggingFaceChatModel
from agent.messages import Message, ToolSchema
from minigpt_service.config import ChatServiceConfig
from minigpt_service.model_service import select_device


def _model_identity(source: str) -> tuple[str, str | None]:
    path = Path(source)
    for part in path.parts:
        if part.startswith("models--"):
            model = part.removeprefix("models--").replace("--", "/")
            revision = path.name if path.parent.name == "snapshots" else None
            return model, revision
    return source, None


def _portable_path(path: Path | None) -> str | None:
    if path is None:
        return None
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _memory_mb(device: torch.device) -> float | None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 2)


def benchmark(config: ChatServiceConfig) -> dict[str, Any]:
    """Load a chat backend and time first-token plus short completion."""
    device = select_device(config.device)
    backend = HuggingFaceChatModel(config, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    load_started = time.perf_counter()
    backend.load()
    load_ms = (time.perf_counter() - load_started) * 1000

    messages = [
        Message(role="system", content="只用一个数字回答。"),
        Message(role="user", content="17乘以6等于多少？"),
    ]
    first = backend.chat(
        messages,
        max_new_tokens=1,
        temperature=0.7,
        top_k=None,
        top_p=None,
        greedy=True,
        seed=42,
    )
    full = backend.chat(
        messages,
        max_new_tokens=16,
        temperature=0.7,
        top_k=None,
        top_p=None,
        greedy=True,
        seed=42,
    )
    structured = backend.chat(
        [Message(role="user", content="请使用 calculator 工具计算 17*6。")],
        [
            ToolSchema(
                name="calculator",
                description="Evaluate a simple arithmetic expression.",
                parameters={
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                    "additionalProperties": False,
                },
            )
        ],
        max_new_tokens=64,
        temperature=0.7,
        top_k=None,
        top_p=None,
        greedy=True,
        seed=42,
    )
    model, revision = _model_identity(config.model_source)
    payload = {
        "model": model,
        "model_revision": revision,
        "adapter_path": _portable_path(config.adapter_path),
        "device": str(device),
        "dtype": config.dtype,
        "quantization": config.quantization,
        "load_ms": round(load_ms, 3),
        "first_token_ms": first.duration_ms,
        "completion_ms": full.duration_ms,
        "prompt_tokens": full.prompt_tokens,
        "generated_tokens": full.generated_tokens,
        "peak_memory_mb": _memory_mb(device),
        "sample_text": full.text,
        "structured_tool_text": structured.text,
        "structured_tool_json_valid": bool(structured.tool_calls),
        "structured_tool_name": (
            structured.tool_calls[0].name if structured.tool_calls else None
        ),
        "structured_tool_arguments": (
            structured.tool_calls[0].arguments if structured.tool_calls else None
        ),
    }
    backend.close()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/stage1_chat_benchmark.json"),
    )
    args = parser.parse_args()
    config = ChatServiceConfig.from_environment()
    if config is None:
        raise SystemExit("Set MINIGPT_CHAT_MODEL_SOURCE before running the benchmark.")
    payload = benchmark(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
