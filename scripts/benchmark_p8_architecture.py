"""Run controlled P8 architecture and tokenizer micro-experiments."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from minigpt.bpe_tokenizer import BPETokenizer
from minigpt.model import MiniGPT, count_parameters
from minigpt.tokenizer import CharacterTokenizer
from minigpt.trainer import compute_language_model_loss

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Variant:
    name: str
    position_encoding: str = "learned"
    norm: str = "layernorm"
    activation: str = "gelu"
    attention_backend: str = "manual"
    qkv_projection: str = "separate"
    qk_norm: bool = False
    num_kv_heads: int | None = None


VARIANTS = (
    Variant("classic"),
    Variant("rope_only", position_encoding="rope"),
    Variant("rmsnorm_only", norm="rmsnorm"),
    Variant("swiglu_only", activation="swiglu"),
    Variant("sdpa_only", attention_backend="sdpa"),
    Variant("fused_qkv_only", qkv_projection="fused"),
    Variant("qk_norm_only", qk_norm=True),
    Variant("gqa_only", num_kv_heads=2),
    Variant(
        "modern_combined",
        position_encoding="rope",
        norm="rmsnorm",
        activation="swiglu",
        attention_backend="sdpa",
        qkv_projection="fused",
    ),
    Variant(
        "modern_qk_gqa",
        position_encoding="rope",
        norm="rmsnorm",
        activation="swiglu",
        attention_backend="sdpa",
        qkv_projection="fused",
        qk_norm=True,
        num_kv_heads=2,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs" / "p8_architecture_benchmark.json",
    )
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用。")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def fixed_batch(
    token_ids: list[int], batch_size: int, block_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = []
    stride = max(1, (len(token_ids) - block_size - 1) // batch_size)
    for index in range(batch_size):
        start = index * stride
        rows.append(token_ids[start : start + block_size + 1])
    batch = torch.tensor(rows, dtype=torch.long, device=device)
    return batch[:, :-1], batch[:, 1:]


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_variant(
    variant: Variant,
    *,
    vocab_size: int,
    train_batch: tuple[torch.Tensor, torch.Tensor],
    val_batch: tuple[torch.Tensor, torch.Tensor],
    block_size: int,
    steps: int,
    device: torch.device,
) -> dict[str, object]:
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
        torch.cuda.reset_peak_memory_stats(device)
    model = MiniGPT(
        vocab_size=vocab_size,
        block_size=block_size,
        embedding_dim=128,
        num_heads=4,
        num_layers=4,
        expansion_factor=4,
        position_encoding=variant.position_encoding,
        norm=variant.norm,
        activation=variant.activation,
        attention_backend=variant.attention_backend,
        qkv_projection=variant.qkv_projection,
        qk_norm=variant.qk_norm,
        num_kv_heads=variant.num_kv_heads,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    x_train, y_train = train_batch
    x_val, y_val = val_batch

    model.eval()
    with torch.no_grad():
        _, initial_val_loss = compute_language_model_loss(model, x_val, y_val)
    synchronize(device)
    started = time.perf_counter()
    last_loss = 0.0
    for _ in range(steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        _, loss = compute_language_model_loss(model, x_train, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_loss = float(loss.detach())
    synchronize(device)
    elapsed = time.perf_counter() - started
    model.eval()
    with torch.no_grad():
        _, final_val_loss = compute_language_model_loss(model, x_val, y_val)
    tokens = steps * x_train.numel()
    return {
        "variant": asdict(variant),
        "parameters": count_parameters(model),
        "steps": steps,
        "processed_tokens": tokens,
        "train_loss_last": last_loss,
        "validation_loss_initial": float(initial_val_loss),
        "validation_loss_final": float(final_val_loss),
        "validation_bpb_final": float(final_val_loss) / math.log(2),
        "elapsed_seconds": elapsed,
        "tokens_per_second": tokens / elapsed,
        "peak_memory_mib": (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda"
            else None
        ),
    }


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.batch_size <= 0 or args.block_size <= 0:
        raise ValueError("steps、batch-size、block-size 必须是正整数。")
    device = choose_device(args.device)
    train_text = (ROOT / "data/processed/train.txt").read_text(encoding="utf-8")
    val_text = (ROOT / "data/processed/val.txt").read_text(encoding="utf-8")
    tokenizer = CharacterTokenizer.load(ROOT / "data/processed/tokenizer.json")
    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text, allow_unknown=True)
    train_batch = fixed_batch(train_ids, args.batch_size, args.block_size, device)
    val_batch = fixed_batch(val_ids, args.batch_size, args.block_size, device)

    results = [
        run_variant(
            variant,
            vocab_size=tokenizer.vocab_size,
            train_batch=train_batch,
            val_batch=val_batch,
            block_size=args.block_size,
            steps=args.steps,
            device=device,
        )
        for variant in VARIANTS
    ]

    bpe_training_text = train_text[:100_000]
    bpe = BPETokenizer.train(bpe_training_text, vocab_size=256)
    bpe_ids = bpe.encode(bpe_training_text)
    payload = {
        "date": "2026-08-25",
        "author": "李林知",
        "status": (
            "controlled_fixed_batch_short_budget"
            if args.steps >= 300
            else "micro_benchmark_not_full_training"
        ),
        "device": str(device),
        "torch_version": torch.__version__,
        "controlled_budget": {
            "seed": 42,
            "batch_size": args.batch_size,
            "block_size": args.block_size,
            "steps": args.steps,
            "architecture_embedding_dim": 128,
            "architecture_layers": 4,
            "architecture_heads": 4,
        },
        "architecture_results": results,
        "tokenizer_comparison": {
            "characters": len(bpe_training_text),
            "character_vocab_size": tokenizer.vocab_size,
            "character_tokens": len(bpe_training_text),
            "bpe_vocab_size": bpe.vocab_size,
            "bpe_tokens": len(bpe_ids),
            "characters_per_bpe_token": len(bpe_training_text) / len(bpe_ids),
            "token_reduction_rate": 1 - len(bpe_ids) / len(bpe_training_text),
            "roundtrip_ok": bpe.decode(bpe_ids) == bpe_training_text,
        },
        "limitations": [
            "架构结果是固定 Batch 的短步数 micro-benchmark，不替代 2,000 步正式训练。",
            "单 seed、单 GPU，不外推为所有硬件结论。",
            "各变体参数量并非完全相同，吞吐与 Loss 需要结合参数量解释。",
            "BPE 只比较压缩与可逆性，尚未完成相同字符预算的正式 Loss/BPB 对照。",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
