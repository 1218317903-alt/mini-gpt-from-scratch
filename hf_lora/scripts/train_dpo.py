"""Run a small LoRA DPO experiment with a frozen reference model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import PeftModel
from torch.utils.data import DataLoader

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE_ROOT))

from hf_lora.modeling import (  # noqa: E402
    DTYPES,
    inject_lora,
    load_base_model,
    load_tokenizer,
    summarize_parameters,
)
from hf_lora.preference import (  # noqa: E402
    DPOTrainingConfig,
    PreferenceDataCollator,
    PreferenceDataset,
    evaluate_dpo,
    train_dpo,
)
from hf_lora.training import set_seed  # noqa: E402

from hf_lora.data import load_jsonl  # noqa: E402


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping.")
    return value


def _project_path(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small LoRA policy with DPO.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "hf_lora/configs/qwen2_5_0_5b_dpo_4060.yaml",
    )
    parser.add_argument("--model-source", help="Optional local model snapshot.")
    parser.add_argument(
        "--sft-adapter",
        type=Path,
        help="Optional SFT adapter used to initialize both policy and reference.",
    )
    args = parser.parse_args()
    payload = _mapping(
        yaml.safe_load(args.config.read_text(encoding="utf-8")), "config"
    )
    model_config = _mapping(payload.get("model"), "model")
    lora_config = _mapping(payload.get("lora"), "lora")
    dpo_config = _mapping(payload.get("dpo"), "dpo")
    paths = _mapping(payload.get("paths"), "paths")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("The configured DPO experiment requires a CUDA GPU.")
    if torch.cuda.get_device_properties(0).total_memory < 6 * 1024**3:
        raise RuntimeError("The 0.5B DPO recipe expects at least 6 GiB of GPU memory.")
    torch.cuda.reset_peak_memory_stats()

    source = args.model_source or str(model_config["name"])
    dtype_name = str(model_config.get("dtype", "float16"))
    quantization = str(model_config.get("quantization", "4bit"))
    max_length = int(model_config["max_length"])
    set_seed(int(dpo_config.get("seed", 42)))
    tokenizer = load_tokenizer(source)
    collator = PreferenceDataCollator(tokenizer.pad_token_id)
    train_dataset = PreferenceDataset(
        load_jsonl(_project_path(paths["train_data"])), tokenizer, max_length
    )
    validation_dataset = PreferenceDataset(
        load_jsonl(_project_path(paths["validation_data"])), tokenizer, max_length
    )
    generator = torch.Generator().manual_seed(int(dpo_config.get("seed", 42)))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(dpo_config["micro_batch_size"]),
        shuffle=True,
        collate_fn=collator,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(dpo_config["micro_batch_size"]),
        shuffle=False,
        collate_fn=collator,
    )

    policy_base = load_base_model(
        source,
        dtype_name=dtype_name,
        device=device,
        quantization=quantization,
    )
    reference_base = load_base_model(
        source,
        dtype_name=dtype_name,
        device=device,
        quantization=quantization,
    )
    if args.sft_adapter:
        policy = PeftModel.from_pretrained(
            policy_base,
            args.sft_adapter,
            is_trainable=True,
        )
        reference = PeftModel.from_pretrained(
            reference_base,
            args.sft_adapter,
            is_trainable=False,
        )
    else:
        policy = inject_lora(
            policy_base,
            rank=int(lora_config["rank"]),
            alpha=int(lora_config["alpha"]),
            dropout=float(lora_config["dropout"]),
            target_modules=tuple(str(item) for item in lora_config["target_modules"]),
            quantization=quantization,
        )
        reference = reference_base
    reference.eval()
    training_config = DPOTrainingConfig(
        beta=float(dpo_config["beta"]),
        learning_rate=float(dpo_config["learning_rate"]),
        weight_decay=float(dpo_config["weight_decay"]),
        max_optimizer_steps=int(dpo_config["max_optimizer_steps"]),
        gradient_accumulation_steps=int(dpo_config["gradient_accumulation_steps"]),
        max_grad_norm=float(dpo_config["max_grad_norm"]),
        warmup_steps=int(dpo_config.get("warmup_steps", 0)),
        minimum_lr_ratio=float(dpo_config.get("minimum_lr_ratio", 0.1)),
    )
    validation_before = evaluate_dpo(
        policy,
        reference,
        validation_loader,
        device=device,
        amp_dtype=DTYPES[dtype_name],
        beta=training_config.beta,
    )
    history = train_dpo(
        policy,
        reference,
        train_loader,
        device=device,
        amp_dtype=DTYPES[dtype_name],
        config=training_config,
    )
    validation_after = evaluate_dpo(
        policy,
        reference,
        validation_loader,
        device=device,
        amp_dtype=DTYPES[dtype_name],
        beta=training_config.beta,
    )
    artifact_directory = _project_path(paths["artifact_directory"])
    adapter_directory = artifact_directory / "adapter"
    artifact_directory.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(str(adapter_directory), safe_serialization=True)
    tokenizer.save_pretrained(str(adapter_directory))
    report = {
        "method": "DPO",
        "model": source,
        "reference": "same SFT adapter" if args.sft_adapter else "base model",
        "hardware": torch.cuda.get_device_name(0),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "parameter_summary": summarize_parameters(policy).__dict__,
        "config": dpo_config,
        "validation_before": validation_before,
        "validation_after": validation_after,
        "history": history,
        "limitations": [
            "This small teaching dataset is not evidence of general preference alignment.",
            "Run frozen regression evaluation before claiming a quality improvement.",
        ],
    }
    (artifact_directory / "dpo_training.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
