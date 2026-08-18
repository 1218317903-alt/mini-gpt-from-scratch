"""训练 LoRA、保存 Adapter、全新恢复并保存前后对比。"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from torch.utils.data import DataLoader

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE_ROOT))

from hf_lora.config import load_experiment_config  # noqa: E402
from hf_lora.generation import generate_response  # noqa: E402
from hf_lora.modeling import (  # noqa: E402
    DTYPES,
    estimate_lora_parameters,
    inject_lora,
    load_base_model,
    load_tokenizer,
    summarize_parameters,
)
from hf_lora.training import (  # noqa: E402
    evaluate_token_weighted_loss,
    set_seed,
    train_lora,
)

from hf_lora.data import SFTDataCollator, SFTDataset, load_jsonl  # noqa: E402


def resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def evaluate_prompts(model, tokenizer, prompts, *, device, max_new_tokens):
    results = []
    for prompt in prompts:
        messages = [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ]
        results.append(
            {
                "category": prompt["category"],
                "prompt": prompt["user"],
                "response": generate_response(
                    model,
                    tokenizer,
                    messages,
                    device=device,
                    max_new_tokens=max_new_tokens,
                ),
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "hf_lora/configs/qwen2_5_0_5b_lora.yaml",
    )
    parser.add_argument(
        "--model-source",
        help="可传本地快照目录实现严格离线加载。",
    )
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    source = args.model_source or config.model.name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("本教学训练脚本需要 CUDA GPU。")
    set_seed(config.training.seed)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = load_tokenizer(source)
    collator = SFTDataCollator(tokenizer.pad_token_id)
    train_dataset = SFTDataset(
        load_jsonl(resolve_project_path(config.paths.train_data)),
        tokenizer,
        config.model.max_length,
    )
    validation_dataset = SFTDataset(
        load_jsonl(resolve_project_path(config.paths.validation_data)),
        tokenizer,
        config.model.max_length,
    )
    generator = torch.Generator().manual_seed(config.training.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.micro_batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.training.micro_batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    amp_dtype = DTYPES[config.model.dtype]
    base_model = load_base_model(source, dtype_name=config.model.dtype, device=device)
    baseline_loss = evaluate_token_weighted_loss(
        base_model,
        validation_loader,
        device=device,
        amp_dtype=amp_dtype,
    )
    before = evaluate_prompts(
        base_model,
        tokenizer,
        config.generation.prompts,
        device=device,
        max_new_tokens=config.generation.max_new_tokens,
    )
    matched, estimated = estimate_lora_parameters(
        base_model, config.lora.target_modules, config.lora.rank
    )
    model = inject_lora(
        base_model,
        rank=config.lora.rank,
        alpha=config.lora.alpha,
        dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
    )
    summary = summarize_parameters(model)
    if summary.trainable != estimated:
        raise RuntimeError(
            f"LoRA 参数估算与实测不一致：{estimated} != {summary.trainable}"
        )
    history = train_lora(
        model,
        train_loader,
        validation_loader,
        device=device,
        amp_dtype=amp_dtype,
        config=config.training,
    )
    final_loss = evaluate_token_weighted_loss(
        model,
        validation_loader,
        device=device,
        amp_dtype=amp_dtype,
    )
    after = evaluate_prompts(
        model,
        tokenizer,
        config.generation.prompts,
        device=device,
        max_new_tokens=config.generation.max_new_tokens,
    )
    fixed_messages = [
        {"role": "system", "content": config.generation.prompts[0]["system"]},
        {"role": "user", "content": config.generation.prompts[0]["user"]},
    ]
    fixed_inputs = tokenizer.apply_chat_template(
        fixed_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)
    model.eval()
    with torch.no_grad():
        logits_before_save = model(**fixed_inputs).logits.detach().cpu()
    artifact_directory = resolve_project_path(config.paths.artifact_directory)
    adapter_directory = artifact_directory / "adapter"
    artifact_directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_directory, safe_serialization=True)
    tokenizer.save_pretrained(adapter_directory)
    del model, base_model
    gc.collect()
    torch.cuda.empty_cache()

    fresh_base = load_base_model(source, dtype_name=config.model.dtype, device=device)
    restored = PeftModel.from_pretrained(
        fresh_base, adapter_directory, is_trainable=False
    )
    restored.eval()
    with torch.no_grad():
        logits_after_load = restored(**fixed_inputs).logits.detach().cpu()
    maximum_logit_difference = float(
        (logits_before_save - logits_after_load).abs().max().item()
    )
    if not torch.allclose(logits_before_save, logits_after_load, atol=1e-5, rtol=1e-4):
        raise RuntimeError(
            f"Adapter 恢复 logits 不一致，最大误差 {maximum_logit_difference}。"
        )
    restored_results = evaluate_prompts(
        restored,
        tokenizer,
        config.generation.prompts,
        device=device,
        max_new_tokens=config.generation.max_new_tokens,
    )
    comparison = {
        "model": config.model.name,
        "adapter_directory": str(adapter_directory),
        "parameter_summary": summary.__dict__,
        "target_module_count": matched,
        "baseline_validation_loss": baseline_loss,
        "final_validation_loss": final_loss,
        "maximum_reload_logit_difference": maximum_logit_difference,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "generation_config": {
            "do_sample": False,
            "max_new_tokens": config.generation.max_new_tokens,
        },
        "results": [
            {
                "category": first["category"],
                "prompt": first["prompt"],
                "before": first["response"],
                "after": second["response"],
                "restored": third["response"],
            }
            for first, second, third in zip(
                before, after, restored_results, strict=True
            )
        ],
    }
    (artifact_directory / "training_metrics.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (artifact_directory / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
