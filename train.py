"""阶段三训练入口。"""

import argparse
import logging
import math
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from minigpt.config import (
    DataConfig,
    ModelConfig,
    TrainingConfig,
    load_data_config,
    load_model_config,
    load_training_config,
)
from minigpt.dataset import build_language_model_dataset
from minigpt.experiment import (
    LOGGER_NAME,
    ExperimentRun,
    collect_environment,
    configure_training_logger,
    create_run_id,
)
from minigpt.model import MiniGPT, count_parameters
from minigpt.tokenizer import ProjectTokenizer, load_tokenizer
from minigpt.trainer import (
    create_cuda_grad_scaler,
    evaluate_metrics,
    train_accumulation_step,
    train_step,
)
from minigpt.utils import (
    WarmupCosineScheduler,
    compile_model,
    load_checkpoint,
    read_utf8_text,
    save_checkpoint,
    set_random_seed,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "tiny_shakespeare.yaml"
LOGGER = logging.getLogger(LOGGER_NAME)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 MiniGPT。")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="YAML 配置文件路径。",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="从指定 Checkpoint 恢复训练。",
    )
    parser.add_argument(
        "--skip-overfit",
        action="store_true",
        help="跳过极小 Batch 过拟合测试。",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT / "checkpoints",
        help="Checkpoint 输出目录。",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="实验记录目录；默认自动创建 runs/<run-id>。",
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="可选的训练步数覆盖，适合短程冒烟或受限硬件实验。",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="连续多少次验证无改善后停止；0 表示关闭。",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="验证 Loss 至少改善多少才重置 Early Stopping。",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=None,
        help="可选的验证间隔覆盖，适合早停和短程实验。",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=None,
        help="可选的验证 Batch 数覆盖。",
    )
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
    )
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument(
        "--warmup-start-factor",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--cosine-decay",
        action="store_true",
    )
    parser.add_argument(
        "--min-learning-rate",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16", "bf16"),
        default="fp32",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="使用 torch.compile 编译模型执行路径。",
    )
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
    )
    return parser.parse_args()


def select_device() -> torch.device:
    """按 CUDA、MPS、CPU 的优先级选择设备。"""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def build_model(
    tokenizer: ProjectTokenizer,
    data_config: DataConfig,
    model_config: ModelConfig,
    dropout: float = 0.0,
) -> MiniGPT:
    """按照配置创建 MiniGPT。"""
    return MiniGPT(
        vocab_size=tokenizer.vocab_size,
        block_size=data_config.block_size,
        embedding_dim=model_config.embedding_dim,
        num_heads=model_config.num_heads,
        num_layers=model_config.num_layers,
        expansion_factor=model_config.expansion_factor,
        tie_weights=model_config.tie_weights,
        dropout=dropout,
        position_encoding=model_config.position_encoding,
        norm=model_config.norm,
        activation=model_config.activation,
        attention_backend=model_config.attention_backend,
        qkv_projection=model_config.qkv_projection,
        qk_norm=model_config.qk_norm,
        num_kv_heads=model_config.num_kv_heads,
    )


def apply_runtime_overrides(
    args: argparse.Namespace,
    data_config: DataConfig,
    model_config: ModelConfig,
) -> tuple[DataConfig, ModelConfig]:
    """应用实验命令行覆盖，不修改原始 YAML 文件。"""
    original_num_heads = model_config.num_heads
    if args.block_size is not None:
        data_config = replace(data_config, block_size=args.block_size)
    if args.batch_size is not None:
        data_config = replace(data_config, batch_size=args.batch_size)

    model_changes = {
        name: value
        for name, value in {
            "embedding_dim": args.embedding_dim,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
        }.items()
        if value is not None
    }
    if model_changes:
        if (
            args.num_heads is not None
            and model_config.num_kv_heads == original_num_heads
        ):
            model_changes["num_kv_heads"] = args.num_heads
        model_config = replace(model_config, **model_changes)

    positive_values = {
        "block_size": data_config.block_size,
        "batch_size": data_config.batch_size,
        "embedding_dim": model_config.embedding_dim,
        "num_heads": model_config.num_heads,
        "num_layers": model_config.num_layers,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
    }
    if any(type(value) is not int or value <= 0 for value in positive_values.values()):
        raise ValueError("结构覆盖与梯度累积次数必须是正整数。")
    if model_config.embedding_dim % model_config.num_heads != 0:
        raise ValueError("embedding_dim 必须能被 num_heads 整除。")
    if (
        model_config.num_kv_heads is None
        or model_config.num_heads % model_config.num_kv_heads != 0
    ):
        raise ValueError("num_heads 必须能被 num_kv_heads 整除。")
    if (
        model_config.position_encoding == "rope"
        and model_config.embedding_dim // model_config.num_heads % 2 != 0
    ):
        raise ValueError("RoPE 要求 head_dim 为偶数。")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("dropout 必须位于 [0, 1) 中。")
    return data_config, model_config


def resolve_amp(
    precision: str,
    device: torch.device,
) -> tuple[torch.dtype | None, object | None]:
    """把精度配置转换成 autocast dtype 与可选 GradScaler。"""
    if precision == "fp32":
        return None, None
    if device.type != "cuda":
        raise RuntimeError("FP16/BF16 混合精度在本阶段只支持 CUDA。")
    if precision == "fp16":
        return torch.float16, create_cuda_grad_scaler()
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前 CUDA 设备不支持 BF16。")
    return torch.bfloat16, None


def normalize_config(
    config: DataConfig | ModelConfig | TrainingConfig,
) -> dict[str, object]:
    """把 dataclass 配置转换为可保存的字典。"""
    values = asdict(config)
    return {
        name: str(value) if isinstance(value, Path) else value
        for name, value in values.items()
    }


def build_tokenizer_info(
    tokenizer: ProjectTokenizer,
) -> dict[str, object]:
    """收集恢复训练所需的 Tokenizer 信息。"""
    return {
        "vocab_size": tokenizer.vocab_size,
        "itos": tokenizer.itos,
        "unk_token": tokenizer.UNK_TOKEN,
        "unk_id": tokenizer.unk_id,
        "eos_id": getattr(tokenizer, "eos_id", None),
        "pad_id": getattr(tokenizer, "pad_id", None),
        "tokenizer_type": type(tokenizer).__name__,
    }


def run_tiny_batch_overfit_test(
    tokenizer: ProjectTokenizer,
    data_config: DataConfig,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    train_loader: DataLoader,
    device: torch.device,
    dropout: float = 0.0,
) -> None:
    """固定一个小 Batch，确认模型和 Loss 可以学习。"""
    test_model = build_model(
        tokenizer=tokenizer,
        data_config=data_config,
        model_config=model_config,
        dropout=dropout,
    ).to(device)

    test_optimizer = torch.optim.AdamW(
        test_model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    fixed_inputs, fixed_targets = next(iter(train_loader))

    fixed_inputs = fixed_inputs[:1].to(device)
    fixed_targets = fixed_targets[:1].to(device)

    initial_loss: float | None = None
    final_loss: float | None = None
    report_interval = max(
        1,
        training_config.overfit_steps // 5,
    )

    LOGGER.info("开始极小 Batch 过拟合测试。")

    for step in range(
        1,
        training_config.overfit_steps + 1,
    ):
        loss, gradient_norm = train_step(
            model=test_model,
            optimizer=test_optimizer,
            inputs=fixed_inputs,
            targets=fixed_targets,
            grad_clip_norm=training_config.grad_clip_norm,
        )

        loss_value = float(loss.item())
        if initial_loss is None:
            initial_loss = loss_value
        final_loss = loss_value

        if step == 1 or step % report_interval == 0:
            LOGGER.info(
                f"overfit_step={step} "
                f"loss={loss_value:.4f} "
                f"gradient_norm={gradient_norm:.4f}"
            )

    if initial_loss is None or final_loss is None or final_loss >= initial_loss:
        raise RuntimeError("极小 Batch 过拟合失败：Loss 没有下降。")

    LOGGER.info(f"极小 Batch 过拟合通过：{initial_loss:.4f} -> {final_loss:.4f}")


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()

    data_config = load_data_config(config_path)
    model_config = load_model_config(config_path)
    training_config = load_training_config(config_path)
    if args.max_steps is not None:
        if args.max_steps <= 0:
            raise ValueError("max_steps 覆盖必须是正整数。")
        training_config = replace(training_config, max_steps=args.max_steps)
    if args.eval_interval is not None:
        if args.eval_interval <= 0:
            raise ValueError("eval_interval 覆盖必须是正整数。")
        training_config = replace(
            training_config,
            eval_interval=args.eval_interval,
        )
    if args.eval_steps is not None:
        if args.eval_steps <= 0:
            raise ValueError("eval_steps 覆盖必须是正整数。")
        training_config = replace(training_config, eval_steps=args.eval_steps)
    if args.early_stopping_patience < 0:
        raise ValueError("early_stopping_patience 不能为负数。")
    if (
        not math.isfinite(args.early_stopping_min_delta)
        or args.early_stopping_min_delta < 0
    ):
        raise ValueError("early_stopping_min_delta 必须是非负有限数字。")
    data_config, model_config = apply_runtime_overrides(
        args,
        data_config,
        model_config,
    )

    checkpoint_dir = args.checkpoint_dir
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = PROJECT_ROOT / checkpoint_dir
    run_dir = args.run_dir
    if run_dir is None:
        run_dir = PROJECT_ROOT / "runs" / create_run_id(PROJECT_ROOT)
    elif not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir

    config_snapshot = {
        "data": normalize_config(data_config),
        "model": normalize_config(model_config),
        "training": normalize_config(training_config),
        "runtime": {
            "dropout": args.dropout,
            "gradient_accumulation_steps": (args.gradient_accumulation_steps),
            "warmup_steps": args.warmup_steps,
            "warmup_start_factor": args.warmup_start_factor,
            "cosine_decay": args.cosine_decay,
            "minimum_learning_rate": args.min_learning_rate,
            "precision": args.precision,
            "compile": args.compile,
            "compile_mode": args.compile_mode,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_min_delta": args.early_stopping_min_delta,
        },
        "paths": {
            "source_config": str(config_path),
            "run_dir": str(run_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "resume_checkpoint": (None if args.resume is None else str(args.resume)),
        },
    }

    set_random_seed(training_config.seed)

    tokenizer = load_tokenizer(
        data_config.tokenizer_path,
        tokenizer_type=data_config.tokenizer_type,
    )

    train_text = read_utf8_text(data_config.train_path)
    val_text = read_utf8_text(data_config.val_path)

    train_dataset = build_language_model_dataset(
        train_text,
        tokenizer,
        block_size=data_config.block_size,
        packing=data_config.packing,
        sample_separator=data_config.sample_separator,
    )
    val_dataset = build_language_model_dataset(
        val_text,
        tokenizer,
        block_size=data_config.block_size,
        packing=data_config.packing,
        sample_separator=data_config.sample_separator,
        allow_unknown=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    device = select_device()
    amp_dtype, scaler = resolve_amp(args.precision, device)
    experiment = ExperimentRun.create(
        run_dir=run_dir,
        config=config_snapshot,
        environment=collect_environment(
            PROJECT_ROOT,
            command=[sys.executable, *sys.argv],
        ),
        allow_existing=args.resume is not None,
    )
    configure_training_logger(experiment)
    LOGGER.info("run_dir=%s", run_dir)
    LOGGER.info("checkpoint_dir=%s", checkpoint_dir)

    if args.resume is None and not args.skip_overfit:
        run_tiny_batch_overfit_test(
            tokenizer=tokenizer,
            data_config=data_config,
            model_config=model_config,
            training_config=training_config,
            train_loader=train_loader,
            device=device,
            dropout=args.dropout,
        )

        set_random_seed(training_config.seed)

    model = build_model(
        tokenizer=tokenizer,
        data_config=data_config,
        model_config=model_config,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    if not args.cosine_decay and args.min_learning_rate != 0.0:
        raise ValueError("未启用 Cosine Decay 时 min_learning_rate 必须为 0。")

    scheduler: WarmupCosineScheduler | None = None
    if args.warmup_steps > 0 or args.cosine_decay:
        scheduler = WarmupCosineScheduler(
            optimizer,
            peak_learning_rate=training_config.learning_rate,
            max_steps=training_config.max_steps,
            warmup_steps=args.warmup_steps,
            minimum_learning_rate=args.min_learning_rate,
            warmup_start_factor=args.warmup_start_factor,
            cosine_decay=args.cosine_decay,
        )

    latest_path = checkpoint_dir / "latest.pt"
    best_path = checkpoint_dir / "best.pt"

    global_step = 0
    best_validation_loss = float("inf")
    early_stopping_bad_evals = 0
    stopped_early = False

    if args.resume is not None:
        resume_path = args.resume
        if not resume_path.is_absolute():
            resume_path = PROJECT_ROOT / resume_path

        checkpoint = load_checkpoint(
            path=resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            scaler=scaler,
        )

        saved_tokenizer_info = checkpoint.get("tokenizer_info")
        if (
            saved_tokenizer_info is not None
            and saved_tokenizer_info.get("itos") != tokenizer.itos
        ):
            raise ValueError("Checkpoint 的 Tokenizer 与当前 Tokenizer 不一致。")

        global_step = int(checkpoint["global_step"])
        if scheduler is not None and checkpoint.get("scheduler_state_dict") is None:
            scheduler.set_completed_steps(global_step)
        best_validation_loss = float(checkpoint["best_validation_loss"])
        training_state = checkpoint.get("training_state")
        if isinstance(training_state, dict):
            raw_bad_evals = training_state.get("early_stopping_bad_evals", 0)
            if type(raw_bad_evals) is int and raw_bad_evals >= 0:
                early_stopping_bad_evals = raw_bad_evals

        LOGGER.info(f"已恢复训练：global_step={global_step}")
    tokenizer_info = build_tokenizer_info(tokenizer)

    def checkpoint_training_state() -> dict[str, object]:
        return {
            "early_stopping_bad_evals": early_stopping_bad_evals,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_min_delta": args.early_stopping_min_delta,
        }

    batch_iterator = iter(train_loader)
    execution_model = compile_model(
        model,
        enabled=args.compile,
        mode=args.compile_mode,
    )
    processed_tokens = 0
    training_elapsed = 0.0
    last_step = global_step
    LOGGER.info(
        f"device={device} precision={args.precision} "
        f"parameters={count_parameters(model)} "
        f"micro_batch_size={data_config.batch_size} "
        f"accumulation_steps={args.gradient_accumulation_steps} "
        f"compiled={args.compile} compile_mode={args.compile_mode} "
        f"effective_batch_size="
        f"{data_config.batch_size * args.gradient_accumulation_steps} "
        f"tokenizer={data_config.tokenizer_type} packing={data_config.packing}"
        f" early_stopping_patience={args.early_stopping_patience}"
    )
    experiment.record(
        "run_started",
        device=str(device),
        precision=args.precision,
        parameters=count_parameters(model),
        micro_batch_size=data_config.batch_size,
        accumulation_steps=args.gradient_accumulation_steps,
        effective_batch_size=(
            data_config.batch_size * args.gradient_accumulation_steps
        ),
        compiled=args.compile,
        compile_mode=args.compile_mode,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        starting_step=global_step,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    while global_step < training_config.max_steps:
        micro_batches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(args.gradient_accumulation_steps):
            try:
                x_batch, y_batch = next(batch_iterator)
            except StopIteration:
                batch_iterator = iter(train_loader)
                x_batch, y_batch = next(batch_iterator)
            micro_batches.append((x_batch.to(device), y_batch.to(device)))

        learning_rate_used = optimizer.param_groups[0]["lr"]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_start = time.perf_counter()
        train_loss, gradient_norm, optimizer_updated = train_accumulation_step(
            model=execution_model,
            optimizer=optimizer,
            micro_batches=micro_batches,
            grad_clip_norm=training_config.grad_clip_norm,
            amp_dtype=amp_dtype,
            scaler=scaler,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        training_elapsed += time.perf_counter() - step_start
        processed_tokens += sum(inputs.numel() for inputs, _ in micro_batches)

        if not optimizer_updated:
            LOGGER.warning("FP16 梯度出现 Inf/NaN，本次参数更新已跳过。")
            experiment.record("optimizer_step_skipped", reason="inf_or_nan")
            continue

        global_step += 1
        last_step = global_step
        if scheduler is not None:
            scheduler.step()

        tokens_per_second = processed_tokens / max(
            training_elapsed,
            1e-12,
        )
        peak_memory_mib = (
            torch.cuda.max_memory_allocated(device) / (1024**2)
            if device.type == "cuda"
            else None
        )
        experiment.record(
            "train_step",
            step=global_step,
            train_loss=float(train_loss.item()),
            learning_rate=float(learning_rate_used),
            gradient_norm=float(gradient_norm),
            processed_tokens=processed_tokens,
            tokens_per_second=float(tokens_per_second),
            peak_memory_mib=peak_memory_mib,
            training_time_seconds=float(training_elapsed),
        )

        if global_step % training_config.log_interval == 0 or global_step == 1:
            memory_text = (
                "not_available" if peak_memory_mib is None else f"{peak_memory_mib:.2f}"
            )
            LOGGER.info(
                f"step={global_step} "
                f"train_loss={train_loss.item():.4f} "
                f"validation_loss=not_evaluated "
                f"learning_rate={learning_rate_used:.8f} "
                f"gradient_norm={gradient_norm:.4f} "
                f"tokens_per_second={tokens_per_second:.2f} "
                f"peak_memory_mib={memory_text} "
                f"training_time={training_elapsed:.2f}s"
            )

        if global_step % training_config.eval_interval == 0 or global_step == 1:
            validation_metrics = evaluate_metrics(
                model=execution_model,
                data_loader=val_loader,
                device=device,
                max_batches=(
                    None
                    if args.early_stopping_patience > 0
                    else training_config.eval_steps
                ),
                amp_dtype=amp_dtype,
                byte_count=len(val_text.encode("utf-8")),
            )
            validation_loss = validation_metrics.loss

            bpb_text = (
                "not_available"
                if validation_metrics.bpb_utf8 is None
                else f"{validation_metrics.bpb_utf8:.4f}"
            )
            LOGGER.info(
                f"step={global_step} validation_loss={validation_loss:.4f} "
                f"bits_per_token={validation_metrics.bits_per_token:.4f} "
                f"bpb_utf8={bpb_text}"
            )

            is_best = validation_loss < (
                best_validation_loss - args.early_stopping_min_delta
            )
            if is_best:
                early_stopping_bad_evals = 0
            elif args.early_stopping_patience > 0:
                early_stopping_bad_evals += 1
            experiment.record(
                "validation",
                step=global_step,
                validation_loss=float(validation_loss),
                bits_per_token=validation_metrics.bits_per_token,
                bpb_utf8=validation_metrics.bpb_utf8,
                token_count=validation_metrics.token_count,
                byte_count=validation_metrics.byte_count,
                is_best=is_best,
                early_stopping_bad_evals=early_stopping_bad_evals,
            )

            if is_best:
                best_validation_loss = validation_loss
                save_checkpoint(
                    path=best_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    global_step=global_step,
                    best_validation_loss=(best_validation_loss),
                    config=config_snapshot,
                    tokenizer_info=tokenizer_info,
                    seed=training_config.seed,
                    scaler=scaler,
                    training_state=checkpoint_training_state(),
                )
                experiment.record(
                    "checkpoint_saved",
                    step=global_step,
                    kind="best",
                    path=str(best_path),
                    best_validation_loss=float(best_validation_loss),
                )
                LOGGER.info(f"best checkpoint 已保存：{best_path}")

            if (
                args.early_stopping_patience > 0
                and early_stopping_bad_evals >= args.early_stopping_patience
            ):
                stopped_early = True
                experiment.record(
                    "early_stopping",
                    step=global_step,
                    best_validation_loss=float(best_validation_loss),
                    bad_evals=early_stopping_bad_evals,
                    patience=args.early_stopping_patience,
                )
                LOGGER.info(
                    "Early Stopping: 连续 %d 次验证无改善，停止训练。",
                    early_stopping_bad_evals,
                )
                break

        if global_step % training_config.save_interval == 0:
            save_checkpoint(
                path=latest_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                global_step=global_step,
                best_validation_loss=(best_validation_loss),
                config=config_snapshot,
                tokenizer_info=tokenizer_info,
                seed=training_config.seed,
                scaler=scaler,
                training_state=checkpoint_training_state(),
            )
            experiment.record(
                "checkpoint_saved",
                step=global_step,
                kind="latest",
                path=str(latest_path),
                best_validation_loss=float(best_validation_loss),
            )
            LOGGER.info(f"latest checkpoint 已保存：{latest_path}")

    if last_step > 0:
        save_checkpoint(
            path=latest_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            global_step=last_step,
            best_validation_loss=best_validation_loss,
            config=config_snapshot,
            tokenizer_info=tokenizer_info,
            seed=training_config.seed,
            scaler=scaler,
            training_state=checkpoint_training_state(),
        )
        experiment.record(
            "run_completed",
            step=last_step,
            processed_tokens=processed_tokens,
            training_time_seconds=float(training_elapsed),
            best_validation_loss=float(best_validation_loss),
            latest_checkpoint=str(latest_path),
            stopped_early=stopped_early,
        )
        LOGGER.info(f"训练结束，latest checkpoint 已保存：{latest_path}")


if __name__ == "__main__":
    main()
