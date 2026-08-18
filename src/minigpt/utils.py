"""项目通用工具、随机种子和 Checkpoint 工具。"""

import random
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import torch


def compile_model(
    model: torch.nn.Module,
    *,
    enabled: bool = False,
    mode: str = "default",
) -> torch.nn.Module:
    """Optionally compile a model while keeping the original module separate."""
    if not isinstance(enabled, bool):
        raise TypeError("enabled 必须是 bool。")
    valid_modes = {"default", "reduce-overhead", "max-autotune"}
    if mode not in valid_modes:
        raise ValueError(f"compile mode 必须是 {sorted(valid_modes)} 之一。")
    if not enabled:
        return model
    compile_function = getattr(torch, "compile", None)
    if compile_function is None:
        raise RuntimeError("当前 PyTorch 版本不支持 torch.compile。")
    return compile_function(model, mode=mode)


def read_utf8_text(path: Path) -> str:
    """读取非空 UTF-8 文本。"""
    if not path.is_file():
        raise FileNotFoundError(f"找不到文本文件：{path}")

    text = path.read_text(encoding="utf-8")
    if not text:
        raise ValueError(f"文本文件为空：{path}")
    return text


def write_utf8_text(path: Path, text: str) -> None:
    """创建父目录并写入 UTF-8 文本。"""
    if not isinstance(text, str):
        raise TypeError("text 必须是 str。")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def set_random_seed(seed: int) -> None:
    """固定 Python、PyTorch 和 CUDA 的随机种子。"""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed 必须是非负整数。")

    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_random_states() -> dict[str, Any]:
    """保存 Python、CPU 和 CUDA 的随机状态。"""
    states: dict[str, Any] = {
        "python": random.getstate(),
        # Keep RNG tensors on CPU so torch.load(map_location=...) cannot
        # accidentally turn the CPU state into a CUDA tensor.
        "torch": torch.get_rng_state().cpu(),
    }

    if torch.cuda.is_available():
        states["cuda"] = [
            state.cpu()
            for state in torch.cuda.get_rng_state_all()
        ]

    return states


def restore_random_states(
    states: dict[str, Any],
) -> None:
    """恢复 Python、CPU 和 CUDA 的随机状态。"""
    random.setstate(states["python"])
    torch.set_rng_state(states["torch"].detach().cpu())

    if (
        torch.cuda.is_available()
        and "cuda" in states
    ):
        torch.cuda.set_rng_state_all(
            [
                state.detach().cpu()
                for state in states["cuda"]
            ]
        )


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    global_step: int,
    best_validation_loss: float,
    config: dict[str, Any],
    tokenizer_info: dict[str, Any],
    seed: int,
    scaler: Any = None,
) -> None:
    """保存完整训练状态。"""
    if global_step < 0:
        raise ValueError("global_step 不能为负数。")

    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (
            None
            if scheduler is None
            else scheduler.state_dict()
        ),
        "scaler_state_dict": (
            None if scaler is None else scaler.state_dict()
        ),
        "global_step": global_step,
        "best_validation_loss": best_validation_loss,
        "config": config,
        "tokenizer_info": tokenizer_info,
        "seed": seed,
        "random_states": capture_random_states(),
    }

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    device: torch.device,
    scaler: Any = None,
) -> dict[str, Any]:
    """加载模型和训练状态。"""
    checkpoint = load_checkpoint_payload(path, device)

    required_keys = {
        "model_state_dict",
        "optimizer_state_dict",
        "global_step",
        "best_validation_loss",
    }
    missing_keys = required_keys.difference(checkpoint)
    if missing_keys:
        raise ValueError(
            f"Checkpoint 缺少字段：{sorted(missing_keys)}"
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    scheduler_state = checkpoint.get(
        "scheduler_state_dict"
    )
    if (
        scheduler is not None
        and scheduler_state is not None
    ):
        scheduler.load_state_dict(scheduler_state)

    scaler_state = checkpoint.get("scaler_state_dict")
    if scaler is not None and scaler_state is not None:
        scaler.load_state_dict(scaler_state)

    random_states = checkpoint.get("random_states")
    if random_states is not None:
        restore_random_states(random_states)

    return checkpoint


def load_checkpoint_payload(
    path: Path,
    device: torch.device,
) -> dict[str, Any]:
    """只读取 Checkpoint 字典，供构造对应结构的模型使用。"""
    if not path.is_file():
        raise FileNotFoundError(
            f"找不到 Checkpoint：{path}"
        )

    try:
        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            path,
            map_location=device,
        )

    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint 顶层必须是字典。")
    return checkpoint


def checkpoint_config_value(
    checkpoint: dict[str, Any],
    section: str,
    name: str,
    default: Any,
) -> Any:
    """读取 Checkpoint 配置快照，旧 Checkpoint 缺失时使用默认值。"""
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        return default
    values = config.get(section)
    if not isinstance(values, dict):
        return default
    return values.get(name, default)


class WarmupCosineScheduler:
    """按 Optimizer Step 执行线性 Warmup 与可选 Cosine Decay。"""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        peak_learning_rate: float,
        max_steps: int,
        warmup_steps: int = 0,
        minimum_learning_rate: float = 0.0,
        warmup_start_factor: float = 0.1,
        cosine_decay: bool = False,
    ) -> None:
        if peak_learning_rate <= 0:
            raise ValueError("peak_learning_rate 必须大于 0。")
        if type(max_steps) is not int or max_steps <= 0:
            raise ValueError("max_steps 必须是正整数。")
        if type(warmup_steps) is not int or not 0 <= warmup_steps < max_steps:
            raise ValueError("warmup_steps 必须位于 [0, max_steps) 中。")
        if not 0.0 <= minimum_learning_rate <= peak_learning_rate:
            raise ValueError("minimum_learning_rate 必须位于 [0, peak_lr] 中。")
        if not 0.0 <= warmup_start_factor <= 1.0:
            raise ValueError("warmup_start_factor 必须位于 [0, 1] 中。")

        self.optimizer = optimizer
        self.peak_learning_rate = float(peak_learning_rate)
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.minimum_learning_rate = float(minimum_learning_rate)
        self.warmup_start_factor = float(warmup_start_factor)
        self.cosine_decay = bool(cosine_decay)
        self.completed_steps = 0
        self._set_learning_rate(self.learning_rate_for_step(1))

    def learning_rate_for_step(self, step: int) -> float:
        """返回第 ``step`` 次参数更新应使用的学习率。"""
        if type(step) is not int or not 1 <= step <= self.max_steps:
            raise ValueError("step 必须位于 [1, max_steps] 中。")

        if self.warmup_steps > 0 and step <= self.warmup_steps:
            start_lr = (
                self.peak_learning_rate * self.warmup_start_factor
            )
            progress = step / self.warmup_steps
            return start_lr + (
                self.peak_learning_rate - start_lr
            ) * progress

        if not self.cosine_decay:
            return self.peak_learning_rate

        if self.warmup_steps == 0:
            decay_length = self.max_steps - 1
            progress = 1.0 if decay_length == 0 else (step - 1) / decay_length
        else:
            decay_length = self.max_steps - self.warmup_steps
            progress = (step - self.warmup_steps) / decay_length

        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.minimum_learning_rate + (
            self.peak_learning_rate - self.minimum_learning_rate
        ) * cosine_factor

    def _set_learning_rate(self, learning_rate: float) -> None:
        for parameter_group in self.optimizer.param_groups:
            parameter_group["lr"] = learning_rate

    def step(self) -> None:
        """在一次成功的 ``optimizer.step`` 后推进调度器。"""
        if self.completed_steps >= self.max_steps:
            raise RuntimeError("Scheduler 已超过 max_steps。")
        self.completed_steps += 1
        if self.completed_steps < self.max_steps:
            self._set_learning_rate(
                self.learning_rate_for_step(self.completed_steps + 1)
            )

    def set_completed_steps(self, completed_steps: int) -> None:
        """设置恢复进度，并准备下一次更新的学习率。"""
        if type(completed_steps) is not int or not 0 <= completed_steps <= self.max_steps:
            raise ValueError("completed_steps 超出有效范围。")
        self.completed_steps = completed_steps
        if completed_steps < self.max_steps:
            self._set_learning_rate(
                self.learning_rate_for_step(completed_steps + 1)
            )

    def state_dict(self) -> dict[str, Any]:
        return {
            "completed_steps": self.completed_steps,
            "peak_learning_rate": self.peak_learning_rate,
            "max_steps": self.max_steps,
            "warmup_steps": self.warmup_steps,
            "minimum_learning_rate": self.minimum_learning_rate,
            "warmup_start_factor": self.warmup_start_factor,
            "cosine_decay": self.cosine_decay,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = {
            "peak_learning_rate": self.peak_learning_rate,
            "max_steps": self.max_steps,
            "warmup_steps": self.warmup_steps,
            "minimum_learning_rate": self.minimum_learning_rate,
            "warmup_start_factor": self.warmup_start_factor,
            "cosine_decay": self.cosine_decay,
        }
        for name, value in expected.items():
            if state.get(name) != value:
                raise ValueError(
                    f"Checkpoint Scheduler 配置不一致：{name}。"
                )
        self.set_completed_steps(int(state["completed_steps"]))
