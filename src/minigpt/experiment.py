"""可复现训练运行的目录、环境快照、日志与结构化指标。"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any

import torch
import yaml


LOGGER_NAME = "minigpt.train"


def utc_now() -> datetime:
    """返回带时区的 UTC 时间，便于测试替换。"""
    return datetime.now(timezone.utc)


def _run_git(project_root: Path, *args: str) -> str | None:
    """读取 Git 元数据；仓库外或 Git 不可用时返回 None。"""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def create_run_id(project_root: Path, now: datetime | None = None) -> str:
    """生成可排序且低冲突的运行 ID。"""
    timestamp = (now or utc_now()).astimezone(timezone.utc)
    short_sha = _run_git(project_root, "rev-parse", "--short", "HEAD")
    suffix = short_sha or "nogit"
    return f"{timestamp:%Y%m%d-%H%M%S-%f}-{suffix}"


def collect_environment(
    project_root: Path,
    command: list[str] | None = None,
) -> dict[str, Any]:
    """收集复现实验所需的软件、硬件和 Git 信息。"""
    devices: list[dict[str, Any]] = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "compute_capability": [properties.major, properties.minor],
                    "total_memory_bytes": properties.total_memory,
                }
            )

    status = _run_git(project_root, "status", "--short")
    return {
        "schema_version": 1,
        "created_at": utc_now().isoformat(),
        "command": list(command or sys.argv),
        "working_directory": str(Path.cwd()),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "platform": platform.platform(),
        "pytorch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "cudnn_version": (
                torch.backends.cudnn.version()
                if torch.cuda.is_available()
                else None
            ),
        },
        "cuda_devices": devices,
        "git": {
            "commit": _run_git(project_root, "rev-parse", "HEAD"),
            "branch": _run_git(
                project_root,
                "branch",
                "--show-current",
            ),
            "dirty": status is not None,
        },
    }


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = Path(handle.name)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


class ExperimentRun:
    """管理单次训练运行的可复现产物。"""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.metrics_path = run_dir / "metrics.jsonl"
        self.log_path = run_dir / "train.log"

    @classmethod
    def create(
        cls,
        run_dir: Path,
        config: dict[str, Any],
        environment: dict[str, Any],
        *,
        allow_existing: bool = False,
    ) -> "ExperimentRun":
        config_path = run_dir / "config.yaml"
        environment_path = run_dir / "environment.json"
        if not allow_existing and (
            config_path.exists() or environment_path.exists()
        ):
            raise FileExistsError(f"运行目录已包含实验记录：{run_dir}")

        run_dir.mkdir(parents=True, exist_ok=True)
        run = cls(run_dir)
        if allow_existing and config_path.exists():
            resume_stamp = utc_now().strftime("%Y%m%d-%H%M%S-%f")
            environment_path = run_dir / f"environment-resume-{resume_stamp}.json"
        else:
            _atomic_write_text(
                config_path,
                yaml.safe_dump(
                    config,
                    allow_unicode=True,
                    sort_keys=False,
                ),
            )
        _atomic_write_text(
            environment_path,
            json.dumps(environment, ensure_ascii=False, indent=2) + "\n",
        )
        return run

    def record(self, event: str, **values: Any) -> None:
        if not event:
            raise ValueError("event 不能为空。")
        payload = {
            "timestamp": utc_now().isoformat(),
            "event": event,
            **values,
        }
        line = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        with self.metrics_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")
            handle.flush()


def configure_training_logger(run: ExperimentRun) -> logging.Logger:
    """把训练日志同时输出到终端和运行目录。"""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(
        run.log_path,
        mode="a",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger
