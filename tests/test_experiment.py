from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime

import pytest
import torch
import yaml

import train as train_cli
from minigpt.experiment import (
    LOGGER_NAME,
    ExperimentRun,
    collect_environment,
    configure_training_logger,
    create_run_id,
)
from minigpt.tokenizer import CharacterTokenizer


def test_create_run_id_is_sortable_and_has_fallback(tmp_path) -> None:
    now = datetime(2026, 8, 18, 1, 2, 3, 456789, tzinfo=UTC)
    run_id = create_run_id(tmp_path, now=now)
    assert run_id == "20260818-010203-456789-nogit"


def test_experiment_run_writes_snapshots_metrics_and_log(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run = ExperimentRun.create(
        run_dir=run_dir,
        config={"model": {"layers": 2}, "说明": "测试"},
        environment={"schema_version": 1, "device": "cpu"},
    )
    run.record("train_step", step=1, train_loss=1.25)

    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    environment = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
    metric = json.loads((run_dir / "metrics.jsonl").read_text(encoding="utf-8").strip())
    assert config["model"]["layers"] == 2
    assert config["说明"] == "测试"
    assert environment["device"] == "cpu"
    assert metric["event"] == "train_step"
    assert metric["step"] == 1
    assert metric["train_loss"] == pytest.approx(1.25)

    logger = configure_training_logger(run)
    logger.info("structured run ready")
    for handler in logger.handlers:
        handler.flush()
    assert "structured run ready" in run.log_path.read_text(encoding="utf-8")
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()


def test_existing_run_requires_resume_mode(tmp_path) -> None:
    run_dir = tmp_path / "run"
    ExperimentRun.create(run_dir, {"version": 1}, {"attempt": 1})
    with pytest.raises(FileExistsError, match="已包含实验记录"):
        ExperimentRun.create(run_dir, {"version": 2}, {"attempt": 2})

    resumed = ExperimentRun.create(
        run_dir,
        {"version": 2},
        {"attempt": 2},
        allow_existing=True,
    )
    assert resumed.run_dir == run_dir
    assert yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")) == {
        "version": 1
    }
    resume_files = list(run_dir.glob("environment-resume-*.json"))
    assert len(resume_files) == 1


def test_collect_environment_has_reproduction_fields(tmp_path) -> None:
    environment = collect_environment(tmp_path, command=["python", "train.py"])
    assert environment["schema_version"] == 1
    assert environment["command"] == ["python", "train.py"]
    assert environment["python"]["version"]
    assert environment["pytorch"]["version"]
    assert isinstance(environment["pytorch"]["cuda_available"], bool)
    assert environment["git"]["commit"] is None
    assert environment["git"]["dirty"] is False


def test_training_cli_writes_complete_run_artifacts(
    tmp_path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "configs" / "tiny.yaml"
    data_dir = tmp_path / "data"
    run_dir = tmp_path / "runs" / "smoke"
    checkpoint_dir = tmp_path / "checkpoints" / "smoke"
    config_path.parent.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    (data_dir / "train.txt").write_text("abc" * 30, encoding="utf-8")
    (data_dir / "val.txt").write_text("abc" * 10, encoding="utf-8")
    (data_dir / "test.txt").write_text("abc" * 10, encoding="utf-8")
    (data_dir / "raw.txt").write_text("abc" * 50, encoding="utf-8")
    CharacterTokenizer("abc").save(data_dir / "tokenizer.json")
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "raw_path": "data/raw.txt",
                    "train_path": "data/train.txt",
                    "val_path": "data/val.txt",
                    "test_path": "data/test.txt",
                    "tokenizer_path": "data/tokenizer.json",
                    "train_ratio": 0.8,
                    "val_ratio": 0.1,
                    "block_size": 4,
                    "batch_size": 2,
                },
                "model": {
                    "embedding_dim": 8,
                    "num_heads": 2,
                    "num_layers": 1,
                    "expansion_factor": 2,
                    "tie_weights": False,
                },
                "training": {
                    "learning_rate": 0.001,
                    "weight_decay": 0.0,
                    "max_steps": 1,
                    "grad_clip_norm": 1.0,
                    "log_interval": 1,
                    "eval_interval": 1,
                    "eval_steps": 1,
                    "save_interval": 1,
                    "seed": 42,
                    "overfit_steps": 2,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(train_cli, "select_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            str(config_path),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--run-dir",
            str(run_dir),
            "--skip-overfit",
        ],
    )

    try:
        train_cli.main()
    finally:
        logger = logging.getLogger(LOGGER_NAME)
        for handler in logger.handlers:
            handler.close()
        logger.handlers.clear()

    assert (run_dir / "config.yaml").is_file()
    assert (run_dir / "environment.json").is_file()
    assert (run_dir / "train.log").is_file()
    assert (checkpoint_dir / "best.pt").is_file()
    assert (checkpoint_dir / "latest.pt").is_file()
    events = [
        json.loads(line)["event"]
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert events == [
        "run_started",
        "train_step",
        "validation",
        "checkpoint_saved",
        "checkpoint_saved",
        "run_completed",
    ]
