"""End-to-end checks for stages one through four."""

import sys
from pathlib import Path

import pytest
import torch
import yaml
from torch.utils.data import DataLoader

import generate as generate_cli
from minigpt.config import (
    load_data_config,
    load_model_config,
    load_training_config,
)
from minigpt.dataset import CharacterLanguageModelDataset
from minigpt.model import MiniGPT
from minigpt.tokenizer import CharacterTokenizer
from minigpt.trainer import evaluate_loss, train_step
from minigpt.utils import load_checkpoint, save_checkpoint


def make_tiny_model(vocab_size: int = 7) -> MiniGPT:
    return MiniGPT(
        vocab_size=vocab_size,
        block_size=4,
        embedding_dim=8,
        num_heads=2,
        num_layers=1,
        expansion_factor=2,
    )


def test_project_config_loads_all_stage_three_fields() -> None:
    config_path = Path("configs/tiny_shakespeare.yaml")

    data_config = load_data_config(config_path)
    model_config = load_model_config(config_path)
    training_config = load_training_config(config_path)

    assert data_config.block_size > 0
    assert model_config.embedding_dim % model_config.num_heads == 0
    assert training_config.eval_interval > 0
    assert training_config.save_interval > 0


def test_train_step_and_evaluate_loss_are_finite() -> None:
    model = make_tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    dataset = CharacterLanguageModelDataset(
        token_ids=list(range(7)) * 4,
        block_size=4,
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    inputs, targets = next(iter(loader))

    loss, gradient_norm = train_step(
        model=model,
        optimizer=optimizer,
        inputs=inputs,
        targets=targets,
        grad_clip_norm=1.0,
    )

    validation_loss = evaluate_loss(
        model=model,
        data_loader=loader,
        device=torch.device("cpu"),
        max_batches=2,
    )

    assert torch.isfinite(loss)
    assert gradient_norm >= 0
    assert validation_loss > 0
    assert model.training is True


def test_checkpoint_round_trip_restores_model(tmp_path) -> None:
    model = make_tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    checkpoint_path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        global_step=3,
        best_validation_loss=1.25,
        config={"stage": 3},
        tokenizer_info={"itos": ["<unk>", "a"]},
        seed=42,
    )

    restored_model = make_tiny_model()
    checkpoint = load_checkpoint(
        path=checkpoint_path,
        model=restored_model,
        optimizer=None,
        scheduler=None,
        device=torch.device("cpu"),
    )

    for expected, actual in zip(
        model.state_dict().values(),
        restored_model.state_dict().values(),
    ):
        assert torch.equal(expected, actual)
    assert checkpoint["global_step"] == 3
    assert checkpoint["best_validation_loss"] == pytest.approx(1.25)


def test_cli_loads_checkpoint_generates_and_decodes(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    project_root = tmp_path
    config_path = project_root / "configs" / "tiny.yaml"
    tokenizer_path = project_root / "data" / "tokenizer.json"
    checkpoint_path = project_root / "checkpoints" / "best.pt"
    tokenizer = CharacterTokenizer("abc")
    tokenizer.save(tokenizer_path)

    config_path.parent.mkdir(parents=True, exist_ok=True)
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
                    "batch_size": 1,
                },
                "model": {
                    "embedding_dim": 8,
                    "num_heads": 2,
                    "num_layers": 1,
                    "expansion_factor": 2,
                    "tie_weights": False,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    model = make_tiny_model(tokenizer.vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        global_step=1,
        best_validation_loss=1.0,
        config={},
        tokenizer_info={
            "itos": tokenizer.itos,
            "vocab_size": tokenizer.vocab_size,
        },
        seed=42,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate.py",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint_path),
            "--prompt",
            "ab",
            "--max-new-tokens",
            "2",
            "--greedy",
        ],
    )

    generate_cli.main()
    output = capsys.readouterr().out

    assert output.startswith("ab")
