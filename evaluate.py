"""独立评估 MiniGPT Checkpoint。"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from minigpt.config import (
    load_data_config,
    load_model_config,
    load_training_config,
)
from minigpt.dataset import CharacterLanguageModelDataset
from minigpt.model import MiniGPT
from minigpt.tokenizer import CharacterTokenizer
from minigpt.trainer import evaluate_loss
from minigpt.utils import (
    checkpoint_config_value,
    compile_model,
    load_checkpoint,
    load_checkpoint_payload,
    read_utf8_text,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "tiny_shakespeare.yaml"
)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="评估 MiniGPT Checkpoint。"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="YAML 配置文件路径。",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "best.pt",
        help="Checkpoint 路径。",
    )
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="val",
        help="评估验证集或测试集。",
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint

    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path

    data_config = load_data_config(config_path)
    model_config = load_model_config(config_path)
    training_config = load_training_config(config_path)

    tokenizer = CharacterTokenizer.load(
        data_config.tokenizer_path
    )
    device = select_device()
    checkpoint_payload = load_checkpoint_payload(
        checkpoint_path,
        device,
    )
    block_size = checkpoint_config_value(
        checkpoint_payload,
        "data",
        "block_size",
        data_config.block_size,
    )
    batch_size = checkpoint_config_value(
        checkpoint_payload,
        "data",
        "batch_size",
        data_config.batch_size,
    )

    if args.split == "val":
        text_path = data_config.val_path
    else:
        text_path = data_config.test_path

    text = read_utf8_text(text_path)
    dataset = CharacterLanguageModelDataset(
        token_ids=tokenizer.encode(
            text,
            allow_unknown=True,
        ),
        block_size=block_size,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        embedding_dim=checkpoint_config_value(
            checkpoint_payload,
            "model",
            "embedding_dim",
            model_config.embedding_dim,
        ),
        num_heads=checkpoint_config_value(
            checkpoint_payload,
            "model",
            "num_heads",
            model_config.num_heads,
        ),
        num_layers=checkpoint_config_value(
            checkpoint_payload,
            "model",
            "num_layers",
            model_config.num_layers,
        ),
        expansion_factor=checkpoint_config_value(
            checkpoint_payload,
            "model",
            "expansion_factor",
            model_config.expansion_factor,
        ),
        tie_weights=checkpoint_config_value(
            checkpoint_payload,
            "model",
            "tie_weights",
            model_config.tie_weights,
        ),
    )

    model = model.to(device)

    checkpoint = load_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=None,
        scheduler=None,
        device=device,
    )

    saved_tokenizer_info = checkpoint.get(
        "tokenizer_info"
    )
    if (
        saved_tokenizer_info is not None
        and saved_tokenizer_info.get("itos")
        != tokenizer.itos
    ):
        raise ValueError(
            "Checkpoint 的 Tokenizer 与当前 Tokenizer 不一致。"
        )

    execution_model = compile_model(
        model,
        enabled=args.compile,
        mode=args.compile_mode,
    )

    validation_loss = evaluate_loss(
        model=execution_model,
        data_loader=data_loader,
        device=device,
        max_batches=training_config.eval_steps,
    )

    print(f"device: {device}")
    print(f"split: {args.split}")
    print(f"checkpoint: {checkpoint_path}")
    print(
        f"global_step: {checkpoint['global_step']}"
    )
    print(f"loss: {validation_loss:.4f}")


if __name__ == "__main__":
    main()
