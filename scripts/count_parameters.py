"""按 Embedding、Attention、FFN 等模块统计 MiniGPT 参数。"""

from __future__ import annotations

import argparse
from pathlib import Path

from minigpt.config import load_data_config, load_model_config
from minigpt.model import MiniGPT, count_parameters, count_parameters_by_group
from minigpt.tokenizer import CharacterTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统计 MiniGPT 参数分布。")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "tiny_shakespeare.yaml",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Tokenizer 尚未生成时使用的词表大小。",
    )
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument(
        "--tie-weights",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser.parse_args()


def resolve_vocab_size(
    tokenizer_path: Path,
    override: int | None,
) -> int:
    if override is not None:
        if override <= 0:
            raise ValueError("vocab_size 必须是正整数。")
        return override
    if not tokenizer_path.is_file():
        raise FileNotFoundError(
            "Tokenizer 尚未生成；请先准备数据，或传入 --vocab-size。"
        )
    return CharacterTokenizer.load(tokenizer_path).vocab_size


def main() -> None:
    args = parse_args()
    data_config = load_data_config(args.config.resolve())
    model_config = load_model_config(args.config.resolve())

    vocab_size = resolve_vocab_size(
        data_config.tokenizer_path,
        args.vocab_size,
    )
    embedding_dim = args.embedding_dim or model_config.embedding_dim
    num_heads = args.num_heads or model_config.num_heads
    num_layers = args.num_layers or model_config.num_layers
    block_size = args.block_size or data_config.block_size
    tie_weights = (
        model_config.tie_weights if args.tie_weights is None else args.tie_weights
    )

    model = MiniGPT(
        vocab_size=vocab_size,
        block_size=block_size,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        expansion_factor=model_config.expansion_factor,
        tie_weights=tie_weights,
    )
    groups = count_parameters_by_group(model)
    total = count_parameters(model)

    print(f"total_trainable_parameters: {total:,}")
    print(f"tie_weights: {tie_weights}")
    for name, value in groups.items():
        percentage = 100.0 * value / total
        print(f"{name:>16}: {value:>12,} ({percentage:6.2f}%)")


if __name__ == "__main__":
    main()
