"""把指定层、指定 Head 的 Attention 权重保存为热力图。"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from minigpt.config import load_data_config, load_model_config
from minigpt.model import MiniGPT
from minigpt.tokenizer import CharacterTokenizer
from minigpt.utils import (
    checkpoint_config_value,
    load_checkpoint,
    load_checkpoint_payload,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="可视化 MiniGPT Attention。")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "tiny_shakespeare.yaml",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--head", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    return parser.parse_args()


def display_token(token: str, position: int) -> str:
    visible = {
        " ": "<space>",
        "\n": r"\n",
        "\t": r"\t",
    }.get(token, token)
    return f"{position}:{visible}"


def main() -> None:
    args = parse_args()
    if not args.prompt:
        raise ValueError("prompt 不能为空。")

    config_path = args.config.resolve()
    data_config = load_data_config(config_path)
    model_config = load_model_config(config_path)
    tokenizer = CharacterTokenizer.load(data_config.tokenizer_path)

    checkpoint_path = (
        args.checkpoint
        if args.checkpoint.is_absolute()
        else PROJECT_ROOT / args.checkpoint
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_payload = load_checkpoint_payload(checkpoint_path, device)

    block_size = args.block_size or checkpoint_config_value(
        checkpoint_payload, "data", "block_size", data_config.block_size
    )
    embedding_dim = args.embedding_dim or checkpoint_config_value(
        checkpoint_payload,
        "model",
        "embedding_dim",
        model_config.embedding_dim,
    )
    num_heads = args.num_heads or checkpoint_config_value(
        checkpoint_payload, "model", "num_heads", model_config.num_heads
    )
    num_layers = args.num_layers or checkpoint_config_value(
        checkpoint_payload, "model", "num_layers", model_config.num_layers
    )
    if not 0 <= args.layer < num_layers:
        raise ValueError(f"layer 必须位于 [0, {num_layers - 1}]。")
    if not 0 <= args.head < num_heads:
        raise ValueError(f"head 必须位于 [0, {num_heads - 1}]。")

    token_ids = tokenizer.encode(args.prompt)
    if len(token_ids) > block_size:
        raise ValueError(
            f"prompt 有 {len(token_ids)} 个 Token，超过 block_size={block_size}。"
        )

    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        expansion_factor=model_config.expansion_factor,
        tie_weights=model_config.tie_weights,
    )
    model = model.to(device)
    load_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=None,
        scheduler=None,
        device=device,
    )

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        _, attentions = model(input_ids, output_attentions=True)
    matrix = attentions[args.layer][0, args.head].float().cpu()

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "Attention 可视化需要 matplotlib：python -m pip install matplotlib"
        ) from error

    labels = [
        display_token(tokenizer.itos[token_id], index)
        for index, token_id in enumerate(token_ids)
    ]
    figure_size = max(6.0, min(16.0, len(labels) * 0.45))
    figure, axis = plt.subplots(figsize=(figure_size, figure_size))
    image = axis.imshow(
        matrix.numpy(),
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    axis.set_xticks(range(len(labels)), labels, rotation=90)
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Key：被读取的位置")
    axis.set_ylabel("Query：正在读取的位置")
    axis.set_title(f"Layer {args.layer}, Head {args.head}")
    figure.colorbar(image, ax=axis, label="Attention weight")
    figure.tight_layout()

    output_path = (
        args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    print(f"attention_shape: {tuple(matrix.shape)}")
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
