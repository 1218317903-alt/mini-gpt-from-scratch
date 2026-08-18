"""划分文本并保存训练集词表。"""

from pathlib import Path

from minigpt.config import load_data_config
from minigpt.tokenizer import CharacterTokenizer
from minigpt.utils import read_utf8_text, write_utf8_text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "tiny_shakespeare.yaml"


def read_text(path: Path) -> str:
    """读取非空 UTF-8 文本。"""
    return read_utf8_text(path)


def split_text(
    text: str,
    train_ratio: float,
    val_ratio: float,
) -> tuple[str, str, str]:
    """按原始顺序划分训练集、验证集和测试集。"""
    if not text:
        raise ValueError("text 不能为空。")
    if train_ratio <= 0 or val_ratio <= 0:
        raise ValueError("数据集比例必须大于 0。")
    if train_ratio + val_ratio >= 1:
        raise ValueError("训练集和验证集比例之和必须小于 1。")

    train_end = int(len(text) * train_ratio)
    val_end = int(len(text) * (train_ratio + val_ratio))

    if train_end <= 0 or val_end <= train_end or val_end >= len(text):
        raise ValueError("文本过短，无法生成三个非空数据集。")

    return (
        text[:train_end],
        text[train_end:val_end],
        text[val_end:],
    )


def save_text(path: Path, text: str) -> None:
    """保存 UTF-8 文本。"""
    write_utf8_text(path, text)


def main() -> None:
    """生成三个文本数据集和训练集词表。"""
    config = load_data_config(CONFIG_PATH)
    text = read_text(config.raw_path)

    train_text, val_text, test_text = split_text(
        text,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
    )

    save_text(config.train_path, train_text)
    save_text(config.val_path, val_text)
    save_text(config.test_path, test_text)

    tokenizer = CharacterTokenizer(train_text)
    tokenizer.save(config.tokenizer_path)

    print(f"原始文本：{len(text)}")
    print(f"训练集：{len(train_text)}")
    print(f"验证集：{len(val_text)}")
    print(f"测试集：{len(test_text)}")
    print(f"词表大小：{tokenizer.vocab_size}")


if __name__ == "__main__":
    main()
