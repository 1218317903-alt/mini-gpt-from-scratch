"""检查 Dataset 和 DataLoader 的 Batch。"""

from pathlib import Path

from torch.utils.data import DataLoader

from minigpt.config import load_data_config
from minigpt.dataset import CharacterLanguageModelDataset
from minigpt.tokenizer import CharacterTokenizer
from minigpt.utils import read_utf8_text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "tiny_shakespeare.yaml"


def main() -> None:
    """打印形状、类型，并验证 x/y 错位。"""
    config = load_data_config(CONFIG_PATH)
    tokenizer = CharacterTokenizer.load(config.tokenizer_path)

    train_text = read_utf8_text(config.train_path)
    val_text = read_utf8_text(config.val_path)
    test_text = read_utf8_text(config.test_path)

    train_dataset = CharacterLanguageModelDataset(
        tokenizer.encode(train_text),
        config.block_size,
    )
    val_dataset = CharacterLanguageModelDataset(
        tokenizer.encode(val_text, allow_unknown=True),
        config.block_size,
    )
    test_dataset = CharacterLanguageModelDataset(
        tokenizer.encode(test_text, allow_unknown=True),
        config.block_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    x_batch, y_batch = next(iter(train_loader))
    first_x = tokenizer.decode(x_batch[0].tolist())
    first_y = tokenizer.decode(y_batch[0].tolist())

    print(f"词表大小：{tokenizer.vocab_size}")
    print(f"训练集样本数：{len(train_dataset)}")
    print(f"验证集样本数：{len(val_dataset)}")
    print(f"测试集样本数：{len(test_dataset)}")
    print(f"x.shape：{tuple(x_batch.shape)}")
    print(f"y.shape：{tuple(y_batch.shape)}")
    print(f"x.dtype：{x_batch.dtype}")
    print(f"y.dtype：{y_batch.dtype}")
    print(f"第一个 x：{first_x!r}")
    print(f"第一个 y：{first_y!r}")

    assert tuple(x_batch.shape) == (
        config.batch_size,
        config.block_size,
    )
    assert tuple(y_batch.shape) == (
        config.batch_size,
        config.block_size,
    )
    assert x_batch.dtype == y_batch.dtype
    assert (x_batch[:, 1:] == y_batch[:, :-1]).all()

    print("Batch 检查通过。")


if __name__ == "__main__":
    main()
