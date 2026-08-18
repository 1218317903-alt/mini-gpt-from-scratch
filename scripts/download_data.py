"""下载 Tiny Shakespeare 原始文本。"""

from pathlib import Path
from urllib.request import Request, urlopen

from minigpt.config import load_data_config
from minigpt.utils import write_utf8_text

DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "tiny_shakespeare.yaml"


def main() -> None:
    """下载并保存原始文本。"""
    config = load_data_config(CONFIG_PATH)
    output_path = config.raw_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        print(f"文件已存在，跳过下载：{output_path}")
        return

    request = Request(
        DATA_URL,
        headers={"User-Agent": "mini-gpt-from-scratch"},
    )

    with urlopen(request, timeout=30) as response:
        text = response.read().decode("utf-8")

    if not text.strip():
        raise ValueError("下载到的文本为空。")

    write_utf8_text(output_path, text)
    print(f"下载完成：{output_path}")
    print(f"字符数量：{len(text)}")


if __name__ == "__main__":
    main()
