"""下载 Tiny Shakespeare 原始文本。"""

from pathlib import Path
from urllib.request import Request, urlopen


DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "tiny_shakespeare.txt"


def main() -> None:
    """下载并保存原始文本。"""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists():
        print(f"文件已存在，跳过下载：{OUTPUT_PATH}")
        return

    request = Request(
        DATA_URL,
        headers={"User-Agent": "mini-gpt-from-scratch"},
    )

    with urlopen(request, timeout=30) as response:
        content = response.read()

    text = content.decode("utf-8")

    if not text.strip():
        raise ValueError("下载到的文本为空。")

    OUTPUT_PATH.write_text(text, encoding="utf-8")

    print(f"下载完成：{OUTPUT_PATH}")
    print(f"字符数量：{len(text)}")


if __name__ == "__main__":
    main()