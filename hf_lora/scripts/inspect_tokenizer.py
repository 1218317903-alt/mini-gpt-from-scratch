"""展示 Tokenizer 编解码与聊天模板。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from hf_lora.modeling import load_tokenizer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-source", default="Qwen/Qwen2.5-0.5B-Instruct")
    args = parser.parse_args()
    tokenizer = load_tokenizer(args.model_source)
    text = "LoRA 只训练少量增量参数。"
    ids = tokenizer.encode(text, add_special_tokens=False)
    print("text:", text)
    print("input_ids:", ids)
    print("tokens:", tokenizer.convert_ids_to_tokens(ids))
    print("decoded:", tokenizer.decode(ids))
    messages = [
        {"role": "system", "content": "你是 MiniGPT 课程助教。"},
        {"role": "user", "content": "什么是 LoRA？"},
    ]
    print(
        tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    )


if __name__ == "__main__":
    main()
