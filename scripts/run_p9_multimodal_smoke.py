"""Create local media and run real TrOCR -> TTS -> Whisper smoke evaluation."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from agent.multimodal import (
    MultimodalWorkspace,
    Pyttsx3TTSBackend,
    TransformersTrOCRBackend,
    TransformersWhisperBackend,
)

ROOT = Path(__file__).resolve().parents[1]


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.casefold())


def edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, 1):
        current = [left_index]
        for right_index, right_value in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=ROOT / "output/multimodal")
    parser.add_argument("--model-cache", type=Path, default=ROOT / "models/p9")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "docs/p9_multimodal_smoke.json"
    )
    args = parser.parse_args()
    args.workspace.mkdir(parents=True, exist_ok=True)

    expected_ocr = "MINIGPT P9 TEST"
    image = Image.new("RGB", (760, 150), "white")
    draw = ImageDraw.Draw(image)
    font_path = Path("C:/Windows/Fonts/arialbd.ttf")
    font = ImageFont.truetype(str(font_path), 58) if font_path.is_file() else None
    draw.text((28, 38), expected_ocr, fill="black", font=font)
    image_path = args.workspace / "p9_ocr_input.png"
    image.save(image_path)

    workspace = MultimodalWorkspace(
        args.workspace,
        ocr_backend=TransformersTrOCRBackend(args.model_cache / "ocr"),
        asr_backend=TransformersWhisperBackend(args.model_cache / "asr"),
        tts_backend=Pyttsx3TTSBackend(),
    )
    expected_asr = "Mini GPT multimodal test"

    tts_started = time.perf_counter()
    tts = workspace.synthesize_speech(
        expected_asr, "p9_tts_output.wav", rate=165, overwrite=True
    )
    tts_ms = (time.perf_counter() - tts_started) * 1000

    ocr_started = time.perf_counter()
    ocr = workspace.analyze_image("p9_ocr_input.png", include_ocr=True)
    ocr_ms = (time.perf_counter() - ocr_started) * 1000

    asr_started = time.perf_counter()
    asr = workspace.transcribe_audio("p9_tts_output.wav", language="en")
    asr_ms = (time.perf_counter() - asr_started) * 1000

    warm_ocr_started = time.perf_counter()
    workspace.analyze_image("p9_ocr_input.png", include_ocr=True)
    warm_ocr_ms = (time.perf_counter() - warm_ocr_started) * 1000
    warm_asr_started = time.perf_counter()
    workspace.transcribe_audio("p9_tts_output.wav", language="en")
    warm_asr_ms = (time.perf_counter() - warm_asr_started) * 1000

    expected_ocr_normalized = normalize(expected_ocr)
    actual_ocr_normalized = normalize(str(ocr["ocr_text"]))
    expected_asr_normalized = normalize(expected_asr)
    actual_asr_normalized = normalize(str(asr["text"]))
    ocr_cer = edit_distance(expected_ocr_normalized, actual_ocr_normalized) / max(
        1, len(expected_ocr_normalized)
    )
    asr_cer = edit_distance(expected_asr_normalized, actual_asr_normalized) / max(
        1, len(expected_asr_normalized)
    )
    model_bytes = sum(
        path.stat().st_size for path in args.model_cache.rglob("*") if path.is_file()
    )
    payload = {
        "date": "2026-08-25",
        "author": "李林知",
        "status": "real_local_model_smoke",
        "expected_ocr": expected_ocr,
        "actual_ocr": ocr["ocr_text"],
        "ocr_character_error_rate": ocr_cer,
        "expected_asr": expected_asr,
        "actual_asr": asr["text"],
        "asr_character_error_rate": asr_cer,
        "tts": tts,
        "cold_latency_ms": {"ocr": ocr_ms, "tts": tts_ms, "asr": asr_ms},
        "warm_latency_ms": {"ocr": warm_ocr_ms, "asr": warm_asr_ms},
        "model_cache_bytes": model_bytes,
        "model_cache_mib": model_bytes / 1024**2,
        "workspace": str(args.workspace),
        "limitations": [
            "OCR 只验证单行印刷体图片，不代表复杂截图、表格或扫描件能力。",
            "ASR 只验证操作系统 TTS 合成的短英文，不能替代真实噪声语音 WER。",
            "首次运行包含模型下载和加载时间，后续热路径应单独测量。",
            "P9 采用 OCR/ASR/LLM/TTS 级联，不是端到端多模态联合训练。",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
