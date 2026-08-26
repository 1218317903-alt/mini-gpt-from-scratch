"""P9 confined image, ASR, TTS, routing, and HTTP tests."""

from __future__ import annotations

import base64
import io
import wave
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from agent.builtin_tools import build_default_registry
from agent.messages import Message
from agent.multimodal import MultimodalConversationPipeline, MultimodalWorkspace
from agent.routing import HeuristicAgentRouter
from minigpt_service.app import create_app


class FakeOCR:
    def recognize(self, path: Path) -> str:
        assert path.is_file()
        return "MiniGPT P9 OCR"


class FakeASR:
    def transcribe(self, path: Path, *, language: str | None = None) -> str:
        assert language == "zh"
        assert path.is_file()
        return "这是语音转写"


class FakeTTS:
    def synthesize(self, text: str, path: Path, *, rate: int) -> None:
        assert text
        assert rate == 180
        with wave.open(str(path), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16_000)
            stream.writeframes(b"\x00\x00" * 1600)


def png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (96, 48), color="white").save(buffer, format="PNG")
    return buffer.getvalue()


def wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(b"\x00\x00" * 1600)
    return buffer.getvalue()


def workspace(tmp_path: Path) -> MultimodalWorkspace:
    return MultimodalWorkspace(
        tmp_path,
        ocr_backend=FakeOCR(),
        asr_backend=FakeASR(),
        tts_backend=FakeTTS(),
    )


def test_upload_image_ocr_audio_and_tts_closed_loop(tmp_path: Path) -> None:
    media = workspace(tmp_path)
    upload = media.save_upload("screen.png", png_bytes())
    assert upload["validated"] is True
    image = media.analyze_image("screen.png")
    assert image["ocr_text"] == "MiniGPT P9 OCR"
    assert image["regions"] == [{"x": 0, "y": 0, "width": 96, "height": 48}]

    media.save_upload("input.wav", wav_bytes())
    transcript = media.transcribe_audio("input.wav", language="zh")
    assert transcript["text"] == "这是语音转写"
    speech = media.synthesize_speech("你好", "answer.wav")
    assert speech["validated"] is True
    assert speech["duration_seconds"] == 0.1


def test_media_workspace_rejects_traversal_and_silent_overwrite(
    tmp_path: Path,
) -> None:
    media = workspace(tmp_path)
    for filename in ("../escape.png", "folder/image.png", "image.exe"):
        try:
            media.save_upload(filename, png_bytes())
        except ValueError:
            pass
        else:
            raise AssertionError("unsafe filename should fail")
    media.save_upload("safe.png", png_bytes())
    try:
        media.save_upload("safe.png", png_bytes())
    except FileExistsError:
        pass
    else:
        raise AssertionError("silent overwrite should fail")


def test_multimodal_tools_are_registered_and_tts_requires_confirmation(
    tmp_path: Path,
) -> None:
    registry = build_default_registry(multimodal_workspace=workspace(tmp_path))
    assert {"analyze_image", "transcribe_audio", "synthesize_speech"}.issubset(
        set(registry.names())
    )
    tts = registry.get("synthesize_speech")
    assert tts is not None
    assert tts.side_effect is True


def test_router_selects_multimodal_tools() -> None:
    router = HeuristicAgentRouter()
    image = router.route(
        [Message(role="user", content="请 OCR 这张截图")], ["analyze_image"]
    )
    speech = router.route(
        [Message(role="user", content="把回答文字转语音")], ["synthesize_speech"]
    )
    assert image.kind == "multimodal"
    assert image.allowed_tools == ("analyze_image",)
    assert speech.kind == "multimodal"


def test_multimodal_http_upload_and_operations(tmp_path: Path) -> None:
    client = TestClient(create_app(multimodal_workspace=workspace(tmp_path)))
    capabilities = client.get("/multimodal/capabilities")
    assert capabilities.status_code == 200
    assert capabilities.json()["ocr"] is True

    upload = client.post(
        "/multimodal/upload",
        json={
            "filename": "screen.png",
            "content_base64": base64.b64encode(png_bytes()).decode("ascii"),
        },
    )
    assert upload.status_code == 200
    analyzed = client.post(
        "/multimodal/image/analyze",
        json={"filename": "screen.png", "include_ocr": True},
    )
    assert analyzed.status_code == 200
    assert analyzed.json()["result"]["ocr_text"] == "MiniGPT P9 OCR"

    audio_upload = client.post(
        "/multimodal/upload",
        json={
            "filename": "input.wav",
            "content_base64": base64.b64encode(wav_bytes()).decode("ascii"),
        },
    )
    assert audio_upload.status_code == 200
    asr = client.post(
        "/multimodal/audio/transcribe",
        json={"filename": "input.wav", "language": "zh"},
    )
    assert asr.status_code == 200
    tts = client.post(
        "/multimodal/speech/synthesize",
        json={"text": "你好", "filename": "answer.wav"},
    )
    assert tts.status_code == 200


def test_multimodal_http_is_explicitly_disabled_without_workspace() -> None:
    client = TestClient(create_app())
    assert client.get("/multimodal/capabilities").json()["enabled"] is False
    response = client.post(
        "/multimodal/image/analyze",
        json={"filename": "screen.png", "include_ocr": False},
    )
    assert response.status_code == 503


class FakeResponder:
    def respond(self, text: str) -> str:
        return f"回答：{text[:20]}"


def test_multimodal_pipeline_reports_each_stage_separately(tmp_path: Path) -> None:
    media = workspace(tmp_path)
    media.save_upload("screen.png", png_bytes())
    media.save_upload("input.wav", wav_bytes())
    pipeline = MultimodalConversationPipeline(media, FakeResponder())
    image_result = pipeline.answer_image("screen.png", "图片写了什么？")
    assert image_result["grounding"]["ocr_text"] == "MiniGPT P9 OCR"
    assert set(image_result["latency_ms"]) == {"vision_ocr", "llm"}
    voice_result = pipeline.voice_to_voice("input.wav", "reply.wav", language="zh")
    assert set(voice_result["latency_ms"]) == {"asr", "llm", "tts"}
    assert voice_result["audio"]["validated"] is True
