"""Confined image/audio tools with replaceable OCR, ASR, and TTS backends."""

from __future__ import annotations

import hashlib
import re
import wave
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from agent.tools.base import ToolSpec
from agent.tools.registry import ToolRegistry

_SAFE_MEDIA_FILENAME = re.compile(r"^[\w\-. ]{1,128}$", re.UNICODE)
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
_AUDIO_SUFFIXES = {".wav"}


class ImageAnalyzeArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")
    filename: str = Field(min_length=1, max_length=128)
    include_ocr: bool = True


class AudioTranscribeArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")
    filename: str = Field(min_length=1, max_length=128)
    language: str | None = Field(default=None, max_length=16)


class SpeechSynthesisArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=4000)
    filename: str = Field(min_length=1, max_length=128)
    rate: int = Field(default=180, ge=80, le=360)
    overwrite: bool = False


class OCRBackend(Protocol):
    def recognize(self, path: Path) -> str: ...


class ASRBackend(Protocol):
    def transcribe(self, path: Path, *, language: str | None = None) -> str: ...


class TTSBackend(Protocol):
    def synthesize(self, text: str, path: Path, *, rate: int) -> None: ...


class TextResponder(Protocol):
    def respond(self, text: str) -> str: ...


@dataclass(frozen=True)
class MediaCapabilities:
    image_metadata: bool
    ocr: bool
    asr: bool
    tts: bool


class AgentTextResponder:
    """Adapt an AgentOrchestrator to the narrow multimodal text-stage protocol."""

    def __init__(self, agent: Any) -> None:
        self.agent = agent

    def respond(self, text: str) -> str:
        from agent.messages import Message

        result = self.agent.run([Message(role="user", content=text)], greedy=True)
        return str(result.text)


class TransformersTrOCRBackend:
    """Lazy local/downloadable TrOCR adapter; model is loaded only on first use."""

    def __init__(
        self,
        cache_dir: Path,
        model_name: str = "microsoft/trocr-small-printed",
        device: str = "auto",
    ) -> None:
        self.cache_dir = cache_dir
        self.model_name = model_name
        self.device_name = device
        self._processor: Any | None = None
        self._model: Any | None = None

    def _load(self) -> tuple[Any, Any, str]:
        if self._processor is None or self._model is None:
            import torch
            from transformers import (
                DeiTImageProcessorPil,
                TrOCRProcessor,
                VisionEncoderDecoderModel,
                XLMRobertaTokenizer,
            )

            self.cache_dir.mkdir(parents=True, exist_ok=True)
            image_processor = DeiTImageProcessorPil.from_pretrained(
                self.model_name, cache_dir=self.cache_dir
            )
            tokenizer = XLMRobertaTokenizer.from_pretrained(
                self.model_name, cache_dir=self.cache_dir
            )
            self._processor = TrOCRProcessor(image_processor, tokenizer)
            model: Any = VisionEncoderDecoderModel.from_pretrained(
                self.model_name, cache_dir=self.cache_dir
            )
            selected = (
                "cuda"
                if self.device_name == "auto" and torch.cuda.is_available()
                else ("cpu" if self.device_name == "auto" else self.device_name)
            )
            model.to(selected).eval()
            self._model = model
        return self._processor, self._model, str(next(self._model.parameters()).device)

    def recognize(self, path: Path) -> str:
        from PIL import Image

        processor, model, device = self._load()
        image = Image.open(path).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(
            device
        )
        generated_ids = model.generate(pixel_values, max_new_tokens=128)
        return str(processor.batch_decode(generated_ids, skip_special_tokens=True)[0])


class TransformersWhisperBackend:
    """Lazy Whisper adapter using the existing Transformers dependency."""

    def __init__(
        self,
        cache_dir: Path,
        model_name: str = "openai/whisper-tiny",
        device: str = "auto",
    ) -> None:
        self.cache_dir = cache_dir
        self.model_name = model_name
        self.device_name = device
        self._processor: Any | None = None
        self._model: Any | None = None

    def _load(self) -> tuple[Any, Any, str]:
        if self._processor is None or self._model is None:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._processor = AutoProcessor.from_pretrained(
                self.model_name, cache_dir=self.cache_dir
            )
            self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_name, cache_dir=self.cache_dir
            )
            selected = (
                "cuda"
                if self.device_name == "auto" and torch.cuda.is_available()
                else ("cpu" if self.device_name == "auto" else self.device_name)
            )
            self._model.to(selected).eval()
        return self._processor, self._model, str(next(self._model.parameters()).device)

    @staticmethod
    def _read_mono_wav(path: Path) -> tuple[Any, int]:
        import numpy as np

        with wave.open(str(path), "rb") as stream:
            channels = stream.getnchannels()
            sample_width = stream.getsampwidth()
            sample_rate = stream.getframerate()
            frames = stream.readframes(stream.getnframes())
        if sample_width != 2:
            raise ValueError("Whisper adapter currently requires 16-bit PCM WAV.")
        samples = np.frombuffer(frames, dtype="<i2").astype("float32") / 32768.0
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)
        return samples, sample_rate

    def transcribe(self, path: Path, *, language: str | None = None) -> str:
        import numpy as np

        processor, model, device = self._load()
        samples, sample_rate = self._read_mono_wav(path)
        if sample_rate != 16_000:
            target_length = max(1, round(len(samples) * 16_000 / sample_rate))
            source_positions = np.linspace(0.0, 1.0, num=len(samples), endpoint=False)
            target_positions = np.linspace(0.0, 1.0, num=target_length, endpoint=False)
            samples = np.interp(target_positions, source_positions, samples).astype(
                "float32"
            )
            sample_rate = 16_000
        inputs = processor(
            samples, sampling_rate=sample_rate, return_tensors="pt"
        ).input_features.to(device)
        options: dict[str, object] = {"max_new_tokens": 128}
        if language:
            options["language"] = language
            options["task"] = "transcribe"
        generated_ids = model.generate(inputs, **options)
        return str(processor.batch_decode(generated_ids, skip_special_tokens=True)[0])


class Pyttsx3TTSBackend:
    """Offline TTS adapter using operating-system voices through pyttsx3."""

    def synthesize(self, text: str, path: Path, *, rate: int) -> None:
        import pyttsx3  # type: ignore[import-untyped]

        engine = pyttsx3.init()
        try:
            engine.setProperty("rate", rate)
            engine.save_to_file(text, str(path))
            engine.runAndWait()
        finally:
            engine.stop()
        if not path.is_file() or path.stat().st_size <= 44:
            raise RuntimeError("TTS backend did not create a valid WAV file.")


class MultimodalWorkspace:
    """Keep media access inside one explicit directory and expose bounded tasks."""

    def __init__(
        self,
        root: Path,
        *,
        ocr_backend: OCRBackend | None = None,
        asr_backend: ASRBackend | None = None,
        tts_backend: TTSBackend | None = None,
        max_image_bytes: int = 20 * 1024 * 1024,
        max_audio_bytes: int = 50 * 1024 * 1024,
    ) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.ocr_backend = ocr_backend
        self.asr_backend = asr_backend
        self.tts_backend = tts_backend
        self.max_image_bytes = max_image_bytes
        self.max_audio_bytes = max_audio_bytes

    @property
    def capabilities(self) -> MediaCapabilities:
        return MediaCapabilities(
            image_metadata=True,
            ocr=self.ocr_backend is not None,
            asr=self.asr_backend is not None,
            tts=self.tts_backend is not None,
        )

    def _resolve(self, filename: str, allowed_suffixes: set[str]) -> Path:
        if not _SAFE_MEDIA_FILENAME.fullmatch(filename):
            raise ValueError("filename must be a plain safe file name.")
        path = (self.root / filename).resolve()
        if path.parent != self.root:
            raise ValueError("Media path must remain inside the configured workspace.")
        if path.suffix.lower() not in allowed_suffixes:
            allowed = ", ".join(sorted(allowed_suffixes))
            raise ValueError(
                f"Unsupported media extension; expected one of: {allowed}."
            )
        return path

    @staticmethod
    def _check_existing(path: Path, max_bytes: int) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Media file does not exist: {path.name}")
        size = path.stat().st_size
        if size <= 0 or size > max_bytes:
            raise ValueError(
                "Media file is empty or exceeds the configured size limit."
            )

    def save_upload(
        self, filename: str, content: bytes, *, overwrite: bool = False
    ) -> dict[str, object]:
        suffix = Path(filename).suffix.lower()
        allowed = _IMAGE_SUFFIXES | _AUDIO_SUFFIXES
        path = self._resolve(filename, allowed)
        limit = (
            self.max_image_bytes if suffix in _IMAGE_SUFFIXES else self.max_audio_bytes
        )
        if not content or len(content) > limit:
            raise ValueError(
                "Uploaded media is empty or exceeds the configured size limit."
            )
        if path.exists() and not overwrite:
            raise FileExistsError("Refusing to overwrite an existing media file.")
        temporary = path.with_suffix(path.suffix + ".uploading")
        temporary.write_bytes(content)
        try:
            if suffix in _IMAGE_SUFFIXES:
                from PIL import Image

                with Image.open(temporary) as image:
                    image.verify()
            else:
                with wave.open(str(temporary), "rb") as stream:
                    if stream.getnframes() <= 0:
                        raise ValueError("Uploaded WAV contains no audio frames.")
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return {
            "filename": path.name,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(content).hexdigest(),
            "validated": True,
        }

    def analyze_image(
        self, filename: str, include_ocr: bool = True
    ) -> dict[str, object]:
        from PIL import Image

        path = self._resolve(filename, _IMAGE_SUFFIXES)
        self._check_existing(path, self.max_image_bytes)
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
            image_format = image.format or path.suffix.removeprefix(".").upper()
        ocr_text = ""
        if include_ocr:
            if self.ocr_backend is None:
                raise RuntimeError("OCR backend is not configured.")
            ocr_text = self.ocr_backend.recognize(path).strip()
        return {
            "filename": path.name,
            "format": image_format,
            "width": width,
            "height": height,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "ocr_text": ocr_text,
            "regions": (
                [{"x": 0, "y": 0, "width": width, "height": height}] if ocr_text else []
            ),
        }

    def transcribe_audio(
        self, filename: str, language: str | None = None
    ) -> dict[str, object]:
        path = self._resolve(filename, _AUDIO_SUFFIXES)
        self._check_existing(path, self.max_audio_bytes)
        if self.asr_backend is None:
            raise RuntimeError("ASR backend is not configured.")
        with wave.open(str(path), "rb") as stream:
            duration = stream.getnframes() / stream.getframerate()
            sample_rate = stream.getframerate()
            channels = stream.getnchannels()
        return {
            "filename": path.name,
            "text": self.asr_backend.transcribe(path, language=language).strip(),
            "duration_seconds": duration,
            "sample_rate": sample_rate,
            "channels": channels,
        }

    def synthesize_speech(
        self,
        text: str,
        filename: str,
        rate: int = 180,
        overwrite: bool = False,
    ) -> dict[str, object]:
        path = self._resolve(filename, _AUDIO_SUFFIXES)
        if path.exists() and not overwrite:
            raise FileExistsError("Refusing to overwrite an existing audio file.")
        if self.tts_backend is None:
            raise RuntimeError("TTS backend is not configured.")
        self.tts_backend.synthesize(text, path, rate=rate)
        self._check_existing(path, self.max_audio_bytes)
        with wave.open(str(path), "rb") as stream:
            duration = stream.getnframes() / stream.getframerate()
            sample_rate = stream.getframerate()
        return {
            "filename": path.name,
            "bytes": path.stat().st_size,
            "duration_seconds": duration,
            "sample_rate": sample_rate,
            "validated": True,
        }


class MultimodalConversationPipeline:
    """Compose independent media and language stages while preserving timings."""

    def __init__(
        self, workspace: MultimodalWorkspace, responder: TextResponder
    ) -> None:
        self.workspace = workspace
        self.responder = responder

    def answer_image(
        self, filename: str, question: str, *, include_ocr: bool = True
    ) -> dict[str, object]:
        started = perf_counter()
        image = self.workspace.analyze_image(filename, include_ocr=include_ocr)
        vision_ms = (perf_counter() - started) * 1000
        grounded_prompt = (
            f"用户问题：{question}\n"
            f"图片文件：{image['filename']}，尺寸 {image['width']}x{image['height']}。\n"
            f"OCR：{image['ocr_text']}\n"
            "只根据上述图片元数据和 OCR 回答；若证据不足请明确说明。"
        )
        started = perf_counter()
        answer = self.responder.respond(grounded_prompt)
        llm_ms = (perf_counter() - started) * 1000
        return {
            "answer": answer,
            "grounding": image,
            "latency_ms": {"vision_ocr": vision_ms, "llm": llm_ms},
        }

    def voice_to_voice(
        self,
        input_filename: str,
        output_filename: str,
        *,
        language: str | None = None,
        overwrite: bool = False,
    ) -> dict[str, object]:
        started = perf_counter()
        transcript = self.workspace.transcribe_audio(input_filename, language=language)
        asr_ms = (perf_counter() - started) * 1000
        started = perf_counter()
        answer = self.responder.respond(str(transcript["text"]))
        llm_ms = (perf_counter() - started) * 1000
        started = perf_counter()
        audio = self.workspace.synthesize_speech(
            answer, output_filename, overwrite=overwrite
        )
        tts_ms = (perf_counter() - started) * 1000
        return {
            "transcript": transcript,
            "answer": answer,
            "audio": audio,
            "latency_ms": {"asr": asr_ms, "llm": llm_ms, "tts": tts_ms},
        }


def register_multimodal_tools(
    registry: ToolRegistry, workspace: MultimodalWorkspace
) -> ToolRegistry:
    registry.register(
        ToolSpec(
            name="analyze_image",
            description="Validate an image and optionally extract grounded OCR text.",
            arguments_model=ImageAnalyzeArguments,
            handler=workspace.analyze_image,
            timeout_seconds=120.0,
        )
    )
    registry.register(
        ToolSpec(
            name="transcribe_audio",
            description="Transcribe a confined WAV file with the configured ASR backend.",
            arguments_model=AudioTranscribeArguments,
            handler=workspace.transcribe_audio,
            timeout_seconds=120.0,
        )
    )
    registry.register(
        ToolSpec(
            name="synthesize_speech",
            description="Create and validate a WAV response with the configured TTS backend.",
            arguments_model=SpeechSynthesisArguments,
            handler=workspace.synthesize_speech,
            side_effect=True,
            timeout_seconds=120.0,
        )
    )
    return registry
