"""HTTP routes for the confined local multimodal workspace."""

from __future__ import annotations

import base64
import binascii
from dataclasses import asdict
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException

from minigpt_service.schemas import (
    AudioTranscribeRequest,
    ImageAnalyzeRequest,
    MediaOperationResponse,
    MediaUploadRequest,
    MultimodalCapabilitiesResponse,
    SpeechSynthesisRequest,
)

if TYPE_CHECKING:
    from agent.multimodal import MultimodalWorkspace


def register_multimodal_routes(
    app: FastAPI,
    workspace: MultimodalWorkspace | None,
) -> None:
    """Register media routes while keeping workspace checks in one module."""

    def require_workspace() -> MultimodalWorkspace:
        if workspace is None:
            raise HTTPException(
                status_code=503,
                detail="Multimodal workspace is not configured.",
            )
        return workspace

    @app.get(
        "/multimodal/capabilities",
        response_model=MultimodalCapabilitiesResponse,
        summary="Inspect configured local media capabilities",
    )
    def multimodal_capabilities() -> MultimodalCapabilitiesResponse:
        if workspace is None:
            return MultimodalCapabilitiesResponse(
                enabled=False,
                image_metadata=False,
                ocr=False,
                asr=False,
                tts=False,
            )
        return MultimodalCapabilitiesResponse(
            enabled=True,
            **asdict(workspace.capabilities),
        )

    @app.post(
        "/multimodal/upload",
        response_model=MediaOperationResponse,
        summary="Upload and validate a confined image or WAV file",
    )
    def upload_media(request: MediaUploadRequest) -> MediaOperationResponse:
        current = require_workspace()
        try:
            content = base64.b64decode(request.content_base64, validate=True)
            result = current.save_upload(
                request.filename,
                content,
                overwrite=request.overwrite,
            )
        except binascii.Error as error:
            raise HTTPException(
                status_code=400,
                detail="Invalid base64 media.",
            ) from error
        except FileExistsError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except (ValueError, OSError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return MediaOperationResponse(operation="upload", result=result)

    @app.post(
        "/multimodal/image/analyze",
        response_model=MediaOperationResponse,
        summary="Validate an image and optionally run grounded OCR",
    )
    def analyze_image(request: ImageAnalyzeRequest) -> MediaOperationResponse:
        try:
            result = require_workspace().analyze_image(
                request.filename,
                include_ocr=request.include_ocr,
            )
        except (FileNotFoundError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return MediaOperationResponse(operation="image", result=result)

    @app.post(
        "/multimodal/audio/transcribe",
        response_model=MediaOperationResponse,
        summary="Transcribe a confined WAV file",
    )
    def transcribe_audio(
        request: AudioTranscribeRequest,
    ) -> MediaOperationResponse:
        try:
            result = require_workspace().transcribe_audio(
                request.filename,
                language=request.language,
            )
        except (FileNotFoundError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return MediaOperationResponse(operation="asr", result=result)

    @app.post(
        "/multimodal/speech/synthesize",
        response_model=MediaOperationResponse,
        summary="Create a confined offline WAV response",
    )
    def synthesize_speech(
        request: SpeechSynthesisRequest,
    ) -> MediaOperationResponse:
        try:
            result = require_workspace().synthesize_speech(
                request.text,
                request.filename,
                rate=request.rate,
                overwrite=request.overwrite,
            )
        except FileExistsError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return MediaOperationResponse(operation="tts", result=result)
