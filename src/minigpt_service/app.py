"""FastAPI application factory for the MiniGPT inference service."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from time import perf_counter
from typing import TYPE_CHECKING
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import RequestResponseEndpoint

from minigpt_service.config import ChatServiceConfig, ServiceConfig
from minigpt_service.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    ReadyResponse,
)

if TYPE_CHECKING:
    from minigpt_service.chat_model import ChatModel
    from minigpt_service.model_service import ModelService

SERVICE_NAME = "MiniGPT Inference Service"
SERVICE_VERSION = "0.2.0"
REQUEST_ID_HEADER = "X-Request-ID"
logger = logging.getLogger("minigpt_service.http")


def _new_request_id() -> str:
    return uuid4().hex


def _request_id_from_header(request: Request) -> str:
    candidate = request.headers.get(REQUEST_ID_HEADER)
    if (
        candidate
        and len(candidate) <= 128
        and all(character not in "\r\n" for character in candidate)
    ):
        return candidate
    return _new_request_id()


def _request_id(request: Request) -> str:
    value = getattr(request.state, "request_id", None)
    if isinstance(value, str) and value:
        return value
    value = _new_request_id()
    request.state.request_id = value
    return value


def _error_response(
    request: Request,
    *,
    status_code: int,
    detail: object,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    request_id = _request_id(request)
    response_headers = dict(headers or {})
    response_headers[REQUEST_ID_HEADER] = request_id
    return JSONResponse(
        status_code=status_code,
        content=jsonable_encoder(
            ErrorResponse(detail=detail, request_id=request_id).model_dump()
        ),
        headers=response_headers,
    )


def _log_request(
    request: Request,
    *,
    status_code: int,
    duration_ms: float,
) -> None:
    payload = {
        "event": "http_request",
        "request_id": _request_id(request),
        "method": request.method,
        "path": request.url.path,
        "status_code": status_code,
        "duration_ms": round(duration_ms, 3),
    }
    logger.info(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def create_app(
    model_service: ModelService | None = None,
    chat_model: ChatModel | None = None,
) -> FastAPI:
    """Create a fresh FastAPI application instance."""

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            if model_service is not None:
                model_service.load()
                _app.state.model_service = model_service
            if chat_model is not None:
                chat_model.load()
                _app.state.chat_model = chat_model
            yield
        finally:
            if model_service is not None:
                model_service.close()
            if chat_model is not None:
                chat_model.close()

    service_app = FastAPI(
        title=SERVICE_NAME,
        version=SERVICE_VERSION,
        description=(
            "HTTP inference service for the from-scratch MiniGPT and "
            "Hugging Face chat backends."
        ),
        lifespan=lifespan,
    )

    @service_app.middleware("http")
    async def request_context(
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        request.state.request_id = _request_id_from_header(request)
        started = perf_counter()
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = _request_id(request)
        _log_request(
            request,
            status_code=response.status_code,
            duration_ms=(perf_counter() - started) * 1000,
        )
        return response

    @service_app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request,
        error: HTTPException,
    ) -> JSONResponse:
        return _error_response(
            request,
            status_code=error.status_code,
            detail=error.detail,
            headers=error.headers,
        )

    @service_app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        error: RequestValidationError,
    ) -> JSONResponse:
        return _error_response(
            request,
            status_code=422,
            detail=error.errors(),
        )

    @service_app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request,
        error: Exception,
    ) -> JSONResponse:
        request_id = _request_id(request)
        logger.exception(
            json.dumps(
                {
                    "event": "http_exception",
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "exception": type(error).__name__,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return _error_response(
            request,
            status_code=500,
            detail="Internal server error.",
        )

    @service_app.get(
        "/health",
        response_model=HealthResponse,
        summary="Check whether the service process is alive",
    )
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            service=SERVICE_NAME,
            version=SERVICE_VERSION,
        )

    @service_app.get(
        "/ready",
        response_model=ReadyResponse,
        summary="Check whether a model is ready for inference",
    )
    def ready() -> ReadyResponse:
        model_ready = model_service is not None and model_service.is_ready
        chat_ready = chat_model is not None and chat_model.is_ready
        if not model_ready and not chat_ready:
            raise HTTPException(
                status_code=503,
                detail="No model backend is ready.",
            )
        return ReadyResponse(status="ready")

    @service_app.post(
        "/generate",
        response_model=GenerateResponse,
        summary="Generate text from a prompt",
    )
    def generate_text(request: GenerateRequest) -> GenerateResponse:
        if model_service is None or not model_service.is_ready:
            raise HTTPException(
                status_code=503,
                detail="Model service is not ready.",
            )

        try:
            generated_text = model_service.generate_text(
                request.prompt,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                greedy=request.greedy,
                use_kv_cache=request.use_kv_cache,
                seed=request.seed,
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

        # MiniGPT is character-level, so the continuation length is the
        # generated string length minus the prompt length.
        generated_tokens = max(0, len(generated_text) - len(request.prompt))
        return GenerateResponse(
            prompt=request.prompt,
            generated_text=generated_text,
            generated_tokens=generated_tokens,
            finish_reason="length",
        )

    @service_app.post(
        "/chat",
        response_model=ChatResponse,
        summary="Generate an assistant response from chat messages",
    )
    def chat_text(request: ChatRequest) -> ChatResponse:
        if chat_model is None or not chat_model.is_ready:
            raise HTTPException(
                status_code=503,
                detail="Chat model service is not ready.",
            )

        try:
            result = chat_model.chat(
                request.messages,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                greedy=request.greedy,
                seed=request.seed,
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

        return ChatResponse(
            message=ChatMessage(role="assistant", content=result.text),
            generated_text=result.text,
            prompt_tokens=result.prompt_tokens,
            generated_tokens=result.generated_tokens,
            finish_reason=result.finish_reason,
            model=result.model,
        )

    return service_app


def create_app_from_environment() -> FastAPI:
    """Create the production app using ``MINIGPT_*`` environment variables."""
    service_config = ServiceConfig.from_environment()
    chat_config = ChatServiceConfig.from_environment()
    if service_config is None and chat_config is None:
        return create_app()

    model_service = None
    chat_model = None
    if service_config is not None:
        from minigpt_service.model_service import ModelService

        model_service = ModelService(service_config)
    if chat_config is not None:
        from minigpt_service.chat_model import create_huggingface_chat_model
        from minigpt_service.model_service import select_device

        chat_model = create_huggingface_chat_model(
            chat_config,
            select_device(chat_config.device),
        )

    return create_app(model_service, chat_model)


app = create_app_from_environment()
