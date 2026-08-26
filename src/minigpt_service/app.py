"""FastAPI application factory for the MiniGPT inference service."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)
from starlette.middleware.base import RequestResponseEndpoint

from minigpt_service.agent_runtime import AgentRunManager
from minigpt_service.config import ChatServiceConfig, ServiceConfig
from minigpt_service.conversation_routes import build_conversation_router
from minigpt_service.conversation_runtime import ConversationRunManager
from minigpt_service.conversations import (
    ConversationContextBuilder,
    ConversationRepository,
    SQLiteConversationRepository,
)
from minigpt_service.domain_runtime import DomainRuntime
from minigpt_service.multimodal_routes import register_multimodal_routes
from minigpt_service.schemas import (
    AgentCancelResponse,
    AgentEventResponse,
    AgentMetricsResponse,
    AgentRequest,
    AgentResponse,
    AgentTraceResponse,
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
    from agent.domain_adapter import DomainAdapterRegistry
    from agent.multimodal import MultimodalWorkspace
    from agent.orchestrator import AgentOrchestrator, AgentProgressEvent, AgentResult
    from minigpt_service.chat_model import ChatModel
    from minigpt_service.model_service import ModelService

SERVICE_NAME = "MiniGPT Inference Service"
SERVICE_VERSION = "0.6.0"
REQUEST_ID_HEADER = "X-Request-ID"
logger = logging.getLogger("minigpt_service.http")

_DEMO_HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>MiniGPT Agent Demo</title><style>
body{font:16px system-ui;max-width:920px;margin:40px auto;padding:0 20px;color:#172033;background:#f6f8fb}
main{background:white;border:1px solid #dce3ed;border-radius:14px;padding:24px;box-shadow:0 8px 30px #20304a12}
textarea{width:100%;min-height:110px;box-sizing:border-box;padding:12px;border:1px solid #b9c4d4;border-radius:8px}
button{margin-top:10px;padding:10px 18px;border:0;border-radius:8px;background:#2457d6;color:white;cursor:pointer}
pre{white-space:pre-wrap;background:#101828;color:#dbeafe;padding:16px;border-radius:8px;min-height:160px}
.muted{color:#667085;font-size:14px}</style></head><body><main>
<h1>MiniGPT 可观测 Agent</h1><p class="muted">展示 route、model、tool、plan 与 final SSE 事件；服务端执行仍受步骤、并发、确认和引用白名单约束。</p>
<textarea id="prompt">请查内部资料后计算相关数值。</textarea><br><button id="send">开始</button>
<pre id="events">等待请求...</pre></main><script>
document.querySelector('#send').onclick=async()=>{const out=document.querySelector('#events');out.textContent='';
const body={messages:[{role:'user',content:document.querySelector('#prompt').value}],greedy:true,max_steps:6};
const response=await fetch('/agent/chat/stream',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body)});
const reader=response.body.getReader(),decoder=new TextDecoder();while(true){const {value,done}=await reader.read();if(done)break;out.textContent+=decoder.decode(value);}};
</script></body></html>"""
_STATIC_ROOT = Path(__file__).with_name("static")


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
    agent: AgentOrchestrator | None = None,
    multimodal_workspace: MultimodalWorkspace | None = None,
    domain_registry: DomainAdapterRegistry | None = None,
    conversation_repository: ConversationRepository | None = None,
    conversation_context_builder: ConversationContextBuilder | None = None,
) -> FastAPI:
    """Create a fresh FastAPI application instance."""

    run_manager = AgentRunManager(max_concurrency=2)
    conversation_run_manager = ConversationRunManager(max_concurrency=2)
    domain_runtime = DomainRuntime(domain_registry, chat_model)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            if model_service is not None:
                model_service.load()
                _app.state.model_service = model_service
            if chat_model is not None:
                chat_model.load()
                _app.state.chat_model = chat_model
            if agent is not None:
                _app.state.agent = agent
            if multimodal_workspace is not None:
                _app.state.multimodal_workspace = multimodal_workspace
            if domain_registry is not None:
                _app.state.domain_registry = domain_registry
            if conversation_repository is not None:
                conversation_repository.initialize()
                _app.state.conversation_repository = conversation_repository
            yield
        finally:
            if agent is not None:
                agent.close()
            if model_service is not None:
                model_service.close()
            if chat_model is not None:
                chat_model.close()
            if conversation_repository is not None:
                conversation_repository.close()

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

    @service_app.get("/demo", response_class=HTMLResponse, include_in_schema=False)
    def demo() -> HTMLResponse:
        return HTMLResponse(_DEMO_HTML)

    @service_app.get("/chat-ui", response_class=FileResponse, include_in_schema=False)
    def chat_ui() -> FileResponse:
        """Serve the persistent local chat interface."""
        return FileResponse(_STATIC_ROOT / "chat.html", media_type="text/html")

    @service_app.get("/static/chat.css", include_in_schema=False)
    def chat_css() -> FileResponse:
        return FileResponse(_STATIC_ROOT / "chat.css", media_type="text/css")

    @service_app.get("/static/chat.js", include_in_schema=False)
    def chat_javascript() -> FileResponse:
        return FileResponse(
            _STATIC_ROOT / "chat.js",
            media_type="application/javascript",
        )

    @service_app.get(
        "/ready",
        response_model=ReadyResponse,
        summary="Check whether a model is ready for inference",
    )
    def ready() -> ReadyResponse:
        model_ready = model_service is not None and model_service.is_ready
        chat_ready = chat_model is not None and chat_model.is_ready
        agent_ready = agent is not None and agent.is_ready
        if (
            not model_ready
            and not chat_ready
            and not agent_ready
            and multimodal_workspace is None
        ):
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
            from agent.messages import as_message

            result = domain_runtime.run(
                request,
                lambda: chat_model.chat(
                    [as_message(message) for message in request.messages],
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    top_k=request.top_k,
                    top_p=request.top_p,
                    greedy=request.greedy,
                    seed=request.seed,
                ),
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
            duration_ms=result.duration_ms,
        )

    @service_app.post(
        "/agent/chat",
        response_model=AgentResponse,
        summary="Answer a request with bounded tool calls",
    )
    def agent_chat(request: AgentRequest) -> AgentResponse:
        if agent is None or not agent.is_ready:
            raise HTTPException(
                status_code=503,
                detail="Agent service is not ready.",
            )

        run_id = _new_request_id()
        token = run_manager.start(run_id)
        if token is None:
            raise HTTPException(status_code=429, detail="Agent service is busy.")
        result = None
        try:
            result = domain_runtime.run(
                request,
                lambda: agent.run(
                    request.messages,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    top_k=request.top_k,
                    top_p=request.top_p,
                    greedy=request.greedy,
                    seed=request.seed,
                    max_steps=request.max_steps,
                    approved_side_effects=request.approved_side_effects,
                    cancellation_token=token,
                ),
            )
        except ValueError as error:
            run_manager.finish(run_id, failed=True)
            raise HTTPException(status_code=400, detail=str(error)) from error
        except RuntimeError as error:
            run_manager.finish(run_id, failed=True)
            raise HTTPException(status_code=503, detail=str(error)) from error
        except Exception:
            run_manager.finish(run_id, failed=True)
            raise
        run_manager.finish(run_id, result=result)
        return _agent_response(result, run_id=run_id)

    @service_app.post(
        "/agent/chat/stream",
        summary="Stream route, model, tool, plan, and final Agent events",
    )
    async def agent_chat_stream(
        payload: AgentRequest,
        http_request: Request,
    ) -> StreamingResponse:
        if agent is None or not agent.is_ready:
            raise HTTPException(status_code=503, detail="Agent service is not ready.")
        run_id = _request_id(http_request)
        try:
            token = run_manager.start(run_id)
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        if token is None:
            raise HTTPException(status_code=429, detail="Agent service is busy.")
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[AgentProgressEvent] = asyncio.Queue()

        def on_progress(event: AgentProgressEvent) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, event)

        async def execute() -> AgentResult:
            def run_agent() -> AgentResult:
                return agent.run(
                    payload.messages,
                    max_new_tokens=payload.max_new_tokens,
                    temperature=payload.temperature,
                    top_k=payload.top_k,
                    top_p=payload.top_p,
                    greedy=payload.greedy,
                    seed=payload.seed,
                    max_steps=payload.max_steps,
                    approved_side_effects=payload.approved_side_effects,
                    cancellation_token=token,
                    progress_callback=on_progress,
                )

            return await asyncio.to_thread(
                domain_runtime.run,
                payload,
                run_agent,
            )

        task = asyncio.create_task(execute())

        async def stream() -> AsyncIterator[str]:
            result: AgentResult | None = None
            failed = False
            yield _sse("start", {"run_id": run_id})
            try:
                while not task.done() or not queue.empty():
                    if await http_request.is_disconnected():
                        token.cancel()
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=0.1)
                    except TimeoutError:
                        continue
                    yield _sse("progress", asdict(event))
                result = await task
                yield _sse(
                    "result", _agent_response(result, run_id=run_id).model_dump()
                )
            except Exception as error:
                failed = True
                logger.exception("Agent SSE run failed: %s", type(error).__name__)
                yield _sse("error", {"run_id": run_id, "detail": "Agent run failed."})
            finally:
                if not task.done():
                    token.cancel()
                    try:
                        result = await task
                    except Exception:
                        failed = True
                run_manager.finish(run_id, result=result, failed=failed)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @service_app.post(
        "/agent/cancel/{run_id}",
        response_model=AgentCancelResponse,
        summary="Request cooperative cancellation for an active Agent run",
    )
    def cancel_agent(run_id: str) -> AgentCancelResponse:
        return AgentCancelResponse(
            run_id=run_id,
            cancellation_requested=run_manager.cancel(run_id),
        )

    @service_app.get(
        "/agent/metrics",
        response_model=AgentMetricsResponse,
        summary="Read process-local Agent operational metrics",
    )
    def agent_metrics() -> AgentMetricsResponse:
        return AgentMetricsResponse(**asdict(run_manager.snapshot()))

    register_multimodal_routes(service_app, multimodal_workspace)
    domain_runtime.register_routes(service_app)
    if conversation_repository is not None:
        service_app.include_router(
            build_conversation_router(
                conversation_repository,
                conversation_context_builder or ConversationContextBuilder(),
                chat_model,
                conversation_run_manager,
            )
        )
    return service_app


def _agent_response(result: object, *, run_id: str) -> AgentResponse:
    """Serialize an AgentResult in one place for sync and SSE endpoints."""
    from agent.orchestrator import AgentResult

    if not isinstance(result, AgentResult):
        raise TypeError("Expected AgentResult.")
    return AgentResponse(
        message=ChatMessage(role="assistant", content=result.text),
        generated_text=result.text,
        model=result.model,
        steps=result.steps,
        tool_calls=result.tool_calls,
        finish_reason=result.finish_reason,
        trace=[
            AgentTraceResponse(
                step=trace.step,
                call_id=trace.call_id,
                name=trace.name,
                arguments=trace.arguments,
                status=trace.status,
                content=trace.content,
                duration_ms=trace.duration_ms,
                confirmation_id=trace.confirmation_id,
                argument_repaired=trace.argument_repaired,
                attempts=trace.attempts,
            )
            for trace in result.trace
        ],
        events=[
            AgentEventResponse(
                step=event.step,
                phase=event.phase,
                input=event.input,
                output=event.output,
                duration_ms=event.duration_ms,
                error=event.error,
            )
            for event in result.events
        ],
        run_id=run_id,
        route=result.route.kind if result.route else None,
        prompt_tokens=result.usage.prompt_tokens,
        generated_tokens=result.usage.generated_tokens,
        model_duration_ms=result.usage.model_duration_ms,
        tool_duration_ms=result.usage.tool_duration_ms,
        total_duration_ms=result.usage.total_duration_ms,
        allowed_citations=list(result.citation.allowed) if result.citation else [],
        unsupported_citations=(
            list(result.citation.unsupported) if result.citation else []
        ),
    )


def _sse(event: str, data: object) -> str:
    return (
        f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"
    )


def create_app_from_environment() -> FastAPI:
    """Create the production app using ``MINIGPT_*`` environment variables."""
    service_config = ServiceConfig.from_environment()
    chat_config = ChatServiceConfig.from_environment()
    multimodal_workspace = _multimodal_from_environment()
    domain_registry = _domain_registry_from_environment()
    conversation_repository = SQLiteConversationRepository(
        Path(os.getenv("MINIGPT_CONVERSATION_DB", "output/conversations.db"))
    )
    conversation_context_builder = ConversationContextBuilder(
        max_input_tokens=int(os.getenv("MINIGPT_CHAT_MAX_INPUT_TOKENS", "4096")),
        reserved_output_tokens=int(
            os.getenv("MINIGPT_CONTEXT_RESERVED_OUTPUT_TOKENS", "512")
        ),
        summary_token_budget=int(os.getenv("MINIGPT_CONTEXT_SUMMARY_TOKENS", "768")),
        recent_turns=int(os.getenv("MINIGPT_CONTEXT_RECENT_TURNS", "6")),
    )
    if service_config is None and chat_config is None:
        return create_app(
            multimodal_workspace=multimodal_workspace,
            domain_registry=domain_registry,
            conversation_repository=conversation_repository,
            conversation_context_builder=conversation_context_builder,
        )

    model_service = None
    chat_model = None
    agent = None
    if service_config is not None:
        from minigpt_service.model_service import ModelService

        model_service = ModelService(service_config)
    if chat_config is not None:
        from agent import (
            AgentOrchestrator,
            ChatModelAgentAdapter,
            HeuristicAgentRouter,
            build_default_registry,
        )
        from agent.llm import create_huggingface_chat_model
        from minigpt_service.model_service import select_device

        chat_model = create_huggingface_chat_model(
            chat_config,
            select_device(chat_config.device),
        )
        document_workspace = os.getenv("MINIGPT_DOCUMENT_WORKSPACE")
        agent = AgentOrchestrator(
            ChatModelAgentAdapter(chat_model),
            build_default_registry(
                document_workspace=(
                    Path(document_workspace) if document_workspace else None
                ),
                multimodal_workspace=multimodal_workspace,
            ),
            router=HeuristicAgentRouter(),
        )
    elif model_service is not None:
        from agent import (
            AgentOrchestrator,
            ChatModelAgentAdapter,
            HeuristicAgentRouter,
            build_default_registry,
        )
        from agent.llm import MiniGPTChatModel

        chat_model = MiniGPTChatModel(model_service, owns_service=False)
        document_workspace = os.getenv("MINIGPT_DOCUMENT_WORKSPACE")
        agent = AgentOrchestrator(
            ChatModelAgentAdapter(chat_model),
            build_default_registry(
                document_workspace=(
                    Path(document_workspace) if document_workspace else None
                ),
                multimodal_workspace=multimodal_workspace,
            ),
            router=HeuristicAgentRouter(),
        )

    return create_app(
        model_service,
        chat_model,
        agent,
        multimodal_workspace,
        domain_registry,
        conversation_repository,
        conversation_context_builder,
    )


def _multimodal_from_environment() -> MultimodalWorkspace | None:
    root = os.getenv("MINIGPT_MULTIMODAL_WORKSPACE")
    if not root:
        return None
    from agent.multimodal import (
        MultimodalWorkspace,
        Pyttsx3TTSBackend,
        TransformersTrOCRBackend,
        TransformersWhisperBackend,
    )

    cache = Path(
        os.getenv(
            "MINIGPT_MULTIMODAL_MODEL_CACHE",
            str(Path(root) / ".model-cache"),
        )
    )
    ocr = (
        TransformersTrOCRBackend(cache / "ocr")
        if os.getenv("MINIGPT_ENABLE_OCR") == "1"
        else None
    )
    asr = (
        TransformersWhisperBackend(cache / "asr")
        if os.getenv("MINIGPT_ENABLE_ASR") == "1"
        else None
    )
    tts = Pyttsx3TTSBackend() if os.getenv("MINIGPT_ENABLE_TTS") == "1" else None
    return MultimodalWorkspace(
        Path(root), ocr_backend=ocr, asr_backend=asr, tts_backend=tts
    )


def _domain_registry_from_environment() -> DomainAdapterRegistry | None:
    root = os.getenv("MINIGPT_DOMAIN_ROOT")
    if not root:
        return None
    from agent.domain_adapter import DomainAdapterRegistry

    registry = DomainAdapterRegistry(Path(root))
    registry.load()
    return registry


app = create_app_from_environment()
