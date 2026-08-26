"""Persistent multi-turn conversation HTTP and SSE routes."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterator, Sequence
from dataclasses import asdict
from typing import Any, cast
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from agent.llm.base import ChatModel, ChatResult, ChatStreamChunk, StreamingChatModel
from agent.messages import Message
from minigpt_service.conversation_runtime import ConversationRunManager
from minigpt_service.conversations import (
    ContextBuildResult,
    Conversation,
    ConversationContextBuilder,
    ConversationRepository,
    StoredMessage,
)
from minigpt_service.conversations.models import MessageStatus
from minigpt_service.schemas import (
    ConversationCancelResponse,
    ConversationCreateRequest,
    ConversationGenerationResponse,
    ConversationListResponse,
    ConversationMessageRequest,
    ConversationResponse,
    ConversationUpdateRequest,
    StoredMessageResponse,
)


def build_conversation_router(
    repository: ConversationRepository,
    context_builder: ConversationContextBuilder,
    chat_model: ChatModel | None,
    run_manager: ConversationRunManager,
) -> APIRouter:
    """Build conversation routes around injected storage and model services."""
    router = APIRouter(prefix="/conversations", tags=["conversations"])

    @router.post("", response_model=ConversationResponse, status_code=201)
    def create_conversation(payload: ConversationCreateRequest) -> ConversationResponse:
        conversation = repository.create_conversation(
            title=payload.title,
            model=payload.model,
            system_prompt=payload.system_prompt,
        )
        return _conversation_response(conversation)

    @router.get("", response_model=ConversationListResponse)
    def list_conversations(
        limit: int = 50,
        offset: int = 0,
    ) -> ConversationListResponse:
        try:
            conversations = repository.list_conversations(limit=limit, offset=offset)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return ConversationListResponse(
            conversations=[_conversation_response(item) for item in conversations]
        )

    @router.get("/{conversation_id}", response_model=ConversationResponse)
    def get_conversation(conversation_id: str) -> ConversationResponse:
        conversation = _require_conversation(repository, conversation_id)
        return _conversation_response(
            conversation,
            repository.list_messages(conversation_id),
        )

    @router.patch("/{conversation_id}", response_model=ConversationResponse)
    def update_conversation(
        conversation_id: str,
        payload: ConversationUpdateRequest,
    ) -> ConversationResponse:
        try:
            conversation = repository.update_conversation(
                conversation_id,
                title=payload.title,
                model=payload.model,
                system_prompt=payload.system_prompt,
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        return _conversation_response(
            conversation,
            repository.list_messages(conversation_id),
        )

    @router.delete("/{conversation_id}", status_code=204)
    def delete_conversation(conversation_id: str) -> None:
        if not repository.delete_conversation(conversation_id):
            raise HTTPException(status_code=404, detail="Conversation not found.")

    @router.post(
        "/{conversation_id}/messages",
        response_model=ConversationGenerationResponse,
    )
    def create_message(
        conversation_id: str,
        payload: ConversationMessageRequest,
    ) -> ConversationGenerationResponse:
        model = _require_chat_model(chat_model)
        conversation, user_message, assistant_message, context = _start_exchange(
            repository,
            context_builder,
            conversation_id,
            payload,
        )
        try:
            result = model.chat(
                context.messages,
                max_new_tokens=payload.max_new_tokens,
                temperature=payload.temperature,
                top_k=payload.top_k,
                top_p=payload.top_p,
                greedy=payload.greedy,
                seed=payload.seed,
            )
            completed = repository.update_message(
                assistant_message.id,
                content=result.text,
                status="complete",
                model=result.model,
                prompt_tokens=result.prompt_tokens,
                generated_tokens=result.generated_tokens,
                finish_reason=result.finish_reason,
            )
        except (RuntimeError, ValueError) as error:
            repository.update_message(
                assistant_message.id,
                content="Generation failed.",
                status="error",
                finish_reason="error",
            )
            raise HTTPException(status_code=503, detail=str(error)) from error
        assert completed is not None
        current = _require_conversation(repository, conversation_id)
        return ConversationGenerationResponse(
            conversation=_conversation_response(
                current,
                repository.list_messages(conversation_id),
            ),
            user_message=_message_response(user_message),
            assistant_message=_message_response(completed),
            compressed=context.compressed,
            omitted_messages=context.omitted_messages,
            estimated_context_tokens=context.estimated_tokens,
        )

    @router.post("/{conversation_id}/messages/stream")
    async def stream_message(
        conversation_id: str,
        payload: ConversationMessageRequest,
        request: Request,
    ) -> StreamingResponse:
        model = _require_chat_model(chat_model)
        (
            conversation,
            user_message,
            assistant_message,
            context,
        ) = await asyncio.to_thread(
            _start_exchange,
            repository,
            context_builder,
            conversation_id,
            payload,
        )
        del conversation
        run_id = uuid4().hex
        try:
            cancellation = run_manager.start(run_id, conversation_id)
        except ValueError as error:
            repository.update_message(
                assistant_message.id,
                content="Generation rejected because this conversation is busy.",
                status="error",
                finish_reason="busy",
            )
            raise HTTPException(status_code=409, detail=str(error)) from error
        if cancellation is None:
            repository.update_message(
                assistant_message.id,
                content="Generation rejected because the service is busy.",
                status="error",
                finish_reason="busy",
            )
            raise HTTPException(status_code=429, detail="Conversation service is busy.")

        async def event_stream() -> AsyncIterator[str]:
            queue: asyncio.Queue[ChatStreamChunk | BaseException | None] = (
                asyncio.Queue()
            )
            loop = asyncio.get_running_loop()
            accumulated: list[str] = []

            def produce() -> None:
                try:
                    for chunk in _stream_backend(
                        model,
                        context.messages,
                        payload,
                        cancellation.is_cancelled,
                    ):
                        if chunk.text:
                            accumulated.append(chunk.text)
                            if sum(len(piece) for piece in accumulated) % 256 < len(
                                chunk.text
                            ):
                                repository.update_message(
                                    assistant_message.id,
                                    content="".join(accumulated),
                                )
                        loop.call_soon_threadsafe(queue.put_nowait, chunk)
                except BaseException as error:
                    loop.call_soon_threadsafe(queue.put_nowait, error)
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, None)

            worker = asyncio.create_task(asyncio.to_thread(produce))
            terminal: ChatStreamChunk | None = None
            failed = False
            yield _sse(
                "start",
                {
                    "run_id": run_id,
                    "conversation_id": conversation_id,
                    "user_message": _message_response(user_message).model_dump(),
                    "assistant_message": _message_response(
                        assistant_message
                    ).model_dump(),
                    "compressed": context.compressed,
                    "omitted_messages": context.omitted_messages,
                    "estimated_context_tokens": context.estimated_tokens,
                },
            )
            try:
                while True:
                    if await request.is_disconnected():
                        cancellation.cancel()
                    try:
                        item = await asyncio.wait_for(queue.get(), timeout=0.1)
                    except TimeoutError:
                        continue
                    if item is None:
                        break
                    if isinstance(item, BaseException):
                        raise item
                    if item.done:
                        terminal = item
                        continue
                    yield _sse("delta", {"text": item.text})

                await worker
                final_text = "".join(accumulated).strip()
                if terminal is None:
                    raise RuntimeError("Streaming backend returned no terminal event.")
                status: MessageStatus = (
                    "cancelled" if terminal.finish_reason == "cancelled" else "complete"
                )
                completed = repository.update_message(
                    assistant_message.id,
                    content=final_text or "Generation produced no text.",
                    status=status,
                    model=terminal.model,
                    prompt_tokens=terminal.prompt_tokens,
                    generated_tokens=terminal.generated_tokens,
                    finish_reason=terminal.finish_reason,
                )
                assert completed is not None
                yield _sse(
                    "done",
                    {
                        "run_id": run_id,
                        "message": _message_response(completed).model_dump(),
                        "duration_ms": terminal.duration_ms,
                    },
                )
            except BaseException as error:
                failed = True
                cancellation.cancel()
                await worker
                partial = "".join(accumulated).strip()
                repository.update_message(
                    assistant_message.id,
                    content=partial or "Generation failed.",
                    status="error",
                    finish_reason="error",
                )
                yield _sse(
                    "error",
                    {
                        "run_id": run_id,
                        "detail": str(error) or type(error).__name__,
                    },
                )
            finally:
                if not failed and cancellation.is_cancelled and terminal is None:
                    repository.update_message(
                        assistant_message.id,
                        content="".join(accumulated).strip() or "Generation cancelled.",
                        status="cancelled",
                        finish_reason="cancelled",
                    )
                run_manager.finish(run_id, conversation_id)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @router.post(
        "/runs/{run_id}/cancel",
        response_model=ConversationCancelResponse,
    )
    def cancel_run(run_id: str) -> ConversationCancelResponse:
        return ConversationCancelResponse(
            run_id=run_id,
            cancellation_requested=run_manager.cancel(run_id),
        )

    return router


def _start_exchange(
    repository: ConversationRepository,
    context_builder: ConversationContextBuilder,
    conversation_id: str,
    payload: ConversationMessageRequest,
) -> tuple[Conversation, StoredMessage, StoredMessage, ContextBuildResult]:
    conversation = _require_conversation(repository, conversation_id)
    if payload.client_request_id:
        existing = repository.get_message_by_request_id(payload.client_request_id)
        if existing is not None:
            raise HTTPException(
                status_code=409, detail="Request has already been used."
            )
    user_message = repository.append_message(
        conversation_id,
        role="user",
        content=payload.content,
    )
    if conversation.title == "新对话":
        generated_title = " ".join(payload.content.split())[:40]
        conversation = cast(
            Conversation,
            repository.update_conversation(conversation_id, title=generated_title),
        )
    context = context_builder.build(
        conversation,
        repository.list_messages(conversation_id),
        reserved_output_tokens=payload.max_new_tokens,
    )
    if (
        context.compressed
        and context.summary
        and context.summarized_through_sequence
        > conversation.summarized_through_sequence
    ):
        conversation = repository.update_summary(
            conversation_id,
            summary=context.summary,
            summarized_through_sequence=context.summarized_through_sequence,
        )
    assistant_message = repository.append_message(
        conversation_id,
        role="assistant",
        content="",
        status="generating",
        model=conversation.model,
        request_id=payload.client_request_id,
    )
    return conversation, user_message, assistant_message, context


def _stream_backend(
    model: ChatModel,
    messages: Sequence[Message],
    payload: ConversationMessageRequest,
    is_cancelled: Any,
) -> Iterator[ChatStreamChunk]:
    if hasattr(model, "stream_chat"):
        streaming_model = cast(StreamingChatModel, model)
        yield from streaming_model.stream_chat(
            messages,
            max_new_tokens=payload.max_new_tokens,
            temperature=payload.temperature,
            top_k=payload.top_k,
            top_p=payload.top_p,
            greedy=payload.greedy,
            seed=payload.seed,
            is_cancelled=is_cancelled,
        )
        return
    if is_cancelled():
        yield ChatStreamChunk(done=True, finish_reason="cancelled")
        return
    result: ChatResult = model.chat(
        messages,
        max_new_tokens=payload.max_new_tokens,
        temperature=payload.temperature,
        top_k=payload.top_k,
        top_p=payload.top_p,
        greedy=payload.greedy,
        seed=payload.seed,
    )
    if result.text:
        yield ChatStreamChunk(text=result.text, model=result.model)
    yield ChatStreamChunk(
        done=True,
        prompt_tokens=result.prompt_tokens,
        generated_tokens=result.generated_tokens,
        finish_reason=result.finish_reason,
        model=result.model,
        duration_ms=result.duration_ms,
    )


def _require_chat_model(chat_model: ChatModel | None) -> ChatModel:
    if chat_model is None or not chat_model.is_ready:
        raise HTTPException(status_code=503, detail="Chat model service is not ready.")
    return chat_model


def _require_conversation(
    repository: ConversationRepository,
    conversation_id: str,
) -> Conversation:
    conversation = repository.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return conversation


def _conversation_response(
    conversation: Conversation,
    messages: Sequence[StoredMessage] = (),
) -> ConversationResponse:
    return ConversationResponse(
        **asdict(conversation),
        messages=[_message_response(message) for message in messages],
    )


def _message_response(message: StoredMessage) -> StoredMessageResponse:
    return StoredMessageResponse(
        id=message.id,
        sequence=message.sequence,
        role=message.role,
        content=message.content,
        status=message.status,
        model=message.model,
        prompt_tokens=message.prompt_tokens,
        generated_tokens=message.generated_tokens,
        finish_reason=message.finish_reason,
        request_id=message.request_id,
        created_at=message.created_at,
        updated_at=message.updated_at,
    )


def _sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
