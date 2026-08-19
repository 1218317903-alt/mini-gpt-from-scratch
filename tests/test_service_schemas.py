"""Tests for the stage-eight request and response contracts."""

import pytest
from pydantic import ValidationError

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


def test_generate_request_defaults_match_cli_generation() -> None:
    request = GenerateRequest(prompt="To be")

    assert request.max_new_tokens == 100
    assert request.temperature == 1.0
    assert request.top_k is None
    assert request.top_p is None
    assert request.greedy is False
    assert request.use_kv_cache is False
    assert request.seed is None


@pytest.mark.parametrize(
    "payload",
    [
        {"prompt": ""},
        {"prompt": "x", "max_new_tokens": 0},
        {"prompt": "x", "temperature": 0},
        {"prompt": "x", "top_k": -1},
        {"prompt": "x", "top_p": 0},
        {"prompt": "x", "top_p": 1.1},
        {"prompt": "x", "unknown": True},
    ],
)
def test_generate_request_rejects_invalid_values(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        GenerateRequest.model_validate(payload)


def test_generate_response_has_stable_shape() -> None:
    response = GenerateResponse(
        prompt="To be",
        generated_text="To be or not to be",
        generated_tokens=5,
        finish_reason="length",
    )

    assert response.model_dump() == {
        "prompt": "To be",
        "generated_text": "To be or not to be",
        "generated_tokens": 5,
        "finish_reason": "length",
    }


def test_chat_request_defaults_and_message_validation() -> None:
    request = ChatRequest.model_validate(
        {"messages": [{"role": "user", "content": "你好"}]}
    )

    assert request.max_new_tokens == 128
    assert request.temperature == 0.7
    assert request.greedy is False
    assert request.top_k is None
    assert request.top_p is None

    with pytest.raises(ValidationError, match="user message"):
        ChatRequest.model_validate(
            {"messages": [{"role": "system", "content": "only system"}]}
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"messages": []},
        {"messages": [{"role": "user", "content": ""}]},
        {"messages": [{"role": "unknown", "content": "x"}]},
        {"messages": [{"role": "user", "content": "x"}], "temperature": 0},
        {"messages": [{"role": "user", "content": "x"}], "top_p": 1.1},
        {"messages": [{"role": "user", "content": "x"}], "extra": True},
    ],
)
def test_chat_request_rejects_invalid_values(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ChatRequest.model_validate(payload)


def test_chat_response_has_stable_shape() -> None:
    response = ChatResponse(
        message=ChatMessage(role="assistant", content="你好！"),
        generated_text="你好！",
        prompt_tokens=12,
        generated_tokens=4,
        finish_reason="stop",
        model="Qwen/Qwen2.5-0.5B-Instruct",
    )

    assert response.model_dump() == {
        "message": {"role": "assistant", "content": "你好！", "name": None},
        "generated_text": "你好！",
        "prompt_tokens": 12,
        "generated_tokens": 4,
        "finish_reason": "stop",
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
    }


def test_health_and_ready_responses_have_stable_shape() -> None:
    assert HealthResponse(
        status="ok", service="MiniGPT Inference Service", version="0.2.0"
    ).model_dump() == {
        "status": "ok",
        "service": "MiniGPT Inference Service",
        "version": "0.2.0",
    }
    assert ReadyResponse(status="ready").model_dump() == {"status": "ready"}


def test_error_response_has_stable_shape() -> None:
    response = ErrorResponse(detail="not ready", request_id="trace-123")

    assert response.model_dump() == {
        "detail": "not ready",
        "request_id": "trace-123",
    }
