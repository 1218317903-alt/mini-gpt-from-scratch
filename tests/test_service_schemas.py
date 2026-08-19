"""Tests for the stage-eight request and response contracts."""

import pytest
from pydantic import ValidationError

from minigpt_service.schemas import (
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


def test_health_and_ready_responses_have_stable_shape() -> None:
    assert HealthResponse(
        status="ok", service="MiniGPT Inference Service", version="0.1.0"
    ).model_dump() == {
        "status": "ok",
        "service": "MiniGPT Inference Service",
        "version": "0.1.0",
    }
    assert ReadyResponse(status="ready").model_dump() == {"status": "ready"}


def test_error_response_has_stable_shape() -> None:
    response = ErrorResponse(detail="not ready", request_id="trace-123")

    assert response.model_dump() == {
        "detail": "not ready",
        "request_id": "trace-123",
    }
