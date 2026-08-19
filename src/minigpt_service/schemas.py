"""Pydantic contracts for the MiniGPT HTTP service."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class GenerateRequest(BaseModel):
    """Input accepted by the future text-generation endpoint."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(
        min_length=1,
        max_length=4096,
        description="Non-empty prompt text.",
        examples=["To be, or not to be"],
    )
    max_new_tokens: int = Field(
        default=100,
        ge=1,
        le=512,
        description="Maximum number of tokens to generate.",
    )
    temperature: float = Field(
        default=1.0,
        gt=0,
        description="Sampling temperature; must be greater than zero.",
    )
    top_k: int | None = Field(
        default=None,
        ge=0,
        description="Top-k filtering; zero or null disables filtering.",
    )
    top_p: float | None = Field(
        default=None,
        gt=0,
        le=1,
        description="Nucleus sampling probability threshold.",
    )
    greedy: bool = Field(
        default=False,
        description="Use deterministic greedy decoding.",
    )
    use_kv_cache: bool = Field(
        default=False,
        description="Use the model's key/value cache during decoding.",
    )
    seed: int | None = Field(
        default=None,
        description="Optional random seed for reproducible sampling.",
    )


class GenerateResponse(BaseModel):
    """Stable response shape returned by text generation."""

    model_config = ConfigDict(extra="forbid")

    prompt: str
    generated_text: str
    generated_tokens: int = Field(ge=0)
    finish_reason: Literal["length", "stop"]


class HealthResponse(BaseModel):
    """Response returned by the liveness endpoint."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"]
    service: str
    version: str


class ReadyResponse(BaseModel):
    """Response returned when the model is ready to serve requests."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ready"]


class ErrorResponse(BaseModel):
    """Stable error envelope returned by the service."""

    model_config = ConfigDict(extra="forbid")

    detail: Any
    request_id: str = Field(min_length=1)
