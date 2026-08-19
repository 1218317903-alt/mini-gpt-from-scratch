"""Pydantic contracts for the MiniGPT HTTP service."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ChatRole = Literal["system", "user", "assistant", "tool"]


class ChatMessage(BaseModel):
    """One message in a chat conversation."""

    model_config = ConfigDict(extra="forbid")

    role: ChatRole
    content: str = Field(max_length=32768)
    name: str | None = Field(default=None, max_length=128)


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


class ChatRequest(BaseModel):
    """Input accepted by the conversational Hugging Face backend."""

    model_config = ConfigDict(extra="forbid")

    messages: list[ChatMessage] = Field(min_length=1, max_length=64)
    max_new_tokens: int = Field(
        default=128,
        ge=1,
        le=1024,
        description="Maximum number of new response tokens.",
    )
    temperature: float = Field(
        default=0.7,
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
    seed: int | None = Field(
        default=None,
        description="Optional random seed for reproducible sampling.",
    )

    @model_validator(mode="after")
    def validate_messages(self) -> ChatRequest:
        if not any(message.role == "user" for message in self.messages):
            raise ValueError("messages must contain at least one user message.")
        if any(not message.content.strip() for message in self.messages):
            raise ValueError("message content must not be empty.")
        return self


class ChatResponse(BaseModel):
    """Stable response shape returned by the conversational backend."""

    model_config = ConfigDict(extra="forbid")

    message: ChatMessage
    generated_text: str
    prompt_tokens: int = Field(ge=0)
    generated_tokens: int = Field(ge=0)
    finish_reason: Literal["length", "stop"]
    model: str


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
