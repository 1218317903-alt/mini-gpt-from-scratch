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
    tool_call_id: str | None = Field(default=None, min_length=1, max_length=128)


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
    domain_id: str | None = Field(
        default=None,
        min_length=3,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_-]{2,63}$",
        description="Optional private-domain Adapter selected for this request only.",
    )
    domain_access_token: str | None = Field(
        default=None,
        min_length=1,
        max_length=512,
        repr=False,
        description="Authorization token paired with domain_id.",
    )

    @model_validator(mode="after")
    def validate_messages(self) -> ChatRequest:
        if not any(message.role == "user" for message in self.messages):
            raise ValueError("messages must contain at least one user message.")
        if any(not message.content.strip() for message in self.messages):
            raise ValueError("message content must not be empty.")
        if (self.domain_id is None) != (self.domain_access_token is None):
            raise ValueError(
                "domain_id and domain_access_token must be provided together."
            )
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
    duration_ms: float = Field(ge=0)


class ConversationCreateRequest(BaseModel):
    """Create one persistent local chat thread."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(default="新对话", min_length=1, max_length=200)
    model: str | None = Field(default=None, max_length=512)
    system_prompt: str | None = Field(default=None, max_length=32768)


class ConversationUpdateRequest(BaseModel):
    """Update mutable conversation metadata."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, min_length=1, max_length=200)
    model: str | None = Field(default=None, max_length=512)
    system_prompt: str | None = Field(default=None, max_length=32768)


class ConversationMessageRequest(BaseModel):
    """Append a user turn and generate a persistent assistant response."""

    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1, max_length=32768)
    client_request_id: str | None = Field(default=None, min_length=8, max_length=128)
    max_new_tokens: int = Field(default=512, ge=1, le=2048)
    temperature: float = Field(default=0.7, gt=0)
    top_k: int | None = Field(default=None, ge=0)
    top_p: float | None = Field(default=None, gt=0, le=1)
    greedy: bool = False
    seed: int | None = None


class StoredMessageResponse(BaseModel):
    """Persistent message returned to the chat client."""

    model_config = ConfigDict(extra="forbid")

    id: str
    sequence: int = Field(ge=1)
    role: ChatRole
    content: str
    status: Literal["complete", "generating", "cancelled", "error", "interrupted"]
    model: str | None = None
    prompt_tokens: int | None = Field(default=None, ge=0)
    generated_tokens: int | None = Field(default=None, ge=0)
    finish_reason: str | None = None
    request_id: str | None = None
    created_at: str
    updated_at: str


class ConversationResponse(BaseModel):
    """Conversation metadata and optional ordered messages."""

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    model: str | None = None
    system_prompt: str | None = None
    summary: str | None = None
    summarized_through_sequence: int = Field(ge=0)
    created_at: str
    updated_at: str
    archived: bool = False
    messages: list[StoredMessageResponse] = Field(default_factory=list)


class ConversationListResponse(BaseModel):
    """Paginated recent conversations."""

    model_config = ConfigDict(extra="forbid")

    conversations: list[ConversationResponse]


class ConversationGenerationResponse(BaseModel):
    """Completed persistent user/assistant exchange."""

    model_config = ConfigDict(extra="forbid")

    conversation: ConversationResponse
    user_message: StoredMessageResponse
    assistant_message: StoredMessageResponse
    compressed: bool
    omitted_messages: int = Field(ge=0)
    estimated_context_tokens: int = Field(ge=0)


class ConversationCancelResponse(BaseModel):
    """Result of requesting cancellation for a conversation generation."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    cancellation_requested: bool


class AgentRequest(ChatRequest):
    """Input accepted by the bounded tool-using Agent endpoint."""

    max_steps: int = Field(
        default=6,
        ge=1,
        le=6,
        description="Maximum number of model/tool loop iterations.",
    )
    approved_side_effects: list[str] = Field(
        default_factory=list,
        max_length=16,
        description="Approval IDs bound to exact side-effect tool calls.",
    )


class AgentTraceResponse(BaseModel):
    """One tool execution in an Agent response trace."""

    model_config = ConfigDict(extra="forbid")

    step: int = Field(ge=1)
    call_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments: dict[str, Any]
    status: Literal["ok", "error", "confirmation_required"]
    content: str
    duration_ms: float = Field(ge=0)
    confirmation_id: str | None = None
    argument_repaired: bool = False
    attempts: int = Field(default=1, ge=1, le=2)


class AgentEventResponse(BaseModel):
    """One sanitized model or tool event from the complete Agent trace."""

    model_config = ConfigDict(extra="forbid")

    step: int = Field(ge=1)
    phase: Literal["model", "tool"]
    input: Any
    output: Any
    duration_ms: float = Field(ge=0)
    error: str | None = None


class AgentResponse(BaseModel):
    """Stable response returned by the bounded Agent endpoint."""

    model_config = ConfigDict(extra="forbid")

    message: ChatMessage
    generated_text: str
    model: str
    steps: int = Field(ge=1)
    tool_calls: int = Field(ge=0)
    finish_reason: Literal[
        "stop",
        "max_steps",
        "confirmation_required",
        "cancelled",
        "citation_guard",
    ]
    trace: list[AgentTraceResponse]
    events: list[AgentEventResponse] = Field(default_factory=list)
    run_id: str | None = None
    route: str | None = None
    prompt_tokens: int = Field(default=0, ge=0)
    generated_tokens: int = Field(default=0, ge=0)
    model_duration_ms: float = Field(default=0, ge=0)
    tool_duration_ms: float = Field(default=0, ge=0)
    total_duration_ms: float = Field(default=0, ge=0)
    allowed_citations: list[str] = Field(default_factory=list)
    unsupported_citations: list[str] = Field(default_factory=list)


class AgentCancelResponse(BaseModel):
    """Result of requesting cooperative cancellation."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    cancellation_requested: bool


class AgentMetricsResponse(BaseModel):
    """Process-local aggregate Agent service metrics."""

    model_config = ConfigDict(extra="forbid")

    running: int = Field(ge=0)
    accepted: int = Field(ge=0)
    completed: int = Field(ge=0)
    cancelled: int = Field(ge=0)
    failed: int = Field(ge=0)
    rejected_busy: int = Field(ge=0)
    avg_duration_ms: float = Field(ge=0)


class MediaUploadRequest(BaseModel):
    """Base64 upload kept inside the configured multimodal workspace."""

    model_config = ConfigDict(extra="forbid")
    filename: str = Field(min_length=1, max_length=128)
    content_base64: str = Field(min_length=1, max_length=72_000_000)
    overwrite: bool = False


class ImageAnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    filename: str = Field(min_length=1, max_length=128)
    include_ocr: bool = True


class AudioTranscribeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    filename: str = Field(min_length=1, max_length=128)
    language: str | None = Field(default=None, max_length=16)


class SpeechSynthesisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=4000)
    filename: str = Field(min_length=1, max_length=128)
    rate: int = Field(default=180, ge=80, le=360)
    overwrite: bool = False


class MultimodalCapabilitiesResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool
    image_metadata: bool
    ocr: bool
    asr: bool
    tts: bool


class MediaOperationResponse(BaseModel):
    """Stable envelope for media upload, OCR, ASR, and TTS results."""

    model_config = ConfigDict(extra="forbid")
    operation: Literal["upload", "image", "asr", "tts"]
    result: dict[str, Any]


class DomainCatalogResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool
    domains: list[dict[str, Any]]
    active_domain: str | None = None


class DomainAuthorizationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    access_token: str = Field(min_length=1, max_length=512)


class DomainAuthorizationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    domain_id: str
    authorized: bool
    base_model: str
    adult_content_enabled: bool
    activated: bool = False


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
