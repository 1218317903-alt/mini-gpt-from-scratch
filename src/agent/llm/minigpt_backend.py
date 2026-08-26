"""Character-level MiniGPT backend implementing the unified ChatModel API."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from threading import Lock

import torch

from agent.llm.base import ChatResult, ChatStreamChunk
from agent.llm.tool_format import parse_tool_calls, prepare_messages
from agent.messages import Message, ToolSchema
from minigpt.generation import generate
from minigpt_service.model_service import ModelService


class MiniGPTChatModel:
    """Adapt the handwritten MiniGPT service to the chat-model contract."""

    def __init__(
        self,
        service: ModelService,
        *,
        owns_service: bool = False,
    ) -> None:
        self.service = service
        self._owns_service = owns_service
        self._inference_lock = getattr(service, "inference_lock", Lock())

    @property
    def is_ready(self) -> bool:
        """Return whether the wrapped MiniGPT service is ready."""
        return self.service.is_ready

    @property
    def model_name(self) -> str:
        """Return a stable backend identifier."""
        return "minigpt"

    def load(self) -> None:
        """Load the underlying MiniGPT service if needed."""
        self.service.load()

    def close(self) -> None:
        """Close the service only when this backend owns it."""
        if self._owns_service:
            self.service.close()

    def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSchema] | None = None,
        *,
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
        top_p: float | None,
        greedy: bool,
        seed: int | None,
    ) -> ChatResult:
        """Render a chat prompt and generate a character-level continuation."""
        if (
            not self.is_ready
            or self.service.model is None
            or self.service.tokenizer is None
            or self.service.block_size is None
        ):
            raise RuntimeError("MiniGPT chat model is not ready.")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than zero.")
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero.")
        if top_k is not None and top_k < 0:
            raise ValueError("top_k must be zero, positive, or null.")
        if top_p is not None and not 0 < top_p <= 1:
            raise ValueError("top_p must be in the interval (0, 1].")

        prepared = prepare_messages(messages, tuple(tools) if tools else None)
        prompt = render_minigpt_prompt(prepared)
        prompt_ids = self.service.tokenizer.encode(prompt, allow_unknown=True)
        input_ids = torch.tensor(
            [prompt_ids],
            dtype=torch.long,
            device=self.service.device,
        )
        started = time.perf_counter()
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.service.device).manual_seed(seed)
        with self._inference_lock:
            generated_ids = generate(
                model=self.service.model,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                block_size=self.service.block_size,
                do_sample=not greedy,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                use_kv_cache=False,
                generator=generator,
            )
        duration_ms = (time.perf_counter() - started) * 1000
        full_ids = generated_ids[0].detach().cpu().tolist()
        continuation_ids = [int(token_id) for token_id in full_ids[len(prompt_ids) :]]
        text = self.service.tokenizer.decode(continuation_ids).strip()
        return ChatResult(
            text=text,
            prompt_tokens=len(prompt_ids),
            generated_tokens=len(continuation_ids),
            finish_reason="length",
            model=self.model_name,
            duration_ms=round(duration_ms, 3),
            tool_calls=parse_tool_calls(text),
        )

    def stream_chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSchema] | None = None,
        *,
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
        top_p: float | None,
        greedy: bool,
        seed: int | None,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> Iterator[ChatStreamChunk]:
        """Expose a compatible one-block stream for the research backend."""
        if is_cancelled is not None and is_cancelled():
            yield ChatStreamChunk(
                done=True,
                finish_reason="cancelled",
                model=self.model_name,
            )
            return
        result = self.chat(
            messages,
            tools,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            greedy=greedy,
            seed=seed,
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

    def count_chat_tokens(self, messages: Sequence[Message]) -> int:
        """Count rendered prompt IDs for the research backend."""
        if not self.is_ready or self.service.tokenizer is None:
            raise RuntimeError("MiniGPT chat model is not ready.")
        prompt = render_minigpt_prompt(prepare_messages(messages, None))
        return len(self.service.tokenizer.encode(prompt, allow_unknown=True))


def render_minigpt_prompt(messages: Sequence[Message]) -> str:
    """Render a deterministic multi-role prompt for the character model."""
    lines: list[str] = []
    for message in messages:
        header: str = message.role
        if message.role == "tool" and message.name:
            header = f"tool:{message.name}"
            if message.tool_call_id:
                header = f"{header}:{message.tool_call_id}"
        lines.append(f"{header}: {message.content}")
    lines.append("assistant:")
    return "\n".join(lines)
