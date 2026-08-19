"""Unified chat-model protocol and Hugging Face inference backend."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from minigpt.utils import set_random_seed
from minigpt_service.config import ChatServiceConfig
from minigpt_service.schemas import ChatMessage

ChatFinishReason = Literal["length", "stop"]


@dataclass(frozen=True)
class ChatResult:
    """Backend-independent result returned by a chat model."""

    text: str
    prompt_tokens: int
    generated_tokens: int
    finish_reason: ChatFinishReason
    model: str


class ChatModel(Protocol):
    """Lifecycle and inference contract used by the HTTP application."""

    @property
    def is_ready(self) -> bool:
        """Return whether the backend is loaded and ready."""

    def load(self) -> None:
        """Load model weights and tokenizer."""

    def close(self) -> None:
        """Release model resources."""

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
        top_p: float | None,
        greedy: bool,
        seed: int | None,
    ) -> ChatResult:
        """Generate one assistant response for a conversation."""


class HuggingFaceChatModel:
    """Inference-only Hugging Face Causal LM with optional PEFT Adapter."""

    _DTYPES: dict[str, torch.dtype] = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    def __init__(self, config: ChatServiceConfig, device: torch.device) -> None:
        if config.dtype not in self._DTYPES:
            raise ValueError("dtype must be float16, bfloat16, or float32.")
        self.config = config
        self.device = device
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    @property
    def is_ready(self) -> bool:
        """Return whether both tokenizer and model are loaded."""
        return self._tokenizer is not None and self._model is not None

    @property
    def model_name(self) -> str:
        """Return the configured base model identifier."""
        return self.config.model_source

    def load(self) -> None:
        """Load the tokenizer, base model, and optional LoRA Adapter."""
        if self.is_ready:
            return

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_source,
            use_fast=True,
            local_files_only=self.config.local_files_only,
        )
        if tokenizer.chat_template is None:
            raise ValueError("Tokenizer has no chat template.")
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer has neither PAD nor EOS Token.")
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        model: Any = AutoModelForCausalLM.from_pretrained(
            self.config.model_source,
            dtype=self._DTYPES[self.config.dtype],
            low_cpu_mem_usage=True,
            local_files_only=self.config.local_files_only,
        )
        model = model.to(self.device)
        if self.config.adapter_path is not None:
            model = PeftModel.from_pretrained(
                model,
                str(self.config.adapter_path),
                is_trainable=False,
            )
        model.eval()
        model.config.use_cache = True

        self._tokenizer = tokenizer
        self._model = model

    def close(self) -> None:
        """Release references and clear CUDA allocator state."""
        self._model = None
        self._tokenizer = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
        top_p: float | None,
        greedy: bool,
        seed: int | None,
    ) -> ChatResult:
        """Apply the model's chat template and generate one response."""
        if not self.is_ready or self._model is None or self._tokenizer is None:
            raise RuntimeError("Hugging Face chat model is not ready.")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than zero.")
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero.")
        if top_k is not None and top_k < 0:
            raise ValueError("top_k must be zero, positive, or null.")
        if top_p is not None and not 0 < top_p <= 1:
            raise ValueError("top_p must be in the interval (0, 1].")

        if seed is not None:
            set_random_seed(seed)

        payload = [
            {"role": message.role, "content": message.content} for message in messages
        ]
        inputs = self._tokenizer.apply_chat_template(
            payload,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        if prompt_tokens > self.config.max_input_tokens:
            raise ValueError(
                "Conversation exceeds max_input_tokens "
                f"({self.config.max_input_tokens})."
            )

        context_limit = getattr(self._model.config, "max_position_embeddings", None)
        if isinstance(context_limit, int) and (
            prompt_tokens + max_new_tokens > context_limit
        ):
            raise ValueError(
                "Conversation plus max_new_tokens exceeds the model context "
                f"window ({context_limit})."
            )

        was_training = self._model.training
        previous_cache = bool(getattr(self._model.config, "use_cache", True))
        self._model.eval()
        self._model.config.use_cache = True
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": not greedy,
            "use_cache": True,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if self._tokenizer.eos_token_id is not None:
            generation_kwargs["eos_token_id"] = self._tokenizer.eos_token_id
        if not greedy:
            generation_kwargs["temperature"] = temperature
            if top_k is not None and top_k > 0:
                generation_kwargs["top_k"] = top_k
            if top_p is not None:
                generation_kwargs["top_p"] = top_p

        try:
            output = self._model.generate(**inputs, **generation_kwargs)
            response_ids = output[0, prompt_tokens:]
            response_token_ids = [int(token_id) for token_id in response_ids]
            eos_token_id = self._tokenizer.eos_token_id
            stopped = eos_token_id is not None and eos_token_id in response_token_ids
            text = self._tokenizer.decode(
                response_ids,
                skip_special_tokens=True,
            ).strip()
            return ChatResult(
                text=text,
                prompt_tokens=prompt_tokens,
                generated_tokens=len(response_token_ids),
                finish_reason="stop" if stopped else "length",
                model=self.model_name,
            )
        finally:
            self._model.config.use_cache = previous_cache
            if was_training:
                self._model.train()


def create_huggingface_chat_model(
    config: ChatServiceConfig,
    device: torch.device,
) -> HuggingFaceChatModel:
    """Construct the backend without loading weights yet."""
    return HuggingFaceChatModel(config, device)
