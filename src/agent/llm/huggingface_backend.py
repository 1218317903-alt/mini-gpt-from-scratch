"""Hugging Face Instruct model backend with optional LoRA and quantization."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from threading import Lock, Thread
from typing import Any

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

from agent.llm.base import ChatResult, ChatStreamChunk
from agent.llm.tool_format import parse_tool_calls, prepare_messages
from agent.messages import Message, ToolSchema, message_payload
from minigpt_service.config import ChatServiceConfig


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
        self._inference_lock = Lock()
        self._active_adapter: str | None = None
        self._default_adapter: str | None = None

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

        model_source = self._resolved_model_source()
        tokenizer = AutoTokenizer.from_pretrained(
            model_source,
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

        load_kwargs: dict[str, Any] = {
            "dtype": self._DTYPES[self.config.dtype],
            "low_cpu_mem_usage": True,
            "local_files_only": self.config.local_files_only,
        }
        quantization_config = self._quantization_config()
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
            load_kwargs["device_map"] = "auto"

        model: Any = AutoModelForCausalLM.from_pretrained(
            model_source,
            **load_kwargs,
        )
        if quantization_config is None:
            model = model.to(self.device)
        if self.config.adapter_path is not None:
            model = PeftModel.from_pretrained(
                model,
                str(self.config.adapter_path),
                is_trainable=False,
            )
            self._default_adapter = "default"
            self._active_adapter = "default"
        model.eval()
        model.config.use_cache = True

        self._tokenizer = tokenizer
        self._model = model

    def _resolved_model_source(self) -> str:
        """Resolve a cached Hub ID to a snapshot path in strict offline mode."""
        source = self.config.model_source
        if not self.config.local_files_only or Path(source).exists():
            return source
        try:
            from huggingface_hub import snapshot_download

            return snapshot_download(repo_id=source, local_files_only=True)
        except Exception as error:
            raise RuntimeError(
                f"Model is not available in the local Hugging Face cache: {source}"
            ) from error

    def close(self) -> None:
        """Release references and clear CUDA allocator state."""
        self._model = None
        self._tokenizer = None
        self._active_adapter = None
        self._default_adapter = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    @property
    def active_adapter(self) -> str | None:
        """Return the selected PEFT adapter name, or ``None`` for the base model."""
        return self._active_adapter

    def activate_adapter(self, domain_id: str, adapter_path: Path) -> None:
        """Load/select an already-authorized Adapter under the inference lock."""
        if not self.is_ready or self._model is None:
            raise RuntimeError("Hugging Face chat model is not ready.")
        if not adapter_path.is_dir():
            raise FileNotFoundError("Adapter directory does not exist.")
        with self._inference_lock:
            if isinstance(self._model, PeftModel):
                if domain_id not in self._model.peft_config:
                    self._model.load_adapter(str(adapter_path), adapter_name=domain_id)
            else:
                self._model = PeftModel.from_pretrained(
                    self._model,
                    str(adapter_path),
                    adapter_name=domain_id,
                    is_trainable=False,
                )
            self._model.set_adapter(domain_id)
            self._model.eval()
            self._active_adapter = domain_id

    def deactivate_adapter(self) -> None:
        """Return to the configured default Adapter or unload to the base model."""
        if not self.is_ready or self._model is None:
            raise RuntimeError("Hugging Face chat model is not ready.")
        with self._inference_lock:
            if not isinstance(self._model, PeftModel):
                self._active_adapter = None
                return
            if self._default_adapter is not None:
                self._model.set_adapter(self._default_adapter)
                self._active_adapter = self._default_adapter
            else:
                self._model = self._model.unload()
                self._model.eval()
                self._active_adapter = None

    @torch.inference_mode()
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

        tool_list = tuple(tools) if tools else None
        prepared = prepare_messages(messages, tool_list)
        payload = [message_payload(message) for message in prepared]
        inputs = self._apply_chat_template(payload, tool_list)
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

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": not greedy,
            "use_cache": True,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        eos_token_id = self._eos_token_ids()
        if eos_token_id is not None:
            generation_kwargs["eos_token_id"] = eos_token_id
        if not greedy:
            generation_kwargs["temperature"] = temperature
            if top_k is not None and top_k > 0:
                generation_kwargs["top_k"] = top_k
            if top_p is not None:
                generation_kwargs["top_p"] = top_p
            if seed is not None:
                generation_kwargs["generator"] = torch.Generator(
                    device=self.device
                ).manual_seed(seed)

        with self._inference_lock:
            was_training = self._model.training
            previous_cache = bool(getattr(self._model.config, "use_cache", True))
            self._model.eval()
            self._model.config.use_cache = True
            try:
                started = time.perf_counter()
                output = self._model.generate(**inputs, **generation_kwargs)
                duration_ms = (time.perf_counter() - started) * 1000
                response_ids = output[0, prompt_tokens:]
                response_token_ids = [int(token_id) for token_id in response_ids]
                stopped = self._stopped_by_eos(response_token_ids, eos_token_id)
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
                    duration_ms=round(duration_ms, 3),
                    tool_calls=parse_tool_calls(text),
                )
            finally:
                self._model.config.use_cache = previous_cache
                if was_training:
                    self._model.train()

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
        """Generate incremental decoded text and stop at a decode boundary."""
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

        tool_list = tuple(tools) if tools else None
        prepared = prepare_messages(messages, tool_list)
        payload = [message_payload(message) for message in prepared]
        inputs = self._apply_chat_template(payload, tool_list)
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

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": not greedy,
            "use_cache": True,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        eos_token_id = self._eos_token_ids()
        if eos_token_id is not None:
            generation_kwargs["eos_token_id"] = eos_token_id
        if not greedy:
            generation_kwargs["temperature"] = temperature
            if top_k is not None and top_k > 0:
                generation_kwargs["top_k"] = top_k
            if top_p is not None:
                generation_kwargs["top_p"] = top_p
            if seed is not None:
                generation_kwargs["generator"] = torch.Generator(
                    device=self.device
                ).manual_seed(seed)

        cancellation = is_cancelled or (lambda: False)
        generation_kwargs["stopping_criteria"] = StoppingCriteriaList(
            [_CallbackStoppingCriteria(cancellation)]
        )
        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=None,
        )
        generation_kwargs["streamer"] = streamer
        error: list[BaseException] = []
        started = time.perf_counter()
        model = self._model
        assert model is not None

        def generate_in_thread() -> None:
            try:
                with torch.inference_mode(), self._inference_lock:
                    model.generate(**inputs, **generation_kwargs)
            except BaseException as generated_error:
                error.append(generated_error)
                streamer.on_finalized_text("", stream_end=True)

        worker = Thread(target=generate_in_thread, daemon=True)
        worker.start()
        pieces: list[str] = []
        for text in streamer:
            if text:
                pieces.append(text)
                yield ChatStreamChunk(text=text, model=self.model_name)
        worker.join(timeout=5.0)
        if worker.is_alive():
            raise RuntimeError("Streaming generation did not stop in time.")
        if error:
            raise RuntimeError("Streaming generation failed.") from error[0]

        complete_text = "".join(pieces).strip()
        encoded = self._tokenizer.encode(complete_text, add_special_tokens=False)
        duration_ms = (time.perf_counter() - started) * 1000
        cancelled = cancellation()
        yield ChatStreamChunk(
            done=True,
            prompt_tokens=prompt_tokens,
            generated_tokens=len(encoded),
            finish_reason=(
                "cancelled"
                if cancelled
                else "stop"
                if len(encoded) < max_new_tokens
                else "length"
            ),
            model=self.model_name,
            duration_ms=round(duration_ms, 3),
        )

    def count_chat_tokens(self, messages: Sequence[Message]) -> int:
        """Count the exact chat-template input tokens for context budgeting."""
        if not self.is_ready or self._tokenizer is None:
            raise RuntimeError("Hugging Face chat model is not ready.")
        prepared = prepare_messages(messages, None)
        payload = [message_payload(message) for message in prepared]
        encoded = self._tokenizer.apply_chat_template(
            payload,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        return int(encoded["input_ids"].shape[-1])

    def _apply_chat_template(
        self,
        payload: list[dict[str, Any]],
        tools: Sequence[ToolSchema] | None,
    ) -> Any:
        assert self._tokenizer is not None
        template_kwargs: dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        if tools:
            template_kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in tools
            ]
        try:
            encoded = self._tokenizer.apply_chat_template(payload, **template_kwargs)
        except TypeError:
            template_kwargs.pop("tools", None)
            encoded = self._tokenizer.apply_chat_template(payload, **template_kwargs)
        return encoded.to(self.device)

    def _eos_token_ids(self) -> int | list[int] | None:
        assert self._tokenizer is not None
        token_ids: list[int] = []
        eos_token_id = self._tokenizer.eos_token_id
        if isinstance(eos_token_id, int):
            token_ids.append(eos_token_id)
        elif isinstance(eos_token_id, list):
            token_ids.extend(int(token_id) for token_id in eos_token_id)
        extra = self._safe_token_id("<|im_end|>")
        if extra is not None and extra not in token_ids:
            token_ids.append(extra)
        if not token_ids:
            return None
        if len(token_ids) == 1:
            return token_ids[0]
        return token_ids

    def _safe_token_id(self, token: str) -> int | None:
        assert self._tokenizer is not None
        convert = getattr(self._tokenizer, "convert_tokens_to_ids", None)
        if convert is None:
            return None
        token_id = convert(token)
        unk_id = getattr(self._tokenizer, "unk_token_id", None)
        if not isinstance(token_id, int) or token_id < 0:
            return None
        if unk_id is not None and token_id == unk_id:
            return None
        return token_id

    def _quantization_config(self) -> Any | None:
        if self.config.quantization == "none":
            return None
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as error:
            raise RuntimeError(
                "bitsandbytes quantization requires transformers BitsAndBytesConfig."
            ) from error
        compute_dtype = self._DTYPES[self.config.dtype]
        if self.config.quantization == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        return BitsAndBytesConfig(load_in_8bit=True)

    @staticmethod
    def _stopped_by_eos(
        response_token_ids: list[int],
        eos_token_id: int | list[int] | None,
    ) -> bool:
        if eos_token_id is None:
            return False
        stop_ids = (
            {eos_token_id} if isinstance(eos_token_id, int) else set(eos_token_id)
        )
        return any(token_id in stop_ids for token_id in response_token_ids)


def create_huggingface_chat_model(
    config: ChatServiceConfig,
    device: torch.device,
) -> HuggingFaceChatModel:
    """Construct the backend without loading weights yet."""
    return HuggingFaceChatModel(config, device)


class _CallbackStoppingCriteria(StoppingCriteria):
    """Stop generation when the service cancellation callback becomes true."""

    def __init__(self, callback: Callable[[], bool]) -> None:
        self._callback = callback

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs: Any,
    ) -> bool:
        del input_ids, scores, kwargs
        return self._callback()
