"""Checkpoint-backed MiniGPT model lifecycle management."""

from __future__ import annotations

from threading import Lock
from typing import Any

import torch

from minigpt.config import DataConfig, ModelConfig, load_data_config, load_model_config
from minigpt.generation import generate
from minigpt.model import MiniGPT
from minigpt.tokenizer import ProjectTokenizer, decode_token_ids, load_tokenizer
from minigpt.utils import (
    checkpoint_config_value,
    compile_model,
    load_checkpoint,
    load_checkpoint_payload,
)
from minigpt_service.config import ServiceConfig


def select_device(preference: str = "auto") -> torch.device:
    """Resolve a configured device, preferring CUDA then MPS for auto mode."""
    if preference not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("device must be one of: auto, cpu, cuda, mps.")
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if preference == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ModelService:
    """Own one loaded MiniGPT model and its Tokenizer."""

    def __init__(self, config: ServiceConfig) -> None:
        self.config = config
        self.device = select_device(config.device)
        self.data_config: DataConfig | None = None
        self.model_config: ModelConfig | None = None
        self.tokenizer: ProjectTokenizer | None = None
        self.model: torch.nn.Module | None = None
        self.block_size: int | None = None
        self.checkpoint: dict[str, Any] | None = None
        self.inference_lock = Lock()

    @property
    def is_ready(self) -> bool:
        """Return whether the model and Tokenizer are loaded."""
        return self.model is not None and self.tokenizer is not None

    def load(self) -> None:
        """Load configuration, Tokenizer, Checkpoint, and model exactly once."""
        if self.is_ready:
            return

        data_config = load_data_config(self.config.config_path)
        model_config = load_model_config(self.config.config_path)
        tokenizer = load_tokenizer(
            data_config.tokenizer_path,
            tokenizer_type=data_config.tokenizer_type,
        )
        checkpoint = load_checkpoint_payload(self.config.checkpoint_path, self.device)
        self._validate_checkpoint_tokenizer(checkpoint, tokenizer)

        model = self._build_model(tokenizer, data_config, model_config, checkpoint)
        model = model.to(self.device)
        load_checkpoint(
            path=self.config.checkpoint_path,
            model=model,
            optimizer=None,
            scheduler=None,
            device=self.device,
        )
        model.eval()
        execution_model = compile_model(
            model,
            enabled=self.config.compile_model,
            mode=self.config.compile_mode,
        )

        self.data_config = data_config
        self.model_config = model_config
        self.tokenizer = tokenizer
        self.block_size = model.block_size
        self.checkpoint = checkpoint
        self.model = execution_model

    def close(self) -> None:
        """Release references held by the service and clear accelerator cache."""
        self.model = None
        self.tokenizer = None
        self.data_config = None
        self.model_config = None
        self.block_size = None
        self.checkpoint = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    @staticmethod
    def _validate_checkpoint_tokenizer(
        checkpoint: dict[str, Any],
        tokenizer: ProjectTokenizer,
    ) -> None:
        saved_info = checkpoint.get("tokenizer_info")
        if saved_info is None:
            return
        if not isinstance(saved_info, dict):
            raise ValueError("Checkpoint tokenizer_info must be an object.")
        if saved_info.get("itos") != tokenizer.itos:
            raise ValueError(
                "Checkpoint Tokenizer does not match the current Tokenizer."
            )

    @staticmethod
    def _build_model(
        tokenizer: ProjectTokenizer,
        data_config: DataConfig,
        model_config: ModelConfig,
        checkpoint: dict[str, Any],
    ) -> MiniGPT:
        return MiniGPT(
            vocab_size=tokenizer.vocab_size,
            block_size=checkpoint_config_value(
                checkpoint, "data", "block_size", data_config.block_size
            ),
            embedding_dim=checkpoint_config_value(
                checkpoint, "model", "embedding_dim", model_config.embedding_dim
            ),
            num_heads=checkpoint_config_value(
                checkpoint, "model", "num_heads", model_config.num_heads
            ),
            num_layers=checkpoint_config_value(
                checkpoint, "model", "num_layers", model_config.num_layers
            ),
            expansion_factor=checkpoint_config_value(
                checkpoint,
                "model",
                "expansion_factor",
                model_config.expansion_factor,
            ),
            tie_weights=checkpoint_config_value(
                checkpoint, "model", "tie_weights", model_config.tie_weights
            ),
            position_encoding=checkpoint_config_value(
                checkpoint,
                "model",
                "position_encoding",
                model_config.position_encoding,
            ),
            norm=checkpoint_config_value(
                checkpoint, "model", "norm", model_config.norm
            ),
            activation=checkpoint_config_value(
                checkpoint, "model", "activation", model_config.activation
            ),
            attention_backend=checkpoint_config_value(
                checkpoint,
                "model",
                "attention_backend",
                model_config.attention_backend,
            ),
            qkv_projection=checkpoint_config_value(
                checkpoint,
                "model",
                "qkv_projection",
                model_config.qkv_projection,
            ),
            qk_norm=checkpoint_config_value(
                checkpoint, "model", "qk_norm", model_config.qk_norm
            ),
            num_kv_heads=checkpoint_config_value(
                checkpoint,
                "model",
                "num_kv_heads",
                model_config.num_kv_heads,
            ),
        )

    @torch.inference_mode()
    def generate_text(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        greedy: bool = False,
        use_kv_cache: bool = False,
        seed: int | None = None,
    ) -> str:
        """Generate text using the loaded model; endpoint wiring comes next."""
        if not self.is_ready or self.model is None or self.tokenizer is None:
            raise RuntimeError("ModelService is not ready.")
        prompt_ids = self.tokenizer.encode(prompt, allow_unknown=False)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        if self.block_size is None:
            raise RuntimeError("ModelService block size is not initialized.")
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        with self.inference_lock:
            generated_ids = generate(
                model=self.model,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                block_size=self.block_size,
                do_sample=not greedy,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                use_kv_cache=use_kv_cache,
                generator=generator,
                eos_token_id=getattr(self.tokenizer, "eos_id", None),
            )
        return decode_token_ids(self.tokenizer, generated_ids[0].cpu().tolist())
