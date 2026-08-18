"""Command-line text generation for a trained MiniGPT checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch

from minigpt.config import load_data_config, load_model_config
from minigpt.generation import generate
from minigpt.model import MiniGPT
from minigpt.tokenizer import CharacterTokenizer
from minigpt.utils import (
    checkpoint_config_value,
    compile_model,
    load_checkpoint,
    load_checkpoint_payload,
    set_random_seed,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "tiny_shakespeare.yaml"
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best.pt"


def select_device() -> torch.device:
    """Select CUDA, MPS, or CPU in that order."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    """Parse command-line generation arguments."""
    parser = argparse.ArgumentParser(
        description="Generate text with a trained MiniGPT checkpoint.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="YAML configuration path.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Checkpoint path.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Non-empty prompt text.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Maximum number of new Tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling Temperature.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k candidate count; omit to disable.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p probability mass; omit to disable.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed.",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use deterministic Greedy Decoding.",
    )
    parser.add_argument(
        "--kv-cache",
        action="store_true",
        help="Use KV Cache for prompt prefill and Token-by-Token decode.",
    )
    parser.add_argument(
        "--benchmark-kv-cache",
        action="store_true",
        help="Benchmark Greedy generation with and without KV Cache.",
    )
    parser.add_argument(
        "--benchmark-repeats",
        type=int,
        default=5,
        help="Measured repetitions for the KV Cache benchmark.",
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
    )
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_generation(
    model: MiniGPT,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    repeats: int,
    device: torch.device,
) -> None:
    """Compare steady-state Greedy generation with one changed variable."""
    if type(repeats) is not int or repeats <= 0:
        raise ValueError("benchmark_repeats must be a positive integer.")
    if max_new_tokens <= 0:
        raise ValueError("KV Cache benchmark requires max_new_tokens > 0.")
    if input_ids.size(1) + max_new_tokens > model.block_size:
        raise ValueError("KV Cache benchmark length exceeds block_size.")

    results: dict[bool, list[float]] = {False: [], True: []}
    outputs: dict[bool, torch.Tensor] = {}
    peak_memory_mib: dict[bool, float | None] = {}
    for use_cache in (False, True):
        generate(
            model,
            input_ids,
            max_new_tokens=max_new_tokens,
            block_size=model.block_size,
            do_sample=False,
            use_kv_cache=use_cache,
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for _ in range(repeats):
            synchronize(device)
            started = time.perf_counter()
            output = generate(
                model,
                input_ids,
                max_new_tokens=max_new_tokens,
                block_size=model.block_size,
                do_sample=False,
                use_kv_cache=use_cache,
            )
            synchronize(device)
            results[use_cache].append(time.perf_counter() - started)
            outputs[use_cache] = output
        peak_memory_mib[use_cache] = (
            torch.cuda.max_memory_allocated(device) / (1024**2)
            if device.type == "cuda"
            else None
        )

    if not torch.equal(outputs[False], outputs[True]):
        raise RuntimeError("KV Cache benchmark outputs differ from baseline.")

    baseline = sum(results[False]) / repeats
    cached = sum(results[True]) / repeats
    baseline_rate = max_new_tokens / baseline
    cached_rate = max_new_tokens / cached
    print(f"kv_cache_baseline_seconds: {baseline:.6f}")
    print(f"kv_cache_cached_seconds: {cached:.6f}")
    print(f"kv_cache_baseline_tokens_per_second: {baseline_rate:.2f}")
    print(f"kv_cache_cached_tokens_per_second: {cached_rate:.2f}")
    print(f"kv_cache_speedup: {cached_rate / baseline_rate:.3f}x")
    print(f"kv_cache_baseline_peak_memory_mib: {peak_memory_mib[False]}")
    print(f"kv_cache_cached_peak_memory_mib: {peak_memory_mib[True]}")
    print("kv_cache_outputs_equal: True")


def resolve_project_path(path: Path) -> Path:
    """Resolve a relative path from the project root."""
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def build_model(
    tokenizer: CharacterTokenizer,
    data_config: object,
    model_config: object,
    checkpoint: dict[str, object] | None = None,
) -> MiniGPT:
    """Construct the same model architecture used during training."""
    checkpoint = {} if checkpoint is None else checkpoint
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
    )


def validate_checkpoint_tokenizer(
    checkpoint: dict[str, object],
    tokenizer: CharacterTokenizer,
) -> None:
    """Ensure that checkpoint and runtime Tokenizer use the same vocabulary."""
    saved_info = checkpoint.get("tokenizer_info")
    if saved_info is None:
        return
    if not isinstance(saved_info, dict):
        raise ValueError("Checkpoint tokenizer_info must be an object.")
    if saved_info.get("itos") != tokenizer.itos:
        raise ValueError(
            "Checkpoint Tokenizer does not match the current Tokenizer."
        )


def main() -> None:
    """Load a checkpoint and print one generated completion."""
    args = parse_args()
    if not args.prompt:
        raise ValueError("Prompt cannot be empty.")

    config_path = resolve_project_path(args.config)
    checkpoint_path = resolve_project_path(args.checkpoint)

    data_config = load_data_config(config_path)
    model_config = load_model_config(config_path)
    tokenizer = CharacterTokenizer.load(data_config.tokenizer_path)
    device = select_device()
    checkpoint_payload = load_checkpoint_payload(
        checkpoint_path,
        device,
    )

    prompt_ids = tokenizer.encode(args.prompt, allow_unknown=False)
    input_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
    )

    model = build_model(
        tokenizer=tokenizer,
        data_config=data_config,
        model_config=model_config,
        checkpoint=checkpoint_payload,
    )

    model = model.to(device)
    input_ids = input_ids.to(device)

    checkpoint = load_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=None,
        scheduler=None,
        device=device,
    )
    validate_checkpoint_tokenizer(checkpoint, tokenizer)
    execution_model = compile_model(
        model,
        enabled=args.compile,
        mode=args.compile_mode,
    )

    # Apply the user-requested seed after checkpoint loading, because loading
    # a checkpoint may restore the training-time RNG state.
    if args.seed is not None:
        set_random_seed(args.seed)

    if args.benchmark_kv_cache:
        benchmark_generation(
            execution_model,
            input_ids,
            max_new_tokens=args.max_new_tokens,
            repeats=args.benchmark_repeats,
            device=device,
        )

    generated_ids = generate(
        model=execution_model,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        block_size=model.block_size,
        do_sample=not args.greedy,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        use_kv_cache=args.kv_cache,
    )

    generated_text = tokenizer.decode(
        generated_ids[0].detach().cpu().tolist(),
    )
    print(generated_text)


if __name__ == "__main__":
    main()
