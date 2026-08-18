"""固定配置的公平生成比较。"""

from __future__ import annotations

from typing import Any

import torch


@torch.no_grad()
def generate_response(
    model: torch.nn.Module,
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    was_training = model.training
    previous_cache = bool(getattr(model.config, "use_cache", False))
    model.eval()
    model.config.use_cache = True
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    response_ids = output[0, inputs["input_ids"].shape[1] :]
    response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
    model.config.use_cache = previous_cache
    if was_training:
        model.train()
    return response
