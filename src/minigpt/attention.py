"""Decoder-only Causal Self-Attention。"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

KVCache = tuple[torch.Tensor, torch.Tensor]


def split_heads(
    hidden_states: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    """将 [B,T,C] 拆成 [B,H,T,D]。"""
    if hidden_states.ndim != 3:
        raise ValueError("hidden_states 必须是 [B,T,C]。")
    if type(num_heads) is not int or num_heads <= 0:
        raise ValueError("num_heads 必须是正整数。")

    batch_size, sequence_length, embedding_dim = hidden_states.shape
    if embedding_dim % num_heads != 0:
        raise ValueError("embedding_dim 必须能被 num_heads 整除。")

    head_dim = embedding_dim // num_heads
    hidden_states = hidden_states.contiguous().view(
        batch_size,
        sequence_length,
        num_heads,
        head_dim,
    )

    return hidden_states.transpose(1, 2)


def merge_heads(hidden_states: torch.Tensor) -> torch.Tensor:
    """将 [B,H,T,D] 合并成 [B,T,C]。"""
    if hidden_states.ndim != 4:
        raise ValueError("hidden_states 必须是 [B,H,T,D]。")

    batch_size, num_heads, sequence_length, head_dim = hidden_states.shape

    hidden_states = hidden_states.transpose(1, 2)
    hidden_states = hidden_states.contiguous()

    return hidden_states.view(
        batch_size,
        sequence_length,
        num_heads * head_dim,
    )


class QKVProjection(nn.Module):
    """使用三个独立 Linear 生成 Q、K、V。"""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()

        if type(embedding_dim) is not int or embedding_dim <= 0:
            raise ValueError("embedding_dim 必须是正整数。")

        self.embedding_dim = embedding_dim
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回形状均为 [B,T,C] 的 Q、K、V。"""
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states 必须是 [B,T,C]。")
        if hidden_states.size(-1) != self.embedding_dim:
            raise ValueError("输入最后一维必须等于 embedding_dim。")

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        return query, key, value


class FusedQKVProjection(nn.Module):
    """用一次 Linear 生成 Q/K/V，再沿最后一维切分。"""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        if type(embedding_dim) is not int or embedding_dim <= 0:
            raise ValueError("embedding_dim 必须是正整数。")
        self.embedding_dim = embedding_dim
        self.qkv_proj = nn.Linear(embedding_dim, 3 * embedding_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states 必须是 [B,T,C]。")
        if hidden_states.size(-1) != self.embedding_dim:
            raise ValueError("输入最后一维必须等于 embedding_dim。")
        return self.qkv_proj(hidden_states).chunk(3, dim=-1)


def apply_rotary_position_embedding(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    position_offset: int = 0,
    base: float = 10_000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """对当前 ``[B,H,T,D]`` Q/K 应用 RoPE，支持 KV Cache 偏移。"""
    if query.shape != key.shape or query.ndim != 4:
        raise ValueError("query 和 key 必须是相同形状的 [B,H,T,D]。")
    if query.size(-1) % 2 != 0:
        raise ValueError("RoPE 要求 head_dim 为偶数。")
    if type(position_offset) is not int or position_offset < 0:
        raise ValueError("position_offset 必须是非负整数。")
    if base <= 0:
        raise ValueError("RoPE base 必须大于 0。")

    head_dim = query.size(-1)
    positions = torch.arange(
        position_offset,
        position_offset + query.size(-2),
        device=query.device,
        dtype=torch.float32,
    )
    inverse_frequency = 1.0 / (
        base
        ** (
            torch.arange(0, head_dim, 2, device=query.device, dtype=torch.float32)
            / head_dim
        )
    )
    angles = torch.outer(positions, inverse_frequency)
    cosine = angles.cos().to(dtype=query.dtype)[None, None, :, :]
    sine = angles.sin().to(dtype=query.dtype)[None, None, :, :]

    def rotate(tensor: torch.Tensor) -> torch.Tensor:
        even = tensor[..., 0::2]
        odd = tensor[..., 1::2]
        rotated = torch.stack(
            (even * cosine - odd * sine, even * sine + odd * cosine),
            dim=-1,
        )
        return rotated.flatten(-2)

    return rotate(query), rotate(key)


def scaled_dot_product_scores(
    query: torch.Tensor,
    key: torch.Tensor,
) -> torch.Tensor:
    """计算缩放后的 QKᵀ，返回 [B,H,T,T]。"""
    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("query 和 key 必须是 [B,H,T,D]。")
    if query.shape[:2] != key.shape[:2]:
        raise ValueError("query 和 key 的 Batch 与 Head 维必须一致。")
    if query.size(-1) != key.size(-1):
        raise ValueError("query 和 key 的 Head Dim 必须一致。")

    head_dim = query.size(-1)
    scores = torch.matmul(
        query,
        key.transpose(-2, -1),
    )
    return scores / math.sqrt(head_dim)


def apply_causal_mask(scores: torch.Tensor) -> torch.Tensor:
    """对 ``[B,H,Q,K]`` 分数应用支持历史 Cache 的因果 Mask。"""
    if scores.ndim != 4:
        raise ValueError("scores 必须是 [B,H,T,T]。")
    query_length = scores.size(-2)
    key_length = scores.size(-1)
    if query_length <= 0 or key_length < query_length:
        raise ValueError("Key 长度必须大于等于正的 Query 长度。")

    past_length = key_length - query_length
    query_positions = torch.arange(query_length, device=scores.device) + past_length
    key_positions = torch.arange(key_length, device=scores.device)
    mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)

    return scores.masked_fill(~mask, float("-inf"))


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    dropout_probability: float = 0.0,
    training: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """执行 Causal Attention，并返回 Dropout 前的可解释权重。"""
    if key.shape != value.shape:
        raise ValueError("key 和 value 的形状必须一致。")
    if not 0.0 <= dropout_probability < 1.0:
        raise ValueError("dropout_probability 必须位于 [0, 1) 中。")

    scores = scaled_dot_product_scores(query, key)
    scores = apply_causal_mask(scores)
    weights = torch.softmax(scores, dim=-1)
    dropped_weights = F.dropout(
        weights,
        p=dropout_probability,
        training=training,
    )
    output = torch.matmul(dropped_weights, value)
    return output, weights


class MultiHeadCausalSelfAttention(nn.Module):
    """教学版 Multi-Head Causal Self-Attention。"""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        block_size: int,
        dropout: float = 0.0,
        attention_backend: str = "manual",
        qkv_projection: str = "separate",
        position_encoding: str = "learned",
    ) -> None:
        super().__init__()

        if type(embedding_dim) is not int or embedding_dim <= 0:
            raise ValueError("embedding_dim 必须是正整数。")
        if type(num_heads) is not int or num_heads <= 0:
            raise ValueError("num_heads 必须是正整数。")
        if type(block_size) is not int or block_size <= 0:
            raise ValueError("block_size 必须是正整数。")
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim 必须能被 num_heads 整除。")
        if not isinstance(dropout, (int, float)) or isinstance(dropout, bool):
            raise TypeError("dropout 必须是数字。")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout 必须位于 [0, 1) 中。")
        if attention_backend not in {"manual", "sdpa"}:
            raise ValueError("attention_backend 必须是 manual 或 sdpa。")
        if qkv_projection not in {"separate", "fused"}:
            raise ValueError("qkv_projection 必须是 separate 或 fused。")
        if position_encoding not in {"learned", "rope"}:
            raise ValueError("position_encoding 必须是 learned 或 rope。")
        if position_encoding == "rope" and embedding_dim // num_heads % 2 != 0:
            raise ValueError("RoPE 要求 head_dim 为偶数。")

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.block_size = block_size
        self.head_dim = embedding_dim // num_heads
        self.dropout_probability = float(dropout)
        self.attention_backend = attention_backend
        self.position_encoding = position_encoding

        self.qkv_projection: QKVProjection | FusedQKVProjection
        if qkv_projection == "fused":
            self.qkv_projection = FusedQKVProjection(embedding_dim)
        else:
            self.qkv_projection = QKVProjection(embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        self.output_dropout = nn.Dropout(self.dropout_probability)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, KVCache]
        | tuple[torch.Tensor, KVCache, torch.Tensor]
    ):
        """输入当前 Token，可选接收并返回本层 K/V Cache。"""
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states 必须是 [B,T,C]。")
        if hidden_states.size(-1) != self.embedding_dim:
            raise ValueError("输入最后一维必须等于 embedding_dim。")
        current_length = hidden_states.size(1)
        if current_length > self.block_size:
            raise ValueError("输入序列长度超过 block_size。")

        query, key, value = self.qkv_projection(hidden_states)
        query = split_heads(query, self.num_heads)
        key = split_heads(key, self.num_heads)
        value = split_heads(value, self.num_heads)

        past_length = 0
        if past_key_value is not None:
            past_key, past_value = past_key_value
            expected_prefix = (
                hidden_states.size(0),
                self.num_heads,
            )
            if past_key.ndim != 4 or past_value.ndim != 4:
                raise ValueError("缓存的 key/value 必须是 [B,H,S,D]。")
            if past_key.shape != past_value.shape:
                raise ValueError("缓存的 key/value 形状必须一致。")
            if past_key.shape[:2] != expected_prefix:
                raise ValueError("缓存的 Batch 或 Head 数与当前输入不一致。")
            if past_key.size(-1) != self.head_dim:
                raise ValueError("缓存的 Head Dim 与模型不一致。")
            if past_key.device != key.device or past_key.dtype != key.dtype:
                raise ValueError("缓存的设备与 dtype 必须和当前 K/V 一致。")
            if past_key.size(2) + current_length > self.block_size:
                raise ValueError("缓存长度与当前输入之和超过 block_size。")
            past_length = past_key.size(2)

        if self.position_encoding == "rope":
            query, key = apply_rotary_position_embedding(
                query,
                key,
                position_offset=past_length,
            )

        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat((past_key, key), dim=2)
            value = torch.cat((past_value, value), dim=2)

        if self.attention_backend == "sdpa" and not output_attentions:
            query_length = query.size(-2)
            key_length = key.size(-2)
            query_positions = (
                torch.arange(query_length, device=query.device)
                + key_length
                - query_length
            )
            key_positions = torch.arange(key_length, device=query.device)
            causal_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
            attention_output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=causal_mask,
                dropout_p=self.dropout_probability if self.training else 0.0,
            )
            attention_weights = None
        else:
            attention_output, attention_weights = scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_probability=self.dropout_probability,
                training=self.training,
            )

        attention_output = merge_heads(attention_output)
        attention_output = self.output_dropout(self.out_proj(attention_output))
        present_key_value = (key, value)
        if use_cache and output_attentions:
            assert attention_weights is not None
            return attention_output, present_key_value, attention_weights
        if use_cache:
            return attention_output, present_key_value
        if output_attentions:
            assert attention_weights is not None
            return attention_output, attention_weights
        return attention_output
