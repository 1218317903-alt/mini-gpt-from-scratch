"""Decoder-only MiniGPT 模型结构。"""

from __future__ import annotations

import torch
from torch import nn

from minigpt.attention import KVCache, MultiHeadCausalSelfAttention


ModelKVCache = tuple[KVCache, ...]


class TokenEmbedding(nn.Module):
    """将 Token ID 转换为 [B,T,C]。"""

    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        super().__init__()
        if type(vocab_size) is not int or vocab_size <= 0:
            raise ValueError("vocab_size 必须是正整数。")
        if type(embedding_dim) is not int or embedding_dim <= 0:
            raise ValueError("embedding_dim 必须是正整数。")

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids 必须是 [B,T]。")
        if token_ids.dtype != torch.long:
            raise TypeError("token_ids 必须是 torch.long。")
        if token_ids.numel() > 0:
            if token_ids.min() < 0:
                raise ValueError("token_ids 不能包含负数。")
            if token_ids.max() >= self.vocab_size:
                raise ValueError("token_ids 超出词表范围。")
        return self.embedding(token_ids)


class PositionEmbedding(nn.Module):
    """可学习的绝对位置 Embedding。"""

    def __init__(self, block_size: int, embedding_dim: int) -> None:
        super().__init__()
        if type(block_size) is not int or block_size <= 0:
            raise ValueError("block_size 必须是正整数。")
        if type(embedding_dim) is not int or embedding_dim <= 0:
            raise ValueError("embedding_dim 必须是正整数。")

        self.block_size = block_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(block_size, embedding_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_offset: int = 0,
    ) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states 必须是 [B,T,C]。")
        if hidden_states.size(-1) != self.embedding_dim:
            raise ValueError("输入最后一维必须等于 embedding_dim。")

        sequence_length = hidden_states.size(1)
        if type(position_offset) is not int or position_offset < 0:
            raise ValueError("position_offset 必须是非负整数。")
        if position_offset + sequence_length > self.block_size:
            raise ValueError("输入序列长度超过 block_size。")

        positions = torch.arange(
            position_offset,
            position_offset + sequence_length,
            device=hidden_states.device,
        )
        position_vectors = self.embedding(positions).unsqueeze(0)
        return hidden_states + position_vectors


class FeedForward(nn.Module):
    """Transformer Block 中的两层前馈网络。"""

    def __init__(
        self,
        embedding_dim: int,
        expansion_factor: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if type(embedding_dim) is not int or embedding_dim <= 0:
            raise ValueError("embedding_dim 必须是正整数。")
        if type(expansion_factor) is not int or expansion_factor <= 0:
            raise ValueError("expansion_factor 必须是正整数。")
        if not isinstance(dropout, (int, float)) or isinstance(dropout, bool):
            raise TypeError("dropout 必须是数字。")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout 必须位于 [0, 1) 中。")

        hidden_dim = embedding_dim * expansion_factor
        self.fc_in = nn.Linear(embedding_dim, hidden_dim)
        self.activation = nn.GELU()
        self.fc_out = nn.Linear(hidden_dim, embedding_dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states 必须是 [B,T,C]。")
        return self.dropout(
            self.fc_out(self.activation(self.fc_in(hidden_states)))
        )


class TransformerBlock(nn.Module):
    """Pre-Norm Transformer Block。"""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        block_size: int,
        expansion_factor: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(embedding_dim)
        self.attention = MultiHeadCausalSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            block_size=block_size,
            dropout=dropout,
        )
        self.layer_norm_2 = nn.LayerNorm(embedding_dim)
        self.feed_forward = FeedForward(
            embedding_dim=embedding_dim,
            expansion_factor=expansion_factor,
            dropout=dropout,
        )

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
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states 必须是 [B,T,C]。")

        attention_result = self.attention(
            self.layer_norm_1(hidden_states),
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        if use_cache and output_attentions:
            attention_output, present_key_value, attention_weights = attention_result
        elif use_cache:
            attention_output, present_key_value = attention_result
            attention_weights = None
        elif output_attentions:
            attention_output, attention_weights = attention_result
            present_key_value = None
        else:
            attention_output = attention_result
            attention_weights = None
            present_key_value = None

        hidden_states = hidden_states + attention_output
        hidden_states = hidden_states + self.feed_forward(
            self.layer_norm_2(hidden_states)
        )
        if use_cache and output_attentions:
            assert present_key_value is not None
            assert attention_weights is not None
            return hidden_states, present_key_value, attention_weights
        if use_cache:
            assert present_key_value is not None
            return hidden_states, present_key_value
        if output_attentions:
            assert attention_weights is not None
            return hidden_states, attention_weights
        return hidden_states


class LanguageModelHead(nn.Module):
    """将隐藏状态映射到词表 logits。"""

    def __init__(self, embedding_dim: int, vocab_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(
            embedding_dim,
            vocab_size,
            bias=False,
        )

    @property
    def weight(self) -> torch.Tensor:
        """返回词表投影权重。"""
        return self.projection.weight

    def tie_weights(self, weight: nn.Parameter) -> None:
        """让输出投影与 Token Embedding 共享权重。"""
        if weight.shape != self.projection.weight.shape:
            raise ValueError("共享权重的形状必须一致。")
        self.projection.weight = weight

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states 必须是 [B,T,C]。")
        return self.projection(hidden_states)


class MiniGPT(nn.Module):
    """教学版 Decoder-only MiniGPT。"""

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
        expansion_factor: int = 4,
        tie_weights: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        values = {
            "vocab_size": vocab_size,
            "block_size": block_size,
            "embedding_dim": embedding_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "expansion_factor": expansion_factor,
        }
        if any(type(value) is not int or value <= 0 for value in values.values()):
            raise ValueError("模型整数配置必须全部是正整数。")
        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim 必须能被 num_heads 整除。"
            )
        if not isinstance(dropout, (int, float)) or isinstance(dropout, bool):
            raise TypeError("dropout 必须是数字。")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout 必须位于 [0, 1) 中。")

        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.expansion_factor = expansion_factor
        self.dropout_probability = float(dropout)

        self.token_embedding = TokenEmbedding(
            vocab_size,
            embedding_dim,
        )
        self.position_embedding = PositionEmbedding(
            block_size,
            embedding_dim,
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    block_size=block_size,
                    expansion_factor=expansion_factor,
                    dropout=self.dropout_probability,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(embedding_dim)
        self.lm_head = LanguageModelHead(embedding_dim, vocab_size)

        self.apply(self._init_weights)

        if tie_weights:
            self.lm_head.tie_weights(
                self.token_embedding.embedding.weight
            )

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """初始化 Embedding、Linear 和 LayerNorm。"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        token_ids: torch.Tensor,
        *,
        past_key_values: ModelKVCache | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, tuple[torch.Tensor, ...]]
        | tuple[torch.Tensor, ModelKVCache]
        | tuple[torch.Tensor, ModelKVCache, tuple[torch.Tensor, ...]]
    ):
        if token_ids.ndim != 2:
            raise ValueError("token_ids 必须是 [B,T]。")
        if token_ids.dtype != torch.long:
            raise TypeError("token_ids 必须是 torch.long。")
        if token_ids.size(1) <= 0:
            raise ValueError("token_ids 的序列长度必须大于 0。")
        if token_ids.size(1) > self.block_size:
            raise ValueError("输入序列长度超过 block_size。")

        if past_key_values is None:
            layer_past_values: tuple[KVCache | None, ...] = (
                (None,) * self.num_layers
            )
            past_length = 0
        else:
            if len(past_key_values) != self.num_layers:
                raise ValueError("past_key_values 数量必须等于 num_layers。")
            layer_past_values = past_key_values
            cache_lengths = {key.size(2) for key, _ in past_key_values}
            if len(cache_lengths) != 1:
                raise ValueError("所有层的 KV Cache 长度必须一致。")
            past_length = next(iter(cache_lengths))

        if past_length + token_ids.size(1) > self.block_size:
            raise ValueError("缓存长度与当前输入之和超过 block_size。")

        hidden_states = self.token_embedding(token_ids)
        hidden_states = self.position_embedding(
            hidden_states,
            position_offset=past_length,
        )

        attention_weights: list[torch.Tensor] = []
        present_key_values: list[KVCache] = []
        for block, layer_past in zip(self.blocks, layer_past_values):
            block_result = block(
                hidden_states,
                past_key_value=layer_past,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            if use_cache and output_attentions:
                hidden_states, layer_present, block_weights = block_result
                present_key_values.append(layer_present)
                attention_weights.append(block_weights)
            elif use_cache:
                hidden_states, layer_present = block_result
                present_key_values.append(layer_present)
            elif output_attentions:
                hidden_states, block_weights = block_result
                attention_weights.append(block_weights)
            else:
                hidden_states = block_result

        hidden_states = self.final_layer_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        model_cache = tuple(present_key_values)
        weights_tuple = tuple(attention_weights)
        if use_cache and output_attentions:
            return logits, model_cache, weights_tuple
        if use_cache:
            return logits, model_cache
        if output_attentions:
            return logits, weights_tuple
        return logits


def count_parameters(model: nn.Module) -> int:
    """统计可训练参数数量。"""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def count_parameters_by_group(model: nn.Module) -> dict[str, int]:
    """按教学模块统计独立可训练参数，并避免重复计算共享权重。"""
    groups = {
        "embedding": 0,
        "attention": 0,
        "feed_forward": 0,
        "layer_norm": 0,
        "lm_head": 0,
        "other": 0,
    }

    for name, parameter in model.named_parameters(remove_duplicate=True):
        if not parameter.requires_grad:
            continue
        if name.startswith(("token_embedding.", "position_embedding.")):
            group = "embedding"
        elif ".attention." in name:
            group = "attention"
        elif ".feed_forward." in name:
            group = "feed_forward"
        elif "layer_norm" in name:
            group = "layer_norm"
        elif name.startswith("lm_head."):
            group = "lm_head"
        else:
            group = "other"
        groups[group] += parameter.numel()

    grouped_total = sum(groups.values())
    total = count_parameters(model)
    if grouped_total != total:
        raise RuntimeError(
            "参数分组总和与独立可训练参数总量不一致："
            f"{grouped_total} != {total}"
        )
    return groups
