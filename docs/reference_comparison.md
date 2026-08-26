# 阶段二实现对比

## 本项目

- 三个独立的 `q_proj`、`k_proj`、`v_proj`
- 可学习的绝对 Position Embedding
- 手写 Causal Mask
- 手写 Multi-Head Attention
- Pre-Norm Transformer Block
- 不使用 Flash Attention、KV Cache 或 `nn.MultiheadAttention`

## 独立 QKV 与 fused QKV

本项目使用：

```text
Q = XWQ + bQ
K = XWK + bK
V = XWV + bV
```

高效实现可能使用一个 `Linear(C, 3C)`，再沿最后一维拆分为 Q、K、V。
当 fused 权重由三个独立权重正确拼接时，两种方式在数学上等价。

独立实现更适合教学，因为三套参数和三个张量清晰可见；fused 实现通常更适合性能优化，因为可以减少模块调用并使用更大的矩阵乘法。

## 常见实现取舍

- 性能导向实现常用 fused QKV，以减少模块调用和运行开销。
- 教学导向实现通常强调模块化，让 Q/K/V、形状变化、Mask 和 Attention 计算清晰可见。
- 两种组织方式在权重正确映射时数学上等价，主要差别在可读性、调用开销和内核利用率。

本项目同时保留独立 QKV 和 fused QKV，用受控基准记录可读性与性能差异。
