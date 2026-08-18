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

## nanoGPT、minGPT、build-nanogpt

- nanoGPT 风格的实现常见 fused QKV，以减少运行开销。
- minGPT 风格强调模块化和教学可读性，具体 QKV 组织方式可以因版本不同而变化。
- build-nanogpt 常把形状变化、Mask 和 Attention 计算逐步展开，适合学习底层过程。

本项目保留独立 QKV，记录设计差异，不复制这些项目的完整代码。
