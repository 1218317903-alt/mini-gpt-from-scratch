# P4：3B QLoRA 训练与重载验收说明

作者：李林知
日期：2026-08-25

## 我完成了什么

我在 RTX 4060 Laptop 8GB 显存环境中，以 Qwen2.5-3B-Instruct 为基座完成了 Agent
v1.1 的 QLoRA 训练。基座使用 NF4 4-bit 与 double quantization，LoRA 使用 rank 8、
alpha 16，并注入 q/k/v/o projection。训练长度为 80 个 optimizer steps，micro batch
为 1，梯度累积为 4，序列上限为 640，seed 为 20260825。

配置与原始记录：

- `hf_lora/configs/agent_v1_qwen2_5_3b_qlora.yaml`
- `hf_lora/artifacts/agent_v1_1_qwen2_5_3b_qlora/training_metrics.json`
- `hf_lora/artifacts/agent_v1_1_qwen2_5_3b_qlora/adapter/`

## 训练结果

| 项目 | 实测值 |
|---|---:|
| Optimizer steps | 80 |
| 训练 step 累计耗时 | 277.75 秒 |
| 峰值 allocated VRAM | 4,028.70MiB |
| 最后一步 train loss | 0.004049 |
| Final validation loss | 0.160835 |
| Adapter 目录大小 | 约 26.25MiB |

Loss 的下降证明优化器有效拟合了 assistant-only 监督目标，但我没有把它当作 Agent 能力
结论。真正的候选选择来自 100 条 dev 的端到端指标：任务成功率 78%、JSON 合法率 100%、
端到端 RAG Recall@5 63.64%、Citation Recall 63.64%、不必要工具调用率 0%。随后唯一一次
150 条冻结终测得到任务成功率 74%、RAG Recall@5 88.64%，同时暴露多步骤成功率 0% 和
恢复率 0%。

## 为什么没有 `comparison.json`

训练后的严格重载检查比较了 NF4 模型重新量化前后的 logits，最大差异约为 0.628，脚本
因此在写出 `comparison.json` 前按旧规则停止。重新加载 4-bit 基座会再次执行量化，不能像
FP16 非量化权重那样期待逐位 logits 相同；这一现象不能直接证明 Adapter 损坏。

我采用的证据边界是：

1. 不宣称 3B QLoRA 通过了严格 logits 一致性。
2. Adapter 被新的模型进程独立加载，并完成 100 条 dev 与 150 条 frozen 推理。
3. 250 条 Agent 推理没有 runtime error，证明产物可加载、可运行。
4. 训练脚本已将量化模型的后续重载验收改为 greedy generation 一致性；非量化模型仍可
   使用严格 logits 比较。

我没有为补一份报告而做第三次相同训练，因为这不会增加关于模型能力的新证据，还会浪费
计算预算。后续若更换数据或配方重新训练，应使用修正后的验收规则，并同时保留生成文本、
token IDs 和 Adapter hash。

## 结果应如何解释

QLoRA 的核心不是“把 LoRA 也量化”，而是冻结的 base weights 用 4-bit 存储与计算，梯度
只更新低秩 Adapter。这样可以在消费级 GPU 上训练 3B 模型。NF4 针对近似正态分布的权重
设计量化桶，double quantization 再压缩量化常数。代价是量化加载并非完全逐位确定，因此
验收应覆盖可重载性、确定性生成、任务指标和错误分层，而不能只依赖一个 logits epsilon。
