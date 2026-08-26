# 现代文本训练线：架构对照与 Early Stopping 记录

本记录补齐 P8 文本训练线的正式随机 Batch 对照，并验证小语料下的 Early Stopping
是否能在验证集开始恶化时停止训练。它只描述本项目的 MiniGPT、Tiny Shakespeare
数据和当前 RTX 4060 Laptop GPU 运行结果，不把单机单 seed 结果外推到其他硬件或
更大的模型。

## 1. 实验范围与复现命令

两组实验使用同一份 BPE 词表、数据切分、seed、有效 Batch 和优化器配置；唯一主要
结构差异是普通多头 Attention 与 QK Norm + GQA。

| 项目 | 设置 |
|---|---|
| 原始数据 | `data/raw/tiny_shakespeare.txt`，1,155,394 字符 |
| Tokenizer | 教学型 BPE，词表 512，启用 EOS/PAD 与 sequence packing |
| 模型主干 | 384 hidden、6 层、6 query heads、RoPE、RMSNorm、SwiGLU、SDPA、fused QKV |
| 训练设备 | NVIDIA GeForce RTX 4060 Laptop GPU，8,188 MiB 显存 |
| 精度 / seed | BF16 / 42 |
| Micro Batch / 累积 | 16 / 4，等效 Batch 64 |
| 调度 | 峰值学习率 `3e-4`，warmup 200，Cosine，最低 `3e-5` |
| 早停规则 | 每 50 步完整遍历验证集；连续 3 次无至少 `0.01` Loss 改善即停止 |

正式 Early Stopping 运行：

```powershell
.\.venv\Scripts\python.exe train.py `
  --config configs\tiny_shakespeare_4060_bpe.yaml `
  --precision bf16 --gradient-accumulation-steps 4 --max-steps 500 `
  --eval-interval 50 --early-stopping-patience 3 `
  --early-stopping-min-delta 0.01 --warmup-steps 200 `
  --cosine-decay --min-learning-rate 0.00003 --skip-overfit `
  --checkpoint-dir checkpoints\formal_early_stop_bpe_20260826 `
  --run-dir runs\formal_early_stop_bpe_20260826
```

QK Norm + GQA 只将配置和输出目录替换为：

```powershell
--config configs\tiny_shakespeare_4060_qknorm_gqa.yaml
--checkpoint-dir checkpoints\formal_early_stop_qknorm_gqa_20260826
--run-dir runs\formal_early_stop_qknorm_gqa_20260826
```

每个 run 目录都保存最终配置、环境快照、JSONL 指标和日志；Checkpoint 目录默认被
`.gitignore` 忽略，不进入源码提交。

## 2. 结果一：同预算 2,000 步控制实验

这是早停改动前已经完成的同预算对照，用来比较架构质量、吞吐和显存。两组都运行到
2,000 个 optimizer step，最佳 Checkpoint 用当前 `evaluate.py` 重新评估测试集，因而
额外补充了统一的 UTF-8 BPB。

| 配置 | 参数量 | 最佳验证 Loss（step） | 测试 Loss | 测试 UTF-8 BPB | 训练时间 | 末步吞吐 | 峰值显存 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 现代 BPE 基线 | 14,366,592 | **3.5146（250）** | **3.8696** | **2.6783** | 417.05 s | **78,521 tok/s** | 894.11 MiB |
| QK Norm + GQA | 13,183,872 | 3.7801（250） | 4.0984 | 2.8367 | 456.86 s | 71,680 tok/s | 947.50 MiB |

在这个短序列、小模型和单卡环境中，GQA 的参数量少 8.23%，但测试 Loss 高 0.2288
（约 5.91%），训练时间多 9.54%，末步吞吐低 8.72%，峰值分配显存多 53.39 MiB。
因此 GQA 在本实验中只保留为可选结构：它的主要价值是减少 KV Cache 的 K/V 头数，
而不是已被当前小规模训练证明有质量或训练速度优势。

## 3. 结果二：完整验证集 Early Stopping

这一轮每次验证都完整遍历验证集，不再用固定的前若干 Batch 估计 BPB；Checkpoint
同时保存 `early_stopping_bad_evals`，恢复训练时会继续累计，而不是把早停计数清零。

| 配置 | 最佳验证 Loss / BPB（step） | 测试 Loss / BPB | 实际停止步数 | 实际训练 Token | 末步吞吐 | 峰值显存 |
|---|---:|---:|---:|---:|---:|---:|
| 现代 BPE 基线 | **3.4873 / 2.3778（250）** | **3.8404 / 2.6581** | 400 | 6,549,504 | 67,197 tok/s | 894.11 MiB |
| QK Norm + GQA | 3.6993 / 2.5224（200） | 3.9799 / 2.7547 | 350 | 5,730,816 | 77,205 tok/s | 949.50 MiB |

验证曲线给出了清晰的过拟合拐点：

- 普通基线在 step 250 达到最佳验证 Loss 3.4873，随后 step 300、350、400 连续恶化，
  因而在 step 400 停止。
- GQA 在 step 200 达到最佳验证 Loss 3.6993，随后连续三次恶化，在 step 350 停止。

早停运行的吞吐不能直接与 2,000 步运行作严格性能排名，因为 GPU 时钟和运行长度会
影响累计吞吐；速度结论以同预算控制实验为准。早停的确定性收益是避免在验证集已经
恶化后继续消耗训练步数，并把停止依据和恢复状态写入实验记录。

## 4. 生成与缓存检查

两个最佳 Checkpoint 都能完成固定 seed 的生成，输出仍明显受小语料和模型规模限制，
不能将随机生成样例当作语言能力指标。KV Cache 的一致性检查在两组模型上均通过：
普通路径与缓存路径的 greedy 输出完全一致。

本次短 Prompt、64 个新 Token、3 次重复的本机观测如下；它只用于功能和数量级检查：

| 配置 | 无缓存 | 有缓存 | 观测加速 |
|---|---:|---:|---:|
| 现代 BPE 基线 | 105.35 tok/s | 110.73 tok/s | 1.051x |
| QK Norm + GQA | 95.32 tok/s | 94.58 tok/s | 0.992x |

短序列下缓存收益会被 Python 调度和测量噪声抵消；不能据此推断长上下文收益。真正
部署时仍应在目标 Prompt 长度、生成长度和并发量下单独基准。

## 5. 结论与下一步

1. 当前默认应保留现代 BPE 基线；它在本机控制实验中同时取得更低测试 Loss、更低
   BPB、更高吞吐和更低峰值显存。
2. Early Stopping 已经具备可复现实现：完整验证、最小改善阈值、连续失败计数、
   Checkpoint 恢复和 `run_completed.stopped_early` 均有记录。后续正式训练建议默认
   打开它，并把最佳 Checkpoint 用独立测试集评估。
3. 当前瓶颈不是继续堆叠 Attention 变体，而是数据规模与覆盖范围。Tiny Shakespeare
   只有约 1.15 MB 原始文本，验证集和测试集分别约 52.7k / 53.5k BPE Token；在没有
   明确许可证、清洗规则和预算登记前，不自动下载更大的外部语料。
4. 下一项有明确收益且可归因的工作是：选择许可证清晰的多百万 Token 公共语料，新增
   数据 manifest、去重与字符级切分统计，再用同一配置重跑基线和 GQA。完成前不把
   当前小语料结果外推为通用模型结论。

