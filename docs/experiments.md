# 阶段一到四实验记录

## 实验目标

验证以下闭环：

```text
数据准备 → Tokenizer → Dataset → MiniGPT → 训练 → Checkpoint
→ 验证 Loss → Prompt 编码 → 自回归生成 → Tokenizer 解码
```

## 自动化测试

当前测试覆盖阶段一到四的单元、集成和端到端路径：

```powershell
uv run pytest -q
```

测试包括：

- Dataset 的 `x/y` 错位和 DataLoader Batch
- Tokenizer 保存、加载、未知字符和非法 ID
- Q/K/V、Attention、Causal Mask 和多头形状
- MiniGPT 前向、反向、Batch 独立性和权重共享
- 单步训练、验证 Loss 和 Checkpoint 往返
- Greedy、随机采样、Temperature、Top-k、Top-p
- 固定种子复现
- `block_size` 截断和每步单 Token 追加
- Checkpoint 加载、Tokenizer 一致性和 CLI 解码

## 采样参数对比

使用同一个 Checkpoint、Prompt 和 seed，比较：

| 模式 | Temperature | Top-k | Top-p | 预期特征 |
|---|---:|---:|---:|---|
| Greedy | 不适用 | 不适用 | 不适用 | 最稳定、确定性最高 |
| 保守采样 | 0.5 | 20 | 0.90 | 候选集中、变化较少 |
| 平衡采样 | 0.8 | 40 | 0.95 | 连贯性与多样性折中 |
| 原始采样 | 1.0 | 关闭 | 关闭 | 使用完整分布 |
| 发散采样 | 1.2 | 100 | 1.00 | 多样性更高、噪声风险更大 |

命令示例：

```powershell
python generate.py `
  --checkpoint checkpoints/best.pt `
  --prompt "ROMEO:" `
  --max-new-tokens 300 `
  --temperature 0.8 `
  --top-k 40 `
  --top-p 0.95 `
  --seed 42
```

## 观察指标

生成质量不能只看训练 Loss，应同时观察：

- 验证集 Loss
- Greedy 与采样结果的差异
- 固定 seed 的复现性
- 重复片段数量
- 字符级语法和拼写连续性
- 长上下文被截断后的连贯性
- 不同 Temperature 下的多样性

## 已完成的闭环烟囱测试

使用临时小型字符语料完成了：

```text
训练 2 steps
→ 保存 best.pt/latest.pt
→ evaluate.py 计算验证 Loss
→ generate.py 加载 best.pt
→ 使用 Temperature/Top-k/Top-p/seed 生成并解码
```

实际烟囱测试中训练和评估进程均成功退出，生成结果可被 Tokenizer 解码。该测试使用的是教学用极小语料，不能代表 Tiny Shakespeare 的最终生成质量。

## 结果记录模板

```text
checkpoint:
prompt:
max_new_tokens:
temperature:
top_k:
top_p:
seed:
validation_loss:
generated_text:
重复现象:
连贯性观察:
```

## 阶段五：实验规范

阶段五先运行基线，再一次只改变一个主要变量。首个 Tiny Shakespeare FP32 正式基线
已经完成，结果见 [`tiny_shakespeare_baseline.md`](tiny_shakespeare_baseline.md)；尚未
运行的消融结果不得提前填写。

### 基线候选

```text
block_size: 64
batch_size: 32
embedding_dim: 128
num_heads: 4
num_layers: 4
expansion_factor: 4
dropout: 0.0
learning_rate: 0.0003
precision: fp32
gradient_accumulation_steps: 1
warmup_steps: 0
cosine_decay: false
seed: 42
```

### 单次实验记录模板

```text
实验名称:
实验目标:
设备与软件环境:
基线配置:
唯一修改变量:
训练预算口径: Optimizer Steps / 总 Token / 总时间
参数量:
训练 Loss 定义与结果:
最佳验证 Loss:
最佳结果 Step:
最后验证 Loss:
tokens/s 测量区间:
峰值 allocated 显存:
checkpoint:
生成 Prompt 与采样配置:
生成样例:
实验结论:
是否需要多 seed 复验:
```

训练入口会将其中的机器可采集字段自动写入运行目录：

- `config.yaml`：合并 YAML 与命令行覆盖后的最终配置
- `environment.json`：Python、PyTorch、CUDA、GPU、Git commit 与 dirty 状态
- `metrics.jsonl`：逐 step 训练指标、验证结果与 Checkpoint 事件
- `train.log`：带时间戳的控制台日志

JSONL 每行是独立 JSON 对象，因此训练中断后，已经刷新的指标仍然可以读取。

### 结构与 Dropout 消融

先把基线值放在 `--values` 第一位，再放变体值：

```powershell
python scripts/run_ablation.py --experiment dropout --values 0.0 0.1
python scripts/run_ablation.py --experiment num_heads --values 4 8
python scripts/run_ablation.py --experiment num_layers --values 4 6
python scripts/run_ablation.py --experiment d_model --values 128 256
python scripts/run_ablation.py --experiment block_size --values 64 128
```

每次运行使用独立 Checkpoint 目录，并保存 `command.txt` 与 `run.log`。

### 梯度累积对比

```text
真实大 Batch：micro_batch_size=16，accumulation_steps=1
累积方案：  micro_batch_size=4， accumulation_steps=4
```

两组有效 Batch 都是 16。应比较峰值显存、tokens/s、训练 Loss 和验证 Loss。

### Warmup 与 Cosine 的单变量顺序

1. 固定学习率基线。
2. 只加入 Warmup，Warmup 后保持峰值学习率。
3. 使用相同 Warmup，再单独加入 Cosine Decay。

不能直接把“固定学习率”和“Warmup + Cosine”之间的全部差异归因于某一项。

Tiny Shakespeare 的三组同预算正式实验已经完成，结果见
[`learning_rate_schedule_comparison.md`](learning_rate_schedule_comparison.md)。在当前
2,000 步短预算下，固定学习率取得最低验证与测试 Loss，因此保留为默认方案。

### 混合精度对比

保持模型、数据、Batch、训练步数和调度策略一致，分别比较：

```text
fp32
fp16（CUDA + GradScaler）
bf16（支持 BF16 的 CUDA 设备）
```

验证 Loss 建议统一使用 FP32 评估路径；混合精度推理吞吐应作为另一项测量。

Tiny Shakespeare 的 FP32、FP16、BF16 同预算正式对照已经完成，结果见
[`mixed_precision_comparison.md`](mixed_precision_comparison.md)。当前 RTX 4060 上的
单次运行观察是：两种混合精度均节省约 17.29% allocated 峰值显存，但没有提升这个
81.8 万参数小模型的训练吞吐；该结果不应外推到更大模型或其他硬件。

# KV Cache 基准记录

运行示例：

```powershell
python generate.py --config configs/tiny_shakespeare.yaml --checkpoint checkpoints/best.pt --prompt "To be" --max-new-tokens 64 --greedy --benchmark-kv-cache
```

基准固定模型、Prompt、生成长度、精度和设备，只改变是否启用 KV Cache。程序分别预热普通路径和缓存路径，并在 CUDA 计时前后同步。应记录普通/缓存耗时、tokens/s 与加速比；这些数据不应与 `torch.compile` 或混合精度同时启用后归因给 KV Cache。

# torch.compile 基准记录

训练、评估和生成入口支持 `--compile` 与 `--compile-mode {default,reduce-overhead,max-autotune}`。Checkpoint 始终由未包装的原始模型保存，编译包装仅负责执行前向。对比时必须分开报告首次编译耗时与预热后的稳态耗时。

当前 Windows CUDA 环境的实测限制：PyTorch 2.13.0+cu126 已进入 TorchInductor，但缺少可用 Triton，因而 `--compile` 无法完成 CUDA Kernel 生成。该项属于运行时依赖限制，不能填写虚假的 compile 加速结果；安装与当前 PyTorch/Windows/CUDA 组合兼容的后端后再运行稳态基准。若遇到内部模板 GBK 解码错误，可使用 `python -X utf8` 启动，但 UTF-8 模式不能替代缺失的 Triton。

# 本地合成闭环记录（2026-08-12）

以下只验证代码闭环，不代替真实语料消融实验：

- 阶段一至五全量测试：47 项通过，包含 KV Cache 与 compile 调用契约测试。
- 合成字符语料闭环：Tokenizer → Dataset → 两个 Micro-batch 梯度累积 → AdamW → Warmup/Cosine → Checkpoint 保存/恢复 → 验证 Loss → 普通/缓存生成，全部通过。
- 四次合成训练 Loss：`2.5841, 2.5694, 2.5784, 2.5193`；验证 Loss：`2.4582`。短序列存在波动，结论仅是训练链路可运行。
- CUDA KV Cache 合成基准：4 层、`d_model=64`、Prompt 48、生成 64、5 次测量；普通路径 `288.02 tokens/s`，缓存路径 `319.73 tokens/s`，加速 `1.110x`。
- CUDA AMP 冒烟：FP16 与 BF16 均成功完成一次参数更新；该记录只证明功能正确，不构成真实训练吞吐/显存结论。
- 参数统计示例（词表 65、配置文件结构）：总参数 `818,176`；Embedding `2.02%`、Attention `32.29%`、FFN `64.39%`、LayerNorm `0.28%`、LM Head `1.02%`。

Dropout、Heads、Layers、`d_model`、`block_size` 的真实 Loss、显存、tokens/s 和生成样例仍需要同一语料、相同训练预算的正式运行，不能从合成冒烟测试推断。
