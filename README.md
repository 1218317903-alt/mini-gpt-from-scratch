# mini-gpt-from-scratch

[![CI](https://github.com/1218317903-alt/mini-gpt-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/1218317903-alt/mini-gpt-from-scratch/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)

一个从零实现、经过测试并具备可复现实验链路的 Decoder-only MiniGPT 项目。项目先用
PyTorch 手写字符级 Transformer、训练与生成，再把同一套 Causal LM 知识迁移到真实
Hugging Face 模型的 response-only SFT 与 PEFT LoRA。

这不是对 nanoGPT 的文件级复制，也不把一次实验包装成“大模型训练”。项目重点是展示
对数据流、Attention、训练状态、推理缓存、实验设计和 Python 工程质量的完整理解。

## 当前结果

Tiny Shakespeare 正式基线固定 818,432 个参数、seed 42、2,000 步和 4,096,000 个
训练 Token，在 RTX 4060 Laptop GPU 上得到：

| 实验 | 最佳验证 Loss | 测试 Loss | 关键观察 |
|---|---:|---:|---|
| FP32 固定 LR | **1.6666** | **1.8800** | 当前推荐默认方案 |
| FP16 固定 LR | 1.6666 | 1.8800 | 显存 -17.29%，小模型吞吐未提升 |
| BF16 固定 LR | 1.6660 | 1.8813 | 与 FP16 同显存，数值路径更简单 |
| FP32 + Warmup | 1.6804 | 1.8878 | 2,000 步短预算下略差 |
| FP32 + Warmup + Cosine | 1.7705 | 1.9438 | 后期学习率下降过早，训练不足 |

这些是单 seed、本机实验；仓库明确保留这一限制，不把结果外推到其他模型和硬件。

- [FP32 基线报告](docs/tiny_shakespeare_baseline.md)
- [FP32 / FP16 / BF16 对照](docs/mixed_precision_comparison.md)
- [固定 LR / Warmup / Cosine 对照](docs/learning_rate_schedule_comparison.md)
- [阶段七工程化验收](docs/stage7_completion.md)

## 项目能力

- 字符级 Tokenizer、未知字符策略、顺序 train/validation/test 划分
- `[B,T] → [B,T,V]` 的 Decoder-only Transformer 与 Causal LM Loss
- 独立 Q/K/V、多头因果注意力、Pre-Norm、Residual、FFN、可选权重共享
- AdamW、梯度裁剪、梯度累积、Warmup、Cosine、FP16/BF16
- 极小 Batch 过拟合检查、最佳/最新 Checkpoint、训练恢复和 RNG 状态
- Greedy、Temperature、Top-k、Top-p、固定 seed 和滑动上下文生成
- KV Cache、缓存一致性测试、速度/显存基准入口和 Attention 可视化
- Hugging Face 聊天模板、response-only labels、动态 Padding、PEFT LoRA
- 配置快照、软硬件环境、Git 状态、JSONL 指标、日志和原子 Checkpoint
- uv 锁文件、Ruff、mypy、pytest、分支覆盖率和 GitHub Actions

## 模型主线

```text
Token IDs [B,T]
  → Token Embedding + Position Embedding [B,T,C]
  → N × Pre-Norm Transformer Block
      → Q/K/V [B,H,T,D]
      → Scaled Dot-Product + Causal Mask
      → Multi-Head Merge + Residual
      → LayerNorm + FFN + Residual
  → Final LayerNorm
  → LM Head [B,T,V]
  → Shifted Cross Entropy
```

训练时所有位置并行预测下一个 Token；生成时从最后一个位置逐 Token 解码。KV Cache
路径复用历史 K/V，普通路径保留滑动上下文，便于做正确性与性能对照。

## 快速开始

要求 Python 3.11。推荐安装 [uv](https://docs.astral.sh/uv/) 后使用锁文件创建环境：

```powershell
uv sync --locked --extra dev --extra hf --extra viz
```

Windows 会从 PyTorch 官方 CUDA 12.6 索引安装 `torch`；其他平台使用 PyPI 默认构建。

准备 Tiny Shakespeare：

```powershell
uv run python scripts/download_data.py
uv run python scripts/prepare_data.py
uv run python scripts/inspect_batch.py
```

运行正式 FP32 配方：

```powershell
uv run python train.py `
  --config configs/tiny_shakespeare_baseline.yaml `
  --run-dir runs/my-baseline `
  --checkpoint-dir runs/my-baseline/checkpoints `
  --precision fp32
```

评估与生成：

```powershell
uv run python evaluate.py `
  --config configs/tiny_shakespeare_baseline.yaml `
  --checkpoint runs/my-baseline/checkpoints/best.pt `
  --split test

uv run python generate.py `
  --config configs/tiny_shakespeare_baseline.yaml `
  --checkpoint runs/my-baseline/checkpoints/best.pt `
  --prompt "ROMEO:" `
  --max-new-tokens 300 `
  --temperature 0.8 `
  --top-k 40 `
  --top-p 0.95 `
  --seed 42
```

## 可复现实验

每次训练创建独立运行目录：

```text
runs/<run-name>/
├── config.yaml          # YAML + CLI 覆盖后的最终配置
├── environment.json     # Python / PyTorch / CUDA / GPU / Git
├── metrics.jsonl        # 逐 step 指标和 Checkpoint 事件
├── train.log            # 带时间戳日志
└── checkpoints/
    ├── best.pt
    └── latest.pt
```

已有实验目录不会被静默覆盖。Checkpoint 使用同目录临时文件和原子替换，并保存模型、
Optimizer、Scheduler、GradScaler、Tokenizer、随机状态和全局步数。

恢复训练：

```powershell
uv run python train.py `
  --config configs/tiny_shakespeare_baseline.yaml `
  --resume runs/my-baseline/checkpoints/latest.pt `
  --run-dir runs/my-baseline `
  --checkpoint-dir runs/my-baseline/checkpoints
```

## 工程质量

提交前运行与 CI 相同的检查：

```powershell
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest -q --cov=minigpt --cov=hf_lora --cov-report=term-missing
```

当前结果为 58 项测试通过、分支覆盖率 64.62%；覆盖率低于 60% 时 CI 失败。完整开发
约定见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## Hugging Face LoRA

[`hf_lora/`](hf_lora/README.md) 是独立阶段，不改写手写 MiniGPT。它使用真实小型
Causal LM 展示：

- Chat Template 与 response-only SFT labels
- Token 加权梯度累积与验证 Loss
- LoRA 目标模块、参数量推导和真实可训练参数统计
- Adapter 保存、全新基座恢复与保存前后 logits 一致性
- 固定 Prompt 的训练前、训练后、重新加载后对比

## 目录结构

```text
configs/                  MiniGPT 数据、模型与训练配置
data/                     本地原始/处理数据（Git 忽略）
docs/                     原理、实验报告与学习路线
hf_lora/                  Hugging Face SFT + PEFT LoRA
scripts/                  数据、分析、可视化与消融入口
src/minigpt/              手写 MiniGPT 包
tests/                    单元、集成与端到端测试
train.py                  训练与实验追踪入口
evaluate.py               独立验证/测试入口
generate.py               生成与 KV Cache 基准入口
pyproject.toml / uv.lock  环境、工具配置与依赖锁
```

## 已完成阶段

1. 数据流与字符级 Tokenizer
2. Decoder-only MiniGPT
3. 训练、评估与 Checkpoint
4. 自回归生成与采样
5. 分析、KV Cache 与训练优化
6. Hugging Face SFT 与 PEFT LoRA
7. 工程化、CI、实验追踪与真实单变量实验

详细路线见 [docs/learning_plan.md](docs/learning_plan.md)。下一阶段是模型服务化，然后再
进入 RAG 与 Agent；先掌握底层接口和检索机制，再考虑 LangChain 等编排框架。

## 当前限制

- 字符级模型和 409.6 万 Token 预算只能学习局部文本结构，不具备通用语言理解能力。
- 没有 EOS Token，生成由 `max_new_tokens` 停止。
- KV Cache 第一版不支持超过 `block_size` 后直接滑动，需要重新 Prefill。
- 没有 Flash Attention、量化、分布式训练或在线推理服务。
- Windows CUDA 环境缺少兼容 Triton，当前 `torch.compile` 无法形成有效基准。
