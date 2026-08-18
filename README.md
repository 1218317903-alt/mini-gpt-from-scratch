# mini-gpt-from-scratch

[![CI](https://github.com/1218317903-alt/mini-gpt-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/1218317903-alt/mini-gpt-from-scratch/actions/workflows/ci.yml)

从零实现一个字符级 Decoder-only MiniGPT，并将底层知识迁移到真实开源模型，目前覆盖阶段一到阶段六：

1. 数据流与字符级 Tokenizer
2. Causal Self-Attention 与 Decoder-only MiniGPT
3. 训练、Loss、验证、Checkpoint
4. 自回归文本生成与基础采样
5. 参数分析、Attention 可视化、消融实验与训练基础优化
6. Hugging Face 生态、SFT、PEFT LoRA、Adapter 保存与恢复

阶段六位于独立目录 [`hf_lora/`](hf_lora/README.md)，不会改写前五阶段的 MiniGPT。它使用小型真实预训练 Causal LM，将字符级数据流、Causal LM Loss、生成和训练知识迁移到聊天模板、response-only SFT 与 LoRA。

本项目用于教学和实验。阶段五已包含可选 `torch.compile`、KV Cache 与缓存一致性/速度测试；
首个真实语料 FP32 基线已经完成，详见
[`docs/tiny_shakespeare_baseline.md`](docs/tiny_shakespeare_baseline.md)。后续消融仍需按
相同数据与训练预算逐项运行。

## 环境

项目统一使用 Python 3.11 和根目录 `pyproject.toml` 管理依赖。推荐使用
[`uv`](https://docs.astral.sh/uv/) 创建虚拟环境、安装依赖和运行命令：

```powershell
uv sync --extra dev
```

`.python-version` 会让 `uv` 选择 Python 3.11。Windows 使用 NVIDIA GPU 时，
项目会从 PyTorch 官方 CUDA 12.6 索引安装 `torch`；其他平台使用 PyPI 默认构建。
安装后可检查环境：

```powershell
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

需要 Hugging Face LoRA 和 Attention 可视化时安装完整开发环境：

```powershell
uv sync --extra dev --extra hf --extra viz
```

所有命令均通过项目虚拟环境运行，无需手动设置 `PYTHONPATH`：

```powershell
uv run python train.py --config configs/tiny_shakespeare.yaml
```

## 阶段一：数据准备

下载 Tiny Shakespeare：

```powershell
python scripts/download_data.py
```

划分 train/validation/test，并生成 Tokenizer：

```powershell
python scripts/prepare_data.py
```

检查 Dataset 与 DataLoader：

```powershell
python scripts/inspect_batch.py
```

字符级 Tokenizer 的词表包含 `<unk>` 和训练文本中出现的字符。未知字符默认报错；在验证集和测试集上可以显式使用 `allow_unknown=True` 映射到 `<unk>`。

## 阶段二：模型

模型计算主线：

```text
[B,T]
  → Token Embedding [B,T,C]
  → Position Embedding [B,T,C]
  → Q/K/V [B,T,C]
  → Split Heads [B,H,T,D]
  → Attention Scores [B,H,T,T]
  → Causal Mask
  → Merge Heads [B,T,C]
  → FFN / Residual / LayerNorm
  → Language Model Head [B,T,V]
```

## 阶段三：训练与评估

运行训练入口：

```powershell
python train.py --config configs/tiny_shakespeare.yaml
```

训练默认会进行极小 Batch 过拟合检查，并保存：

```text
checkpoints/best.pt
checkpoints/latest.pt
```

每次训练还会自动创建 `runs/<run-id>/`，记录实际生效配置、软硬件环境、
Git 状态、结构化指标和文本日志：

```text
runs/<run-id>/
├── config.yaml
├── environment.json
├── metrics.jsonl
└── train.log
```

需要把一次正式实验的全部产物放在同一目录时，可以同时指定运行目录与
Checkpoint 目录：

```powershell
uv run python train.py `
  --config configs/tiny_shakespeare.yaml `
  --run-dir runs/baseline `
  --checkpoint-dir runs/baseline/checkpoints
```

已有 `config.yaml` 或 `environment.json` 的运行目录不会被新实验静默覆盖；
只有配合 `--resume` 时才允许继续写入，并额外保存本次恢复的环境快照。

跳过过拟合检查：

```powershell
python train.py --skip-overfit
```

恢复训练：

```powershell
uv run python train.py `
  --resume checkpoints/latest.pt `
  --run-dir runs/resumed-training
```

指定 Checkpoint 输出目录，便于实验隔离：

```powershell
python train.py --checkpoint-dir checkpoints/experiment-a
```

评估验证集或测试集：

```powershell
python evaluate.py --checkpoint checkpoints/best.pt --split val
python evaluate.py --checkpoint checkpoints/best.pt --split test
```

Checkpoint 保存模型权重、训练进度、配置快照、Tokenizer 信息和随机状态。加载时会检查当前 Tokenizer 是否与 Checkpoint 一致。

## 阶段四：文本生成

默认命令使用随机采样：

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

Greedy Decoding：

```powershell
python generate.py `
  --checkpoint checkpoints/best.pt `
  --prompt "ROMEO:" `
  --max-new-tokens 300 `
  --greedy
```

生成流程是：

```text
Prompt 编码
  → 最近 block_size 上下文
  → 模型前向传播
  → 取最后位置 logits
  → Temperature
  → Top-k
  → Top-p
  → Softmax / Greedy / Sampling
  → 追加一个 Token
  → 重复 max_new_tokens 次
  → Tokenizer 解码
```

当前字符级 Tokenizer 没有 BOS/EOS Token，因此空 Prompt 会明确报错，生成只由 `max_new_tokens` 控制停止。超过 `block_size` 时，模型只使用最近上下文，但返回结果仍保留完整序列。

## 阶段五：分析与训练优化

参数分类统计：

```powershell
python scripts/count_parameters.py
```

Tokenizer 尚未生成时，可以显式给出教学用词表大小：

```powershell
python scripts/count_parameters.py --vocab-size 65
```

Attention 可视化：

```powershell
python scripts/visualize_attention.py `
  --checkpoint checkpoints/best.pt `
  --prompt "ROMEO:" `
  --layer 0 `
  --head 0 `
  --output artifacts/attention-layer0-head0.png
```

梯度累积、Warmup、Cosine 与混合精度示例：

```powershell
python train.py `
  --gradient-accumulation-steps 4 `
  --warmup-steps 10 `
  --cosine-decay `
  --min-learning-rate 0.00003 `
  --precision fp16
```

这条组合命令用于展示接口。正式实验必须按单变量原则分别与 FP32 固定学习率基线比较。

## 测试

```powershell
uv run pytest -q
```

CI 会同时统计 `minigpt` 与 `hf_lora` 的分支覆盖率，并以 60% 作为当前防回退
基线。查看本地覆盖率明细：

```powershell
uv run pytest -q --cov=minigpt --cov=hf_lora --cov-report=term-missing
```

测试覆盖：

- 数据集错位和 Batch 形状
- Tokenizer 编解码与未知字符
- Attention、Causal Mask、Batch 独立性
- MiniGPT 输出形状、反向传播和权重共享
- 训练一步、验证 Loss、Checkpoint 往返
- Logits、Softmax、Greedy、随机采样
- Temperature、Top-k、Top-p 参数校验
- 固定种子复现
- `eval()` / `no_grad()`
- `block_size` 截断、空 Prompt、Token 解码
- 训练 → Checkpoint → 评估 → CLI 生成闭环

## 当前限制

- 字符级建模的语义能力和生成质量有限。
- 没有 EOS Token，不能按句子自然停止。
- KV Cache 第一版不支持超过 `block_size` 后直接滑动；超长缓存生成需要重新 Prefill。
- 没有 Beam Search、Speculative Decoding、Flash Attention 或量化。
- `torch.compile` 依赖当前平台可用的 Inductor 后端；本机 Windows CUDA 环境缺少可用 Triton。

## 阶段五推理与编译开关

缓存生成：

```powershell
python generate.py --checkpoint checkpoints/best.pt --prompt "To be" --max-new-tokens 64 --greedy --kv-cache
```

缓存速度对比：

```powershell
python generate.py --checkpoint checkpoints/best.pt --prompt "To be" --max-new-tokens 64 --greedy --benchmark-kv-cache
```

可选编译训练：

```powershell
python train.py --skip-overfit --compile --compile-mode default
```

KV Cache 第一版要求 Prompt 与新生成 Token 的总长度不超过 `block_size`。普通生成仍支持滑动上下文；二者保留为独立路径，以便一致性和速度对比。
