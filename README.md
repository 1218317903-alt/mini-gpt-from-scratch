# mini-gpt-from-scratch

[![CI](https://github.com/1218317903-alt/mini-gpt-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/1218317903-alt/mini-gpt-from-scratch/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11.16-3776AB?logo=python&logoColor=white)
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
- [Container 启动说明](docs/container.md)
- [阶段八模型服务化验收](docs/stage8_completion.md)
- [阶段一真实聊天后端验收](docs/stage1_chat_backend.md)
- [阶段三工具调用与 Agent 验收](docs/stage2_agent_tools.md)
- [阶段二 Agent 评测集](docs/stage2_agent_eval.md)
- [阶段四 0.5B Agent 基线](docs/stage4_agent_baseline.md)
- [求职项目总路线图](docs/career_project_roadmap.md)
- [研究与工程日志](docs/research_log.md)
- [P0/P1 第一人称研究与验收报告](docs/p0_p1_completion.md)
- [P2–P5 第一人称完成报告](docs/p2_p5_completion.md)
- [P0–P5 求职项目闭环审计](docs/p0_p5_closed_loop.md)
- [P6/P7 第一人称研究与验收报告](docs/p6_p7_completion.md)
- [P0–P7 求职项目闭环审计](docs/p0_p7_closed_loop.md)
- [P8–P10 第一人称研究与验收报告](docs/p8_p10_completion.md)
- [P0–P10 求职项目闭环审计](docs/p0_p10_closed_loop.md)
- [3B QLoRA 训练与重载验收说明](docs/p4_3b_qlora_training_summary.md)
- [冻结测试集人工审核表](docs/frozen_eval_review_checklist.md)

### P0–P5 纯文本 Agent 冻结结果

最终候选为 Qwen2.5-3B-Instruct 4-bit + Agent v1.1 QLoRA + heuristic router + 默认词法
Retriever。在只运行一次的 150 条 `agent-frozen-test-v1` 上：

| 指标 | 结果 |
|---|---:|
| 任务成功率 | 74.00%（95% CI 67.33%～81.33%） |
| JSON 合法率 | 100.00% |
| 不必要工具调用率 | 0.00% |
| 端到端 RAG Recall@5 | 88.64% |
| Citation Precision / Recall | 92.11% / 79.55% |
| 单工具成功率 | 70.91% |
| 多步骤成功率 / 恢复率 | 0.00% / 0.00% |

150 条均正常结束，没有 runtime error。该结果证明训练、路由、工具、检索、引用与冻结评测
形成了完整闭环，同时也明确暴露多步骤、参数和恢复短板；仓库不把它包装成生产级“完美
Agent”。完整失败分析见 [P0–P5 闭环审计](docs/p0_p5_closed_loop.md)。

### P6/P7 可靠性与真实文件交付

P6 没有重跑 P0–P5 frozen v1，而是在同一 100 条 3B dev 上和保存的 P4 预测做 paired
comparison。只改变编排可靠性层后，任务成功率 78% → 79%（paired 95% CI 0～+3pp），
multi-step 13/15 → 14/15；单工具、参数和恢复没有改善，Citation Recall 63.64% →
59.09%。因此项目把它记录为“系统约束增强、模型效果弱提升且引用有回退”，不宣称显著
精度收益。

P7 提供六个受控 Word/PDF 工具。最终项目报告同时导出 DOCX/PDF，写后回读并把两种格式
的共 4 个页面逐页渲染检查；检查中实际修复了 PDF 中英文字体字距和列表 glyph 问题。完整
思路、失败过程与边界见 [P6/P7 验收报告](docs/p6_p7_completion.md)。

### P8–P10 现代架构、多模态与私有领域控制面

P8 在保持经典配置兼容的前提下加入 RoPE、RMSNorm、SwiGLU、PyTorch SDPA、fused QKV、
教学型 BPE 和样本 packing。固定 307,200 Token 的受控短预算中，SDPA 吞吐提高 21.40%，
fused QKV 提高 15.55%；组合现代配置反而下降 26.13%。所有变体都只是在同一固定 Batch 上
过拟合，验证 Loss 变差，因此这里只形成 kernel/架构结论，不宣称精度提高。BPE 在 100,000
字符样本上减少 48.656% Token，但尚未做同字符预算的正式 BPB 对照。

P9 完成受控上传、图片 OCR、WAV ASR、离线 TTS、Agent 工具与 HTTP 接口，并使用真实本地
模型完成单行印刷体图片和系统合成短英文 smoke。热路径 OCR/ASR 分别约 75.56/74.09ms，
模型缓存 383.99MiB，位于 D 盘；这不是复杂文档 OCR、噪声语音或联合多模态模型评测。

P10 完成独立领域 Manifest、数据治理门禁、Token 授权、路径隔离、私有 catalog 与 PEFT
Adapter 激活控制面。可选成人模板默认关闭，且没有下载相关语料或训练权重；本阶段只证明
治理和隔离边界，不代表领域效果。完整证据与失败记录见
[P8–P10 验收报告](docs/p8_p10_completion.md)。

## 项目能力

- 字符级 Tokenizer、未知字符策略、顺序 train/validation/test 划分
- `[B,T] → [B,T,V]` 的 Decoder-only Transformer 与 Causal LM Loss
- 独立 Q/K/V、多头因果注意力、Pre-Norm、Residual、FFN、可选权重共享
- AdamW、梯度裁剪、梯度累积、Warmup、Cosine、FP16/BF16
- 极小 Batch 过拟合检查、最佳/最新 Checkpoint、训练恢复和 RNG 状态
- Greedy、Temperature、Top-k、Top-p、固定 seed 和滑动上下文生成
- KV Cache、缓存一致性测试、速度/显存基准入口和 Attention 可视化
- Hugging Face 聊天模板、response-only labels、动态 Padding、PEFT LoRA
- 多轮 Agent SFT、全 assistant turn 监督、NF4 QLoRA 与训练污染检查
- 统一 `ChatModel` 接口：MiniGPT、原始 HF 模型和 LoRA Adapter 走同一套 `/chat`
- 三个安全工具、参数校验、超时/确认/重复调用保护、脱敏追踪与 `POST /agent/chat`
- direct/calculator/time/retrieval/multi-step 路由与按路由过滤工具 Schema
- multi-step 计划完成门禁、一次保守参数修复、只读有限重试、引用白名单和上下文预算
- Agent SSE 进度事件、run ID、协作式取消、并发上限、聚合指标与本地 `/demo`
- lexical/dense/hybrid 可插拔检索、BGE 中文 embedding 与 weighted RRF
- 受控 DOCX/PDF 创建、读取、摘要、报告导出、写后回读与逐页视觉 QA
- 可选 RoPE/RMSNorm/SwiGLU/SDPA/fused QKV、教学型 BPE 与序列 packing
- 受控图片/音频上传、真实本地 OCR/ASR、离线 TTS 和模态级联 Agent 工具
- 私有领域 Manifest、数据治理、授权 catalog 与按需 PEFT Adapter 激活控制面
- 独立于训练集的 Agent 评测集、真实模型 runner、失败分层和 Recall@5 校验
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

本地开发与 CI 固定 Python 3.11.16，项目不接受 Python 3.12/3.13。推荐安装
[uv](https://docs.astral.sh/uv/) 后使用锁文件创建环境：

```powershell
uv sync --locked --extra dev --extra hf --extra service --extra viz --extra documents --extra multimodal
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

启动可观测服务并显式启用文档、多模态和私有领域工作区：

```powershell
$env:MINIGPT_DOCUMENT_WORKSPACE = "D:\mini-gpt-from-scratch\output\agent-documents"
$env:MINIGPT_MULTIMODAL_WORKSPACE = "D:\mini-gpt-from-scratch\output\multimodal"
$env:MINIGPT_MULTIMODAL_MODEL_CACHE = "D:\mini-gpt-from-scratch\models\p9"
$env:MINIGPT_DOMAIN_ROOT = "D:\mini-gpt-from-scratch\configs\domains"
# 按需显式开启；未设置时不会加载对应模型。
$env:MINIGPT_ENABLE_OCR = "1"
$env:MINIGPT_ENABLE_ASR = "1"
$env:MINIGPT_ENABLE_TTS = "1"
uv run uvicorn minigpt_service.app:app --host 127.0.0.1 --port 8000
```

然后访问 `http://127.0.0.1:8000/demo`。是否加载 MiniGPT 或 Hugging Face 后端仍由现有服务
环境配置决定；工作区只有显式配置后才注册。可复现实验和治理 smoke：

```powershell
uv run python scripts/benchmark_p8_architecture.py
uv run python scripts/run_p9_multimodal_smoke.py
uv run python scripts/run_p10_governance_smoke.py
uv run python scripts/export_p0_p10_report.py
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
uv run python evals/run_agent_eval.py --check-only
uv run pytest -q --cov=minigpt --cov=hf_lora --cov=agent `
  --cov=minigpt_service --cov=evals --cov-report=term-missing `
  --cov-report=json:coverage.json
uv run python scripts/check_critical_coverage.py coverage.json
```

测试数量和覆盖率随功能迭代，以 CI 的最新结果为准；分支覆盖率低于 70% 时 CI 失败，关键
训练、编排、领域和服务模块还必须分别达到最低线。
Windows 用户也可运行 `scripts/check.ps1` 完成环境同步和全部检查。完整开发约定见
[CONTRIBUTING.md](CONTRIBUTING.md)。

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
configs/                  MiniGPT 与聊天后端配置
data/                     本地原始/处理数据（Git 忽略）
docs/                     原理、实验报告与学习路线
evals/                    Agent 评测集、指标和 runner
hf_lora/                  Hugging Face SFT + PEFT LoRA
scripts/                  数据、分析、可视化与消融入口
src/agent/                统一 ChatModel、工具与 Agent
src/minigpt/              手写 MiniGPT 包
src/minigpt_service/      API 工厂、领域运行时、多模态路由、取消、并发与指标
output/                   本地生成产物（Git 忽略，仅保留目录占位）
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
8. FastAPI 模型服务、容器启动与基础负载测试
9. Hugging Face 聊天后端、LoRA 恢复与 `/chat` 接口
10. 工具调用、工具执行器、Agent 循环与执行轨迹
11. P0–P5 冻结评测、3B QLoRA 主候选与 RAG v2
12. P6 可靠性门禁、SSE/取消/并发/指标与可观测 Demo
13. P7 六个受控 Word/PDF 工具、真实产物与逐页视觉 QA
14. P8 现代架构开关、BPE/packing 与受控短预算基准
15. P9 受控 OCR/ASR/TTS 多模态级联与真实本地 smoke
16. P10 私有领域治理、授权与请求级 PEFT Adapter 隔离

详细路线见 [docs/learning_plan.md](docs/learning_plan.md)。P0–P10 已完成一次端到端求职项目
闭环；其中 P6 的模型效果、P8 的长训练、P9 的真实复杂场景和 P10 的领域训练均保留为条件
验收或后续工作，阶段编号不等于生产成熟度。

从 2026-08-25 起，后续工作按[求职项目总路线图](docs/career_project_roadmap.md)推进，
所有训练、解码、RAG 和服务改动统一记入[研究与工程日志](docs/research_log.md)。路线优先
完成纯文本 Agent、监督微调、冻结评测、RAG、可观测服务和 Word/PDF 工具；图像、语音
以及私有领域 Adapter 只在文本主线稳定后扩展。新增模型、数据、缓存和实验产物统一放在
D 盘，合计硬上限 40GB。

## 快速审阅路径

- 先看本页的当前结果、能力边界和复现命令。
- 再看 [P0–P10 闭环审计](docs/p0_p10_closed_loop.md)了解证据等级与失败项。
- 查看 [工程收敛说明](docs/engineering_hardening.md)了解 CI、服务拆分和并发隔离策略。
- 最后运行 `scripts/check.ps1`，复现格式、类型、评测清单、测试及覆盖率门禁。

## 当前限制

- 字符级模型和 409.6 万 Token 预算只能学习局部文本结构，不具备通用语言理解能力。
- 没有 EOS Token，生成由 `max_new_tokens` 停止。
- KV Cache 第一版不支持超过 `block_size` 后直接滑动，需要重新 Prefill。
- 服务层已提供 `/generate`、`/chat`、同步与 SSE `/agent/chat`；SSE 当前是阶段级进度事件，
  还不是真正逐 Token streaming，协作式取消也不能抢占进行中的 HF generate。
- 还没有长期记忆、用户鉴权、Prometheus 持久指标或跨进程 trace 聚合。
- 已支持 4-bit QLoRA，但没有 Flash Attention 或分布式训练。
- P0–P5 冻结测试中 multi-step 与 recovery 均为 0%，参数准确率 57.53%；P6 只在 dev 做
  可靠性回归，没有重跑 frozen，因此尚不适合生产使用。
- P7 支持结构化报告，不支持任意 Office 模板编辑或修订批注；P9 OCR 也不等于扫描版
  Office/PDF 的版面理解与结构恢复。
- P8 尚未完成 2,000 步随机 Batch 的同预算正式实验；短预算结果不能作为质量提升证据。
- P9 当前是 OCR/ASR/TTS 级联，不是端到端视觉语言或语音到语音模型；真实噪声鲁棒性未知。
- P10 尚未训练真实私有领域 Adapter；单进程请求已经隔离，但仍没有多进程租户 worker、
  外部鉴权或合规审计服务。
- Windows CUDA 环境缺少兼容 Triton，当前 `torch.compile` 无法形成有效基准。
