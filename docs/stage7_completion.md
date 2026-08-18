# 阶段七：工程化与可复现实验封版

完成日期：2026-08-18

## 阶段目标

把“能运行的学习代码”提升为可安装、可测试、可复现、可审查、可由 CI 验收的公开
工程项目，为后续推理服务、RAG 和 Agent 学习提供稳定底座。

## 验收结果

| 验收项 | 状态 | 证据 |
|---|---|---|
| 统一 Python 与依赖 | 通过 | Python 3.11、`pyproject.toml`、`.python-version`、`uv.lock` |
| 锁文件安装 | 通过 | `uv sync --locked --extra dev --extra hf --extra viz` |
| 自动格式化 | 通过 | Ruff 0.16.3，46 个 Python 文件统一格式 |
| Lint | 通过 | `ruff check .` 零问题 |
| 静态类型 | 通过 | mypy 2.3.1，25 个源文件零问题 |
| 自动测试 | 通过 | 58 passed |
| 覆盖率门槛 | 通过 | 分支覆盖率 64.62%，CI 门槛 60% |
| GitHub CI | 通过 | Push / Pull Request 自动执行锁文件安装、格式、Lint、类型与测试 |
| 实验追踪 | 通过 | 配置、环境、Git 状态、JSONL 指标、日志、原子 Checkpoint |
| 真实语料基线 | 通过 | Tiny Shakespeare，409.6 万训练 Token |
| 单变量实验 | 通过 | FP32/FP16/BF16；固定 LR/Warmup/Cosine |
| 公开文档 | 通过 | README、复现命令、真实指标、限制与实验报告 |

## 最终推荐训练配方

当前硬件与短预算下推荐：

```text
precision: fp32
learning_rate: 3e-4 fixed
batch_size: 32
block_size: 64
embedding_dim: 128
num_heads: 4
num_layers: 4
max_steps: 2000
seed: 42
```

理由：固定 LR 的验证 Loss `1.6666`、测试 Loss `1.8800`，优于本阶段另外两种调度；
FP32 在当前 81.8 万参数小模型上的单次吞吐最高。显存受限时可选择 BF16，它把
PyTorch allocated 峰值显存降低约 17.29%，质量基本相当。

## 实验索引

- [`tiny_shakespeare_baseline.md`](tiny_shakespeare_baseline.md)：FP32 基线
- [`mixed_precision_comparison.md`](mixed_precision_comparison.md)：FP32 / FP16 / BF16
- [`learning_rate_schedule_comparison.md`](learning_rate_schedule_comparison.md)：固定 LR / Warmup / Cosine
- [`experiments.md`](experiments.md)：实验规范与历史记录

## 阶段边界

本阶段完成的是训练项目的工程化底座，不等于已经具备生产在线服务能力。当前尚未
包含 HTTP API、容器镜像、服务监控、RAG、工具调用或 Agent 编排；这些属于后续阶段。
明确边界比在简历中夸大“生产级”更可信。

## 下一阶段

阶段八先学习模型服务化：FastAPI 请求模型、生命周期管理、健康检查、结构化日志、
Docker、基础负载测试和错误处理。完成服务层后，再学习 Embedding、向量检索与 RAG，
最后进入工具调用和 Agent；暂不依赖 LangChain，以便先掌握底层机制。
