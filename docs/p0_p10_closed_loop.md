# MiniGPT P0–P10 整体闭环验收

> 2026-08-26 后续工程收敛已经完成请求级 Adapter 隔离、严格评测开关、服务拆分和双层覆盖率
> 门禁；最新工程验证为 263 passed、72.06%。详见
> [工程收敛与并发隔离说明](engineering_hardening.md)。下文保留 P0–P10 首次闭环时的原始数字。

- 作者：李林知
- 日期：2026-08-25
- 范围：治理、评测、路由、SFT/QLoRA、RAG、服务、文档、现代架构、多模态、私有领域

## 1. 最终结论

项目已形成一条从手写 Decoder-only Transformer 到真实 3B QLoRA Agent 的完整技术主线，
并继续扩展到可观测服务、真实 Word/PDF、现代架构开关、真实 OCR/ASR/TTS smoke 和隔离的
私有领域 Adapter 控制面。

整体达到“可审计研究工程”的预期，但没有达到“生产级通用多模态 Agent”。P0–P7 的模型
指标没有因 P8–P10 被重新包装：唯一 frozen 仍是任务 74%、RAG Recall@5 88.64%、multi-step
和 recovery 0%；P6 只在 dev 条件验收。P8 是固定 Batch 短预算架构实验，P9 是清晰单行图像
与合成语音 smoke，P10 没有训练真实领域 Adapter。

## 2. P0–P10 状态表

| 阶段 | 核心证据 | 验收状态 |
|---|---|---|
| P0 | 治理、资源边界、冻结基线 | 完成 |
| P1 | Dev/Frozen manifest、fingerprint、污染检查、CI | 完成 |
| P2 | 两阶段确定性 Router 与消融 | 完成 |
| P3 | 2,400 train + 300 validation Agent SFT 数据 | 完成 |
| P4 | 0.5B LoRA、3B NF4 QLoRA、重载检查 | 条件完成 |
| P5 | lexical/dense/hybrid + weighted RRF 对照 | 组件完成 |
| P6 | 计划门禁、恢复、引用、SSE/取消/指标 | 工程完成、模型条件验收 |
| P7 | 六个 Word/PDF 工具、回读和 4 页视觉 QA | 完成 |
| P8 | RoPE/RMSNorm/SwiGLU/SDPA/fused QKV/BPE/packing | 正式对照完成，扩大复验待执行 |
| P9 | OCR/ASR/TTS、上传、工具、服务、真实 smoke | 基础闭环完成 |
| P10 | 私有 Manifest、授权、数据门禁、PEFT 激活 | 治理完成、训练未执行 |

## 3. 新增实测结论

P8 在固定 307,200 Token 预算中，SDPA 同参数量吞吐 +21.40%，峰值显存约 -3.89MiB；
fused QKV 同参数量吞吐 +15.55%。组合现代架构因 SwiGLU 参数量增加和小序列 RoPE 开销，
吞吐反而 -26.13%。所有变体过拟合固定 Batch，验证 Loss 上升，所以没有精度提升结论。

BPE 在 100,000 字符上把 Token 数减少 48.656%，roundtrip 完整。后续已完成同一 BPE 数据
上的 2,000 步随机 Batch 对照和完整验证集 Early Stopping；多 seed、相同字符/FLOPs 预算
与更大语料复验仍待执行，详见 [现代架构与 Early Stopping 报告](modern_architecture_early_stopping_2026-08-26.md)。

P9 真实 OCR 对单行 `MINIGPT P9 TEST` 得到 CER 0%；Whisper tiny 对系统合成短英文得到
9.52% 字符错误率。热路径 OCR/ASR 分别约 75.56/74.09ms，冷启动分别约 13.92/6.53 秒；
模型缓存 383.99MiB，全部在 D 盘。

P10 标准数据、满足成年人和同意条件的合成治理数据通过；禁止记录、错误 Token、路径穿越、
隐私未审核和不合格成人 manifest 被拒绝。成人模板默认关闭且没有权重或语料。

## 4. 资源验收

本轮 P9 模型缓存约 383.99MiB；Python 新依赖主要为 pywin32 6.6MiB、sentencepiece 1.2MiB
及小型运行包。相对 40GB 上限余量充足，所有新模型、媒体、实验结果位于 D 盘项目范围。
未新增大型语料库，也没有重复下载 3B 基座。

## 5. 安全和隐私验收

- 文档、媒体和领域路径均使用显式根目录与 confinement。
- 上传校验扩展名、真实格式、大小和覆盖策略。
- TTS、文档创建等副作用工具需要确认。
- 私有 catalog 不返回路径和 Token hash，授权使用 SHA-256 与常数时间比较。
- 领域能力默认关闭，不修改公开默认模型。
- 成人领域需要 restricted、18+、同意、来源、许可证、隐私与禁止内容门禁。
- 静态关键词门禁不替代人工、法律或隐私审核，这一点在报告中明确保留。

## 6. 技术价值与证据边界

项目现在可以完整讲述模型数学、训练系统、SFT/QLoRA、Agent 编排、RAG、评测治理、服务、
文件交付、现代 kernel、模态级联和私有化控制。公开数字必须分别注明 frozen、dev、固定
Batch benchmark 或 smoke，不能把不同证据混成一个“准确率”。

仍需继续完成的生产化工作包括：多 seed、相同字符/FLOPs 预算和更大语料的现代训练复验、
困难 OCR/真实噪声 ASR 评测、真实 3B voice-to-voice p50/p95、按租户隔离 Adapter worker、
鉴权/审计日志、Prometheus 和真正 token-level streaming。2,000 步单 seed 对照和完整验证
集 Early Stopping 已单独记录，不替代这些更高等级的验证。

## 7. 最终验收门禁

最终工作树已完成以下门禁：

- `ruff format --check .`：143 个文件均已格式化。
- `ruff check .`：通过。
- `mypy`：75 个源文件无问题。
- Dev/Frozen split validation：100 条 Dev 与 150 条 Frozen 全部通过；无重复、训练污染或
  Dev/Frozen prompt overlap，Frozen fingerprint 保持不变。
- 全量测试：256 passed，1 个第三方 Starlette/httpx 弃用警告。
- 综合覆盖率：71.62%，高于 60% 门槛。

P0–P5 frozen 没有重跑；P8–P10 没有利用 frozen 结果进行选择，因此原冻结证据仍保持隔离。
