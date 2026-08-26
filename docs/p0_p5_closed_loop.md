# P0–P5 求职项目闭环审计

作者：李林知
日期：2026-08-25

## 我对“闭环”的定义

我没有把“写了代码、跑出一个高分样例”当作闭环。对这个项目，闭环必须包含：问题定义、
冻结基线、可复现评测、失败分层、最小可归因实验、数据治理、资源记录、候选选择、隐藏集
验收，以及根据未达标项排下一阶段优先级。

```text
P0 冻结事实与资源边界
  → P1 拆分指标并建立 Dev/Frozen
  → P2 用 Router 修复误调用和检索触发
  → P3 按失败类型构建可执行 Agent 轨迹
  → P4 用 0.5B 快速试错、3B QLoRA 形成候选
  → P5 对 lexical/dense/hybrid 做组件级选择
  → 唯一一次 Frozen Test 验收
  → 失败分层成为 P6 输入，不回头调 P0–P5 冻结结果
```

## 分阶段审计

| 阶段 | 我交付的证据 | 验收判断 | 没有完成或不能宣称的部分 |
|---|---|---|---|
| P0 治理 | 路线图、第一人称日志、基线版本、40GB 预算 | 通过 | 不是所有历史 dirty worktree 都由本轮产生 |
| P1 评测 | 100 dev、150 frozen、manifest、CI、失败标签、bootstrap | 通过 | 置信区间仍受样本规模限制 |
| P2 路由 | Router 抽象、工具过滤、free/heuristic 对照 | 通过 | 规则的 dev 静态 100% 不能当泛化成绩；引用白名单未完成 |
| P3 数据 | 2,700 条多轮轨迹、data card、去重/污染检查、role mask 测试 | 通过 | 以合成可验证轨迹为主，尚无大规模人工审核集 |
| P4 微调 | v1.0 失败复盘、0.5B v1.1、3B QLoRA、dev 与 frozen 推理 | 条件通过 | frozen 单工具 70.91% < 85%；多步骤/恢复未通过；QLoRA 严格 logits 重载未通过 |
| P5 RAG | lexical/dense/hybrid、BGE 小模型、Recall/MRR/延迟对照 | 通过 | 当前小语料无区分度，尚无长文档 chunk/rerank 困难集 |

## 从最初基线到最终候选

不同阶段报告并非都在相同 split、相同模型和相同系统上，因此我不把下表全部解释成严格
单变量提升；它展示的是项目演进轨迹。严格可归因的提升以 paired dev 报告为准。

| 版本 | Split | 任务成功率 | 不必要工具率 | RAG Recall@5 | Citation Recall |
|---|---|---:|---:|---:|---:|
| 0.5B + 课程 LoRA，自由调用 | dev 100 | 37.00% | 74.29% | 18.18% | 0.00% |
| 0.5B + 课程 LoRA，heuristic router | dev 100 | 65.00% | 0.00% | 45.45% | 0.00% |
| 0.5B + Agent v1.1 LoRA + router | dev 100 | 61.00% | 0.00% | 72.73% | 36.36% |
| 3B + Agent v1.1 QLoRA + router | dev 100 | 78.00% | 0.00% | 63.64% | 63.64% |
| 最终 3B 候选 | frozen 150 | 74.00% | 0.00% | 88.64% | 79.55% |

关键的严格对照是 0.5B 同模型 Router-only：任务成功率 37% → 65%，paired 95% CI
+19～+38pp。0.5B v1.1 SFT 相对 Router-only 的总体任务变化 -4pp，区间跨 0，所以我没有
声称它提升总体成功率。最终 frozen 与 dev 不是 paired 同题，不能对 78% → 74% 做显著性
推断，只能把它视为泛化验收。

## 冻结测试告诉我的真实水平

最终候选最强的是 direct（90%）、retrieval（100%）和 tool selection（88.89%）。检索触发
Precision 为 100%、Recall 为 93.18%，条件 Retriever Recall@5 为 95.12%，说明原来“检索
器不行”的模糊判断已被拆开并修正：当前主要 RAG 链路确实能工作。

最弱的是 multi-step（0/20）、tool arguments（14/25）和 recovery（0%）。具体表现包括
漏掉第二步调用、只调用一次、参数不匹配、工具失败后没有执行恢复策略。它们解释了为什么
总体 74% 不能被包装成完整生产 Agent，也直接给出下一阶段应先做什么。

## 资源闭环

| 新增稳定资产 | 实际占用 |
|---|---:|
| BGE small 中文 embedding 模型 | 183.35MiB |
| Agent v1.1 数据 | 3.22MiB |
| 全部 LoRA/QLoRA artifacts | 76.46MiB |
| 合计 | 约 263.03MiB |

这些资产都位于 D 盘项目目录，远低于 40GB 上限。3B 基座没有在本阶段重复下载，而是使用
机器上已有的 Hugging Face 缓存；我不会把缓存大小伪装成“本项目新增 0”。如果将来需要把
项目完全迁移到 D 盘，应另做缓存迁移和 hash 登记。

## 工程质量闭环

最终候选具备以下求职可展示证据：

- 自研而非黑盒框架包裹的 Router、Orchestrator、Tool Registry、Executor、Trace 与 Eval。
- 多轮 assistant-only SFT 数据编码、LoRA/QLoRA、NF4 和消费级 GPU 资源权衡。
- Dev/Frozen 分离、manifest fingerprint、训练污染检查、bootstrap CI 与逐 case 失败标签。
- Retriever 组件级 Recall/MRR/延迟对照，以及“dense 无收益就不默认启用”的工程判断。
- 失败实验、拒绝上线的 0.5B Adapter、量化重载限制和冻结集短板均被保留。

最终质量门禁全部通过：Ruff format、Ruff lint、mypy（63 个 source files，0 issue）、
Dev/Frozen split 校验、recorded Agent smoke 和全量 pytest。全量结果为 188 passed、覆盖率
68.45%（门槛 60%）；唯一警告来自 Starlette TestClient 的上游弃用提示。

## 是否符合我最初的预期

符合的部分：项目已经从教学代码升级为有研究问题、有真实模型、有 Agent 数据、有监督微调、
有 RAG、有隐藏集和有失败归因的求职项目。它足以支撑我在面试中解释从 Transformer 原理到
Agent 系统、QLoRA、RAG 与评测治理的完整链路。

不符合“完美结果”的部分：当前还不是成熟对话产品，没有 SSE 流式服务与前端观测；多步骤、
参数修复、恢复策略不达标；Word/PDF 工具属于 P7；图片和语音属于 P9；可选私有领域
Adapter 属于 P10，未进入当前公开主线。

## 我接下来怎样推进 P6

我不会根据已查看的 frozen v1 逐题调规则。下一阶段只使用 dev 和新增的 multi-step/recovery
训练与回归集，优先顺序是：

1. 把 multi-step 从“路由标签”升级为显式状态机，保存前一步工具结果并决定第二步调用。
2. 为参数 Schema 增加确定性校验和最多一次自动修复；单独统计 missing/wrong/type mismatch。
3. 实现 citation 白名单，最终回答只能引用本次 `search_docs` 实际返回的文档 ID。
4. 实现工具瞬时失败的有限重试和用户可解释恢复，禁止无限循环。
5. 加入 SSE、request ID、route/tool trace、TTFT/总延迟/token/tool latency 与取消。
6. 在 dev 达到预设回归门槛后形成新候选；冻结集治理仍保持“发布验收而非调参集”。

Word/PDF 放在 P7，以工具 Schema、显式工作目录、结构校验、渲染和视觉 QA 形成第二个可展示
闭环；图片和语音只在文本回归稳定后引入。这样项目的每次扩展都建立在已验收主线上，而不
会为了功能数量牺牲可解释性。
