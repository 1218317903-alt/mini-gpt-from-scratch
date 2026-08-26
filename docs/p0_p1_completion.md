# P0/P1：从“有功能”到“有证据”的评测底座

完成日期：2026-08-25

## 我为什么先做 P0/P1

我最初很容易被“再加一个 RAG、再换一个模型、再调高一点 Top-k”吸引，因为这些改动在
Demo 中很直观。但我复盘现有结果后发现：如果我不能区分路由失败、检索失败和回答失败，
即使最终分数变化，我也无法解释提升究竟来自哪里。这样的项目能展示功能，却不能证明我
具备模型实验和 Agent 系统诊断能力。

所以我先暂停下载新语料和新模型，把问题改写成：

> 我能否先建立一套不会被自己反复调参污染、能够追踪每个版本、还能说明不确定性的评测
> 底座？

P0 解决“我如何管理研究证据”，P1 解决“我如何判断一个 Agent 版本真的更好”。

## P0：我建立了什么研究纪律

### 1. 我先冻结已经知道的事实

我没有重新包装历史结果，而是把以下事实作为基线写入研究日志：

- Tiny Shakespeare FP32、混合精度和学习率对照。
- 0.5B + 课程 LoRA 的 100 条 Agent 结果。
- 3B 4-bit Base 的同集合结果。
- RTX 4060 Laptop 8GB 显存和 D 盘资源边界。

我把“实测结果”“目标值”“尚未运行的计划”严格分开。研究项目最容易失去可信度
的地方，就是把计划中的目标表写得像真实成绩。

### 2. 我为实验和资源建立编号

我把实验分为 BASE、EVAL、ROUTE、DATA、SFT、RAG、GEN、SERV、DOC、ARCH 和 MM。
以后我能从一个报告 ID 反查：配置、数据版本、Git 状态、命令、指标、显存、耗时和原始
结果。

我还把新增模型、数据、缓存和产物总量限制在 40GB，日常目标 32GB。我的考虑不是磁盘
现在是否够用，而是避免 Hugging Face 缓存、量化模型和重复 checkpoint 在多次实验后失控。

### 3. 我纠正了一个采样参数误区

我原本可能会写“提高 Top-k 后精度从 X 变为 Y”。重新梳理后，我把这个说法改掉了：
Top-k 控制候选 Token 集，主要改变生成的随机性、多样性和错误风险，不是训练精度旋钮。

现在我的实验口径是：工具和确定性任务优先 greedy；开放式创作才比较 Top-k、Top-p 和
Temperature。即使某个 Top-k 在评测上更高，我也只把结论限定为“当前模型、当前任务和
当前采样预算下的观察”。

### 4. P0 产物

- `docs/research_roadmap.md`：P0～P10 总路线。
- `docs/research_log.md`：第一人称研究日志、实验模板和现有基线。
- README：公开路线、资源上限和研究记录入口。

## P1：我如何重新设计评测

### 1. 我没有继续把旧 100 条叫作隐藏测试集

现有 100 条用例的失败列表已经被我查看，而且这些失败会指导 Prompt、路由和 SFT 数据
设计。如果我继续一边看这些失败一边调优，再把它们叫作“冻结测试集”，最终分数一定会
被高估。

所以我做了一个对公开评测更严谨的决定：

- 原 100 条重新定义为 `agent-dev-v1`，用于错误分析和开发。
- 新建 150 条 `agent-frozen-test-v1-candidate`，只用于候选版本发布验收。
- 冻结集问题不能用于 Prompt few-shot、SFT、DPO 或路由规则设计。

### 2. 我使用 manifest 和 SHA-256 防止静默修改

每个 split 都有 `manifest.json`，包含：

- split 名称和版本。
- 用途和 frozen 状态。
- 各类别预期数量。
- 每个 JSONL 文件的 SHA-256。
- 整个 split 的指纹。
- 人工审核状态。

当前指纹：

```text
dev:
b61e5a51213dc377df7c8ed16e1507db81df800f74ad87eb662f37d90b45a1ee

frozen-test candidate:
06fde3d735893adcbc37f52da6c48e6573e5366b3289a5459b228c6918790444
```

如果以后有人直接改动一条问题或 gold answer 而没有明确刷新 manifest，CI 会因为 hash
不一致而失败。这样“测试集变简单”不会悄悄发生。

### 3. 我把 RAG 的一次失败拆成三个阶段

旧版 `rag_recall_at_5` 只能告诉我端到端是否成功。我新增了：

- 检索触发 Precision/Recall：模型是否知道该不该搜索。
- 条件 Retriever Recall@1/3/5：触发后 Retriever 是否找到 gold。
- 端到端 Recall@1/3/5：把未触发也按失败计算。
- Citation Precision/Recall：最终引用是否正确、是否完整。
- Unsupported Answer Rate：需要依据的回答中，有多少没有 gold 引用支持。

重算历史预测后，3B 的端到端 Recall@5 是 50%，但条件 Recall@5 是 91.67%。这改变了我
的下一步判断：当前首先要修的是 route decision，而不是立即下载更大的 embedding 模型。

### 4. 我为什么增加 bootstrap 置信区间

100 条用例上的 52% 只是一个点估计。如果换一批相似问题，真实结果可能不同。因此我对
主要比例指标进行固定 seed 的 case bootstrap，默认重采样 1,000 次，报告 95% 区间。

我还为同一批 case 的模型对比实现了 paired bootstrap。因为 0.5B 和 3B 回答的是相同
问题，成对重采样比把两边当作独立样本更合适。

0.5B → 3B 的部分结果：

| 指标 | 点估计变化 | 95% paired bootstrap 区间 | 我的解释 |
|---|---:|---:|---|
| 最终任务成功率 | +15 个百分点 | +6 ～ +25 | 本集合上有稳定正向证据 |
| 单工具成功率 | +14 个百分点 | +3 ～ +26 | 本集合上有稳定正向证据 |
| 不必要工具率 | -23 个百分点 | -45 ～ 0 | 明显改善，但区间触及 0，继续收集证据 |
| 端到端 RAG Recall@5 | +32 个百分点 | +8 ～ +55 | 本集合上有稳定正向证据 |
| 条件 Retriever Recall@5 | +42 个百分点 | -1 ～ +88 | 点估计高，但样本少、区间宽 |
| Citation Recall | +41 个百分点 | +21 ～ +63 | 本集合上有稳定正向证据 |

这里我没有使用“所有指标都显著提升”这种笼统说法。条件 Recall 的点估计虽然从 50% 到
91.67%，但 paired 区间仍轻微跨过 0，说明触发后的检索样本量还不够，不能过度外推。

### 5. 我增加了逐 case 回归差异

只看总分会掩盖“修好 19 条、同时弄坏 4 条”。现在比较报告会输出：

- improved case IDs
- regressed case IDs
- unchanged pass/fail
- 每种失败标签的 resolved/introduced/persistent
- 全部指标 delta
- paired bootstrap delta 区间

0.5B → 3B 的当前逐题变化为：

```text
improved:       19
regressed:       4
unchanged pass: 33
unchanged fail: 44
```

下一阶段修改 Prompt 或路由时，我会先检查那 4 条退化，而不是只宣传总成功率提高。

### 6. 我如何处理 CI 中的真实模型问题

我没有让 GitHub Actions 每次下载 0.5B 权重。这样会增加网络不确定性、CI 时间和缓存成本，
而且不能证明 GPU 上的真实推理性能。

我采用两层策略：

- CI 回放 5 条真实保存的 0.5B 历史预测，验证解析、评分、置信区间和失败分层链路。
- 本地 3B 全量运行仍使用真实模型和 GPU，结果写入独立报告。

因此我把 CI 能力准确称为“recorded 0.5B offline smoke replay”，不称为“CI 每次运行真实
0.5B 推理”。这个边界会让项目描述更可信。

## P1 资产与机器验收

| 项目 | 结果 |
|---|---|
| Dev 用例 | 100 条 |
| Frozen-test candidate | 150 条 |
| 跨 split 重复 ID | 0 |
| 跨 split 精确问题重复 | 0 |
| 相似度 ≥ 0.92 的近重复 | 0 |
| 与现有 SFT 精确重叠 | 0 |
| gold 文档缺失 | 0 |
| 词法检索 gold Recall 校验失败 | 0 |
| Manifest hash 错误 | 0 |
| 0.5B recorded smoke | 5 条，可在 CI 离线回放 |
| Bootstrap | 固定 seed，1,000 次 |
| 逐 case 回归 | 已实现 |

最终工程验证：Ruff format、Ruff lint、mypy 全部通过；全量测试 **173 passed**，总覆盖率
**69.84%**，高于 60% CI 门槛。保留一条来自 FastAPI TestClient/httpx 依赖的 Starlette
弃用警告，与 P0/P1 评测改动无关。

机器报告：

- `docs/p1_split_validation.json`
- `docs/eval_v2_qwen25_0_5b_lora.json`
- `docs/eval_v2_qwen25_3b_4bit.json`
- `docs/p1_05b_vs_3b_regression.json`

## 人工 Release Gate 已完成

150 条 frozen-test candidate 已通过 Schema、数量、hash、污染、gold 文档和词法 Recall
机器校验。项目所有者李林知于 2026-08-25 完成人工 Release Gate，并将其签发为
`agent-frozen-test-v1`。最终 fingerprint 为
`06fde3d735893adcbc37f52da6c48e6573e5366b3289a5459b228c6918790444`。

本次签发只改变 manifest 的版本和审核元数据，没有修改 JSONL 内容。后续任何测试题改动
都必须提升版本并重新审核，不能沿用 v1 的正式成绩。

## 这一阶段的核心方法

这一阶段不只报告评测数据数量，更重要的是说明评测治理方法：

> 我发现原来的 Agent 指标把路由、检索和引用混在一起，而且旧测试集已经参与过错误分析，
> 所以我先把它降级为 dev，再建立带 SHA-256 指纹的 150 条冻结候选集。我为主要比例指标
> 加入 case bootstrap，为同题模型对比加入 paired bootstrap，同时输出逐 case 回归。这个
> 改动让我确认 3B 的主要 RAG 瓶颈在检索触发，而不是 Retriever 排序，于是我把下一阶段
> 从“下载更大向量模型”改成了“先做显式路由对照”。

这段过程串联了数据污染、统计不确定性、Agent 故障分层、CI 成本和严谨实验口径。
