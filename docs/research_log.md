# MiniGPT 研究与工程日志

作者：李林知

更新日期：2026-08-25

## 使用原则

本日志记录项目从教学实现扩展为完整研究工程的过程。它不是只展示成功结果的宣传页。
以下内容必须保留：

- 基线和环境。
- 假设以及为什么提出该假设。
- 唯一主要变量和受控变量。
- 训练、评测和资源预算。
- 原始指标所在文件。
- 失败、无提升和被推翻的结论。
- 下一步由证据而不是主观感觉决定。

任何 `xxx`、`TBD` 或目标阈值都不是实测结果，不得写入 README 的“当前结果”或公开结论。

从 P0/P1 起，我统一使用第一人称记录研究过程。每条正式记录不仅写“我做了什么”，还要
写清楚：我看到了什么证据、我当时有哪些备选方案、为什么选择当前方案、什么结果推翻或
修正了原判断，以及我下一步准备如何验证。这样日志可以直接作为研究决策的过程证据，
而不是流水账。

## 实验编号

```text
BASE-xxx  基线和复验
EVAL-xxx  评测口径、用例和评分器
ROUTE-xxx Prompt、工具描述和路由
DATA-xxx  数据构建、清洗和配比
SFT-xxx   SFT/LoRA/QLoRA
RAG-xxx   检索、切分、Embedding 和 Rerank
GEN-xxx   Temperature/Top-k/Top-p/停止策略
SERV-xxx  服务延迟、并发、稳定性
DOC-xxx   Word/PDF 工具和视觉验收
ARCH-xxx  从零模型架构和训练优化
MM-xxx    图像/语音多模态
```

实验目录建议使用相同编号，例如 `runs/SFT-003-qwen05b-agent-lora/`。

## 当前冻结事实

### BASE-001：Tiny Shakespeare FP32 正式基线

- 状态：`completed`
- 模型：字符级 MiniGPT，818,432 参数。
- seed：42。
- 预算：2,000 optimizer steps，4,096,000 训练 Token。
- 最佳验证 Loss：1.6666。
- 测试 Loss：1.8800。
- 当前结论：在该单 seed、短训练预算和本机硬件下，FP32 固定学习率是已有对照中的推荐
  配方；不能外推到更大模型和其他设备。
- 证据：`docs/tiny_shakespeare_baseline.md`。

### BASE-002：混合精度对照

- 状态：`completed`
- FP32：验证 1.6666，测试 1.8800。
- FP16：验证 1.6666，测试 1.8800。
- BF16：验证 1.6660，测试 1.8813。
- 观察：FP16/BF16 的 allocated 峰值显存约降低 17.29%，但该 81.8 万参数模型没有获得
  训练吞吐提升。
- 限制：单机、单 seed、小模型；不能得出“混合精度通常不会加速”的结论。
- 证据：`docs/mixed_precision_comparison.md`。

### BASE-003：0.5B LoRA Agent 基线

- 状态：`completed`
- 模型：Qwen2.5-0.5B-Instruct + 现有课程 LoRA。
- 评测：100 条 Agent 用例，greedy，seed 42。
- 工具选择准确率：76.9%。
- JSON 合法率：100.0%。
- 参数准确率：72.5%。
- 最终任务成功率：37.0%。
- 不必要工具调用率：74.3%。
- 端到端 RAG Recall@5：18.2%。
- 引用正确率：0.0%。
- 当前解释：Adapter 的训练内容是课程问答，不针对 Agent 工具轨迹；该结果是后续 Agent
  SFT 的基线，不能证明现有 LoRA 改善了 Agent。
- 证据：`docs/stage4_qwen25_0_5b_agent_eval.json`。

### BASE-004：3B 4-bit Agent 基线

- 状态：`completed`
- 模型：Qwen2.5-3B-Instruct，NF4 4-bit，不加载 Adapter。
- 评测：与 BASE-003 相同的 100 条用例。
- 工具选择准确率：83.1%。
- JSON 合法率：100.0%。
- 参数准确率：82.5%。
- 最终任务成功率：52.0%。
- 不必要工具调用率：51.4%。
- 端到端 RAG Recall@5：50.0%。
- 引用正确率：40.9%。
- 平均耗时：7,596ms/例。
- 当前解释：模型规模改善了所有主要能力指标，但主要短板仍是错误调用工具、未触发检索
  和缺失引用，不是当前词法检索器的排序能力。
- 证据：`docs/stage4_qwen25_3b_4bit_agent_eval.json`。

### BASE-005：资源边界

- 状态：`completed`
- GPU：RTX 4060 Laptop，8GB VRAM。
- D 盘剩余空间：2026-08-25 检查时约 158.9GB。
- 项目新增资产硬上限：40GB；日常目标：32GB。
- 决策：0.5B 用于快速实验，3B 4-bit/QLoRA 用于主结果；未经预算登记不下载大语料或
  新模型。

## 已确定的研究判断

### DEC-001：Top-k 不等于精度

提出背景：计划记录“提高 Top-k 后精度变化”。

判断：Top-k 只限制每一步参与采样的候选 Token 数。`k` 更大通常增加多样性和不确定性，
`k` 更小通常更保守，也可能增加重复；它不是训练精度参数。工具调用、参数生成和冻结问答
评测优先使用 greedy，创作类任务再比较采样配置。

允许记录的结论格式：

> 在固定模型、Prompt、测试集、每题采样次数和其他解码参数后，Top-k 从 20 改为 40，
> 指标 A 从实测值 X 变为 Y，同时重复率从 M 变为 N。该结果只适用于本任务和本配置。

禁止的结论格式：

> 提高 Top-k 可以提高模型精度。

### DEC-002：先修复检索触发，不立即下载大型向量模型

证据：评测语料上的词法 Retriever 基线 Recall@5 为 100%，但 0.5B/3B 的端到端 RAG
Recall@5 分别为 18.2%/50.0%。当前主要错误发生在 Agent 未选择 `search_docs`，或召回后
未输出 citation。

决策：先拆分触发、条件召回和引用指标，并改进路由/监督数据；之后再用小型 embedding
模型做 lexical/dense/hybrid 对照。

### DEC-003：当前 100 条用例不再视为完全隐藏测试集

原因：失败列表和分类指标已经被查看，并将用于下一步设计。继续反复根据这些失败调优再
称为“冻结测试集”会高估泛化能力。

决策：它们作为开发评测基线保留；新建独立冻结测试集用于版本验收。

### DEC-004：暂不引入 LangChain

原因：仓库已有自研 orchestrator、tool registry、executor、context、trace 和 eval runner。
当前优先问题是模型路由、训练数据和评测，不是缺少编排框架。

决策：先完善原生实现；只有出现持久化状态图或第三方集成的明确需求时，才把 LangGraph/
LangChain 作为可选后端对照，不让框架掩盖核心机制。

## 近期实验队列

| ID | 状态 | 研究问题 | 主要变量 | 主要指标 |
|---|---|---|---|---|
| EVAL-001 | completed | 如何拆开 RAG 失败来源？ | 指标口径 | 触发 P/R、条件 Recall、引用 P/R |
| EVAL-002 | completed | 新冻结测试集是否覆盖主要能力？ | 用例分层 | 覆盖率、污染、人工审核 |
| ROUTE-001 | completed | 两阶段路由能否降低误调用？ | route decision | 不必要工具率、单工具成功率 |
| ROUTE-002 | completed | 引用白名单能否减少伪引用？ | citation validator | Citation P/R、任务成功率 |
| DATA-001 | completed | no-tool hard negatives 如何设计？ | 数据类型 | 误调用率、普通问答回退 |
| SFT-001 | completed | Agent SFT v1 为什么失败？ | 工具暴露与数据重复 | 任务成功率、触发、验证 Loss |
| SFT-002 | completed | 修正后的 0.5B Agent SFT 是否值得替换基线？ | SFT v1.1 | 任务成功率、RAG、引用、回退 |
| SFT-003 | completed | 3B QLoRA 能否形成主候选？ | 模型规模与 QLoRA | 任务成功率、RAG、显存、重载 |
| GEN-001 | planned | 解码配置如何影响确定性任务？ | greedy/top-k/top-p | 成功率、格式率、重复率、延迟 |
| RAG-001 | completed | dense/hybrid 是否优于词法基线？ | Retriever | Recall@1/3/5、MRR、延迟、磁盘 |
| SERV-001 | completed | 计划门禁和保守恢复能否修复冻结暴露的可靠性问题？ | orchestrator v2 | 任务、multi-step、参数、恢复、引用 |
| SERV-002 | completed | 服务能否暴露可取消、可限流、可观测的执行过程？ | SSE/runtime | 事件、并发、取消、Token/延迟 |
| DOC-001 | completed | Agent 能否可靠生成 Word 报告？ | docx 工具 | Schema、文件校验、视觉 QA |
| DOC-002 | completed | Agent 能否可靠生成 PDF 报告？ | PDF 工具 | 页数、抽取一致性、视觉 QA |
| ARCH-001 | completed | 现代组件是否在本机小模型上自动更快/更准？ | RoPE/RMSNorm/SwiGLU/SDPA/fused QKV | Loss、BPB、吞吐、显存 |
| TOK-001 | completed | 教学 BPE 能否降低序列长度并保持可逆？ | char vs BPE | Token 数、字符/Token、roundtrip |
| MM-001 | completed | 小型本地 OCR/ASR/TTS 能否形成真实闭环？ | TrOCR/Whisper/system TTS | CER、分段延迟、磁盘 |
| DOMAIN-001 | completed | 私有/成人可选领域能否默认隔离并在训练前 fail closed？ | manifest/授权/数据门禁 | 拒绝率、泄露、默认状态 |

## 已完成实验

### EVAL-001：拆分 RAG 路由、召回与引用失败

- 日期：2026-08-25
- 状态：`completed`
- 主要变量：只改变评分指标，不改变模型、Prompt、用例或历史预测。
- 评测输入：2026-08-20 保存的 0.5B LoRA 和 3B 4-bit 共两份 100 条逐题预测。
- 新增下载：0GB。
- 新增模型运行：无；没有重新加载模型或占用 GPU。

#### 研究问题

我发现旧版 `rag_recall_at_5` 只能说明端到端结果失败，不能区分模型没有调用检索、Retriever
没有召回，还是最终回答没有引用。本实验将三个阶段拆开计量。

#### 指标定义

- `retrieval_trigger_precision`：所有调用 `search_docs` 的用例中，真正需要检索的比例。
- `retrieval_trigger_recall`：所有需要检索的用例中，实际调用 `search_docs` 的比例。
- `conditional_retrieval_recall_at_k`：已经触发检索的必检索用例中，gold 文档进入 Top-k
  的宏平均 Recall；触发后工具失败或没有返回文档仍按零分计入。
- `end_to_end_rag_recall_at_k`：所有必检索用例的宏平均 Recall，未触发检索按零分计入。
- `citation_precision`：逐用例引用中，受 gold 支持的引用数占全部预测引用数的比例；
  非检索题中的伪引用也计为假阳性。
- `citation_recall`：应引用的 gold 文档中，最终实际引用的比例。

旧字段 `rag_recall_at_5`、`citation_accuracy` 和 `ungrounded_rate` 保留，确保历史报告和目标
判断不被破坏。

#### 结果

| 指标 | 0.5B + LoRA | 3B 4-bit |
|---|---:|---:|
| 检索触发 Precision | 61.54% | 44.44% |
| 检索触发 Recall | 36.36% | 54.55% |
| 条件 Retriever Recall@1 | 37.50% | 91.67% |
| 条件 Retriever Recall@3 | 37.50% | 91.67% |
| 条件 Retriever Recall@5 | 50.00% | 91.67% |
| 端到端 RAG Recall@1 | 13.64% | 50.00% |
| 端到端 RAG Recall@3 | 13.64% | 50.00% |
| 端到端 RAG Recall@5 | 18.18% | 50.00% |
| Citation Precision | 0.00% | 56.25% |
| Citation Recall | 0.00% | 40.91% |
| Unsupported Answer Rate | 100.00% | 59.09% |

原始结果：

- `docs/eval_v2_qwen25_0_5b_lora.json`
- `docs/eval_v2_qwen25_3b_4bit.json`

验证：`tests/test_agent_eval.py` 共 9 项通过；Agent 相关回归 21 项通过；全量测试 167 项
通过。Ruff format、Ruff lint 和 `mypy evals/metrics.py` 均通过。全量测试保留一条来自
FastAPI TestClient 依赖的 Starlette 弃用警告，与本次指标改动无关。新增构造用例同时
覆盖漏检索、错误触发、正确召回、正确引用和伪引用。

#### 我的结论

我看到 3B 一旦正确触发检索，gold 文档进入 Top-1/3/5 的比例均为 91.67%，明显高于其 50%的
端到端 Recall@5；它的主要短板是检索触发 Recall 只有 54.55%，并且触发 Precision 只有
44.44%。因此下一项实验应优先改进 route decision，而不是先下载更大的 embedding 模型。

我同时看到 0.5B 不仅触发 Recall 更低，触发后的条件 Recall@5 也只有 50%，说明它同时存在路由和
检索 query/工具执行质量问题。0.5B 仍适合做 CI 和快速 SFT 实验，但当前主结果应继续以
3B 为准。

Citation 指标还让我看到另一层差距：3B 的 Citation Precision 为 56.25%，Recall 为
40.91%。也就是说，既有缺失引用，也有不受 gold 支持的引用。后续需要 citation 白名单
校验与引用监督数据，不能只要求回答里出现任意 `doc-*` 字符串。

#### 我的学习笔记

我把这次学习总结为：端到端指标适合回答“整个系统好不好”，条件指标适合回答“某个组件在获得正确输入后好
不好”。如果只看到端到端 Recall@5=50%，容易误判 Retriever 很差；条件 Recall@5=91.67%
表明大部分损失发生在上游路由。这里仍不能断言 Retriever 已经完美，因为当前语料规模小、
词法基线在该集合上就是 100%，而且条件指标只覆盖已经触发的样本，可能存在选择偏差。

#### 下一步

执行 `ROUTE-001`：在独立开发用例上比较当前单阶段自由工具调用和显式两阶段路由。保持
模型、文档、Retriever、解码和评测不变，主要观察检索触发 Precision/Recall、不必要工具
调用率、单工具成功率以及额外延迟。

### EVAL-002：建立版本化 Dev/Frozen Test 与统计回归

- 日期：2026-08-25
- 状态：`completed`
- 新增下载：0GB。
- 主要产物：100 条 dev、150 条 frozen-test candidate、manifest 指纹、bootstrap、逐 case
  对比、CI 离线 smoke。

#### 我当时的问题

我已经查看过旧 100 条的失败。如果继续用它们调 Prompt/SFT 后再报告为测试成绩，我会
在不知不觉中对测试集过拟合。因此我决定承认它已经是 dev，而不是维护一个形式上的“测试集”。

#### 我的选择

我新建了 150 条冻结候选集，并为两个 split 增加 SHA-256 manifest。我没有把自动生成与
机器校验写成“人工审核完成”，而是保留 `pending_owner_signoff`。我还增加 1,000 次 case
bootstrap 和同题 paired bootstrap，因为单个百分比不足以表达 100 条样本的不确定性。

#### 结果如何改变了我的判断

0.5B → 3B 的任务成功率点估计提高 15 个百分点，paired 95% 区间为 +6～+25；端到端
RAG Recall@5 提高 32 个百分点，区间为 +8～+55。但条件 Retriever Recall@5 的变化区间
为 -1～+88，说明该子集样本少，我不能仅凭 91.67% 的点估计声称 Retriever 已稳定达到
高水平。

#### 最终签发

我于 2026-08-25 以项目所有者李林知的身份批准 150 条冻结测试集，将版本从
`agent-frozen-test-v1-candidate` 提升为 `agent-frozen-test-v1`。最终 fingerprint 保持为：

```text
06fde3d735893adcbc37f52da6c48e6573e5366b3289a5459b228c6918790444
```

签发只改变 manifest 的审核元数据，没有修改任何 JSONL 用例，因此 fingerprint 不变。
从现在开始，任何问题或 gold answer 修改都必须创建新版本并重新签字。

#### 下一步

我会先完成冻结集人工签字，再执行 ROUTE-001。路由实验只在 dev 上迭代；冻结集只在形成
候选版本后运行一次，避免把新的 150 条再次变成开发集。

#### 工程验收

我运行了与 CI 相同的 Ruff format、Ruff lint、mypy、split 校验、0.5B recorded smoke 和
全量覆盖率测试。最终结果为 173 项通过、总覆盖率 69.84%，超过 60% 门槛；没有下载新
模型或语料。唯一警告来自 FastAPI TestClient/httpx 的上游弃用提示，与本阶段改动无关。

### ROUTE-001：用确定性路由把模型从“是否调用工具”中解耦

- 日期：2026-08-25
- 状态：`completed`
- 主要变量：是否在进入模型前执行 `HeuristicAgentRouter` 并只暴露被选中的工具。
- 受控变量：同一 Qwen2.5-0.5B-Instruct、同一课程 LoRA、同一 100 条 dev、同一文档、
  greedy、seed 42。
- 新增下载：0GB。

#### 我为什么先做路由而不是立即训练

P1 告诉我，旧 0.5B 的不必要工具调用率高达 74.29%，3B 在正确触发检索后的条件
Recall@5 又显著高于端到端 Recall@5。如果我直接把所有问题归因于模型权重，就无法知道
训练收益和系统策略收益各占多少。因此我先实现一个可解释的路由层：`direct`、
`calculator`、`current_time`、`retrieval` 和 `multi_step`。路由层默认走 direct，只有命中
明确意图时才给模型看到相应工具 Schema。

#### 静态校准与真实模型实验

我先在 dev 标签上校准规则，静态路由准确率、必需工具召回和 no-tool accuracy 都达到
100%。这个数只能证明规则覆盖了我已经看过的开发集，不是泛化成绩，所以我没有把它写成
“路由模型达到 100%”。真正有意义的是在同一真实模型上的前后对照：

| 指标 | 自由工具调用 | 启用路由 | 变化 |
|---|---:|---:|---:|
| 任务成功率 | 37.00% | 65.00% | +28.00pp |
| 任务成功率 paired 95% CI | - | - | +19～+38pp |
| 工具选择准确率 | 76.92% | 93.85% | +16.92pp |
| 不必要工具调用率 | 74.29% | 0.00% | -74.29pp |
| 检索触发 Recall | 36.36% | 86.36% | +50.00pp |
| 端到端 RAG Recall@5 | 18.18% | 45.45% | +27.27pp |
| Citation Recall | 0.00% | 0.00% | 0.00pp |
| 平均耗时 | 3,907.05ms | 3,047.69ms | -859.36ms |

证据：`docs/p2_qwen25_0_5b_heuristic_eval.json`、
`docs/p2_05b_free_vs_heuristic.json`、`docs/p2_router_static_dev.json`。

#### 我的判断

这个实验接受“先路由能显著降低误调用”的假设。平均延迟下降并不是路由本身神奇加速，
而是 direct 问题不再携带无关工具 Schema，模型生成的无效工具轨迹也减少。与此同时，引用
Recall 仍为 0%，说明“选择了 search_docs”不等于“生成了受支持引用”。因此我把
ROUTE-002 的 citation 白名单保留为独立未完成项，没有把 P2 描述成所有可靠性问题都已解决。

#### 我的学习笔记

规则路由的优点是便宜、确定、可审计；缺点是对新表达方式的召回有限，而且规则在 dev 上
校准后容易显得过于漂亮。模型路由更灵活，但会引入额外推理成本和随机错误。对当前研究
项目，我选择“规则路由作为安全边界，模型负责参数和回答”，并通过冻结集检验规则能否泛化。

### DATA-001：把 SFT 数据从单轮回答升级为可执行 Agent 轨迹

- 日期：2026-08-25
- 状态：`completed`
- 数据版本：`agent_v1.1`。
- 规模：2,400 条 train、300 条 validation，共 2,700 条；实际磁盘约 3.22MiB。
- 类别：direct、calculator、current_time、retrieval、recovery 各 540 条。
- 新增下载：0GB。

#### 我怎样设计监督信号

原管线只适合 `system/user/assistant` 单轮回答。我把它扩展为多轮 role 序列，允许 assistant
先输出工具调用 JSON，tool 返回执行结果，assistant 再给最终答案或恢复回答。Loss 只覆盖
所有 assistant turn，包括工具调用和最终回答；system、user、tool result 与 padding 都设为
`-100`。这样模型学习的是“我应该生成什么”，不会被要求复述用户问题或伪造工具结果。

我没有直接抓取大规模网络语料，而是围绕 P1 失败标签生成小而可验证的轨迹。执行型样本的
参数可以由真实工具规则验证；direct 样本作为 no-tool hard negatives；recovery 样本覆盖
工具失败后的受控回答。这样新增数据只有几 MiB，也更符合 8GB 显存和 40GB 总预算。

#### 污染与重复检查

- 2,700 个 user prompt 全部唯一，内部重复为 0。
- 与 100 条 dev + 150 条 frozen test 的精确重叠为 0。
- 字符三元组 Jaccard 阈值 0.8 时近似重叠为 0。
- 训练到评测的最大相似度为 0.473684。

证据：`hf_lora/data/agent_v1/DATA_CARD.md`、
`hf_lora/data/agent_v1/contamination_report.json`。

#### 我的学习笔记

response-only loss 不只是“把前半段 mask 掉”。Agent 轨迹里可能有多个 assistant turn，
每个 assistant turn 都是监督目标，而 tool turn 必须 mask。否则模型可能学会自己编工具结果。
另外，去重不能只看整行 JSON，因为不同 system 模板包装下的同一句 user prompt 仍可能泄漏；
所以我按规范化 user prompt 做精确检查，再用字符三元组做近似检查。

### SFT-001：保留一次验证 Loss 很漂亮、任务能力却退化的失败实验

- 日期：2026-08-25
- 状态：`completed`，结论为失败，旧 Adapter 仅用于复盘。
- 模型：Qwen2.5-0.5B-Instruct + LoRA。

#### 我犯了什么错误

Agent SFT v1.0 在每条训练样本的 system prompt 里都暴露 calculator、current_time 和
search_docs 三个工具，而且 train/validation 的 user prompt 存在模板级重复。结果验证
Loss 从 0.8424 降到 0.00188，看上去几乎完美；但在真实 dev 上，启用路由后的任务成功率
从 65% 下降到 56%，检索触发 Recall 从 86.36% 下降到 36.36%。

我没有删除这次失败产物，保留在
`hf_lora/artifacts/agent_v1_qwen2_5_0_5b_lora`。它提醒我：验证集和训练模板同分布且重复时，
Loss 只能说明模型记住了模板；每条样本都出现全工具说明时，模型还可能学习“看到工具
Schema 就应该调用某个工具”的伪相关。

#### 我怎样修正

v1.1 只暴露路由选中的工具；direct 样本不出现工具调用指令。我同时重写 prompt 生成逻辑，
保证 2,700 个 user prompt 唯一，并重新执行 dev/frozen 污染检查。这个失败是本项目里很重要的
研究证据：我没有用更低 Loss 覆盖更差的端到端能力，而是让任务评测否决训练结果。

### SFT-002：0.5B v1.1 改善 RAG，但没有成为最终默认模型

- 日期：2026-08-25
- 状态：`completed`，未接受为最终默认模型。
- 配置：LoRA rank 16、alpha 32，target modules 为 q/k/v/o，100 optimizer steps，
  max length 640。
- 训练结果：base validation loss 0.87495，final validation loss 0.002054；峰值显存
  2,396.59MiB；可训练参数 2,162,688（0.4359%）；重载 logits 一致性通过。

#### 结果与取舍

| 指标 | 0.5B Router-only | 0.5B Router + v1.1 LoRA | 变化 |
|---|---:|---:|---:|
| 任务成功率 | 65.00% | 61.00% | -4.00pp |
| 检索触发 Recall | 86.36% | 86.36% | 0.00pp |
| 端到端 RAG Recall@5 | 45.45% | 72.73% | +27.27pp |
| Citation Recall | 0.00% | 36.36% | +36.36pp |
| 不必要工具调用率 | 0.00% | 0.00% | 0.00pp |
| 恢复率 | 75.00% | 100.00% | +25.00pp |

任务成功率变化的 paired 95% CI 跨过 0，因此不能声称总体能力显著下降或提升；但 direct
题的回退是真实存在的。我的决定是：把该 Adapter 保留为“行为训练有效”的证据，不把它
升级为最终默认模型。这个取舍比只挑 RAG 指标更诚实，也说明我同时设置主要指标和回退指标。

证据：`docs/p4_agent_v1_1_05b_dev_eval.json`、
`docs/p4_05b_routed_before_vs_agent_sft_v1_1.json`。

### SFT-003：在 8GB 显存内完成 3B QLoRA 主候选

- 日期：2026-08-25
- 状态：`completed`，作为 P0–P5 冻结终测候选。
- 配置：Qwen2.5-3B-Instruct、NF4 4-bit、LoRA rank 8、alpha 16、q/k/v/o、
  micro batch 1、gradient accumulation 4、80 optimizer steps、seed 20260825。
- 训练资源：训练 step 累计约 277.75 秒；峰值 allocated VRAM 4,028.70MiB；Adapter
  约 26.25MiB。
- Loss：final validation loss 0.160835；最后一步 token-weighted train loss 0.004049。

#### 为什么用 QLoRA

8GB 显存不适合对 3B 模型做全参数训练。我把冻结基座以 NF4 4-bit 加载，只训练低秩
Adapter；double quantization 进一步降低量化常数开销。0.5B 负责发现数据和 Mask 问题，
配方稳定后才把预算用在 3B，这是我对资源约束下实验顺序的选择。

#### Dev 结果

3B QLoRA + heuristic router 在 100 条 dev 上得到：任务成功率 78.00%、JSON 合法率
100%、工具选择准确率 86.15%、单工具成功率 80.00%、不必要工具调用率 0%、条件
Retriever Recall@5 100%、端到端 RAG Recall@5 63.64%、Citation Precision 77.78%、
Citation Recall 63.64%、恢复率 75%。相对历史 3B 自由工具基线的 52% 任务成功率，提高
26 个百分点；但这是“模型 + 路由 + 数据”的系统版本对比，不能把全部增益归因于 QLoRA。

#### 重载检查的诚实边界

训练脚本在严格比较量化模型重载前后的 logits 时发现最大差异约 0.628，因此没有生成
`comparison.json`。原因是重新加载时 NF4 基座会再次量化，量化权重不保证逐位相同；它不等于
Adapter 损坏。Adapter 随后被独立重新加载，并完成 100 条 dev 和 150 条 frozen Agent
推理，0 个 runtime error，这验证了“可加载、可运行”，但不等同于未量化 LoRA 的严格
logits 一致性。代码已把量化模型的重载验收改为 greedy generation 一致性；我没有为了
制造一份漂亮报告再进行第三次相同训练。

证据：`hf_lora/configs/agent_v1_qwen2_5_3b_qlora.yaml`、
`hf_lora/artifacts/agent_v1_1_qwen2_5_3b_qlora/training_metrics.json`、
`docs/p4_agent_v1_1_3b_qlora_dev_eval.json`、
`docs/p4_3b_free_before_vs_routed_qlora.json`。

### RAG-001：小型 dense/hybrid 检索没有击败词法基线

- 日期：2026-08-25
- 状态：`completed`。
- 新增模型：`BAAI/bge-small-zh-v1.5`，MIT License，本地实际 183.35MiB。
- 位置：`D:/mini-gpt-from-scratch/models/bge-small-zh-v1.5`。
- 评测：dev 中 22 条检索用例、15 篇文档。

#### 实现与假设

我实现了可替换的 `LexicalRetriever`、`DenseRetriever` 和基于 weighted reciprocal rank
fusion 的 `HybridRetriever`，并支持为 BGE 中文 query 添加官方建议前缀。我的假设是 dense
可能改善同义表达，hybrid 则兼顾词法精确匹配和语义召回。

#### 结果

| Retriever | Recall@1/3/5 | MRR | 平均延迟 | p95 延迟 |
|---|---:|---:|---:|---:|
| lexical | 100% / 100% / 100% | 1.000 | 0.278ms | 0.448ms |
| dense | 100% / 100% / 100% | 1.000 | 13.851ms | 16.913ms |
| hybrid | 100% / 100% / 100% | 1.000 | 13.295ms | 15.468ms |

证据：`docs/p5_lexical_dense_hybrid_dev_benchmark.json`。

#### 我的决定

三个方案在当前小语料上完全同分，dense 平均延迟约是 lexical 的 49.8 倍，hybrid 约为
47.8 倍。因此我拒绝“用了向量数据库就更先进”的包装，继续把 lexical 设为默认；dense
和 hybrid 作为可插拔实现保留，等待未来更困难、包含同义改写和长文档切分的独立评测。
这里的 100% 也不能外推为 RAG 已完美，它只说明当前 15 篇小语料没有区分三种方案。

### EVAL-003：唯一一次 P0–P5 冻结测试集终测

- 日期：2026-08-25
- 状态：`completed`。
- 所有者：李林知。
- 测试集：`agent-frozen-test-v1`，150 条，fingerprint
  `06fde3d735893adcbc37f52da6c48e6573e5366b3289a5459b228c6918790444`。
- 候选：Qwen2.5-3B-Instruct 4-bit + agent_v1.1 QLoRA + heuristic router + 当前词法
  Retriever。
- 原则：候选形成后只运行一次；本次结果不回流到 P0–P5 调参。

#### 冻结结果

| 指标 | 冻结结果 | 95% bootstrap CI | 预设目标 | 是否达到 |
|---|---:|---:|---:|---|
| 任务成功率 | 74.00% | 67.33%～81.33% | 未单设阈值 | - |
| JSON 合法率 | 100.00% | 100%～100% | ≥95% | 是 |
| 单工具成功率 | 70.91% | 57.89%～82.69% | ≥85% | 否 |
| 不必要工具调用率 | 0.00% | 0%～0% | ≤10% | 是 |
| 检索触发 Precision | 100.00% | 100%～100% | 观察项 | - |
| 检索触发 Recall | 93.18% | 84.78%～100% | 观察项 | - |
| 条件 Retriever Recall@5 | 95.12% | 87.80%～100% | 观察项 | - |
| 端到端 RAG Recall@5 | 88.64% | 78.72%～97.67% | ≥85% | 是 |
| Citation Precision | 92.11% | 81.13%～100% | 观察项 | - |
| Citation Recall | 79.55% | 66.67%～90.70% | 观察项 | - |
| 恢复率 | 0.00% | 0%～0% | ≥70% | 否 |
| 平均耗时 | 6,434.28ms | - | 趋势项 | - |

分层成功率为 direct 90%（27/30）、multi-step 0%（0/20）、retrieval 100%（30/30）、
tool arguments 56%（14/25）、tool selection 88.89%（40/45）。150 条全部正常结束，没有
缺预测、重复预测、未知预测或 runtime error。

#### 结果怎样修正了我的判断

Dev 上任务成功率为 78%，冻结集为 74%，总体差距不大；但内部结构发生了明显变化。
冻结检索题达到 100%，端到端 RAG Recall@5 达到 88.64%，说明 P2 路由、P3 检索轨迹和
当前词法 Retriever 形成了有效链路。另一方面，multi-step 20 条全部失败、参数准确率仅
57.53%、恢复率为 0%，说明当前路由只解决了“先选哪类工具”，没有让 orchestrator 稳定
完成第二步依赖调用、参数传递和失败恢复。

因此我的闭环结论不是“P0–P5 已经完美”，而是“已经形成可复现、可训练、可评测、能检索
并引用的纯文本 Agent 候选，同时冻结测试暴露出清晰的下一阶段工程优先级”。我会把 multi-step
状态机、参数校验/单次修复、citation 白名单和恢复策略放到 P6 的第一批回归里。冻结集 v1
继续封存；后续修复只在 dev 和新增针对性训练集上进行，直到形成新候选，再创建或按治理
规则使用下一版冻结验收。

证据：`docs/p0_p5_final_frozen_3b_qlora_eval.json`、
`evals/results/p0_p5_final_frozen_3b_qlora_predictions.jsonl`。

#### 最终工程验收

我在候选、报告和路线图收口后执行了与 CI 一致的检查：Ruff format check 通过，Ruff
lint 通过，mypy 对 63 个 source files 报告 0 issue，Dev/Frozen manifest 与零重叠检查
通过，recorded Agent smoke 正常退出；全量测试 188 项通过，覆盖率 68.45%，超过 60%
门槛。唯一警告是 FastAPI TestClient 依赖触发的 Starlette 上游弃用提示，与本阶段功能
无关。

### SERV-001：用系统约束收口 multi-step、参数、恢复与引用

- 日期：2026-08-25
- 状态：`completed`，P6 模型效果按 dev 条件验收。
- 所有者：李林知。
- 新增下载：0GB 模型/语料；复用现有 3B QLoRA。
- 冻结纪律：没有重跑 P0–P5 frozen v1，只比较同一 dev 的保存基线与新候选。

#### 我当时怎样想

P0–P5 frozen 的 multi-step 和 recovery 都是 0%。我先问的不是“要不要增加 top-k”，而是
“失败是否来自模型知识不足”。逐项看 trace 后，我发现模型经常知道第一步，却会提前 final；
参数错误多为字段别名或包裹层级；瞬时工具失败也没有按副作用分类。因此这些问题首先属于
编排约束，而不是采样空间。提高 top-k 会增加候选随机性，既不能保证第二步完成，也可能
破坏 JSON 和参数稳定性，所以我没有把它当作本轮变量。

我的假设是：只改变 orchestrator，不重训模型；用计划完成门禁阻止 premature final，允许
一次通过 Pydantic 的保守参数修复，只对无副作用工具做一次瞬时失败重试，并用本轮检索结果
建立 citation whitelist。若任务成功主要受系统状态机限制，multi-step 应改善；若是模型本身
不会生成参数或引用，相关指标不会自动提高。

#### 结果

| 指标 | P4 保存 dev | P6 dev | 我的解释 |
|---|---:|---:|---|
| 任务成功率 | 78.00% | 79.00% | +1pp，95% CI 0～+3pp，弱证据 |
| Multi-step | 86.67%（13/15） | 93.33%（14/15） | `ms-007` 完成第二步 |
| 参数准确率 | 80.00% | 80.00% | 系统修复没有改变模型原始能力分布 |
| Recovery | 75.00% | 75.00% | 保持通过，未新增收益 |
| Citation Precision | 77.78% | 76.47% | -1.31pp |
| Citation Recall | 63.64% | 59.09% | -4.55pp，`ms-007` 新增漏引 |
| 平均总耗时 | 5,348.78ms | 4,840.83ms | 本机趋势，NF4 重载波动需保留 |

任务层没有回退案例，但只有一个新增通过。白名单能拒绝模型编造的来源，却不能自动补齐
没有生成的引用。因此我把 ROUTE-002 判为“安全边界完成、模型引用能力未改善”，没有用
guard 的存在替代 Citation P/R。

#### 我的结论与学习

系统可靠性可以通过不变量获得强证据，例如“副作用工具不会自动重试”“引用只能来自本轮
检索”；模型精度必须依赖样本和置信区间。本轮 +1pp 不能称为显著提升，但 multi-step 门禁
说明 frozen 暴露的问题确实有一部分来自系统状态，而不全是模型能力。

我也再次确认 Top-k 不等于精度。确定性工具任务更需要约束输出空间、校验状态和减少随机性；
只有在开放式生成质量实验中，我才会把 greedy/top-k/top-p 作为单变量对照。

证据：`docs/p6_agent_v1_1_3b_qlora_dev_eval.json`、
`docs/p6_3b_before_vs_reliability_v2.json`。

### SERV-002：从同步 API 到可观测 Agent Runtime

- 日期：2026-08-25
- 状态：`completed`。
- 所有者：李林知。
- 变量：只增加服务运行时、事件和接口，不改变 3B Adapter。

#### 我为什么不只做一个聊天页面

Demo 如果只能看到最后答案，失败时无法解释是 router、模型、工具还是 citation guard。
因此我先定义 `AgentProgressEvent`，让 route、model、tool、plan、final 形成统一事件，再让同步
接口、SSE 和指标共享同一个 `AgentRuntime`。这样 UI 是可观测模型的一位消费者，而不是把
业务逻辑写进前端。

我加入 request/run ID、并发信号量、取消注册表和进程内计数器。默认并发为 2，是针对本机
8GB 显存的保守保护值，不是通用吞吐结论。响应汇总 prompt/generated tokens、模型/工具/总
耗时、finish reason 与引用列表；`/demo` 可以直接观察执行轨迹。

#### 验收边界

自动测试覆盖 SSE 事件顺序、并发释放、取消状态、指标累计和同步接口兼容性。但目前事件流
是阶段级 progress streaming，模型文本在一次 generate 完成后返回；取消在模型/工具边界
生效，不能抢占正在运行的 CUDA kernel；指标只存在当前进程。这些边界进入下一阶段 backlog，
不会因为 API 名叫 stream 就被隐藏。

### DOC-001：把 Word 生成做成受控工具而不是脚本副作用

- 日期：2026-08-25
- 状态：`completed`。
- 所有者：李林知。
- 产物：`output/documents/minigpt_p0_p7_report.docx`。

#### 我的设计

我把文档目录作为显式依赖注入 `DocumentWorkspace`，而不是让模型传任意本地路径。工具只接
受限纯文件名，检查扩展名、大小、目录穿越和覆盖标志；写工具属于 side effect，必须确认。
DOCX 使用 Letter、1 英寸页边距、中文/西文字体、标题元信息、目录、层级标题、明确列宽的
指标表格、引用和动态页码。写入后重新用 python-docx 读取标题、段落和表格，避免只相信保存
调用的返回值。

最终报告为 39,702 bytes，回读得到 27 个段落和 1 个表格。经 LibreOffice 转为 2 页 PDF，
再以 150 DPI 渲染两页 PNG；逐页检查无截断、重叠、乱码和表格越界。

#### 我的学习

Word 是流式文档对象模型，表格宽度、分页和字体替换可能由渲染器重新决定。XML 结构正确只
能证明“可打开”，不能证明“交付排版正确”，所以写后回读与页面渲染必须是两个独立门禁。

### DOC-002：PDF 固定布局、字体嵌入与逐页 QA

- 日期：2026-08-25
- 状态：`completed`。
- 所有者：李林知。
- 产物：`output/pdf/minigpt_p0_p7_report.pdf`。
- 新增文档依赖下载：约 5.7MiB，远低于 40GB；工作目录与产物均在 D 盘。

#### 第一次结果为什么被我拒绝

初版 PDF 为 2 页，pypdf 可以正常抽取文本，所有程序断言都通过。但 150 DPI 页面图显示
`STSong-Light` 在查看器替代后把英文字符拉开，项目名和术语不适合作为正式材料。这证明
“PDF 能打开”和“PDF 可读”不是同一个指标。

我改成优先发现并嵌入真实 Unicode CJK 字体，Windows 使用微软雅黑，其他系统尝试 Noto/
文泉驿，最小环境才 fallback 到 CID 字体。第二次渲染又发现无序列表 glyph 显示为空框，
我改用字体内明确存在的破折号段落，再做第三次逐页检查。

最终 PDF 为 126,675 bytes、2 页，文字、中英文混排、表格、引用与页码均正常，无截断、
重叠、乱码或越界。这个失败修复过程比一张成功截图更能展示我对字体嵌入、查看器替代和
固定页面布局的理解。

#### 工程覆盖

P7 共提供 `create_docx`、`create_pdf`、`read_docx`、`read_pdf`、
`summarize_document`、`export_report` 六个工具。测试覆盖路径穿越、错误扩展名、显式覆盖、
写后回读、摘要、报告导出与工具确认；最终全量工程测试为 227 passed，覆盖率 71.36%，
文档工具模块覆盖率 93%。

### ARCH-001：现代组件不是一键性能升级

- 日期：2026-08-25
- 状态：`completed`，固定 Batch 300 步短预算。
- 所有者：李林知。
- 设备：RTX 4060 Laptop GPU，PyTorch 2.13.0+cu126。

#### 我的问题与控制变量

我想验证 RoPE、RMSNorm、SwiGLU、SDPA 和 fused QKV 在当前 818K 参数教学模型上分别改变
什么，而不是把它们一起打开后把结果都归因于“现代架构”。因此每个 `*_only` 变体只改变
一个组件；seed、Batch、block、层数、宽度、heads、训练步数和数据完全相同。组合变体单独
报告，不参与单变量归因。

300 步结果中，SDPA 同参数吞吐从 93,928 提高到 114,025 tokens/s（+21.40%），峰值分配
显存从 65.75 降到 61.86MiB；fused QKV 为 108,539 tokens/s（+15.55%）。但组合现代架构
只有 69,383 tokens/s（-26.13%），因为 SwiGLU 让参数增加约 30.6%，当前 64 Token 序列下
RoPE/Python 开销也不可忽略。

所有变体都把固定训练 Batch Loss 压到约 0.025，却让验证 Loss 上升。我因此拒绝写“现代
架构提升精度”：这个实验能回答吞吐、内存和过拟合路径，不能回答正式泛化。下一步若要签发
模型结论，需要 2,000 步随机 Batch、相同字符/FLOPs 预算和至少多 seed。

证据：`docs/p8_architecture_benchmark.json`。

### TOK-001：BPE 压缩成立，但精度结论尚未成立

- 日期：2026-08-25
- 状态：`completed`，只签发压缩与可逆性。

我实现确定性 BPE，而不是仅调用现成 tokenizer，这样我能解释 pair count、merge order、未知
字符、序列长度和恢复格式。100,000 字符上，字符 Token 为 100,000，256 词表 BPE 为
51,344，减少 48.656%，平均 1.9476 字符/Token，roundtrip 完全一致。

这个结果不能说明 BPE 的验证 BPB 更好。模型看到的 Token 数减少，也改变 embedding 参数、
序列计算量和字符预算；必须用相同字符或 FLOPs 预算才公平。我的结论只写“压缩有效”，没有
把它包装成“精度提高”。

### MM-001：真实 OCR/ASR/TTS 与依赖失败记录

- 日期：2026-08-25
- 状态：`completed`，基础真实模型 smoke。
- 新增模型缓存：383.99MiB，D 盘。

我先用 Protocol 和 fake backend 完成路径、安全与错误分层测试，再显式下载 TrOCR small 和
Whisper tiny。第一次真实 OCR 失败不是识别错误，而是 Transformers 5 自动 Processor 在当前
环境缺少 sentencepiece/protobuf；补依赖后仍需显式组合 DeiT PIL processor 与 XLM-R tokenizer。
我保留这个过程，因为依赖锁和模型加载同样属于可复现系统。

最终单行印刷体 OCR CER 为 0%；系统 TTS 合成的短英文经 Whisper 回读得到 9.52% 字符错误率。
OCR/ASR 冷启动约 13.92/6.53 秒，热路径约 75.56/74.09ms，TTS 约 525.10ms。这个样本太简单，
不能代表复杂扫描件或噪声语音；P9 的强结论是组件闭环和分段观测完成，弱结论只是该 smoke
在本机可运行。

证据：`docs/p9_multimodal_smoke.json`。

### DOMAIN-001：敏感领域不能靠“不声张”替代治理

- 日期：2026-08-25
- 状态：`completed`，治理和激活控制面完成；没有领域训练。

我把领域能力放在独立 manifest、数据、Adapter、访问 Token 和评测中，默认不加载。Catalog
不泄露路径与 Token hash；错误 Token、禁用 manifest、缺失 Adapter 和越界路径全部 fail
closed。HF 后端可以在推理锁下加载/选择已授权 PEFT Adapter，并可以恢复默认 Adapter/基座。

对于成人可选领域，我没有创建语料或权重。模板必须 restricted、18+、有同意证明，并要求
每条记录确认成年人和自愿情境；涉及未成年人、儿童性内容、强迫或非自愿的记录被拒绝。
静态词表不是充分审核，所以正式流程仍要求来源、许可证、隐私、人工和法律复核。

治理 smoke 中标准合成数据和合规成人治理记录通过，禁止记录被拒绝，错误 Token 被拒绝，
默认公开模型没有修改。状态明确为 `governance_smoke_no_adapter_training`，不能宣称领域质量。

证据：`docs/p10_domain_governance_smoke.json`。

### ACCEPT-001：P0-P10 最终闭环验收

- 日期：2026-08-25
- 状态：`completed`。
- 所有者：李林知。

我在所有功能和报告定稿后重新跑了一遍全仓库门禁，而不是沿用开发中途的局部结果。最终
`ruff format --check` 覆盖 143 个文件，Ruff lint 通过，mypy 检查 75 个源文件无问题；
Dev 100 条与 Frozen 150 条的 manifest、fingerprint、重复、污染和 prompt overlap 检查均
通过。全量 pytest 为 256 passed，只有一个来自 Starlette/httpx 兼容层的第三方弃用警告，
综合覆盖率 71.62%，高于 60% 门槛。

我没有为了让 P8-P10 看起来更好而重跑或调参 frozen v1。最终 DOCX/PDF 先写后回读，再分别
渲染全部 3 页检查；第一轮发现 DOCX 引用编号串号、指标表跨页和 PDF 大块留白，第二轮又
发现 DOCX 引用产生孤立第 4 页，第三轮通过压缩表格内部段距解决。这个过程让我确认“生成
成功”不等于“交付合格”，视觉 QA 也是 Agent 工具能力的一部分。

证据：`docs/p0_p10_closed_loop.md`、`output/documents/minigpt_p0_p10_report.docx`、
`output/pdf/minigpt_p0_p10_report.pdf`。

## 单次实验记录模板

复制本节并追加到日志尾部，不覆盖历史记录。

```markdown
### <ID>：<实验名称>

- 日期：YYYY-MM-DD
- 状态：planned / running / completed / invalid / superseded
- Git commit：
- 工作树是否 dirty：
- 负责人：

#### 研究问题

要回答的唯一主要问题是什么？

#### 背景与原理

为什么这个变量可能有效？涉及什么模型/训练/系统原理？

#### 假设

在看到结果前写下可证伪的预测。

#### 配置

- 基线：
- 唯一主要变量：
- 受控变量：模型、数据切分、训练 Token、seed、Prompt、解码参数、硬件等。
- 数据版本与 hash：
- 模型与 Adapter 版本：
- 训练预算：
- 评测集版本：
- 命令：

#### 资源

- GPU/CPU/RAM：
- 新增下载预计/实际大小：
- 峰值显存：
- 训练/评测耗时：
- 产物目录及大小：

#### 结果

| 指标 | 基线 | 实验 | 绝对变化 | 相对变化 |
|---|---:|---:|---:|---:|
| 主要指标 | | | | |
| 回退指标 | | | | |
| 延迟 | | | | |

原始结果文件：

#### 失败分析

- 改善的 case：
- 退化的 case：
- 新的失败类型：
- 数据污染/运行错误检查：

#### 结论

结论只覆盖本实验能够支持的范围。说明是否接受假设、是否需要多 seed 复验。

#### 学习笔记

用自己的语言说明这次实验涉及的原理、容易误解的点和后续可能追问的问题。

#### 下一步

下一项最小、可归因的实验是什么？
```

## 下载与数据登记模板

```markdown
### ASSET-<ID>：<资产名称>

- 状态：proposed / approved / downloaded / removed
- 来源 URL/仓库：
- 版本/commit/revision：
- 许可证：
- 用途：
- 是否含个人/敏感数据：
- 预计下载大小：
- 实际磁盘大小：
- D 盘目标目录：
- 校验和或 revision：
- 清洗/过滤步骤：
- 删除方式：
- 对 40GB 总预算的影响：
```

## 阶段教学记录方式

推进每个阶段时，日志的“学习笔记”至少回答：

1. 这个组件在数学或系统上做什么。
2. 为什么当前项目需要它。
3. 有哪些替代方案，为什么暂时不选。
4. 用什么实验能够证实它有效。
5. 它可能改善哪些指标，又可能让哪些指标退化。

通过这种方式，代码实现、实验结果和技术理解保持一致。
