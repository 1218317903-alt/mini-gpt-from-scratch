# P2–P5 第一人称完成报告

作者：李林知
日期：2026-08-25

## 结论先行

我已经完成 P2 路由、P3 Agent SFT 数据、P4 0.5B LoRA/3B QLoRA 和 P5 RAG v2 的实现与
实验，并使用唯一一次 150 条冻结测试集对最终候选做了验收。最终候选是：

```text
Qwen2.5-3B-Instruct（NF4 4-bit）
  + agent_v1.1 QLoRA Adapter
  + HeuristicAgentRouter
  + 当前默认 LexicalRetriever
  + 自研 Orchestrator / Tool Registry / Trace / Eval
```

冻结结果为任务成功率 74%、JSON 合法率 100%、不必要工具调用率 0%、RAG Recall@5
88.64%、Citation Precision 92.11%；但单工具成功率 70.91%、参数准确率 57.53%、多步骤
成功率 0%、恢复率 0%。因此我把它定义为“完整、可复现、可演示的纯文本 Agent 候选”，
而不是完美产品。

## P2：路由层

我新增了确定性 `AgentRouter` 抽象和 heuristic 实现，让系统先选择 direct、calculator、
current_time、retrieval 或 multi_step，再只向模型暴露相关工具。Orchestrator 会记录 route，
评测 runner 支持 `free`/`heuristic` 对照。

在相同 0.5B + 课程 LoRA 的 100 条 dev 对照中，任务成功率从 37% 提高到 65%，paired
bootstrap 95% CI 为 +19～+38pp；不必要工具调用率从 74.29% 降到 0%；平均延迟从
3,907ms 降到 3,048ms。引用 Recall 仍是 0%，所以 citation 白名单仍是未完成项。

主要代码与证据：

- `src/agent/routing.py`
- `src/agent/orchestrator.py`
- `evals/evaluate_router.py`
- `docs/p2_05b_free_vs_heuristic.json`

## P3：Agent SFT 数据工程

我把训练 Schema 从单轮问答升级为 system/user/assistant/tool 多轮轨迹，并实现所有
assistant turn 参与 Loss、其余 role 与 padding mask 为 `-100`。数据 v1.1 共 2,700 条：
2,400 train、300 validation，五类样本各 540 条。2,700 个 user prompt 全部唯一，与
dev/frozen 的精确重叠和高相似近似重叠均为 0。

我保留了 v1.0 失败实验：验证 Loss 从 0.8424 降到 0.00188，dev 任务成功率却从 65%
下降到 56%。根因是所有训练样本都暴露全部工具，并且 train/validation 有模板重复。
v1.1 只暴露路由对应工具，direct 不出现工具指令，同时重建唯一 prompt。这是项目里“Loss
不能替代端到端评测”的直接证据。

主要代码与证据：

- `hf_lora/src/hf_lora/data.py`
- `hf_lora/scripts/build_agent_sft_data.py`
- `hf_lora/data/agent_v1/DATA_CARD.md`
- `hf_lora/data/agent_v1/contamination_report.json`

## P4：LoRA/QLoRA 实验

0.5B v1.1 LoRA 把 RAG Recall@5 从 45.45% 提高到 72.73%，Citation Recall 从 0% 提高
到 36.36%，但总体任务成功率从 65% 变为 61%，且 direct 有回退。我没有把它选为默认。

3B QLoRA 在约 4,028.70MiB 峰值显存内完成 80 steps，final validation loss 为 0.160835。
Dev 任务成功率 78%，相对历史 3B 自由工具基线 52% 提高 26pp。因为模型、路由和 SFT
同时变化，我只把它称为系统版本提升，不把 26pp 全部归因于微调。

量化重载的严格 logits 比较未通过，因此我没有伪造或补写 `comparison.json`。Adapter 后续
在新模型实例中完成 dev/frozen 共 250 条推理、0 runtime error，证明可加载运行；精确重载
边界单独记录在 `docs/p4_3b_qlora_training_summary.md`。

## P5：RAG v2

我实现 lexical、dense、hybrid 三种可替换 Retriever、Transformer mean pooling、BGE 中文
query prefix 和 weighted RRF。新增 `bge-small-zh-v1.5` 约 183.35MiB，满足 D 盘和 40GB
上限。

在 22 条 dev 检索用例、15 篇文档上，三种 Retriever 的 Recall@1/3/5 和 MRR 都是 100%。
Lexical 平均延迟 0.278ms，dense 13.851ms，hybrid 13.295ms。由于没有精度增益，我仍把
lexical 作为默认；dense/hybrid 保留为后续困难语料对照。这一决定展示的是基于指标选型，
而不是为了技术名词堆叠强行上线向量检索。

主要代码与证据：

- `src/agent/retrieval.py`
- `evals/benchmark_retrieval.py`
- `docs/p5_lexical_dense_hybrid_dev_benchmark.json`

## 冻结验收

冻结集版本 `agent-frozen-test-v1`，150 条，fingerprint：

```text
06fde3d735893adcbc37f52da6c48e6573e5366b3289a5459b228c6918790444
```

| 指标 | 实测值 | 95% CI / 说明 |
|---|---:|---|
| 任务成功率 | 74.00% | 67.33%～81.33% |
| 工具选择准确率 | 81.90% | 74.50%～89.32% |
| JSON 合法率 | 100.00% | 100%～100% |
| 参数准确率 | 57.53% | 45.71%～69.36% |
| 单工具成功率 | 70.91% | 57.89%～82.69% |
| 不必要工具调用率 | 0.00% | 0%～0% |
| 检索触发 Precision / Recall | 100% / 93.18% | Recall CI 84.78%～100% |
| 条件 Retriever Recall@5 | 95.12% | 87.80%～100% |
| 端到端 RAG Recall@5 | 88.64% | 78.72%～97.67% |
| Citation Precision / Recall | 92.11% / 79.55% | 分别 81.13%～100% / 66.67%～90.70% |
| 恢复率 | 0.00% | 0%～0% |
| 运行稳定性 | 150/150 | 0 runtime error |

原始报告：`docs/p0_p5_final_frozen_3b_qlora_eval.json`。

## 我对 P2–P5 的验收判断

- P2 通过“路由架构与误调用控制”验收，但 citation validator 未完成。
- P3 通过“多轮 Schema、assistant-only mask、版本与污染检查”验收。
- P4 通过“消费级 GPU 训练出可运行候选”验收；未达到单工具 85% 目标，量化重载也只通过
  可加载/端到端运行验收，没有通过严格 logits 一致性。
- P5 通过“lexical/dense/hybrid 可插拔实现和同语料对照”验收；dense 没有证据成为默认。
- 整体纯文本链路可演示，但多步骤、参数生成和失败恢复是进入 P6 前必须优先解决的缺口。
