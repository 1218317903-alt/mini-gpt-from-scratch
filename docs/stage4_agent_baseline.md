# 阶段四：真实 Agent 基线

2026-08-20 在 RTX 4060 Laptop GPU 上用同一套 100 条冻结用例完成了两条基线：
`Qwen/Qwen2.5-0.5B-Instruct` + 仓库 LoRA Adapter，以及原始
`Qwen/Qwen2.5-3B-Instruct` + NF4 4-bit。两次都使用 greedy、seed 42、每轮最多 128
个新 Token；3B 使用 `bitsandbytes 0.50.1`，不加载 0.5B Adapter。

完整机器可读报告见 [stage4_qwen25_0_5b_agent_eval.json](stage4_qwen25_0_5b_agent_eval.json)。
逐条预测位于 `evals/results/qwen25_0_5b_lora_predictions.jsonl`，属于可再生本地结果，
不提交版本库。

## 0.5B + LoRA 结果

| 指标 | 结果 | 目标 | 状态 |
|---|---:|---:|---|
| JSON 合法率 | 100.0% | ≥ 95% | 通过 |
| 单工具任务成功率 | 68.6% | ≥ 85% | 未通过 |
| 不必要工具调用率 | 74.3% | ≤ 10% | 未通过 |
| RAG Recall@5 | 18.2% | ≥ 85% | 未通过 |
| 异常后恢复率 | 75.0% | ≥ 70% | 通过 |
| 工具选择准确率 | 76.9% | — | 记录 |
| 参数准确率 | 72.5% | — | 记录 |
| 最终任务成功率 | 37.0% | — | 记录 |
| 引用正确率 | 0.0% | — | 记录 |
| 平均耗时 | 3907 ms | — | 记录 |
| 平均生成 Token | 61.29 | — | 记录 |

词法检索器在同一语料上的独立 Recall@5 是 100%，而 0.5B 系统 Recall@5 只有 18.2%。因此
当前主要故障不是排序器找不到文档，而是 0.5B 模型经常不选 `search_docs`、选错工具，
或在拿到结果后不输出 citation。

分类成功率为：直接回答 0/20、多步骤 7/15、检索 0/15、参数构造 10/15、工具选择
20/35。无工具的 35 条题中有 26 条发生不必要调用；22 条含检索语料的任务中 18 条
没有召回 gold 文档，全部 22 条都没有正确引用。100 条均完成，99 条正常停止，1 条达到
最大步数，没有 Python/服务运行时错误。

## 3B 4-bit 对照

完整报告见 [stage4_qwen25_3b_4bit_agent_eval.json](stage4_qwen25_3b_4bit_agent_eval.json)，
加载/延迟基准见 [stage4_qwen25_3b_4bit_benchmark.json](stage4_qwen25_3b_4bit_benchmark.json)。

| 指标 | 0.5B + LoRA | 3B 4-bit | 目标 |
|---|---:|---:|---:|
| 工具选择准确率 | 76.9% | 83.1% | — |
| JSON 合法率 | 100.0% | 100.0% | ≥ 95% |
| 参数准确率 | 72.5% | 82.5% | — |
| 最终任务成功率 | 37.0% | 52.0% | — |
| 单工具任务成功率 | 68.6% | 82.9% | ≥ 85% |
| 不必要工具调用率 | 74.3% | 51.4% | ≤ 10% |
| RAG Recall@5 | 18.2% | 50.0% | ≥ 85% |
| 引用正确率 | 0.0% | 40.9% | — |
| 异常后恢复率 | 75.0% | 75.0% | ≥ 70% |
| 多步骤平均调用次数 | 1.53 | 1.27 | — |
| 平均耗时 | 3907ms | 7596ms | — |
| 平均生成 Token | 61.29 | 76.20 | — |

3B 使所有能力指标改善，但仍只通过 JSON 合法率和恢复率两个门槛；单工具成功率
接近目标但差 2.1 个百分点。3B 的直接回答成功率为 2/20，检索成功率为 5/15；无工具
题中 18/35 仍发生不必要调用，22 条含检索语料的任务中 11 条没有召回 gold 文档，
13 条没有正确 citation。加载耗时 4.26 秒，首 Token 454ms，smoke test 峰值显存
约 2053MiB；完整评测平均单例耗时较高，主要来自 128 Token 上限和多轮 Agent 调用。

这说明模型规模确实解决了一部分错工具和参数问题，但当前主要架构问题仍在工具路由、
检索触发和引用约束，而不是基础执行器或显存容量。

## 恢复题口径

评测专用 `EvalToolExecutor` 只对标记为恢复题的用例注入一次瞬时失败。它等待模型首次
调用预期工具，返回标准化 `error`，随后所有调用委托给生产 `ToolExecutor`。只有同一
工具在错误之后成功才记作恢复。4 条恢复题中 3 条通过；0.5B 失败的 `ms-014` 达到最大
轮数，3B 失败的 `ms-011` 没有完成重试。
生产服务不会启用故障注入。

## 结论与下一步

这份基线验证了最小 Agent 循环的结构化调用、参数校验、工具执行、错误回传、重试和
记录链路，也区分出了模型规模收益与架构短板。下一步按以下顺序推进：

1. 新建独立开发集用于 prompt、工具描述和路由策略调优；本次 100 条继续作为冻结测试集，
   避免对已查看失败用例过拟合。
2. 优先降低无工具题误调用，并为检索结果加入可验证的 citation 输出约束；随后再考虑
   embedding/reranker。当前词法检索基线已是 100%，此时先换向量库不会修复主要问题。
3. 将相同评测接入 CI 的 0.5B smoke 子集；3B 全量作为本地回归基线，避免每次 CI 占用
   数分钟和显存。

复现前需设置 `MINIGPT_CHAT_MODEL_SOURCE`、`MINIGPT_CHAT_ADAPTER_PATH`、设备和离线加载
环境变量，然后运行：

```powershell
.\.venv\Scripts\python.exe evals\run_agent_eval.py --run-agent `
  --predictions-output evals\results\qwen25_0_5b_lora_predictions.jsonl `
  --output docs\stage4_qwen25_0_5b_agent_eval.json `
  --max-new-tokens 128 --greedy --seed 42
```

3B 对照命令（本地缓存已存在时）：

```powershell
$env:MINIGPT_CHAT_MODEL_SOURCE='Qwen/Qwen2.5-3B-Instruct'
$env:MINIGPT_CHAT_ADAPTER_PATH=$null
$env:MINIGPT_CHAT_LOCAL_FILES_ONLY='true'
$env:MINIGPT_CHAT_DTYPE='float16'
$env:MINIGPT_CHAT_QUANTIZATION='4bit'
$env:MINIGPT_DEVICE='cuda'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
.\.venv\Scripts\python.exe evals\run_agent_eval.py --run-agent `
  --predictions-output evals\results\qwen25_3b_4bit_predictions.jsonl `
  --output docs\stage4_qwen25_3b_4bit_agent_eval.json `
  --max-new-tokens 128 --greedy --seed 42
```
