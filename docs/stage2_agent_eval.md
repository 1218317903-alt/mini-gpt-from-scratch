# 阶段二：Agent 评测集

当前状态：**评测集、静态校验、真实 Agent runner 和首个 0.5B 基线均已完成。**
目标阈值是后续 Agent 迭代的质量门槛；首个基线只通过 2/5 个门槛，详见
[阶段四 Agent 基线](stage4_agent_baseline.md)。

原第一版 100 条用例的失败已被用于分析，因此从 2026-08-25 起正式定位为开发集
`agent-dev-v1`，用于 Prompt、路由和训练数据设计，不再作为隐藏测试集。另建 150 条
`agent-frozen-test-v1` 作为候选版本 Release 验收集；它已经通过机器校验，并由项目所有
者李林知于 2026-08-25 正式审核签发。任何内容修改都必须提升版本并重新签字。

## 目录

```text
evals/
├── cases/
│   ├── direct_answer.jsonl      # 20 条不需要工具的问答
│   ├── tool_selection.jsonl     # 20 条单工具 + 15 条干扰题
│   ├── tool_arguments.jsonl     # 15 条参数易错题
│   ├── retrieval.jsonl          # 15 条青梧实验室文档检索题
│   └── multi_step.jsonl         # 15 条多步任务 / 失败恢复
├── frozen_cases/                # 150 条冻结测试候选 + manifest
├── fixtures/                    # 可提交的离线预测回放
├── metrics.py
├── validate_splits.py
└── run_agent_eval.py
```

检索题使用虚构的“青梧实验室”文档，避免与 `hf_lora/data` 中的课程问答重叠。

## 指标

- 工具选择准确率
- 工具参数 JSON 合法率
- 工具参数内容准确率
- 任务最终成功率
- 多步骤任务平均调用次数
- 检索触发 Precision / Recall
- 条件 Retriever Recall@1 / 3 / 5
- 端到端 RAG Recall@1 / 3 / 5
- 关键词基线 Recall@5
- Citation Precision / Recall
- 无依据回答比例
- 平均延迟和生成 Token 数

第一版目标：

- JSON 合法率 ≥ 95%
- 单工具任务成功率 ≥ 85%
- 不必要工具调用率 ≤ 10%
- RAG Recall@5 ≥ 85%
- 工具执行异常后可恢复率 ≥ 70%

## 运行

```powershell
uv run python evals/run_agent_eval.py --check-only
uv run python evals/run_agent_eval.py --predictions path/to/predictions.jsonl
uv run python evals/run_agent_eval.py --run-agent `
  --predictions-output evals/results/predictions.jsonl `
  --output docs/agent_eval_report.json --greedy --seed 42
```

`--check-only` 会核对条数、ID 唯一性、与训练集不重叠，以及检索题的 gold 文档能
被词重叠检索器打进 Top-5。

评分模式要求预测文件覆盖全部 case，且不能包含重复或未知 ID；缺失预测按失败计入
分母。`rag_recall_at_5` 只统计预测文件实际返回的文档 ID，关键词检索器的结果单独
记录为 `baseline_rag_recall_at_5`，不会替被评系统补全检索结果。

2026-08-25 起，评测同时保留旧版 `rag_recall_at_5` 和 `citation_accuracy` 以兼容历史
报告，并新增失败分层指标：`retrieval_trigger_*` 衡量是否正确决定调用检索；
`conditional_retrieval_recall_at_*` 只在已经触发检索的必检索用例上衡量返回文档；
`end_to_end_rag_recall_at_*` 将未触发检索按零分计入；`citation_precision/recall` 则衡量
最终引用的纯度和完整性。这样可以区分路由、Retriever 与回答生成三个故障层。

`--run-agent` 从 `MINIGPT_CHAT_*` 加载真实 HF/LoRA 后端，把用例内嵌语料合并为只读
检索库，并逐条刷新 JSONL。中断后可加 `--resume` 续跑；`--case-id` 和 `--limit` 可做
烟测。恢复题由评测专用执行器对第一个预期工具注入一次确定性瞬时错误，只有同名工具
随后成功才记为恢复；这不会改变生产 `ToolExecutor`。

2026-08-20 审查结果：100 条全部通过静态校验，分类数量正确，训练集重叠为 0，词法
检索基线 Recall@5 为 100%。真实 0.5B Agent 预测覆盖 100 条且无运行时错误。人工审核
属性仍需由维护者签字确认，生成脚本本身不能替代人工审核。
