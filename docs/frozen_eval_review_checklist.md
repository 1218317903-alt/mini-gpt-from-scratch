# Frozen Test v1 人工审核与签字表

对象：`evals/frozen_cases/`

当前版本：`agent-frozen-test-v1`

审核状态：`approved`

审核人：李林知

审核日期：2026-08-25

当前指纹：

```text
06fde3d735893adcbc37f52da6c48e6573e5366b3289a5459b228c6918790444
```

## 我为什么必须亲自做这一步

自动测试可以证明 JSON 合法、数量正确、文档存在、词法检索能召回，也可以发现精确重复；
但它不能完全判断题目是否自然、gold answer 是否存在歧义、`answer_contains` 是否会误判，
也不能替我承担最终数据质量责任。

因此我会把机器校验和人工审核分开。只有下列项目全部确认后，我才把 manifest 的
`review.status` 从 `pending_owner_signoff` 改为 `approved`，并把版本从 candidate 提升为
正式 frozen-test v1。

审核完成后使用显式命令重新签发 manifest，例如：

```powershell
uv run python evals/build_manifest.py `
  --cases-dir evals/frozen_cases `
  --split frozen_test `
  --version agent-frozen-test-v1 `
  --purpose "Release-only generalization test; never use for prompt or SFT tuning." `
  --provenance "P1 candidate reviewed and approved by the project owner." `
  --frozen --review-status approved `
  --reviewer "项目所有者姓名" `
  --signed-at "YYYY-MM-DD"
```

签发后再次运行 `uv run python evals/validate_splits.py`，并把新的 fingerprint 填入本表。

## 文件级审核

| 文件 | 条数 | 用途 | 我的审核状态 |
|---|---:|---|---|
| `direct_answer.jsonl` | 30 | 无工具直接回答 | 已批准 |
| `tool_selection.jsonl` | 45 | 需要工具和 no-tool 干扰题 | 已批准 |
| `tool_arguments.jsonl` | 25 | 参数格式和内容 | 已批准 |
| `retrieval.jsonl` | 30 | 检索、答案和引用 | 已批准 |
| `multi_step.jsonl` | 20 | 多工具、检索计算、恢复 | 已批准 |

## 每条用例检查项

我对每条用例逐项确认：

- [x] 用户问题自然、清楚，没有依赖未提供的背景。
- [x] `expect_tool` 与真实任务一致。
- [x] `expected_tools` 顺序和任务要求一致。
- [x] `forbidden_tools` 不会禁止合理工具。
- [x] `expected_arguments` 是工具实际接受的 Schema。
- [x] 数学答案已经独立复算。
- [x] `answer_contains` 不会因为过短字符串产生明显误判。
- [x] Retrieval gold 文档确实支持答案，不只是包含相同关键词。
- [x] Citation ID 与 gold 文档一致。
- [x] Multi-step 的最少调用次数与任务一致。
- [x] Recovery 用例的失败注入对象正确。
- [x] 问题没有复制 dev、SFT 或未来 Prompt few-shot。
- [x] 不包含真实个人隐私、账号、密码或未经授权的数据。
- [x] 难度合理，不用冷僻知识刻意卡模型。

## 分类级复核记录

```text
direct_answer:
发现的问题：
修订的 case IDs：
复核结论：

tool_selection:
发现的问题：
修订的 case IDs：
复核结论：

tool_arguments:
发现的问题：
修订的 case IDs：
复核结论：

retrieval:
发现的问题：
修订的 case IDs：
复核结论：

multi_step:
发现的问题：
修订的 case IDs：
复核结论：
```

如果审核中修改任何 JSONL，旧指纹立即失效。我需要重新运行 builder/manifest、全套 split
校验和测试，然后记录新指纹，不能只手工改 manifest。

## 最终签字

```text
审核人：李林知
审核日期：2026-08-25
最终版本：agent-frozen-test-v1
最终 fingerprint：06fde3d735893adcbc37f52da6c48e6573e5366b3289a5459b228c6918790444
全部 150 条是否逐条检查：是
是否允许用于 Release 验收：是
备注：项目所有者已明确同意签发；机器校验和内容审核状态均已记录。
```

该签字已完成。后续任何用例内容修改都会改变 fingerprint，必须提升版本、重新审核并重新
签发；不得在保持 `agent-frozen-test-v1` 名称的同时修改问题或 gold answer。
