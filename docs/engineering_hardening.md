# 工程收敛与并发隔离说明

日期：2026-08-26

本轮不增加新的模型能力，目标是把 P0–P10 工作区收敛成更容易审查、回归和发布的工程版本。

## 1. 评测门禁语义

CI 中的 5 条 recorded prediction 是评分器 fixture，只验证用例加载、预测清单和指标计算链路；
它不代表模型达到发布质量目标。`evals/run_agent_eval.py` 现在提供两个明确模式：

```powershell
# 只要求预测完整、无重复、无未知 ID
uv run python evals/run_agent_eval.py --predictions <predictions.jsonl>

# 在清单完整之外，要求全部配置目标通过
uv run python evals/run_agent_eval.py --predictions <predictions.jsonl> --require-targets
```

CI 步骤名称已改为 `Replay evaluator fixture (inventory semantics only)`，避免把评分器 smoke
误读成模型质量回归。

## 2. 请求级领域 Adapter

持久化进程级 Adapter 会让并发请求共享可变模型状态。本轮将领域推理改为一个原子事务：

```text
授权 → 获取领域运行锁 → 激活 Adapter → 推理 → 恢复默认模型 → 释放锁
```

普通请求也经过同一运行锁，因此不会在领域 Adapter 临时激活期间进入模型。调用示例：

```json
{
  "messages": [{"role": "user", "content": "分析这份受控领域材料"}],
  "domain_id": "private_example",
  "domain_access_token": "runtime-secret"
}
```

`domain_id` 和 `domain_access_token` 必须同时出现。旧的 `/domains/{domain_id}/activate` 与
`/domains/deactivate` 路由保留为 deprecated API，但固定返回 409，不再修改进程级状态。
单进程内的交叉请求问题因此被关闭；多进程部署仍需外部鉴权、租户到 worker 的绑定和审计。

## 3. 服务模块边界

- `app.py`：生命周期、通用错误、MiniGPT/HF Chat 和 Agent/SSE。
- `domain_runtime.py`：领域授权、请求级 Adapter 事务和领域路由。
- `multimodal_routes.py`：上传、OCR、ASR、TTS 与能力探测路由。
- `agent_runtime.py`：并发槽位、取消 Token 和聚合指标。

拆分后应用工厂的语句数从 347 降到 263；领域和多模态路由可以单独演进与覆盖。

## 4. 覆盖率和产物治理

全局分支覆盖率门槛从 60% 提高到 70%。CI 同时生成 `coverage.json`，再由
`scripts/check_critical_coverage.py` 检查训练、编排、领域和服务关键模块的独立最低线，避免
高覆盖模块掩盖关键路径回退。

`output/` 现在整体被 Git 忽略，只保留 `.gitkeep`。DOCX、PDF、音频和图片等可再生产物不随
源码提交；正式交付物应通过 Release 或带 SHA256、生成命令和 commit 的 artifact manifest
发布。

## 5. 验证结果

- Ruff format/check：通过。
- mypy：通过。
- Dev/Frozen inventory：100/150 条通过，无污染或 split 重叠。
- pytest：263 passed。
- 综合分支覆盖率：72.06%，超过 70% 门槛。
- 关键模块覆盖率门禁：通过。
- 已知第三方警告：FastAPI/Starlette TestClient 提示未来迁移到 httpx2；不影响当前测试结果。

本轮没有重跑真实 3B frozen 模型评测，也没有改变已有模型指标。下一项模型工作仍应优先解决
参数准确率、multi-step 和 recovery，而不是继续扩展新的能力阶段。
