# Contributing

## 开发环境

```powershell
uv sync --locked --extra dev --extra hf --extra service --extra viz --extra documents --extra multimodal
```

项目固定使用 Python 3.11.16。`uv` 会读取根目录 `.python-version`；执行
`uv run python --version` 应输出 `Python 3.11.16`。依赖只在根目录 `pyproject.toml`
声明，并由 `uv.lock` 锁定；
修改依赖后必须同步提交锁文件。

CI 中的 5 条 recorded prediction 只回放评分器并检查预测清单完整性，不代表模型达到质量
目标。对完整候选结果执行发布门禁时，应额外传入 `--require-targets`。

## 提交前检查

```powershell
uv run ruff format .
uv run ruff check .
uv run mypy
uv run python evals/run_agent_eval.py --check-only
uv run pytest -q --cov=minigpt --cov=hf_lora --cov=agent `
  --cov=minigpt_service --cov=evals --cov-report=term-missing `
  --cov-report=json:coverage.json
uv run python scripts/check_critical_coverage.py coverage.json
```

也可以直接运行 `scripts/check.ps1`。CI 使用同一锁文件和同一组命令，覆盖率低于
70% 会失败；此外，关键训练、编排、领域和服务模块还有独立最低线，避免聚合覆盖率掩盖
核心路径回退。

## 实验规范

- 一次只改变一个主要变量，并固定 seed、数据划分和训练 Token 预算。
- 正式训练使用独立 `runs/<name>/`，不得覆盖旧实验。
- 记录最佳验证 Loss、独立测试 Loss、硬件环境、Git commit 和生成参数。
- 单次计时与单 seed 结果必须标注限制，不能写成跨硬件或统计显著结论。
- Checkpoint、模型权重和逐步日志不提交 Git；提交轻量配置、结果摘要和复现命令。
