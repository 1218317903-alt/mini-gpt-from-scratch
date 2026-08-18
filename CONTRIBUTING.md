# Contributing

## 开发环境

```powershell
uv sync --locked --extra dev --extra hf --extra viz
```

项目要求 Python 3.11。依赖只在根目录 `pyproject.toml` 声明，并由 `uv.lock` 锁定；
修改依赖后必须同步提交锁文件。

## 提交前检查

```powershell
uv run ruff format .
uv run ruff check .
uv run mypy
uv run pytest -q --cov=minigpt --cov=hf_lora --cov-report=term-missing
```

CI 使用同一锁文件和同一组命令。覆盖率低于 60% 会失败。

## 实验规范

- 一次只改变一个主要变量，并固定 seed、数据划分和训练 Token 预算。
- 正式训练使用独立 `runs/<name>/`，不得覆盖旧实验。
- 记录最佳验证 Loss、独立测试 Loss、硬件环境、Git commit 和生成参数。
- 单次计时与单 seed 结果必须标注限制，不能写成跨硬件或统计显著结论。
- Checkpoint、模型权重和逐步日志不提交 Git；提交轻量配置、结果摘要和复现命令。
