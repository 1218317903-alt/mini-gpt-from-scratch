# 阶段八：模型服务化验收

验收日期：2026-08-19

## 验收结论

阶段八的八项实现均已落地。服务层代码、接口契约、错误处理和健康检查已通过当前可用
环境中的测试；真实模型加载、Docker 镜像构建和模型生成压测需要恢复 Python 3.11
PyTorch 环境并启动 Docker daemon 后再做最终环境验收。

## 验收清单

| 验收项 | 状态 | 证据 |
|---|---|---|
| FastAPI 应用工厂与 ASGI 入口 | 通过 | `src/minigpt_service/app.py` |
| 生成请求与响应模型 | 通过 | `schemas.py`；Schema 测试 |
| ModelService 生命周期 | 条件通过 | 实现与基础状态测试已添加；当前解释器无 PyTorch，未执行真实 Checkpoint 加载 |
| `POST /generate` | 条件通过 | 路由转发、错误状态和响应测试通过；真实模型生成待 PyTorch 环境 |
| `/health` 与 `/ready` | 通过 | 真实本地 HTTP 请求均返回预期状态 |
| 请求追踪与统一异常处理 | 通过 | `X-Request-ID`、JSON 日志、400/422/500 测试通过 |
| Docker 镜像与容器启动 | 条件通过 | Dockerfile、`.dockerignore` 和启动说明已完成；本机 Docker daemon 未启动 |
| 基础负载测试与性能记录 | 通过（健康端点） | [stage8_health_load.json](stage8_health_load.json) |

## 当前环境测试

阶段八服务相关测试共 **29 passed**。本机 Python 3.13 可以运行不依赖 PyTorch 的服务、
Schema、错误处理、环境配置和负载测试；系统 Python 没有 `torch`，项目 `.venv` 的
Python 3.11 启动器指向已不存在的旧安装路径，因此完整项目测试未能执行。

本次直接运行完整测试时，在前 3 个依赖 PyTorch 的测试收集阶段即因 `ModuleNotFoundError:
No module named 'torch'` 停止；这属于环境阻塞，不是阶段八代码测试失败。

本地健康端点负载测试命令：

```powershell
python scripts/load_test_service.py `
  --url http://127.0.0.1:8765/health `
  --requests 100 `
  --concurrency 8 `
  --output docs/stage8_health_load.json
```

本次记录（Windows 本机、健康端点、100 请求、并发 8）为：100/100 成功，吞吐约
251.732 req/s，p50 8.532 ms，p95 222.233 ms，p99 231.815 ms。该数据只描述当前
健康检查链路，不代表模型生成性能。

## 完整环境验收命令

恢复 Python 3.11 后执行：

```powershell
uv sync --locked --extra dev --extra hf --extra service --extra viz
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest -q --cov=minigpt --cov=hf_lora --cov-report=term-missing
docker build -t minigpt-service:local .
```

挂载配置、Tokenizer 和 Checkpoint 后，再对 `/ready` 与 `/generate` 做模型端到端和
负载测试。阶段八不包含 RAG、工具调用或 Agent 编排。

## 阶段 0 环境复验

验收日期：2026-08-19

本次复验没有下载或安装依赖，直接复用了本机已有环境：项目 `.venv` 提供 Python
3.11.9、PyTorch 2.13.0+cpu、Transformers 5.15.0、PEFT、Datasets、Ruff 和 Mypy；
本机 Python 3.11 的 site-packages 提供 FastAPI 0.115.6 与 Uvicorn 0.34.0。

现有 FastAPI/Uvicorn 版本分别低于 `pyproject.toml` 声明的 0.116/0.35 最低版本；在不
下载依赖的前提下，本次仅验证了兼容性和实际运行结果，严格的锁文件环境复现仍需后续
在允许安装依赖时补做。

结果如下：

| 检查项 | 结果 |
|---|---|
| `ruff format --check .` | 通过，63 个文件已格式化 |
| `ruff check .` | 通过 |
| `mypy` | 通过，31 个源文件 |
| 完整 pytest | 91 passed，83.53 秒 |
| 分支覆盖率 | 64.27%，超过 60% 门槛 |
| 真实 Checkpoint 加载 | 通过，CPU，`is_ready=True` |
| `/health` | HTTP 200 |
| `/ready` | HTTP 200 |
| `/generate` | HTTP 200，`ROMEO:` 生成成功 |
| Docker daemon | 待启动；Docker CLI 可用但 `com.docker.service` 当前为 Stopped |

阶段八的代码和环境验收已具备进入下一阶段的条件；Docker 镜像构建需在启动 Docker
Desktop/daemon 后补做。
