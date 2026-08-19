# 阶段八进度记录

## 已完成

### 实现一：FastAPI 服务骨架

- 新增 `src/minigpt_service/app.py`
- 提供 `create_app()` 和全局 `app`
- 接入 `fastapi` 与 `uvicorn[standard]` 服务依赖
- CI 安装 `service` 依赖组

### 实现二：请求与响应模型

- 新增 `src/minigpt_service/schemas.py`
- `GenerateRequest` 统一约束 Prompt、生成长度、Temperature、Top-k、Top-p、Greedy、KV Cache 和随机种子
- `GenerateResponse` 固定生成结果、生成 Token 数量和结束原因字段
- 默认禁止未知字段，避免客户端拼写错误静默进入服务

### 实现三：模型生命周期管理

- 新增 `ServiceConfig` 和 `ModelService`
- 统一执行配置、Tokenizer、Checkpoint、权重和 `eval()` 加载流程
- 支持 CPU、CUDA、MPS 和 `auto` 设备选择
- 支持可选 `torch.compile`
- FastAPI lifespan 启动时加载、关闭时释放模型
- 增加 Tokenizer 一致性校验，避免词表与权重错配

### 实现四：文本生成接口

- 新增 `POST /generate`
- 将采样、KV Cache 和随机种子等参数转发给 `ModelService`
- 模型未加载或推理运行时不可用时返回 503
- 请求参数或生成参数无效时返回 400
- 返回统一的生成文本、生成 Token 数和结束原因

### 实现五：健康检查与就绪检查

- 新增 `GET /health` 存活检查，服务进程可响应时返回 200
- 新增 `GET /ready` 就绪检查，模型未加载时返回 503
- 新增健康与就绪响应模型，固定返回结构

### 实现六：结构化日志、请求追踪与异常处理

- 使用或生成 `X-Request-ID`，并在响应与错误响应中返回
- 记录包含请求 ID、路径、状态码和耗时的 JSON 访问日志
- 统一处理 HTTP 异常、请求校验错误和未预期异常
- 错误响应固定为 `detail` 与 `request_id` 字段

### 实现七：Docker 镜像与容器启动

- 新增基于 `uv.lock` 的 Python 3.11 Dockerfile
- 新增 `.dockerignore`，排除本地缓存、训练产物和数据文件
- 支持通过 `MINIGPT_*` 环境变量配置模型路径、设备和编译选项
- 容器内置 `/health` Docker HEALTHCHECK；未挂载模型时仍可启动健康检查
- 新增容器构建和运行说明

### 实现八：基础负载测试、性能记录与阶段验收

- 新增 `scripts/load_test_service.py`
- 支持并发请求、成功率、状态码统计、吞吐量和 p50/p95/p99 延迟
- 支持将性能记录写入 JSON，并以 p95 阈值作为可选验收门槛
- 新增负载测试指标单元测试和阶段八验收报告

## 当前边界

目前已经注册 `/generate`、`/health` 和 `/ready` 路由，并提供容器启动入口。
阶段八八项实现均已落地；完整模型推理压测仍依赖可用的 PyTorch、Tokenizer 和 Checkpoint。

## 验证

- Ruff format/lint：历史阶段通过；本轮新增脚本待恢复可用的锁定环境复核
- Python 语法编译：通过
- Pydantic、应用路由、错误处理、容器配置和负载指标测试：通过（29 passed）
- 服务生命周期基础测试：已添加；当前解释器缺少 PyTorch，尚未执行
- 服务 API/Schema 测试：通过（29 passed；当前解释器提示 Starlette 的 httpx TestClient 弃用警告）
- `/generate` 路由已接入；完整模型推理端到端测试待恢复可用的 PyTorch 环境后执行
- `/health`、`/ready`、请求追踪和统一错误响应测试已添加
- Dockerfile 静态检查未能执行：本机 Docker daemon 未启动；Docker CLI 本身可用
- 健康端点负载测试：100 请求、并发 8，成功率 100%，吞吐 251.732 req/s，p95 222.233 ms

## 环境备注

项目 `.venv` 中已存在 pytest，但其 `python.exe` 启动器仍引用已不存在的旧 Python 3.11
安装路径。服务依赖已写入 `pyproject.toml` 和 `uv.lock`；完整项目测试应在修复该解释器
路径后，通过 `uv sync --locked --extra dev --extra service` 执行。

为避免 Schema/OpenAPI 访问被模型依赖阻断，`minigpt_service` 的模型服务组件采用懒加载；
只有实际导入 `ModelService` 或启动带模型的应用时才会加载 PyTorch 相关模块。
