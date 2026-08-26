# Container 启动

镜像使用锁定的 `uv.lock` 安装 Python 3.11 运行时依赖（包含 `service` 与 `hf` extra），
默认监听 `0.0.0.0:8000`。
没有提供模型路径时，容器仍可启动并响应 `/health`，但 `/ready`、`/generate`、`/chat`
和 `/agent/chat` 会返回 503；这便于编排系统区分“进程活着”和“模型已就绪”。

构建镜像：

```powershell
docker build -t minigpt-service:local .
```

使用本地配置、Tokenizer 数据和 Checkpoint 启动：

```powershell
docker run --rm `
  -p 8000:8000 `
  -e MINIGPT_CONFIG_PATH=/app/configs/tiny_shakespeare_baseline.yaml `
  -e MINIGPT_CHECKPOINT_PATH=/models/best.pt `
  -v ${PWD}/data:/app/data:ro `
  -v ${PWD}/runs/my-baseline/checkpoints:/models:ro `
  minigpt-service:local
```

使用 Hugging Face 聊天后端时，模型权重和 Adapter 建议作为只读目录挂载，并设置：

```powershell
docker run --rm `
  -p 8000:8000 `
  -e MINIGPT_CHAT_MODEL_SOURCE=/models/Qwen2.5-0.5B-Instruct `
  -e MINIGPT_CHAT_ADAPTER_PATH=/models/adapter `
  -e MINIGPT_CHAT_LOCAL_FILES_ONLY=true `
  -e MINIGPT_CHAT_DTYPE=float32 `
  -e MINIGPT_CONVERSATION_DB=/data/conversations.db `
  -v ${PWD}/models:/models:ro `
  -v ${PWD}/output:/data `
  minigpt-service:local
```

容器环境变量：

- `MINIGPT_CONFIG_PATH`：模型配置 YAML 路径
- `MINIGPT_CHECKPOINT_PATH`：Checkpoint 路径；必须与配置路径同时提供
- `MINIGPT_DEVICE`：`auto`、`cpu`、`cuda` 或 `mps`
- `MINIGPT_COMPILE`：`true`/`false`，默认 `false`
- `MINIGPT_COMPILE_MODE`：`default`、`reduce-overhead` 或 `max-autotune`
- `MINIGPT_CHAT_MODEL_SOURCE`：Hugging Face 模型 ID 或本地快照目录；设置后启用 `/chat`
- `MINIGPT_CHAT_ADAPTER_PATH`：可选 PEFT LoRA Adapter 目录
- `MINIGPT_CHAT_DTYPE`：`float16`、`bfloat16` 或 `float32`，默认 `float32`
- `MINIGPT_CHAT_MAX_INPUT_TOKENS`：聊天输入 Token 上限，默认 `4096`
- `MINIGPT_CHAT_LOCAL_FILES_ONLY`：`true`/`false`，严格离线加载时设为 `true`
- `MINIGPT_CONVERSATION_DB`：SQLite 会话数据库，默认 `output/conversations.db`
- `MINIGPT_CONTEXT_RESERVED_OUTPUT_TOKENS`：为模型回复预留的上下文 Token，默认 `512`
- `MINIGPT_CONTEXT_SUMMARY_TOKENS`：自动摘要预算，默认 `768`
- `MINIGPT_CONTEXT_RECENT_TURNS`：压缩时优先保留的最近对话轮数，默认 `6`

启动后访问 `/chat-ui` 使用持久化聊天页面。SQLite 数据库目录必须是可写挂载；模型目录仍可
保持只读。
