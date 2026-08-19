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
  -v ${PWD}/models:/models:ro `
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
