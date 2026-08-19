# Container 启动

镜像使用锁定的 `uv.lock` 安装 Python 3.11 运行时依赖，默认监听 `0.0.0.0:8000`。
没有提供模型路径时，容器仍可启动并响应 `/health`，但 `/ready` 和 `/generate` 会返回
503；这便于编排系统区分“进程活着”和“模型已就绪”。

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

容器环境变量：

- `MINIGPT_CONFIG_PATH`：模型配置 YAML 路径
- `MINIGPT_CHECKPOINT_PATH`：Checkpoint 路径；必须与配置路径同时提供
- `MINIGPT_DEVICE`：`auto`、`cpu`、`cuda` 或 `mps`
- `MINIGPT_COMPILE`：`true`/`false`，默认 `false`
- `MINIGPT_COMPILE_MODE`：`default`、`reduce-overhead` 或 `max-autotune`
