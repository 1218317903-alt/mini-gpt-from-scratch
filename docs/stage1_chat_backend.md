# 阶段一：真实聊天模型后端验收

验收日期：2026-08-19

## 目标

把阶段六的 Hugging Face Instruct 模型接入阶段八 FastAPI 服务，同时保留手写
MiniGPT 的 `/generate` 路径。两个后端通过统一的聊天接口和生命周期管理运行，便于
后续加入工具调用、RAG 和会话记忆。

## 已实现

- `ChatMessage`、`ChatRequest`、`ChatResponse` 请求/响应契约。
- `ChatModel` Protocol 和 `ChatResult` 后端无关结果结构。
- `HuggingFaceChatModel`：Tokenizer Chat Template、基座模型、可选 PEFT LoRA Adapter。
- `float16`、`bfloat16`、`float32` 推理类型，以及 `cpu/cuda/mps/auto` 设备选择。
- EOS/长度停止原因、Prompt/生成 Token 统计、温度、Top-k、Top-p、Greedy 和 seed。
- 输入 Token 预算和模型上下文窗口校验。
- `POST /chat`，同时保留 `POST /generate`。
- 环境变量创建聊天后端：`MINIGPT_CHAT_*`。

## 启动配置

使用 Hugging Face 模型时设置：

```powershell
$env:MINIGPT_CHAT_MODEL_SOURCE = "Qwen/Qwen2.5-0.5B-Instruct"
$env:MINIGPT_CHAT_ADAPTER_PATH = "hf_lora/artifacts/qwen2_5_0_5b_lora/adapter"
$env:MINIGPT_CHAT_LOCAL_FILES_ONLY = "true"
$env:MINIGPT_CHAT_DTYPE = "float32"
$env:MINIGPT_DEVICE = "cpu"
uv run uvicorn minigpt_service.app:app --host 127.0.0.1 --port 8000
```

不设置 `MINIGPT_CHAT_MODEL_SOURCE` 时，服务仍可以只启动手写 MiniGPT 或健康检查
模式。基座模型和 Adapter 可分别配置；Adapter 由全新基座加载后恢复，避免依赖训练
进程中的模型对象。

## 接口示例

```powershell
$body = @{
  messages = @(
    @{ role = "system"; content = "你是一个简洁准确的助手。" },
    @{ role = "user"; content = "17乘以6等于多少？" }
  )
  max_new_tokens = 32
  greedy = $true
  seed = 42
} | ConvertTo-Json -Depth 5

Invoke-RestMethod http://127.0.0.1:8000/chat `
  -Method Post -ContentType "application/json" -Body $body
```

## 验收结果

本机已有环境按锁文件升级后，版本为 FastAPI 0.141.1、Uvicorn 0.52.3、PyTorch
2.13.0+cu126、Transformers 5.15.0、PEFT 0.20.0、Datasets 4.8.5。

- 阶段一新增聊天、配置和 HTTP 测试：43 passed。
- 完整测试：108 passed，分支覆盖率 64.62%；Ruff 和 Mypy 均通过。
- Qwen2.5-0.5B-Instruct 使用 Hugging Face 本地缓存离线加载成功。
- LoRA Adapter 使用仓库现有 `hf_lora/artifacts/.../adapter` 离线恢复成功。
- `/health`、`/ready`、`/chat` 真实 TestClient 请求均返回 200。
- 多轮消息能够传入同一个 Chat Template 上下文，第二轮算术回答返回 `102`。

## 当前边界

阶段一只解决“真实聊天模型接入服务”，还没有 Agent 工具循环、RAG、长期记忆、流式
输出或权限确认。这些功能分别在后续阶段实现。
