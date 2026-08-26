# 阶段一：真实聊天模型后端审查

审查日期：2026-08-20

当前状态：**0.5B 路线的代码、Adapter 恢复和本机硬件实测已完成；3B/4-bit 路线尚未
实测。**

## 目标

把真实 Instruct 模型接入服务，同时保留手写 MiniGPT 的 `/generate` 路径。Agent
不再关心底层是字符级 MiniGPT、原始 Hugging Face 模型还是 LoRA Adapter。

统一接口位于 `src/agent/`：

```text
src/agent/
├── llm/
│   ├── base.py
│   ├── minigpt_backend.py
│   └── huggingface_backend.py
└── messages.py
```

`ChatModel.chat(messages, tools=None, ...)` 接受 `system` / `user` /
`assistant` / `tool` 四种角色，并返回文本、可选工具调用、Token 统计和生成耗时。

## 已实现

- 后端无关的 `Message`、`ToolSchema`、`ChatModel` 和 `ChatResult`。
- `HuggingFaceChatModel`：Chat Template、基座模型、可选 PEFT LoRA Adapter。
- `MiniGPTChatModel`：把字符级 MiniGPT 接到同一套 `/chat` 与 Agent 接口。
- `float16` / `bfloat16` / `float32`，以及可选 `4bit` / `8bit` 量化推理。
- EOS / `<|im_end|>` 停止、输入 Token 预算、上下文窗口校验。
- Prompt / 生成 Token 数和 `duration_ms`。
- `POST /chat` 直接接收多轮消息，不再要求调用方拼接 Prompt。
- 模型分两条线：
  - `Qwen2.5-0.5B-Instruct`：CI、接口测试和快速实验。
  - `Qwen2.5-3B-Instruct` + 4-bit：8GB 显存上的本地实用 Agent 主模型。

## 启动配置

```powershell
$env:MINIGPT_CHAT_MODEL_SOURCE = "Qwen/Qwen2.5-0.5B-Instruct"
$env:MINIGPT_CHAT_ADAPTER_PATH = "hf_lora/artifacts/qwen2_5_0_5b_lora/adapter"
$env:MINIGPT_CHAT_LOCAL_FILES_ONLY = "true"
$env:MINIGPT_CHAT_DTYPE = "float32"
$env:MINIGPT_DEVICE = "cpu"
uv run uvicorn minigpt_service.app:app --host 127.0.0.1 --port 8000
```

8GB 显存上的本地主模型：

```powershell
$env:MINIGPT_CHAT_MODEL_SOURCE = "Qwen/Qwen2.5-3B-Instruct"
$env:MINIGPT_CHAT_DTYPE = "float16"
$env:MINIGPT_CHAT_QUANTIZATION = "4bit"
$env:MINIGPT_DEVICE = "cuda"
```

配置说明见 `configs/chat_models.yaml`。不设置 `MINIGPT_CHAT_MODEL_SOURCE` 但提供
MiniGPT Checkpoint 时，`/chat` 和 Agent 会走 `MiniGPTChatModel`。

## 基准

```powershell
$env:MINIGPT_CHAT_MODEL_SOURCE = "Qwen/Qwen2.5-0.5B-Instruct"
$env:MINIGPT_CHAT_LOCAL_FILES_ONLY = "true"
uv run python scripts/benchmark_chat.py --output docs/stage1_chat_benchmark.json
```

记录项：加载耗时、首 Token 延迟（`max_new_tokens=1`）、短补全耗时、峰值显存。

## 验收状态

- Adapter 保存、全新基座恢复及 logits 一致性检查已实现在
  `hf_lora/scripts/train_and_compare.py`。本机产物的最大 logits 差为 `0.0`，且三条固定
  Prompt 的 `after` 与 `restored` 文本完全一致。
- `/chat` 会把完整三轮上下文送入 Chat Template。
- 模型输出可解析为合法工具 JSON。
- Token 数和生成耗时随 `/chat` 返回。

2026-08-20 本机 0.5B + LoRA + CUDA FP32 实测：加载 `3677.046 ms`，首个生成 Token
调用 `278.112 ms`，短补全 `140.367 ms`，峰值显存 `1920.75 MiB`。真实模型同时输出
了可解析的 `calculator` JSON 调用，参数为 `{"expression": "17*6"}`。完整机器可读
记录见 `stage1_chat_benchmark.json`。

待完成：3B/4-bit 配置尚未在本机实测（本机没有该模型缓存，本次审查没有下载模型）。
流式输出仍是可选项，不阻塞本阶段核心验收。
