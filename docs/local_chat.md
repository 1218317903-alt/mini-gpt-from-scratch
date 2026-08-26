# 本地多轮聊天与自动上下文压缩

## 启动

```powershell
$env:MINIGPT_CHAT_MODEL_SOURCE = "Qwen/Qwen2.5-3B-Instruct"
$env:MINIGPT_CHAT_DTYPE = "float16"
$env:MINIGPT_CHAT_QUANTIZATION = "4bit"
$env:MINIGPT_CONVERSATION_DB = "D:\mini-gpt-from-scratch\output\conversations.db"
uv run uvicorn minigpt_service.app:app --host 127.0.0.1 --port 8000
```

浏览器打开 `http://127.0.0.1:8000/chat-ui`。

## 自动上下文压缩

每次生成前，服务端读取完整会话，但只向模型发送受预算约束的上下文：

1. 保留 system prompt。
2. 复用此前已经持久化的摘要。
3. 优先保留最近若干轮消息。
4. 超出预算时，把更早且已经完成的消息压入摘要。
5. 摘要和 `summarized_through_sequence` 写回 SQLite，服务重启后继续复用。
6. 如果最新用户消息自身超过预算，返回明确错误，不静默截断用户输入。

当前默认摘要器是确定性提取摘要：保留角色和主要文本并裁剪到摘要预算，因此无需再次调用
模型，不会因为摘要模型不可用而阻塞聊天。后续可以注入模型摘要器，但应继续保留确定性
摘要作为失败回退，并对关键事实、待办、用户约束和专有名词做回归测试。

可调环境变量：

| 变量 | 默认值 | 作用 |
|---|---:|---|
| `MINIGPT_CHAT_MAX_INPUT_TOKENS` | `4096` | 模型最大输入预算 |
| `MINIGPT_CONTEXT_RESERVED_OUTPUT_TOKENS` | `512` | 为本次回复预留的 Token |
| `MINIGPT_CONTEXT_SUMMARY_TOKENS` | `768` | 持久化摘要预算 |
| `MINIGPT_CONTEXT_RECENT_TURNS` | `6` | 压缩时优先保留的最近轮数 |

单次请求的 `max_new_tokens` 会覆盖默认回复预留量，使上下文构建与实际生成预算一致。

## 会话接口

```text
POST   /conversations
GET    /conversations
GET    /conversations/{conversation_id}
PATCH  /conversations/{conversation_id}
DELETE /conversations/{conversation_id}

POST   /conversations/{conversation_id}/messages
POST   /conversations/{conversation_id}/messages/stream
POST   /conversations/runs/{run_id}/cancel
```

流式接口输出稳定 SSE 事件：

```text
start -> delta... -> done
                  -> error
```

`start` 提供 run ID、消息 ID、上下文 Token 估算和本次压缩数量；`delta` 是增量文本；
`done` 提供最终消息、Token 用量、完成原因和耗时。

## 持久化与恢复

- SQLite 启用 foreign keys、WAL 和 5 秒 busy timeout。
- 每个仓库操作使用短连接和短事务；模型生成期间不持有数据库事务。
- assistant 消息先以 `generating` 状态创建，生成期间分段保存，结束后改成 `complete`、
  `cancelled` 或 `error`。
- 服务启动时，遗留的 `generating` 消息自动改成 `interrupted`。
- 同一个 conversation 同时只能运行一个生成任务。
- `client_request_id` 使用数据库唯一约束，防止网络重试重复创建回复。

## 当前边界

- 第一版是单机单用户设计；SQLite 不用于多节点共享写入。
- Hugging Face 后端提供真正的生成增量和 stopping criteria；MiniGPT 研究后端使用单块流式
  兼容回退。
- 页面当前使用纯文本安全渲染，尚未加入 Markdown/代码高亮、编辑后重试和回复分支。
- 摘要是对早期对话的压缩上下文，不是用户长期偏好记忆；长期记忆应单独授权和管理。
