# 阶段三：工具调用与最小 Agent 循环

## 目标

在阶段一聊天后端之上加入一个可测试、可限制、可观察的最小 Agent。模型只能请求
注册表中的工具；工具参数经过 Pydantic 校验，执行有超时，循环有最大轮数。

## 已实现

- `ToolSpec`、`ToolDefinition`、`ToolCall` 和 `ToolExecutionResult` 契约。
- `ToolRegistry` 显式维护工具 Allow-list，拒绝重复注册。
- `ToolExecutor` 参数校验、异常标准化、超时和执行耗时记录。
- 安全计算器：只允许数字和基本算术运算，不执行 Python、Shell 或函数调用。
- `current_time`：按 IANA 时区返回 ISO-8601 时间。
- `search_docs`：只读检索固定本地文档目录，返回带 citation 的 Top-5 片段。
- `ChatModelAgentAdapter`：把普通聊天模型输出解析为严格 JSON 工具调用。
- `AgentOrchestrator`：默认最多 6 轮、硬上限 8 轮；成功调用不允许原样重复，失败调用
  最多允许一次同参数恢复重试，随后阻断循环。
- `AgentContext`：管理本次请求的消息和重复调用签名。
- `AgentTracer`：记录每个模型/工具步骤的输入、输出、耗时和错误，并对密码、Token、
  API Key 等常见敏感字段脱敏。
- 副作用工具默认不执行；返回与工具名和参数绑定的 `confirmation_id`。调用方明确把该
  ID 放入 `approved_side_effects` 后才允许执行。
- `POST /agent/chat`：返回最终答案、循环步数、工具次数、工具 trace 和完整脱敏 events。

模型请求工具时使用以下协议：

```json
{
  "tool_call": {
    "name": "calculator",
    "arguments": {"expression": "17*6"}
  }
}
```

模型返回普通文本时直接结束；返回工具调用时，执行结果会作为 `tool` 消息加入下一轮
上下文。未知工具、错误参数和工具异常都会作为错误结果反馈给模型，模型可以尝试恢复。

## 接口示例

```powershell
$body = @{
  messages = @(
    @{ role = "user"; content = "现在 UTC 时间是什么？" }
  )
  max_new_tokens = 64
  greedy = $true
  max_steps = 6
} | ConvertTo-Json -Depth 5

Invoke-RestMethod http://127.0.0.1:8000/agent/chat `
  -Method Post -ContentType "application/json" -Body $body
```

## 安全边界

当前默认注册表只包含 `calculator`、`current_time` 和只读 `search_docs`，不包含任意
命令执行、任意 Python、文件写入、网络访问或发送消息工具。副作用确认由执行器强制
实施，不依赖模型 Prompt。

## 验收

- 工具、解析器、编排器和 Agent API 单元测试覆盖正常调用、非法参数、未知工具、超时、
  多轮恢复和最大轮数限制。
- 本阶段不包含长期记忆、流式输出或多 Agent 协作；文档检索是便于理解和测试的词法
  基线，不是生产级向量 RAG。
