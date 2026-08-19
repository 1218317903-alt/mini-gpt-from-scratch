# 阶段二：工具调用与最小 Agent 循环

## 目标

在阶段一聊天后端之上加入一个可测试、可限制、可观察的最小 Agent。模型只能请求
注册表中的工具；工具参数经过 Pydantic 校验，执行有超时，循环有最大轮数。

## 已实现

- `ToolSpec`、`ToolDefinition`、`ToolCall` 和 `ToolExecutionResult` 契约。
- `ToolRegistry` 显式维护工具 Allow-list，拒绝重复注册。
- `ToolExecutor` 参数校验、异常标准化、超时和执行耗时记录。
- 安全计算器：只允许数字和基本算术运算，不执行 Python、Shell 或函数调用。
- `current_time`：按 IANA 时区返回 ISO-8601 时间。
- `ChatModelAgentAdapter`：把普通聊天模型输出解析为严格 JSON 工具调用。
- `AgentOrchestrator`：最多 6 轮模型/工具循环，保存每次工具执行轨迹。
- `POST /agent/chat`：返回最终答案、循环步数、工具次数和工具执行 trace。

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

当前默认注册表只包含 `calculator` 和 `current_time`，不包含任意命令执行、任意文件
读写、网络访问或发送消息工具。后续增加有副作用的工具时，需要额外的权限和用户确认
层，不能只依赖模型 Prompt。

## 验收

- 工具、解析器、编排器和 Agent API 单元测试覆盖正常调用、非法参数、未知工具、超时、
  多轮恢复和最大轮数限制。
- 当前完整测试为 129 项通过，包含 `agent` 的总分支覆盖率为 67.89%；Ruff 和 Mypy 均通过。
- 阶段二暂不包含 RAG、长期记忆、流式输出或多 Agent 协作。
