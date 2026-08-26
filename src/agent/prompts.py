"""Small, explicit prompts and guardrail messages used by the Agent loop."""

from __future__ import annotations

DEFAULT_MAX_STEPS = 6
MAX_ALLOWED_STEPS = 8

MAX_STEPS_MESSAGE = "达到最大 Agent 循环次数，未能完成任务。"
REPEATED_CALL_MESSAGE = (
    "Repeated tool call blocked: the same tool and arguments already succeeded or "
    "the allowed error retry was exhausted. Use the previous result, change the "
    "arguments, or answer the user."
)
CONFIRMATION_MESSAGE = (
    "该工具会产生外部副作用，尚未执行。请用户确认后，使用返回的 "
    "confirmation_id 重新提交请求。"
)
CANCELLED_MESSAGE = "请求已取消；Agent 在安全边界处停止，未继续执行工具。"
CITATION_GUARD_MESSAGE = (
    "回答包含本次检索结果之外的文档引用，已被引用白名单阻止。请重新检索后再回答。"
)
INCOMPLETE_PLAN_MESSAGE = (
    "The requested multi-step plan is incomplete. Before giving a final answer, "
    "call one of these remaining tools exactly once with validated arguments: {tools}."
)
