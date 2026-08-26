"""Automatic context compression tests."""

from dataclasses import replace

from minigpt_service.conversations import ConversationContextBuilder
from minigpt_service.conversations.models import Conversation, StoredMessage


def conversation() -> Conversation:
    return Conversation(
        id="conversation-1",
        title="压缩测试",
        model=None,
        system_prompt="你是一个可靠的助手。",
        summary=None,
        summarized_through_sequence=0,
        created_at="2026-08-26T00:00:00+00:00",
        updated_at="2026-08-26T00:00:00+00:00",
    )


def stored(sequence: int, role: str, content: str) -> StoredMessage:
    return StoredMessage(
        id=f"message-{sequence}",
        conversation_id="conversation-1",
        sequence=sequence,
        role=role,  # type: ignore[arg-type]
        content=content,
        status="complete",
        model=None,
        prompt_tokens=None,
        generated_tokens=None,
        finish_reason=None,
        request_id=None,
        created_at="2026-08-26T00:00:00+00:00",
        updated_at="2026-08-26T00:00:00+00:00",
    )


def test_context_builder_keeps_small_history_without_compression() -> None:
    builder = ConversationContextBuilder(
        max_input_tokens=512,
        reserved_output_tokens=128,
    )
    result = builder.build(
        conversation(),
        [stored(1, "user", "你好"), stored(2, "assistant", "你好，有什么可以帮你？")],
    )

    assert result.compressed is False
    assert [message.role for message in result.messages] == [
        "system",
        "user",
        "assistant",
    ]


def test_context_builder_summarizes_old_turns_and_keeps_latest_user() -> None:
    builder = ConversationContextBuilder(
        max_input_tokens=384,
        reserved_output_tokens=64,
        summary_token_budget=64,
        recent_turns=1,
    )
    messages = [
        stored(
            index, "user" if index % 2 else "assistant", f"第 {index} 条 " + "内容" * 70
        )
        for index in range(1, 7)
    ]
    result = builder.build(conversation(), messages)

    assert result.compressed is True
    assert result.summary
    assert result.summarized_through_sequence > 0
    assert result.messages[-1].content == messages[-1].content
    assert any(
        "Earlier conversation summary" in message.content for message in result.messages
    )


def test_context_builder_reuses_persistent_summary() -> None:
    existing = replace(
        conversation(),
        summary="此前用户询问了注意力机制。",
        summarized_through_sequence=4,
    )
    builder = ConversationContextBuilder(
        max_input_tokens=512,
        reserved_output_tokens=128,
    )
    result = builder.build(
        existing,
        [
            stored(3, "user", "旧消息"),
            stored(4, "assistant", "旧回复"),
            stored(5, "user", "继续解释 GQA"),
        ],
    )

    contents = [message.content for message in result.messages]
    assert any("此前用户询问了注意力机制" in content for content in contents)
    assert "旧消息" not in contents
    assert contents[-1] == "继续解释 GQA"
