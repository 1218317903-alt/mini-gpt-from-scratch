"""Persistence tests for local SQLite conversations."""

import sqlite3
from pathlib import Path

import pytest

from minigpt_service.conversations import SQLiteConversationRepository


def repository(path: Path) -> SQLiteConversationRepository:
    result = SQLiteConversationRepository(path)
    result.initialize()
    return result


def test_sqlite_repository_persists_ordered_conversation(tmp_path: Path) -> None:
    path = tmp_path / "chat.db"
    first = repository(path)
    conversation = first.create_conversation(
        title="模型讨论",
        model="fake-model",
        system_prompt="回答要简洁。",
    )
    first.append_message(conversation.id, role="user", content="什么是 GQA？")
    first.append_message(
        conversation.id,
        role="assistant",
        content="GQA 让多个查询头共享 KV 头。",
        model="fake-model",
    )

    second = repository(path)
    restored = second.get_conversation(conversation.id)
    messages = second.list_messages(conversation.id)

    assert restored is not None
    assert restored.title == "模型讨论"
    assert [message.sequence for message in messages] == [1, 2]
    assert [message.role for message in messages] == ["user", "assistant"]


def test_repository_recovers_generating_message_after_restart(tmp_path: Path) -> None:
    path = tmp_path / "recovery.db"
    first = repository(path)
    conversation = first.create_conversation(title="恢复测试")
    pending = first.append_message(
        conversation.id,
        role="assistant",
        content="",
        status="generating",
    )

    second = repository(path)
    restored = second.list_messages(conversation.id)

    assert restored[0].id == pending.id
    assert restored[0].status == "interrupted"


def test_repository_persists_summary_and_cascades_delete(tmp_path: Path) -> None:
    store = repository(tmp_path / "summary.db")
    conversation = store.create_conversation(title="压缩测试")
    store.append_message(conversation.id, role="user", content="第一条")
    updated = store.update_summary(
        conversation.id,
        summary="用户先前讨论了第一条。",
        summarized_through_sequence=1,
    )

    assert updated.summary == "用户先前讨论了第一条。"
    assert updated.summarized_through_sequence == 1
    assert store.delete_conversation(conversation.id) is True
    assert store.list_messages(conversation.id) == []


def test_repository_rejects_duplicate_idempotency_key(tmp_path: Path) -> None:
    store = repository(tmp_path / "idempotency.db")
    conversation = store.create_conversation(title="幂等测试")
    store.append_message(
        conversation.id,
        role="assistant",
        content="回复一",
        request_id="request-123",
    )

    with pytest.raises(Exception, match="UNIQUE"):
        store.append_message(
            conversation.id,
            role="assistant",
            content="回复二",
            request_id="request-123",
        )


def test_repository_rejects_newer_database_schema(tmp_path: Path) -> None:
    path = tmp_path / "future.db"
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version = 99")

    with pytest.raises(RuntimeError, match="newer"):
        repository(path)
