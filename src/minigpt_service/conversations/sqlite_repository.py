"""SQLite implementation of persistent conversations."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from uuid import uuid4

from agent.messages import MessageRole
from minigpt_service.conversations.models import (
    Conversation,
    MessageStatus,
    StoredMessage,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    model TEXT,
    system_prompt TEXT,
    summary TEXT,
    summarized_through_sequence INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    archived INTEGER NOT NULL DEFAULT 0 CHECK (archived IN (0, 1))
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant', 'tool')),
    content TEXT NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN ('complete', 'generating', 'cancelled', 'error', 'interrupted')
    ),
    model TEXT,
    prompt_tokens INTEGER,
    generated_tokens INTEGER,
    finish_reason TEXT,
    request_id TEXT UNIQUE,
    name TEXT,
    tool_call_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
    UNIQUE (conversation_id, sequence)
);

CREATE INDEX IF NOT EXISTS idx_conversations_updated_at
ON conversations(updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_messages_conversation_sequence
ON messages(conversation_id, sequence);
"""
_SCHEMA_VERSION = 1


def _now() -> str:
    return datetime.now(UTC).isoformat()


class SQLiteConversationRepository:
    """Store local single-user conversations in a WAL-mode SQLite database."""

    def __init__(self, path: Path) -> None:
        if not isinstance(path, Path):
            raise TypeError("path must be a pathlib.Path.")
        self.path = path

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        """Create tables and mark abandoned in-flight generations interrupted."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            current_version = int(
                connection.execute("PRAGMA user_version").fetchone()[0]
            )
            if current_version > _SCHEMA_VERSION:
                raise RuntimeError(
                    "Conversation database schema is newer than this application."
                )
            connection.executescript(_SCHEMA)
            connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
            timestamp = _now()
            connection.execute(
                """
                UPDATE messages
                SET status = 'interrupted', updated_at = ?
                WHERE status = 'generating'
                """,
                (timestamp,),
            )

    def close(self) -> None:
        """Connections are operation-scoped, so no shared handle is retained."""

    def create_conversation(
        self,
        *,
        title: str,
        model: str | None = None,
        system_prompt: str | None = None,
    ) -> Conversation:
        title = title.strip()
        if not title:
            raise ValueError("title must not be empty.")
        conversation_id = uuid4().hex
        timestamp = _now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO conversations (
                    id, title, model, system_prompt, summary,
                    summarized_through_sequence, created_at, updated_at, archived
                ) VALUES (?, ?, ?, ?, NULL, 0, ?, ?, 0)
                """,
                (
                    conversation_id,
                    title,
                    model,
                    system_prompt,
                    timestamp,
                    timestamp,
                ),
            )
            row = self._conversation_row(connection, conversation_id)
        return self._to_conversation(row)

    def list_conversations(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Conversation]:
        if not 1 <= limit <= 200:
            raise ValueError("limit must be between 1 and 200.")
        if offset < 0:
            raise ValueError("offset cannot be negative.")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM conversations
                WHERE archived = 0
                ORDER BY updated_at DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        return [self._to_conversation(row) for row in rows]

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
        return self._to_conversation(row) if row is not None else None

    def update_conversation(
        self,
        conversation_id: str,
        *,
        title: str | None = None,
        model: str | None = None,
        system_prompt: str | None = None,
    ) -> Conversation | None:
        updates: list[str] = []
        values: list[object] = []
        if title is not None:
            title = title.strip()
            if not title:
                raise ValueError("title must not be empty.")
            updates.append("title = ?")
            values.append(title)
        if model is not None:
            updates.append("model = ?")
            values.append(model)
        if system_prompt is not None:
            updates.append("system_prompt = ?")
            values.append(system_prompt or None)
        if not updates:
            return self.get_conversation(conversation_id)
        updates.append("updated_at = ?")
        values.append(_now())
        values.append(conversation_id)
        with self._connect() as connection:
            cursor = connection.execute(
                f"UPDATE conversations SET {', '.join(updates)} WHERE id = ?",
                values,
            )
            if cursor.rowcount == 0:
                return None
            row = self._conversation_row(connection, conversation_id)
        return self._to_conversation(row)

    def update_summary(
        self,
        conversation_id: str,
        *,
        summary: str,
        summarized_through_sequence: int,
    ) -> Conversation:
        if summarized_through_sequence < 0:
            raise ValueError("summarized_through_sequence cannot be negative.")
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE conversations
                SET summary = ?, summarized_through_sequence = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    summary.strip() or None,
                    summarized_through_sequence,
                    _now(),
                    conversation_id,
                ),
            )
            if cursor.rowcount == 0:
                raise KeyError(conversation_id)
            row = self._conversation_row(connection, conversation_id)
        return self._to_conversation(row)

    def delete_conversation(self, conversation_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM conversations WHERE id = ?",
                (conversation_id,),
            )
        return cursor.rowcount > 0

    def append_message(
        self,
        conversation_id: str,
        *,
        role: MessageRole,
        content: str,
        status: MessageStatus = "complete",
        model: str | None = None,
        request_id: str | None = None,
        name: str | None = None,
        tool_call_id: str | None = None,
    ) -> StoredMessage:
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError("unsupported message role.")
        if not content.strip() and status != "generating":
            raise ValueError("message content must not be empty.")
        message_id = uuid4().hex
        timestamp = _now()
        with self._connect() as connection:
            if (
                self._conversation_row(connection, conversation_id, required=False)
                is None
            ):
                raise KeyError(conversation_id)
            next_sequence = cast(
                int,
                connection.execute(
                    """
                    SELECT COALESCE(MAX(sequence), 0) + 1
                    FROM messages WHERE conversation_id = ?
                    """,
                    (conversation_id,),
                ).fetchone()[0],
            )
            connection.execute(
                """
                INSERT INTO messages (
                    id, conversation_id, sequence, role, content, status,
                    model, prompt_tokens, generated_tokens, finish_reason,
                    request_id, name, tool_call_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    conversation_id,
                    next_sequence,
                    role,
                    content,
                    status,
                    model,
                    request_id,
                    name,
                    tool_call_id,
                    timestamp,
                    timestamp,
                ),
            )
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (timestamp, conversation_id),
            )
            row = self._message_row(connection, message_id)
        return self._to_message(row)

    def update_message(
        self,
        message_id: str,
        *,
        content: str | None = None,
        status: MessageStatus | None = None,
        model: str | None = None,
        prompt_tokens: int | None = None,
        generated_tokens: int | None = None,
        finish_reason: str | None = None,
    ) -> StoredMessage | None:
        updates: list[str] = []
        values: list[object] = []
        for column, value in (
            ("content", content),
            ("status", status),
            ("model", model),
            ("prompt_tokens", prompt_tokens),
            ("generated_tokens", generated_tokens),
            ("finish_reason", finish_reason),
        ):
            if value is not None:
                updates.append(f"{column} = ?")
                values.append(value)
        if not updates:
            with self._connect() as connection:
                row = self._message_row(connection, message_id, required=False)
            return self._to_message(row) if row is not None else None
        timestamp = _now()
        updates.append("updated_at = ?")
        values.extend((timestamp, message_id))
        with self._connect() as connection:
            cursor = connection.execute(
                f"UPDATE messages SET {', '.join(updates)} WHERE id = ?",
                values,
            )
            if cursor.rowcount == 0:
                return None
            row = self._message_row(connection, message_id)
            assert row is not None
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (timestamp, row["conversation_id"]),
            )
        return self._to_message(row)

    def list_messages(self, conversation_id: str) -> list[StoredMessage]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM messages
                WHERE conversation_id = ?
                ORDER BY sequence ASC
                """,
                (conversation_id,),
            ).fetchall()
        return [self._to_message(row) for row in rows]

    def get_message_by_request_id(self, request_id: str) -> StoredMessage | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM messages WHERE request_id = ?",
                (request_id,),
            ).fetchone()
        return self._to_message(row) if row is not None else None

    @staticmethod
    def _conversation_row(
        connection: sqlite3.Connection,
        conversation_id: str,
        *,
        required: bool = True,
    ) -> sqlite3.Row | None:
        row = connection.execute(
            "SELECT * FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
        if row is None and required:
            raise KeyError(conversation_id)
        return row

    @staticmethod
    def _message_row(
        connection: sqlite3.Connection,
        message_id: str,
        *,
        required: bool = True,
    ) -> sqlite3.Row | None:
        row = connection.execute(
            "SELECT * FROM messages WHERE id = ?",
            (message_id,),
        ).fetchone()
        if row is None and required:
            raise KeyError(message_id)
        return row

    @staticmethod
    def _to_conversation(row: sqlite3.Row | None) -> Conversation:
        if row is None:
            raise KeyError("conversation row is missing")
        return Conversation(
            id=str(row["id"]),
            title=str(row["title"]),
            model=cast(str | None, row["model"]),
            system_prompt=cast(str | None, row["system_prompt"]),
            summary=cast(str | None, row["summary"]),
            summarized_through_sequence=int(row["summarized_through_sequence"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            archived=bool(row["archived"]),
        )

    @staticmethod
    def _to_message(row: sqlite3.Row | None) -> StoredMessage:
        if row is None:
            raise KeyError("message row is missing")
        return StoredMessage(
            id=str(row["id"]),
            conversation_id=str(row["conversation_id"]),
            sequence=int(row["sequence"]),
            role=cast(MessageRole, row["role"]),
            content=str(row["content"]),
            status=cast(MessageStatus, row["status"]),
            model=cast(str | None, row["model"]),
            prompt_tokens=cast(int | None, row["prompt_tokens"]),
            generated_tokens=cast(int | None, row["generated_tokens"]),
            finish_reason=cast(str | None, row["finish_reason"]),
            request_id=cast(str | None, row["request_id"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            name=cast(str | None, row["name"]),
            tool_call_id=cast(str | None, row["tool_call_id"]),
        )
