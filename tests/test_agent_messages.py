"""Tests for backend-independent chat messages."""

import pytest

from agent.messages import as_message, message_payload


def test_as_message_accepts_mapping_and_object() -> None:
    mapped = as_message(
        {
            "role": "tool",
            "content": "102",
            "name": "calculator",
            "tool_call_id": "call-1",
        }
    )
    copied = as_message(mapped)

    assert copied is mapped
    assert mapped.role == "tool"
    assert message_payload(mapped)["name"] == "calculator"


def test_as_message_rejects_unknown_role() -> None:
    with pytest.raises(ValueError, match="role"):
        as_message({"role": "system2", "content": "x"})
