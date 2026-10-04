"""
Tests for app/messages.py - the structured agent-to-agent message schema.
"""
from __future__ import annotations

import logging

import pytest
from pydantic import ValidationError

from app.messages import AgentMessage, MessageValidationError, validate_message


def _make(sender="orchestrator", receiver="planner", intent="decompose_task", **kw):
    return AgentMessage(sender=sender, receiver=receiver, intent=intent, **kw)


def test_message_has_expected_shape():
    msg = _make(content={"task_description": "fix the bug"}, constraints={"max_subtasks": 4})
    assert msg.sender == "orchestrator"
    assert msg.receiver == "planner"
    assert msg.intent == "decompose_task"
    assert msg.content == {"task_description": "fix the bug"}
    assert msg.constraints == {"max_subtasks": 4}
    # auto-generated fields
    assert isinstance(msg.message_id, str) and msg.message_id
    assert isinstance(msg.timestamp, str) and msg.timestamp


def test_content_and_constraints_default_to_empty_dict():
    msg = _make()
    assert msg.content == {}
    assert msg.constraints == {}


@pytest.mark.parametrize("field_name", ["sender", "receiver", "intent"])
def test_blank_fields_are_rejected(field_name):
    kwargs = {"sender": "orchestrator", "receiver": "planner", "intent": "decompose_task"}
    kwargs[field_name] = "   "
    with pytest.raises(ValidationError):
        AgentMessage(**kwargs)


def test_validate_message_accepts_known_agents():
    msg = _make(sender="orchestrator", receiver="planner")
    validate_message(msg)  # should not raise


def test_validate_message_rejects_unknown_receiver():
    msg = _make(sender="orchestrator", receiver="not_a_real_agent")
    with pytest.raises(MessageValidationError):
        validate_message(msg)


def test_validate_message_rejects_unknown_sender():
    msg = _make(sender="not_a_real_agent", receiver="planner")
    with pytest.raises(MessageValidationError):
        validate_message(msg)


def test_log_sent_and_log_received_do_not_raise(caplog):
    caplog.set_level(logging.INFO, logger="agent.message")
    msg = _make(content={"task_description": "do a thing"})
    msg.log_sent()
    msg.log_received()
    assert any("MESSAGE SENT" in r.message for r in caplog.records)
    assert any("MESSAGE RECEIVED" in r.message for r in caplog.records)
