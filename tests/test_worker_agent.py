"""
Tests for app/worker_agent.py.

Mocks `app.worker_agent.run_agent` (the Assessment 3 single-agent logic
the Worker Agent reuses) so these tests are fast, deterministic, and don't
depend on any LLM/API key.
"""
from __future__ import annotations

import app.worker_agent as worker_mod
from app.messages import AgentMessage
from app.worker_agent import WorkerAgent


def _request(subtask="write the unit tests"):
    return AgentMessage(
        sender="orchestrator", receiver="worker", intent="execute_subtask",
        content={"subtask": subtask},
    )


def test_worker_returns_subtask_result_on_success(monkeypatch):
    fake_result = {
        "due_in_days": 2, "priority": "High", "reasoning": "urgent",
        "suggested_due_date": "2026-10-05", "tool_called": "classify_priority",
        "fallback": False,
    }
    monkeypatch.setattr(worker_mod, "run_agent", lambda task: fake_result)

    agent = WorkerAgent()
    reply = agent.handle_message(_request("urgent production fix"))

    assert reply.sender == "worker"
    assert reply.receiver == "orchestrator"
    assert reply.intent == "subtask_result"
    assert reply.content["subtask"] == "urgent production fix"
    assert reply.content["result"] == fake_result
    assert reply.content["success"] is True


def test_worker_marks_success_false_when_run_agent_used_its_own_fallback(monkeypatch):
    fake_result = {
        "due_in_days": 4, "priority": "Medium", "reasoning": "fallback used",
        "suggested_due_date": "2026-10-07", "tool_called": None,
        "fallback": True, "error": "LLM call failed",
    }
    monkeypatch.setattr(worker_mod, "run_agent", lambda task: fake_result)

    agent = WorkerAgent()
    reply = agent.handle_message(_request())

    assert reply.content["success"] is False
    assert reply.content["result"]["fallback"] is True


def test_worker_has_outer_safety_net_if_run_agent_raises(monkeypatch):
    def _raise(_task):
        raise RuntimeError("completely unexpected failure")

    monkeypatch.setattr(worker_mod, "run_agent", _raise)

    agent = WorkerAgent()
    reply = agent.handle_message(_request("do something"))

    # Must never raise out of handle_message - always a well-formed reply.
    assert reply.intent == "subtask_result"
    assert reply.content["success"] is False
    assert reply.content["result"]["fallback"] is True
    assert "completely unexpected failure" in reply.content["result"]["error"]
