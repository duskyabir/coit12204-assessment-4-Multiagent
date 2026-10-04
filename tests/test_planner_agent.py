"""
Tests for app/planner_agent.py.

Mocks `app.planner_agent.get_llm` (same pattern as Assessment 3's
test_agent.py mocking the LLM) so these tests run fully offline and
deterministically - no real API key or network access required.
"""
from __future__ import annotations

import app.planner_agent as planner_mod
from app.messages import AgentMessage
from app.planner_agent import PlannerAgent, heuristic_decompose


class _FakeLLMResponse:
    """Mimics the `.content` attribute returned by a LangChain chat model."""

    def __init__(self, content):
        self.content = content


class _FakeLLM:
    """
    A stand-in chat model that is plain `Callable`, so LangChain's
    `prompt | llm` composition coerces it into a RunnableLambda. When the
    chain is invoked, this receives the rendered ChatPromptValue and
    returns a fixed, fake response - no real LLM/network call involved.
    """

    def __init__(self, content):
        self._content = content

    def __call__(self, _prompt_value):
        return _FakeLLMResponse(self._content)


def _install_fake_llm(monkeypatch, content):
    monkeypatch.setattr(planner_mod, "get_llm", lambda: _FakeLLM(content))


def test_heuristic_decompose_splits_on_connectors():
    subtasks = heuristic_decompose("fix the bug, write tests and update the docs")
    assert len(subtasks) >= 2
    assert all(isinstance(s, str) and s for s in subtasks)


def test_heuristic_decompose_falls_back_to_generic_steps_for_short_input():
    subtasks = heuristic_decompose("refactor module")
    assert len(subtasks) == 3
    assert any("Investigate" in s for s in subtasks)
    assert any("Implement" in s for s in subtasks)
    assert any("Verify" in s for s in subtasks)


def test_planner_returns_subtasks_from_llm_json(monkeypatch):
    _install_fake_llm(monkeypatch, '{"subtasks": ["Write the parser", "Write the tests"]}')
    agent = PlannerAgent()
    request = AgentMessage(
        sender="orchestrator", receiver="planner", intent="decompose_task",
        content={"task_description": "build a CSV parser"},
    )
    reply = agent.handle_message(request)

    assert reply.sender == "planner"
    assert reply.receiver == "orchestrator"
    assert reply.intent == "subtasks_ready"
    assert reply.content["fallback"] is False
    assert reply.content["subtasks"] == ["Write the parser", "Write the tests"]


def test_planner_falls_back_when_llm_output_is_not_json(monkeypatch):
    _install_fake_llm(monkeypatch, "I cannot help with that.")
    agent = PlannerAgent()
    request = AgentMessage(
        sender="orchestrator", receiver="planner", intent="decompose_task",
        content={"task_description": "urgent production outage, fix asap"},
    )
    reply = agent.handle_message(request)

    assert reply.content["fallback"] is True
    assert reply.content["error"] is not None
    assert len(reply.content["subtasks"]) >= 1


def test_planner_falls_back_when_llm_raises(monkeypatch):
    def _raise():
        raise RuntimeError("no API key configured")

    monkeypatch.setattr(planner_mod, "get_llm", _raise)
    agent = PlannerAgent()
    request = AgentMessage(
        sender="orchestrator", receiver="planner", intent="decompose_task",
        content={"task_description": "plan the release"},
    )
    reply = agent.handle_message(request)

    assert reply.content["fallback"] is True
    assert "no API key configured" in reply.content["error"]
    assert len(reply.content["subtasks"]) >= 1


def test_planner_caps_subtasks_at_four(monkeypatch):
    subtasks = [f"step {i}" for i in range(8)]
    _install_fake_llm(monkeypatch, f'{{"subtasks": {subtasks!r}}}'.replace("'", '"'))
    agent = PlannerAgent()
    request = AgentMessage(
        sender="orchestrator", receiver="planner", intent="decompose_task",
        content={"task_description": "a very large task"},
    )
    reply = agent.handle_message(request)
    assert len(reply.content["subtasks"]) <= 4
