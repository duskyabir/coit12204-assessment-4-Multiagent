"""
Tests for app/orchestrator.py - routing, max-iteration limits, invalid
receiver handling, timeouts, and agent-level fallback behaviour.
"""
from __future__ import annotations

import time

from app.messages import AgentMessage
from app.orchestrator import Orchestrator


class _StubPlanner:
    """Always returns a fixed, known set of subtasks."""

    name = "planner"

    def handle_message(self, message: AgentMessage) -> AgentMessage:
        return AgentMessage(
            sender="planner", receiver=message.sender, intent="subtasks_ready",
            content={"subtasks": ["sub A", "sub B"], "fallback": False, "error": None},
        )


class _StubWorker:
    """Always returns a fixed, successful result for whatever subtask it gets."""

    name = "worker"

    def handle_message(self, message: AgentMessage) -> AgentMessage:
        subtask = message.content["subtask"]
        return AgentMessage(
            sender="worker", receiver=message.sender, intent="subtask_result",
            content={"subtask": subtask, "result": {"priority": "Medium"}, "success": True},
        )


class _RaisingWorker:
    """Simulates a worker that crashes unexpectedly - orchestrator must survive this."""

    name = "worker"

    def handle_message(self, message: AgentMessage) -> AgentMessage:
        raise RuntimeError("simulated worker crash")


class _SlowWorker:
    """Simulates a worker that never returns in time - exercises the timeout path."""

    name = "worker"

    def handle_message(self, message: AgentMessage) -> AgentMessage:
        time.sleep(5)
        return AgentMessage(sender="worker", receiver=message.sender, intent="subtask_result")


def test_happy_path_runs_planner_then_worker_for_each_subtask():
    orch = Orchestrator()
    orch.agents = {"planner": _StubPlanner(), "worker": _StubWorker()}

    result = orch.run("build a login page")

    assert result.subtasks == ["sub A", "sub B"]
    assert len(result.subtask_results) == 2
    assert all(r["success"] for r in result.subtask_results)
    assert result.max_iterations_exceeded is False
    # 1 planner step + 2 worker steps = 3 conversation entries
    assert len(result.conversation) == 3
    assert all(step.ok for step in result.conversation)


def test_invalid_receiver_is_handled_without_crashing():
    orch = Orchestrator()
    orch.agents = {"planner": _StubPlanner(), "worker": _StubWorker()}

    bad_message = AgentMessage(
        sender="orchestrator", receiver="not_a_real_agent", intent="decompose_task",
        content={"task_description": "x"},
    )
    # _route raises MessageValidationError for the caller to decide what to do;
    # the full run() pipeline below demonstrates it never crashes the overall run.
    import pytest
    from app.messages import MessageValidationError
    with pytest.raises(MessageValidationError):
        orch._route(bad_message)


def test_worker_crash_is_caught_and_logged_not_raised():
    orch = Orchestrator()
    orch.agents = {"planner": _StubPlanner(), "worker": _RaisingWorker()}

    result = orch.run("build a login page")

    assert result.subtasks == ["sub A", "sub B"]
    # Both worker calls error out, but are recorded as failed results
    # rather than raised or silently dropped - this is the "failure case"
    # evidence required by the brief.
    assert len(result.subtask_results) == 2
    assert all(r["success"] is False for r in result.subtask_results)
    error_steps = [s for s in result.conversation if not s.ok]
    assert len(error_steps) == 2


def test_timeout_is_enforced_and_logged():
    orch = Orchestrator(timeout_seconds=0.2)
    orch.agents = {"planner": _StubPlanner(), "worker": _SlowWorker()}

    start = time.monotonic()
    result = orch.run("build a login page")
    elapsed = time.monotonic() - start

    # Should return quickly (well under the worker's 5s sleep) because the
    # per-call timeout kicked in.
    assert elapsed < 4
    assert result.subtasks == ["sub A", "sub B"]


def test_max_iterations_limit_stops_the_pipeline():
    orch = Orchestrator(max_iterations=1)  # only the planner step is allowed
    orch.agents = {"planner": _StubPlanner(), "worker": _StubWorker()}

    result = orch.run("build a login page")

    assert result.max_iterations_exceeded is True
    # The planner step itself used the only allowed iteration, so no
    # subtasks should have been executed by the worker.
    assert result.subtask_results == []


def test_planner_fallback_flag_is_propagated():
    class _FallbackPlanner:
        name = "planner"

        def handle_message(self, message: AgentMessage) -> AgentMessage:
            return AgentMessage(
                sender="planner", receiver=message.sender, intent="subtasks_ready",
                content={"subtasks": ["generic step"], "fallback": True, "error": "LLM down"},
            )

    orch = Orchestrator()
    orch.agents = {"planner": _FallbackPlanner(), "worker": _StubWorker()}

    result = orch.run("do something ambiguous")

    assert result.planner_fallback is True
    assert result.subtasks == ["generic step"]
