"""
orchestrator.py
-----------------
Orchestrator for the Planner + Worker multi-agent system (Assessment 4
"Required Features #3 and #4").

Responsibilities:
- Route every message between agents through a single choke point
  (`_route`), so routing, validation, logging and resilience all live
  in one place.
- Track conversation steps (an ordered, inspectable log of every request/
  reply pair, used directly as report/evaluation evidence).
- Enforce a max-iteration budget so a misbehaving agent (e.g. one that
  kept re-queuing work) cannot loop forever.
- Apply a per-call timeout around each agent invocation.
- Handle invalid receivers / invalid senders without crashing.
- Fall back gracefully if an agent raises an unexpected exception.
- Log every message sent, received, timed out, or rejected.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.messages import AgentMessage, MessageValidationError, validate_message
from app.planner_agent import PlannerAgent
from app.worker_agent import WorkerAgent

logger = logging.getLogger("agent.orchestrator")

DEFAULT_MAX_ITERATIONS = 10
DEFAULT_TIMEOUT_SECONDS = 15.0


@dataclass
class ConversationStep:
    """One logged step: the message sent, the reply received (if any), and
    whether the step succeeded."""

    step: int
    request: Dict[str, Any]
    response: Optional[Dict[str, Any]]
    ok: bool
    detail: str = ""


@dataclass
class OrchestratorResult:
    """The full outcome of one orchestrator run - returned to the API layer
    and used directly as evaluation evidence (conversation log, failures,
    iteration count)."""

    task_description: str
    subtasks: List[str] = field(default_factory=list)
    subtask_results: List[Dict[str, Any]] = field(default_factory=list)
    conversation: List[ConversationStep] = field(default_factory=list)
    iterations_used: int = 0
    max_iterations_exceeded: bool = False
    planner_fallback: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_description": self.task_description,
            "subtasks": self.subtasks,
            "subtask_results": self.subtask_results,
            "conversation": [
                {
                    "step": step.step,
                    "ok": step.ok,
                    "detail": step.detail,
                    "request": step.request,
                    "response": step.response,
                }
                for step in self.conversation
            ],
            "iterations_used": self.iterations_used,
            "max_iterations_exceeded": self.max_iterations_exceeded,
            "planner_fallback": self.planner_fallback,
        }


class Orchestrator:
    """
    Coordinates the Planner and Worker agents to complete one task:

    1. Sends {intent: "decompose_task"} to the planner -> gets back subtasks.
    2. Sends {intent: "execute_subtask"} to the worker for each subtask ->
       collects a result for each.

    Every hop goes through `_route`, which validates the message, applies a
    timeout, and never lets an agent-level exception propagate out of the
    orchestrator.
    """

    def __init__(
        self,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.max_iterations = max_iterations
        self.timeout_seconds = timeout_seconds
        self.agents: Dict[str, Any] = {"planner": PlannerAgent(), "worker": WorkerAgent()}

    # ---- low-level message routing ---------------------------------------

    def _route(self, message: AgentMessage) -> AgentMessage:
        """
        Validate and route a single message to its target agent, with a
        timeout. Raises `MessageValidationError` for bad routing (unknown
        sender/receiver) so the caller can log/handle it; for every other
        failure mode (agent exception, timeout) this returns a synthetic
        "error" reply instead of raising, so the orchestrator loop itself
        never crashes.
        """
        validate_message(message)  # raises MessageValidationError if unknown agent
        agent = self.agents[message.receiver]
        message.log_sent()

        # Not using the pool as a context manager: its __exit__ would call
        # shutdown(wait=True), which blocks until a hung worker thread
        # actually finishes - defeating the point of the timeout below.
        # shutdown(wait=False) lets the orchestrator move on immediately
        # while the stuck thread is abandoned in the background (Python
        # threads cannot be forcibly killed).
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            future = pool.submit(agent.handle_message, message)
            try:
                return future.result(timeout=self.timeout_seconds)
            except FutureTimeoutError:
                logger.error(
                    "TIMEOUT | receiver=%s | intent=%s | after %ss",
                    message.receiver, message.intent, self.timeout_seconds,
                )
                return AgentMessage(
                    sender=message.receiver,
                    receiver=message.sender,
                    intent="error",
                    content={"error": f"timeout after {self.timeout_seconds}s"},
                )
            except Exception as exc:  # noqa: BLE001 - defence in depth
                logger.error("AGENT ERROR | receiver=%s | %s", message.receiver, exc)
                return AgentMessage(
                    sender=message.receiver,
                    receiver=message.sender,
                    intent="error",
                    content={"error": str(exc)},
                )
        finally:
            pool.shutdown(wait=False)

    def _step(self, result: OrchestratorResult, message: AgentMessage) -> AgentMessage:
        """
        Run one routed step inside the max-iterations budget, recording it
        in the conversation log either way.
        """
        result.iterations_used += 1
        step_number = result.iterations_used

        if result.iterations_used > self.max_iterations:
            result.max_iterations_exceeded = True
            logger.warning("MAX ITERATIONS EXCEEDED | limit=%s", self.max_iterations)
            result.conversation.append(
                ConversationStep(
                    step=step_number,
                    request=message.model_dump(),
                    response=None,
                    ok=False,
                    detail="max_iterations_exceeded",
                )
            )
            raise RuntimeError("max_iterations_exceeded")

        try:
            reply = self._route(message)
            result.conversation.append(
                ConversationStep(
                    step=step_number,
                    request=message.model_dump(),
                    response=reply.model_dump(),
                    ok=reply.intent != "error",
                    detail="" if reply.intent != "error" else str(reply.content.get("error", "")),
                )
            )
            return reply
        except MessageValidationError as exc:
            logger.error("INVALID MESSAGE | %s", exc)
            result.conversation.append(
                ConversationStep(
                    step=step_number,
                    request=message.model_dump(),
                    response=None,
                    ok=False,
                    detail=str(exc),
                )
            )
            raise

    # ---- public entrypoint -------------------------------------------------

    def run(self, task_description: str) -> OrchestratorResult:
        """Run the full Planner -> Worker(s) pipeline for one task description."""
        result = OrchestratorResult(task_description=task_description)

        plan_request = AgentMessage(
            sender="orchestrator",
            receiver="planner",
            intent="decompose_task",
            content={"task_description": task_description},
            constraints={"max_subtasks": 4},
        )

        try:
            plan_reply = self._step(result, plan_request)
        except (RuntimeError, MessageValidationError):
            # Max iterations hit on the very first step, or an invalid route.
            # Either way, return what we have rather than raising to the API layer.
            return result

        if plan_reply.intent == "error":
            # Planner call errored/timed out even after its own internal
            # fallback (should be rare - PlannerAgent only reaches this via
            # the orchestrator's outer safety net). Nothing more to do.
            return result

        subtasks = [str(s) for s in plan_reply.content.get("subtasks", [])]
        result.subtasks = subtasks
        result.planner_fallback = bool(plan_reply.content.get("fallback", False))

        for subtask in subtasks:
            work_request = AgentMessage(
                sender="orchestrator",
                receiver="worker",
                intent="execute_subtask",
                content={"subtask": subtask},
                constraints={"format": "json"},
            )
            try:
                work_reply = self._step(result, work_request)
            except (RuntimeError, MessageValidationError):
                break

            if work_reply.intent == "error":
                # The worker itself crashed or timed out (caught by _route's
                # defence-in-depth, below the worker's own internal
                # fallback). Record it as a failed subtask result instead
                # of silently dropping it, so this is visible as a
                # documented failure case rather than a vanished subtask.
                result.subtask_results.append(
                    {
                        "subtask": subtask,
                        "result": {"error": work_reply.content.get("error", "unknown error")},
                        "success": False,
                    }
                )
            else:
                result.subtask_results.append(work_reply.content)

        return result
