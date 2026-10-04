"""
worker_agent.py
----------------
Worker Agent: the second of the two specialised agents required by
Assessment 4.

Role: execute a single subtask by producing a structured suggestion for
it (due date, priority, reasoning). This reuses the exact single-agent
logic built and tested in Assessment 3 (`app.agent.run_agent`, with its
own tools and its own internal fallback), rather than reimplementing it -
per the brief, this project extends the Assessment 3 codebase, it does
not replace it.

`run_agent` already never raises (it has its own internal heuristic
fallback), but this agent wraps the call in a second, outer try/except as
a last-resort safety net, consistent with the defence-in-depth resilience
approach used across this project (every layer guards the one below it).
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from app.agent import run_agent
from app.messages import AgentMessage

logger = logging.getLogger("agent.worker")


class WorkerAgent:
    """Executes one subtask. Role name: 'worker'."""

    name = "worker"

    def handle_message(self, message: AgentMessage) -> AgentMessage:
        """
        Expects message.content = {"subtask": "<text>"}.
        Always returns an AgentMessage with intent="subtask_result" and
        content = {"subtask": ..., "result": {...}, "success": bool}.
        """
        message.log_received()
        subtask = str(message.content.get("subtask", ""))

        try:
            result: Dict[str, Any] = run_agent(subtask)
            success = not result.get("fallback", False)
        except Exception as exc:  # noqa: BLE001 - last-resort safety net
            logger.error("Worker failed unexpectedly on subtask %r: %s", subtask, exc)
            result = {
                "priority": "Medium",
                "due_in_days": 4,
                "reasoning": "Worker-level fallback: subtask execution raised an "
                "unexpected error.",
                "fallback": True,
                "error": str(exc),
            }
            success = False

        reply = AgentMessage(
            sender=self.name,
            receiver=message.sender,
            intent="subtask_result",
            content={"subtask": subtask, "result": result, "success": success},
            constraints={"format": "json"},
        )
        reply.log_sent()
        return reply
