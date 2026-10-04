"""
planner_agent.py
-----------------
Planner Agent: the first of the two specialised agents required by
Assessment 4.

Role: given a (possibly broad) task description, decompose it into 2-4
concrete subtasks that a Worker Agent can execute independently.

Reuses the same LLM-resolution strategy as Assessment 3's `app.agent.get_llm`
(Claude -> OpenAI -> offline fallback model), so the planner works with a
real API key or completely offline. If the LLM call fails, returns
unparseable output, or raises for any reason, the planner falls back to a
deterministic heuristic decomposition - it must never crash the
orchestrator, matching the resilience approach used throughout this
project.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, List, Optional

from langchain_core.prompts import ChatPromptTemplate

from app.agent import get_llm
from app.messages import AgentMessage

logger = logging.getLogger("agent.planner")

PLANNER_SYSTEM_PROMPT = """You are the Planner Agent in a multi-agent task \
management system. Given a task description, break it into 2 to 4 concrete, \
actionable subtasks that a separate Worker Agent could execute independently.

Reply with ONLY a JSON object (no prose, no markdown fences) in exactly this \
shape:
{{"subtasks": ["<subtask 1>", "<subtask 2>", ...]}}
"""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_SPLIT_RE = re.compile(r",| and | then |;", re.IGNORECASE)


def _coerce_to_text(raw: Any) -> str:
    """Normalise an LLM response's `.content` to plain text (same pattern as
    app.agent._coerce_output_to_text, needed because Claude can return a
    list of content blocks instead of a plain string)."""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts = []
        for block in raw:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
        return "\n".join(parts)
    return str(raw)


def heuristic_decompose(task_description: str) -> List[str]:
    """
    Deterministic, dependency-free fallback decomposition used when the LLM
    call fails. Guarantees at least one (usually three) subtasks are always
    returned, so a missing/invalid API key never stops the pipeline.
    """
    parts = [p.strip() for p in _SPLIT_RE.split(task_description) if p.strip()]
    if len(parts) >= 2:
        return parts[:4]
    return [
        f"Investigate requirements for: {task_description}",
        f"Implement: {task_description}",
        f"Verify and finalise: {task_description}",
    ]


class PlannerAgent:
    """Decomposes a task description into subtasks. Role name: 'planner'."""

    name = "planner"

    def handle_message(self, message: AgentMessage) -> AgentMessage:
        """
        Expects message.content = {"task_description": "<text>"}.
        Always returns an AgentMessage with intent="subtasks_ready" and
        content = {"subtasks": [...], "fallback": bool, "error": str|None}.
        """
        message.log_received()
        task_description = str(message.content.get("task_description", ""))
        fallback = False
        error: Optional[str] = None

        try:
            llm = get_llm()
            prompt = ChatPromptTemplate.from_messages(
                [("system", PLANNER_SYSTEM_PROMPT), ("human", "{input}")]
            )
            chain = prompt | llm
            response = chain.invoke({"input": task_description})
            text = _coerce_to_text(getattr(response, "content", response))
            match = _JSON_RE.search(text)
            if not match:
                raise ValueError(f"No JSON object found in planner output: {text!r}")
            parsed = json.loads(match.group(0))
            subtasks = parsed["subtasks"]
            if not isinstance(subtasks, list) or not subtasks:
                raise ValueError("'subtasks' must be a non-empty list")
            subtasks = [str(s) for s in subtasks][:4]
        except Exception as exc:  # noqa: BLE001 - deliberate broad catch + fallback
            logger.error("Planner LLM call failed, using heuristic fallback: %s", exc)
            subtasks = heuristic_decompose(task_description)
            fallback = True
            error = str(exc)

        reply = AgentMessage(
            sender=self.name,
            receiver=message.sender,
            intent="subtasks_ready",
            content={"subtasks": subtasks, "fallback": fallback, "error": error},
            constraints={"format": "json"},
        )
        reply.log_sent()
        return reply
