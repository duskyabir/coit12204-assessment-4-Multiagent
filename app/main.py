"""
main.py
-------
FastAPI application exposing the Task Assistant Agent system.

Assessment 3 endpoints (unchanged, kept for backwards compatibility and
because the Worker Agent reuses this exact logic):
- POST /api/agent/suggest_due_date - single-agent endpoint
- GET  /api/agent/state            - single-agent state
- GET  /health                     - liveness check

Assessment 4 endpoints (new - the Planner + Worker multi-agent system):
- POST /api/multi-agent/run        - runs the full orchestrator pipeline
- GET  /api/multi-agent/state      - multi-agent run counters + last result

Both layers share the same design philosophy: pydantic request/response
models for validation, try/except around every call so nothing the agents
do can crash the API process, structured logging, and a small piece of
state tracked across requests.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.agent import run_agent
from app.orchestrator import Orchestrator
from app.state import agent_state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("agent.api")

app = FastAPI(title="Multi-Agent Task Assistant API", version="2.0.0")


class SuggestDueDateRequest(BaseModel):
    task_description: str = Field(
        ..., min_length=1, max_length=2000,
        description="Free-text description of the task to plan.",
    )


class SuggestDueDateResponse(BaseModel):
    suggested_due_date: str
    priority: str
    due_in_days: int
    reasoning: str
    tool_called: str | None = None
    fallback: bool = False


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/agent/suggest_due_date", response_model=SuggestDueDateResponse)
def suggest_due_date(payload: SuggestDueDateRequest) -> SuggestDueDateResponse:
    """
    Run the agent on the given task description and return a suggested
    due date + priority. Updates and logs the agent's internal state.
    """
    logger.info("Received suggest_due_date request: %r", payload.task_description)

    if not payload.task_description.strip():
        raise HTTPException(status_code=422, detail="task_description must not be blank")

    try:
        result = run_agent(payload.task_description)
    except Exception as exc:  # noqa: BLE001
        # run_agent() already has an internal fallback and should not raise,
        # but this is a last-resort safety net for the API layer.
        agent_state.record_failure(
            action="suggest_due_date", tool_called=None, error=str(exc)
        )
        logger.exception("Unhandled error while running agent")
        raise HTTPException(status_code=500, detail="Agent failed to produce a suggestion") from exc

    if result.get("fallback"):
        agent_state.record_failure(
            action="suggest_due_date",
            tool_called=result.get("tool_called"),
            error=result.get("error", "fallback triggered"),
        )
    else:
        agent_state.record_success(
            action="suggest_due_date",
            tool_called=result.get("tool_called"),
            suggestion=result,
        )

    return SuggestDueDateResponse(
        suggested_due_date=result["suggested_due_date"],
        priority=result["priority"],
        due_in_days=int(result["due_in_days"]),
        reasoning=result.get("reasoning", ""),
        tool_called=result.get("tool_called"),
        fallback=result.get("fallback", False),
    )


@app.get("/api/agent/state")
def get_state() -> dict:
    """Expose current agent state - useful for the demo video / evaluation."""
    return agent_state.to_dict()


# ---------------------------------------------------------------------------
# Assessment 4: Planner + Worker multi-agent system
# ---------------------------------------------------------------------------

_orchestrator = Orchestrator()


class MultiAgentState:
    """
    Lightweight state tracker for the multi-agent endpoint, following the
    same pattern as Assessment 3's AgentState: a module-level singleton,
    updated and logged after every run.
    """

    def __init__(self) -> None:
        self.run_count: int = 0
        self.error_count: int = 0
        self.last_result: Optional[Dict[str, Any]] = None

    def record_run(self, result: Dict[str, Any]) -> None:
        self.run_count += 1
        self.last_result = result

        subtask_results = result.get("subtask_results") or []
        any_subtask_failed = any(not r.get("success", False) for r in subtask_results)
        had_issue = (
            result.get("max_iterations_exceeded")
            or not subtask_results
            or any_subtask_failed
        )

        if had_issue:
            self.error_count += 1
            logger.warning(
                "MULTI-AGENT STATE | run_count=%s | error_count=%s | issue detected "
                "(max_iterations_exceeded=%s, planner_fallback=%s, any_subtask_failed=%s)",
                self.run_count, self.error_count,
                result.get("max_iterations_exceeded"), result.get("planner_fallback"),
                any_subtask_failed,
            )
        else:
            logger.info(
                "MULTI-AGENT STATE | run_count=%s | error_count=%s | ok",
                self.run_count, self.error_count,
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_count": self.run_count,
            "error_count": self.error_count,
            "last_result": self.last_result,
        }


multi_agent_state = MultiAgentState()


class RunTaskRequest(BaseModel):
    task_description: str = Field(
        ..., min_length=1, max_length=2000,
        description="Free-text description of the (possibly broad) task to plan and execute.",
    )


class SubtaskResultItem(BaseModel):
    subtask: str
    result: Dict[str, Any]
    success: bool


class ConversationStepItem(BaseModel):
    step: int
    ok: bool
    detail: str = ""
    request: Dict[str, Any]
    response: Optional[Dict[str, Any]] = None


class RunTaskResponse(BaseModel):
    task_description: str
    subtasks: List[str]
    subtask_results: List[SubtaskResultItem]
    conversation: List[ConversationStepItem]
    iterations_used: int
    max_iterations_exceeded: bool
    planner_fallback: bool


@app.post("/api/multi-agent/run", response_model=RunTaskResponse)
def run_multi_agent(payload: RunTaskRequest) -> RunTaskResponse:
    """
    Run the Planner Agent + Worker Agent pipeline, coordinated by the
    Orchestrator, on the given task description.
    """
    logger.info("Received multi-agent run request: %r", payload.task_description)

    if not payload.task_description.strip():
        raise HTTPException(status_code=422, detail="task_description must not be blank")

    try:
        result = _orchestrator.run(payload.task_description)
        result_dict = result.to_dict()
    except Exception as exc:  # noqa: BLE001
        # The orchestrator is designed to never raise - every agent call is
        # wrapped in its own try/except - but this is a last-resort safety
        # net for the API layer, consistent with the single-agent endpoint.
        logger.exception("Unhandled error while running the multi-agent pipeline")
        multi_agent_state.error_count += 1
        raise HTTPException(
            status_code=500, detail="Multi-agent pipeline failed to produce a result"
        ) from exc

    multi_agent_state.record_run(result_dict)
    return RunTaskResponse(**result_dict)


@app.get("/api/multi-agent/state")
def get_multi_agent_state() -> dict:
    """Expose multi-agent run counters + the last result - useful for the
    demo video / evaluation evidence."""
    return multi_agent_state.to_dict()
