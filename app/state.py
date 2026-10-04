"""
state.py
--------
Internal state management for the Task Assistant Agent.

The assessment requires:
- An AgentState class
- At least three tracked state variables
- State updates after each agent run
- Logging of state transitions
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("agent.state")


@dataclass
class ActionRecord:
    """A single logged action taken by the agent."""
    timestamp: str
    action: str
    tool_called: Optional[str]
    success: bool
    detail: str = ""


@dataclass
class AgentState:
    """
    Tracks the agent's internal state across runs.

    Tracked state variables (>= 3 required by the brief):
    1. last_suggestion   - the most recent structured output produced by the agent
    2. error_count       - running count of failed/erroring agent runs
    3. history           - list of ActionRecord entries (full action history)
    4. last_tool_call    - name of the most recently invoked tool
    """

    last_suggestion: Optional[Dict[str, Any]] = None
    error_count: int = 0
    history: List[ActionRecord] = field(default_factory=list)
    last_tool_call: Optional[str] = None

    # ---- state transition helpers -------------------------------------

    def record_success(self, action: str, tool_called: Optional[str], suggestion: Dict[str, Any]) -> None:
        """Update state after a successful agent run."""
        self.last_suggestion = suggestion
        self.last_tool_call = tool_called
        record = ActionRecord(
            timestamp=self._now(),
            action=action,
            tool_called=tool_called,
            success=True,
            detail=str(suggestion),
        )
        self.history.append(record)
        logger.info(
            "STATE TRANSITION | success | action=%s | tool=%s | suggestion=%s",
            action, tool_called, suggestion,
        )

    def record_failure(self, action: str, tool_called: Optional[str], error: str) -> None:
        """Update state after a failed agent run."""
        self.error_count += 1
        self.last_tool_call = tool_called
        record = ActionRecord(
            timestamp=self._now(),
            action=action,
            tool_called=tool_called,
            success=False,
            detail=error,
        )
        self.history.append(record)
        logger.warning(
            "STATE TRANSITION | failure | action=%s | tool=%s | error=%s | error_count=%s",
            action, tool_called, error, self.error_count,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise state for API responses / logging."""
        return {
            "last_suggestion": self.last_suggestion,
            "error_count": self.error_count,
            "last_tool_call": self.last_tool_call,
            "history_length": len(self.history),
            "history": [record.__dict__ for record in self.history[-10:]],  # last 10
        }

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()


# A module-level singleton so the FastAPI app and the agent share one state
# object for the lifetime of the process. (Optional bonus: swap this for
# SQLite/Redis/JSON persistence.)
agent_state = AgentState()
