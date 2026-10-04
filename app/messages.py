"""
messages.py
-----------
Structured message schema for agent-to-agent communication (Assessment 4
"Required Features #2"):

    {
      "sender": "planner",
      "receiver": "worker",
      "intent": "execute_subtask",
      "content": { ... },
      "constraints": { ... }
    }

Every message exchanged between the orchestrator, the Planner Agent and the
Worker Agent is an `AgentMessage`. Messages are:
- schema-validated (pydantic: sender/receiver/intent must be non-blank
  strings, content/constraints must be dicts),
- routing-validated (`validate_message` checks sender/receiver are known
  agents in the system - this is what lets the orchestrator detect and
  reject an invalid receiver instead of crashing), and
- logged, via `log_sent`/`log_received`, to the `agent.message` logger so
  every hop of a conversation has an audit trail (used as report evidence).
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger("agent.message")

# The only valid participants in this system. The orchestrator is a
# participant too (it sends the initial "decompose_task" message and
# receives the final replies), so it must be included here.
KNOWN_AGENTS = {"orchestrator", "planner", "worker"}


class AgentMessage(BaseModel):
    """A single structured message passed between agents."""

    message_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    sender: str
    receiver: str
    intent: str
    content: Dict[str, Any] = Field(default_factory=dict)
    constraints: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @field_validator("sender", "receiver", "intent")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must not be blank")
        return value

    def log_sent(self) -> None:
        logger.info(
            "MESSAGE SENT | id=%s | %s -> %s | intent=%s | content=%s",
            self.message_id, self.sender, self.receiver, self.intent, self.content,
        )

    def log_received(self) -> None:
        logger.info(
            "MESSAGE RECEIVED | id=%s | %s -> %s | intent=%s",
            self.message_id, self.sender, self.receiver, self.intent,
        )


class MessageValidationError(Exception):
    """Raised when a message fails routing-level validation."""


def validate_message(message: AgentMessage) -> None:
    """
    Routing-level validation on top of pydantic's field checks: both the
    sender and the receiver must be known agents in this system. This is
    what the orchestrator relies on to catch an invalid receiver before it
    ever reaches `agents[receiver]` and raises a KeyError.
    """
    if message.sender not in KNOWN_AGENTS:
        raise MessageValidationError(f"Unknown sender: {message.sender!r}")
    if message.receiver not in KNOWN_AGENTS:
        raise MessageValidationError(f"Unknown receiver: {message.receiver!r}")
