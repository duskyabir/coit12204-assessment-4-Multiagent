"""
tools.py
--------
Tools available to the LangChain agent.

The brief requires at least one tool (e.g. lookup_task, summarise_tasks).
We provide two:

1. lookup_task_history  - looks up historical average completion time for
                           a task category (simulated in-memory "database").
2. classify_priority    - deterministic heuristic that classifies a task's
                           priority from its description.

Keeping the tool logic deterministic and dependency-free makes it easy to
unit test independently of the LLM.
"""

from __future__ import annotations

from typing import Dict

from langchain.tools import tool

# Simulated historical data: task category -> average days to complete.
_TASK_HISTORY_DB: Dict[str, int] = {
    "bug_fix": 2,
    "feature": 7,
    "documentation": 3,
    "research": 5,
    "meeting_prep": 1,
    "default": 4,
}

_URGENT_KEYWORDS = {"urgent", "asap", "critical", "outage", "blocked", "production"}
_LOW_PRIORITY_KEYWORDS = {"someday", "minor", "nice to have", "backlog", "cosmetic"}


@tool
def lookup_task_history(task_category: str) -> str:
    """
    Look up the historical average number of days it has taken to complete
    tasks of a given category (e.g. 'bug_fix', 'feature', 'documentation',
    'research', 'meeting_prep'). Returns an integer number of days as a
    string. Falls back to a default estimate if the category is unknown.
    """
    key = task_category.strip().lower().replace(" ", "_")
    days = _TASK_HISTORY_DB.get(key, _TASK_HISTORY_DB["default"])
    return str(days)


@tool
def classify_priority(task_description: str) -> str:
    """
    Classify the priority of a task as 'High', 'Medium', or 'Low' based on
    keywords in its description. This is a deterministic heuristic tool
    (not an LLM call) so it can be tested without mocking.
    """
    text = task_description.lower()
    if any(word in text for word in _URGENT_KEYWORDS):
        return "High"
    if any(word in text for word in _LOW_PRIORITY_KEYWORDS):
        return "Low"
    return "Medium"


def get_tools():
    """Return the list of tools the agent has access to."""
    return [lookup_task_history, classify_priority]
