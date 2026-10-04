"""
agent.py
--------
The single LangChain-based agent for this assessment: a Task Assistant
Agent that, given a task description, suggests a due date and priority.

Satisfies the brief's "Required Features #1":
- Uses LangChain (AgentExecutor + tool-calling agent)
- Uses at least one tool (lookup_task_history, classify_priority)
- Uses a prompt template
- Produces structured output (a dict with due_date, priority, reasoning)
- Handles errors gracefully (falls back to a heuristic if the LLM/agent
  call fails, and always returns a well-formed result)
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.tools import get_tools, classify_priority

logger = logging.getLogger("agent.core")

SYSTEM_PROMPT = """You are a task-planning assistant embedded in a project
management tool. Given a task description, you must:

1. Use the `lookup_task_history` tool to find the typical number of days a
   task of this category takes (guess the category yourself: bug_fix,
   feature, documentation, research, meeting_prep, or default).
2. Use the `classify_priority` tool on the task description.
3. Reply with ONLY a JSON object (no prose, no markdown fences) in exactly
   this shape:
   {{"due_in_days": <integer>, "priority": "<High|Medium|Low>", "reasoning": "<short reasoning>"}}
"""

_PROMPT_TEMPLATE: Optional[ChatPromptTemplate] = None


def get_prompt_template() -> ChatPromptTemplate:
    """LangChain prompt template used by the agent (built once, cached)."""
    global _PROMPT_TEMPLATE
    if _PROMPT_TEMPLATE is None:
        _PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
    return _PROMPT_TEMPLATE


def get_llm() -> BaseChatModel:
    """
    Return the chat model used by the agent.

    Priority order:
    1. If ANTHROPIC_API_KEY is set, use a real Claude chat model (requires
       the optional `langchain-anthropic` package).
    2. Else if OPENAI_API_KEY is set, use a real OpenAI chat model
       (requires the optional `langchain-openai` package).
    3. Otherwise, fall back to a deterministic fake chat model so the app
       is runnable and demoable without any API key or network access -
       this also makes the agent trivially mockable in tests via
       monkeypatch.
    """
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        from langchain_anthropic import ChatAnthropic  # imported lazily, optional dep

        return ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0, api_key=anthropic_key)

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        from langchain_openai import ChatOpenAI  # imported lazily, optional dep

        return ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_key)

    logger.info("No ANTHROPIC_API_KEY or OPENAI_API_KEY set - using offline fallback chat model")
    return GenericFakeChatModel(messages=iter([]))


def build_agent_executor(llm: Optional[BaseChatModel] = None) -> AgentExecutor:
    """Construct the LangChain AgentExecutor with tools + prompt template."""
    llm = llm or get_llm()
    tools = get_tools()
    prompt = get_prompt_template()
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _coerce_output_to_text(raw_output: Any) -> str:
    """
    Normalise the AgentExecutor's `output` field to a plain string.

    Different chat models return this differently: OpenAI-style models
    return a plain string, while Claude (via langchain-anthropic) can
    return a list of content blocks, e.g.
    [{"type": "text", "text": "..."}], especially when the model's reply
    mixes text with tool-use content. This function accepts either shape
    and always returns a single string for downstream JSON parsing.
    """
    if isinstance(raw_output, str):
        return raw_output

    if isinstance(raw_output, list):
        parts = []
        for block in raw_output:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
        return "\n".join(parts)

    return str(raw_output)


def _parse_structured_output(raw_text: Any) -> Dict[str, Any]:
    """Extract the JSON object the agent was instructed to return."""
    text = _coerce_output_to_text(raw_text)
    match = _JSON_RE.search(text)
    if not match:
        raise ValueError(f"No JSON object found in agent output: {text!r}")
    return json.loads(match.group(0))


def _heuristic_fallback(task_description: str) -> Dict[str, Any]:
    """
    Deterministic, dependency-free fallback used when the LLM/agent call
    fails or returns unparseable output. Guarantees the endpoint always
    returns a usable suggestion instead of a hard failure.
    """
    priority = classify_priority.invoke(task_description)
    due_in_days = {"High": 1, "Medium": 4, "Low": 7}.get(priority, 4)
    return {
        "due_in_days": due_in_days,
        "priority": priority,
        "reasoning": "Fallback heuristic used because the agent call failed or "
        "returned an unparseable response.",
        "fallback": True,
    }


def run_agent(task_description: str, executor: Optional[AgentExecutor] = None) -> Dict[str, Any]:
    """
    Run the agent on a single task description and return a structured
    suggestion. Always returns a well-formed dict - never raises - so the
    FastAPI layer can rely on a stable contract. Returns a tuple-like dict
    including metadata about which tool was last invoked, for state
    tracking.
    """
    tool_called: Optional[str] = None

    try:
        executor = executor or build_agent_executor()
        result = executor.invoke({"input": task_description})
        raw_output = result.get("output", "")

        # Track which tool(s) were used, if the executor exposes intermediate steps.
        steps = result.get("intermediate_steps", [])
        if steps:
            tool_called = steps[-1][0].tool

        structured = _parse_structured_output(raw_output)
        structured.setdefault("fallback", False)

        due_date = (
            datetime.now(timezone.utc) + timedelta(days=int(structured["due_in_days"]))
        ).date().isoformat()
        structured["suggested_due_date"] = due_date
        structured["tool_called"] = tool_called
        return structured

    except Exception as exc:  # noqa: BLE001 - deliberate broad catch + graceful fallback
        logger.error("Agent run failed, using fallback: %s", exc)
        structured = _heuristic_fallback(task_description)
        due_date = (
            datetime.now(timezone.utc) + timedelta(days=structured["due_in_days"])
        ).date().isoformat()
        structured["suggested_due_date"] = due_date
        structured["tool_called"] = tool_called
        structured["error"] = str(exc)
        return structured
