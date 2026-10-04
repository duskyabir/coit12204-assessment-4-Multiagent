# Multi-Agent Task Assistant (COIT12204 Assessment 4)

This project extends the Assessment 3 Task Assistant Agent into a
**two-agent collaborative system**: a **Planner Agent** that decomposes a
task description into subtasks, and a **Worker Agent** that executes each
subtask (reusing the exact Assessment 3 agent logic). An **Orchestrator**
coordinates the two agents via a structured message-passing protocol, with
built-in resilience (timeouts, max-iteration limits, invalid-receiver
handling, and fallback behaviour).

## Architecture

```
            ┌───────────────┐
  task ───▶ │  Orchestrator │
            └──────┬────────┘
                   │ 1. decompose_task
                   ▼
            ┌───────────────┐
            │ Planner Agent │ ── subtasks_ready ──▶ back to Orchestrator
            └───────────────┘
                   │ 2. execute_subtask (one message per subtask)
                   ▼
            ┌───────────────┐
            │  Worker Agent │ ── subtask_result ──▶ back to Orchestrator
            └───────────────┘
```

All messages between the Orchestrator, Planner and Worker follow the
structured schema defined in `app/messages.py`:

```json
{
  "sender": "orchestrator",
  "receiver": "planner",
  "intent": "decompose_task",
  "content": { "task_description": "..." },
  "constraints": { "max_subtasks": 4 }
}
```

## Project layout

```
app/
  agent.py          Assessment 3: single-agent LangChain logic (reused by the Worker)
  tools.py          Assessment 3: lookup_task_history, classify_priority tools
  state.py          Assessment 3: AgentState (single-agent state tracking)
  main.py           FastAPI app - both the Assessment 3 and Assessment 4 endpoints
  messages.py       NEW: structured AgentMessage schema + routing validation
  planner_agent.py  NEW: Planner Agent (task -> subtasks)
  worker_agent.py   NEW: Worker Agent (subtask -> suggestion, via app.agent.run_agent)
  orchestrator.py   NEW: Orchestrator (routing, resilience, conversation logging)
tests/
  test_messages.py, test_planner_agent.py, test_worker_agent.py,
  test_orchestrator.py, test_multi_agent_endpoint.py
```

## Setup

```bash
pip install -r requirements.txt
```

Optional, for real LLM calls instead of the offline heuristic fallback:

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # preferred
# or
export OPENAI_API_KEY=sk-...
```

With no key set, both the Planner and the Worker fall back to deterministic
heuristics automatically - the system is fully runnable and demoable
offline.

## Running the API

```bash
uvicorn app.main:app --reload
```

- `POST /api/multi-agent/run` - run the full Planner + Worker pipeline on a
  task description.
- `GET /api/multi-agent/state` - inspect run/error counters and the last
  result.
- `POST /api/agent/suggest_due_date`, `GET /api/agent/state` - the original
  Assessment 3 single-agent endpoints (unchanged, kept for continuity and
  because the Worker Agent reuses this exact code path).
- `GET /health` - liveness check.

### Example request

```bash
curl -X POST http://127.0.0.1:8000/api/multi-agent/run \
  -H "Content-Type: application/json" \
  -d '{"task_description": "fix the urgent production outage, write a postmortem, and update the documentation"}'
```

## Running the tests

```bash
pytest -v
```

28 tests cover the message schema, the Planner and Worker agents
individually (fully mocked, offline), and the Orchestrator's resilience
mechanisms: a happy-path run, an invalid-receiver rejection, a worker
crash, a per-call timeout, and a max-iterations cutoff - plus an
end-to-end integration test against the live FastAPI endpoint.

## Resilience mechanisms

- **Message validation** - every message is schema-validated (pydantic)
  and routing-validated (`sender`/`receiver` must be known agents).
- **Invalid receivers** - rejected with `MessageValidationError` and
  logged, instead of crashing with a `KeyError`.
- **Timeouts** - every agent call runs in its own thread with a timeout
  (default 15s); a hung agent returns a synthetic error reply rather than
  blocking the whole pipeline.
- **Max iterations** - the Orchestrator caps the total number of
  message-passing steps (default 10) to prevent runaway loops.
- **Fallback behaviour** - both agents have their own internal heuristic
  fallback (no LLM/API key required), and the Orchestrator has an outer
  safety net that catches any unexpected agent exception.
- **Logging** - every message sent/received, every timeout, every invalid
  route, and every fallback is logged via Python's `logging` module for
  full auditability (see the `agent.message`, `agent.planner`,
  `agent.worker`, and `agent.orchestrator` loggers).

## AI usage

See the technical report for the academic AI-usage declaration covering
this assessment.
