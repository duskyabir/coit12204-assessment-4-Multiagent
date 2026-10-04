"""
Integration tests for the new /api/multi-agent/* endpoints, using FastAPI's
TestClient (same pattern as Assessment 3's test_endpoints.py) and running
fully offline against the GenericFakeChatModel fallback (no API key
required - this exercises the heuristic fallback paths end-to-end).
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check_still_works():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_multi_agent_run_rejects_blank_task_description():
    response = client.post("/api/multi-agent/run", json={"task_description": "   "})
    assert response.status_code == 422


def test_multi_agent_run_returns_well_formed_result():
    response = client.post(
        "/api/multi-agent/run",
        json={"task_description": "fix the urgent production bug, then write documentation"},
    )
    assert response.status_code == 200
    body = response.json()

    assert body["task_description"]
    assert isinstance(body["subtasks"], list) and len(body["subtasks"]) >= 1
    assert isinstance(body["subtask_results"], list)
    assert len(body["subtask_results"]) == len(body["subtasks"])
    assert isinstance(body["conversation"], list)
    assert len(body["conversation"]) >= 1
    assert "iterations_used" in body
    assert "max_iterations_exceeded" in body
    assert "planner_fallback" in body

    for item in body["subtask_results"]:
        assert "subtask" in item
        assert "result" in item
        assert "success" in item


def test_multi_agent_state_reflects_run_count():
    before = client.get("/api/multi-agent/state").json()

    client.post(
        "/api/multi-agent/run",
        json={"task_description": "plan a small feature release"},
    )

    after = client.get("/api/multi-agent/state").json()
    assert after["run_count"] == before["run_count"] + 1
    assert after["last_result"] is not None
