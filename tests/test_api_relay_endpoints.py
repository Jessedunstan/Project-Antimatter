"""Tests for Antimatter API Multi-Agent Relay & VRAM Arbiter Endpoints."""

import pytest
from starlette.testclient import TestClient
from antimatter.api.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_api_relay_status(client):
    resp = client.get("/relay/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "vram" in data
    assert data["vram"]["usable_vram_mb"] == 7168
    assert "specialists" in data
    assert "steering" in data["specialists"]
    assert "vision" in data["specialists"]


def test_api_relay_consult_steering(client):
    payload = {
        "caller_id": "test_caller",
        "domain": "steering",
        "prompt": "Move forward along clearance path",
        "context_data": {
            "observation": {
                "tick": 1,
                "raycast_distances": [10.0, 10.0, 10.0, 10.0, 10.0],
                "goal_direction": [0.0, 1.0],
            }
        },
    }
    resp = client.post("/relay/consult", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "Steering Tick 1:" in data["advice"]
    assert data["artifacts"]["action"]["action"] == "MOVE_FORWARD"


def test_api_fuzzing_run(client):
    resp = client.post("/fuzz/run", json={"session_id": "api-fuzz-01", "num_ticks": 8})
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == "api-fuzz-01"
    assert data["total_ticks"] == 8
    assert "synthesized_regression_tests" in data
