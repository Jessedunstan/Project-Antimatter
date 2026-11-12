"""Comprehensive Verification Suite for Desktop Client Backend API Endpoints."""

import pytest
from starlette.testclient import TestClient
from antimatter.api.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_api_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["offline"] is True


def test_api_file_operations(client):
    # Write file
    write_resp = client.post(
        "/file/write",
        json={"path": "client_test.txt", "content": "Antimatter Client API Test"},
    )
    assert write_resp.status_code == 200
    assert write_resp.json()["status"] == "ok"

    # Read file
    read_resp = client.get("/file/read?path=client_test.txt")
    assert read_resp.status_code == 200
    assert read_resp.json()["content"] == "Antimatter Client API Test"

    # List files
    files_resp = client.get("/files")
    assert files_resp.status_code == 200
    assert any(f["name"] == "client_test.txt" for f in files_resp.json()["entries"])


def test_api_terminal_execution(client):
    resp = client.post("/terminal/execute", json={"command": "echo 'Client Terminal Online'"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["exit_code"] == 0
    assert "Client Terminal Online" in data["stdout"]


def test_api_subagent_lifecycle(client):
    # 1. Spawn subagent
    spawn_resp = client.post(
        "/subagents/spawn",
        json={
            "name": "tester",
            "role": "QA Bot",
            "prompt": "Run tests",
            "workspace_mode": "inherit",
        },
    )
    assert spawn_resp.status_code == 200
    sub_id = spawn_resp.json()["id"]

    # 2. List subagents
    list_resp = client.get("/subagents")
    assert list_resp.status_code == 200
    assert any(s["id"] == sub_id for s in list_resp.json()["subagents"])

    # 3. Kill subagent
    kill_resp = client.post(f"/subagents/kill?subagent_id={sub_id}")
    assert kill_resp.status_code == 200
    assert kill_resp.json()["status"] == "killed"


def test_api_autocomplete_endpoint(client):
    resp = client.post(
        "/autocomplete",
        json={
            "prefix": "def calculate_area(width, height):\n    return ",
            "suffix": "\n",
            "file_path": "math.py",
        },
    )
    assert resp.status_code == 200
    assert "completion" in resp.json()


def test_api_swarm_run_endpoint(client):
    resp = client.post(
        "/swarm/run",
        json={
            "goal": "Verify system configuration constants",
            "target_file": "antimatter/core/config.py",
            "max_retries": 1,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "task_id" in data
    assert "stages" in data
    assert data["user_goal"] == "Verify system configuration constants"

