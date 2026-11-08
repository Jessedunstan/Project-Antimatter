"""Test suite for Autonomous Agent ReAct Loop, FastAPI Server, and CLI."""

from __future__ import annotations
import tempfile
from pathlib import Path
import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from antimatter.core.config import AntimatterSettings, LLMBackendType
from antimatter.brain.client import LocalLLMClient
from antimatter.brain.types import ChatMessage, ChatRole, ToolCall
from antimatter.tools.registry import ToolRegistry
from antimatter.agent.loop import AgentReActLoop
from antimatter.agent.types import AgentStatus
from antimatter.api.app import app
from antimatter.cli import main


@pytest.mark.asyncio
async def test_agent_react_execution_flow():
    """Verify autonomous ReAct agent dispatches tools and finishes task."""
    settings = AntimatterSettings(llm_backend=LLMBackendType.MOCK)
    client = LocalLLMClient(settings)
    registry = ToolRegistry()

    @registry.register(name="mock_write", description="Write mock data")
    def mock_write(key: str, val: str) -> str:
        return f"Wrote {key}={val}"

    # Override mock client to simulate 1 tool call then final answer
    call_count = 0

    async def mock_chat(messages, tools=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ChatMessage(
                role=ChatRole.ASSISTANT,
                content="I will save the configuration.",
                tool_calls=[ToolCall(id="tc_1", name="mock_write", arguments={"key": "theme", "val": "dark"})],
            )
        else:
            return ChatMessage(
                role=ChatRole.ASSISTANT,
                content="Configuration saved successfully.",
            )

    client.chat = mock_chat

    agent = AgentReActLoop(llm_client=client, tool_registry=registry, max_steps=5)
    trajectory = await agent.run("Configure application theme to dark mode.")

    assert trajectory.status == AgentStatus.COMPLETED
    assert len(trajectory.steps) == 2
    assert "tc_1" in trajectory.steps[0].tool_results
    assert trajectory.steps[0].tool_results["tc_1"] == "Wrote theme=dark"
    assert trajectory.final_response == "Configuration saved successfully."


def test_fastapi_endpoints():
    """Verify local FastAPI health and code intelligence endpoints."""
    client = TestClient(app)

    # 1. Health check
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["offline"] is True

    # 2. Search endpoint
    res_search = client.get("/search?q=database")
    assert res_search.status_code == 200
    assert "results" in res_search.json()

    # 3. Symbols endpoint
    res_sym = client.get("/symbols?file_path=antimatter/context/types.py")
    assert res_sym.status_code == 200
    assert "symbols" in res_sym.json()


def test_cli_commands():
    """Verify Click CLI commands function deterministically."""
    runner = CliRunner()

    # 1. Info
    result = runner.invoke(main, ["info"])
    assert result.exit_code == 0
    assert "Antimatter Sovereign Engine Status" in result.output

    # 2. Symbols
    result_sym = runner.invoke(main, ["symbols", "antimatter/context/types.py"])
    assert result_sym.exit_code == 0
    assert "Symbol Hierarchy" in result_sym.output

    # 3. Check diagnostics
    result_check = runner.invoke(main, ["check", "antimatter/context/types.py"])
    assert result_check.exit_code == 0
    assert "No syntax issues" in result_check.output
