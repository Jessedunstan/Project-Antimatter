"""Comprehensive Test Suite for Track 4: Multi-Agent Orchestrator & Antigravity Runtime."""

import asyncio
import pytest
from pathlib import Path

from antimatter.agent.compactor import ContextCompactor
from antimatter.agent.skills.loader import SkillLoader, Skill
from antimatter.agent.subagent import SubagentSpec, WorkspaceMode
from antimatter.agent.supervisor import SubagentSupervisor
from antimatter.agent.skills.mcp_bridge import MCPClient
from antimatter.agent.loop import AgentReActLoop
from antimatter.agent.types import AgentStatus
from antimatter.brain.types import ChatMessage, ChatRole, ToolCall
from antimatter.tools.registry import ToolRegistry


# ============================================================================
# 1. CONTEXT COMPACTOR TESTS
# ============================================================================

def test_context_compactor_token_estimation():
    compactor = ContextCompactor()
    messages = [
        ChatMessage(role=ChatRole.SYSTEM, content="System prompt"),
        ChatMessage(role=ChatRole.USER, content="Hello world"),
    ]
    tokens = compactor.estimate_tokens(messages)
    assert tokens > 0


def test_context_compactor_compression():
    compactor = ContextCompactor(max_token_budget=80, preserve_recent_turns=2)

    # Build history with verbose tool output
    messages = [
        ChatMessage(role=ChatRole.SYSTEM, content="System prompt instructions"),
        ChatMessage(role=ChatRole.USER, content="Initial user task request"),
        ChatMessage(role=ChatRole.ASSISTANT, content="Running tool...", tool_calls=[ToolCall(id="1", name="run_cmd", arguments={})]),
        ChatMessage(role=ChatRole.TOOL, content="X" * 1000, name="run_cmd", tool_call_id="1"),
        ChatMessage(role=ChatRole.ASSISTANT, content="Second thought"),
        ChatMessage(role=ChatRole.USER, content="Recent turn query"),
    ]

    compacted = compactor.compact(messages)

    # Assert head preserved
    assert compacted[0].role == ChatRole.SYSTEM
    assert compacted[1].role == ChatRole.USER

    # Assert total length compressed
    assert compactor.estimate_tokens(compacted) < compactor.estimate_tokens(messages)


# ============================================================================
# 2. SKILL LOADER TESTS
# ============================================================================

def test_skill_loader_parsing(tmp_path):
    skill_dir = tmp_path / "skills" / "database_helper"
    skill_dir.mkdir(parents=True)
    skill_file = skill_dir / "SKILL.md"

    skill_content = (
        "---\n"
        "name: database_optimizer\n"
        "description: Optimize SQLite and PostgreSQL queries\n"
        "version: 1.2.0\n"
        "tags: [sql, performance]\n"
        "---\n"
        "### Workflow Instructions\n"
        "1. Check query execution plan with EXPLAIN QUERY PLAN.\n"
        "2. Add necessary indexes.\n"
    )
    skill_file.write_text(skill_content, encoding="utf-8")

    loader = SkillLoader([tmp_path / "skills"])
    skills = loader.scan_and_load()

    assert "database_optimizer" in skills
    skill = skills["database_optimizer"]
    assert skill.description == "Optimize SQLite and PostgreSQL queries"
    assert skill.version == "1.2.0"
    assert "EXPLAIN QUERY PLAN" in skill.instructions

    # Verify prompt formatting
    prompt_str = loader.format_all_for_system_prompt()
    assert "### Skill: database_optimizer" in prompt_str


# ============================================================================
# 3. SUBAGENT SUPERVISOR & MAILBOX TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_subagent_spawning_and_mailbox(tmp_path):
    supervisor = SubagentSupervisor(workspace_dir=tmp_path)

    # Create dummy file in parent
    (tmp_path / "base.txt").write_text("parent data", encoding="utf-8")

    # Spawn subagent in branched workspace
    spec = SubagentSpec(
        name="code_auditor",
        role="Security Auditor",
        prompt="Audit base.txt and report findings.",
        workspace_mode=WorkspaceMode.BRANCH,
    )
    worker = supervisor.spawn(spec)

    assert worker.id.startswith("sub_")
    assert worker.workspace_dir != tmp_path
    assert (worker.workspace_dir / "base.txt").exists()

    # Test inter-agent messaging: send message to subagent
    ok, status_msg = await supervisor.send_message(
        recipient_id=worker.id,
        message="Please check for SQL injections.",
        sender_id="parent"
    )
    assert ok is True
    assert worker.inbox.qsize() == 1

    received = await worker.inbox.get()
    assert received["message"] == "Please check for SQL injections."
    assert received["sender_id"] == "parent"

    # List subagents
    telemetry = supervisor.list_subagents()
    assert len(telemetry) == 1
    assert telemetry[0].name == "code_auditor"
    assert telemetry[0].workspace_mode == "branch"

    # Kill subagent and verify cleanup
    branch_dir = worker.workspace_dir
    supervisor.kill_subagent(worker.id)
    assert not branch_dir.exists()
    assert len(supervisor.list_subagents()) == 0


# ============================================================================
# 4. MCP TOOL REGISTRATION TESTS
# ============================================================================

def test_mcp_tool_registration():
    registry = ToolRegistry()
    client = MCPClient(server_cmd=["dummy"], server_name="git_server")

    # Mock discovered tools
    client._discovered_tools = [
        {
            "name": "git_status",
            "description": "Show git repository status",
            "inputSchema": {
                "type": "object",
                "properties": {"repo_path": {"type": "string"}},
                "required": ["repo_path"]
            }
        }
    ]

    client.register_tools_into_registry(registry)

    assert registry.has_tool("git_server_git_status")
    defs = registry.get_definitions()
    git_tool = next((d for d in defs if d.function.name == "git_server_git_status"), None)
    assert git_tool is not None
    assert git_tool.function.description == "Show git repository status"
    assert "repo_path" in git_tool.function.parameters["properties"]
