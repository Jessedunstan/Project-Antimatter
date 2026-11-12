"""Unit tests for Tool-Call Hardening, Argument Coercion, and ReAct Loop Resilience."""

from pathlib import Path
from typing import List, Optional
import pytest

from antimatter.agent.loop import AgentReActLoop
from antimatter.agent.types import AgentStatus
from antimatter.brain.client import LocalLLMClient
from antimatter.brain.tool_extractor import ToolCallExtractor
from antimatter.brain.types import ChatMessage, ChatRole, ToolCall, ToolDefinition
from antimatter.tools import FileSystemManager, MultiLinePatcher, build_default_tool_registry
from antimatter.tools.registry import ToolRegistry
from antimatter.tools.terminal import TerminalSandbox


@pytest.mark.asyncio
async def test_nested_argument_unwrapping_and_aliases():
    registry = ToolRegistry()

    @registry.register(name="edit_file", description="Edit a file.")
    def edit_file(path: str, target: str, replacement: str, replace_all: bool = False) -> str:
        return f"{path}:{target}->{replacement} (all={replace_all})"

    # 1. Nested under 'parameters'
    tc1 = ToolCall(
        id="c1",
        name="edit_file",
        arguments={
            "parameters": {
                "path": "app.py",
                "target": "old",
                "replacement": "new",
                "replace_all": "true",
            }
        },
    )
    res1 = await registry.dispatch(tc1)
    assert res1 == "app.py:old->new (all=True)"

    # 2. Aliases ('file_path', 'old_str', 'new_str') wrapped under 'arguments'
    tc2 = ToolCall(
        id="c2",
        name="edit_file",
        arguments={
            "arguments": {
                "file_path": "utils.py",
                "old_str": "x = 1",
                "new_str": "x = 2",
            }
        },
    )
    res2 = await registry.dispatch(tc2)
    assert res2 == "utils.py:x = 1->x = 2 (all=False)"


@pytest.mark.asyncio
async def test_type_coercion_int_float_bool():
    registry = ToolRegistry()

    @registry.register(name="slice_lines", description="Slice lines.")
    def slice_lines(
        path: str,
        start_line: int,
        end_line: Optional[int] = None,
        ratio: float = 1.0,
        strict: bool = False,
    ) -> dict:
        return {
            "path": path,
            "start_line": start_line,
            "end_line": end_line,
            "ratio": ratio,
            "strict": strict,
            "types": [
                type(start_line).__name__,
                type(end_line).__name__,
                type(ratio).__name__,
                type(strict).__name__,
            ],
        }

    tc = ToolCall(
        id="c3",
        name="slice_lines",
        arguments={
            "path": "main.py",
            "start_line": " 10 ",
            "end_line": "25",
            "ratio": "0.75",
            "strict": "yes",
        },
    )
    raw = await registry.dispatch(tc)
    assert '"start_line": 10' in raw
    assert '"end_line": 25' in raw
    assert '"ratio": 0.75' in raw
    assert '"strict": true' in raw


@pytest.mark.asyncio
async def test_missing_required_arguments_actionable_error():
    registry = ToolRegistry()

    @registry.register(name="write_file", description="Write a file.")
    def write_file(path: str, content: str) -> str:
        return "ok"

    tc = ToolCall(id="c4", name="write_file", arguments={"path": "only_path.py"})
    res = await registry.dispatch(tc)
    assert "Error executing 'write_file'" in res
    assert "Missing required argument(s) for tool 'write_file'" in res
    assert "content: str" in res
    assert "Provided keys: ['path']" in res


def test_tool_registry_subset_and_allowed_tools(tmp_path: Path):
    registry = build_default_tool_registry(tmp_path)
    core_names = ["read_file", "edit_file", "write_file", "run_command", "list_directory"]

    defs = registry.get_definitions(allowed_tools=core_names)
    assert {d.function.name for d in defs} == set(core_names)

    sub = registry.subset(core_names)
    assert {d.function.name for d in sub.get_definitions()} == set(core_names)
    assert not sub.has_tool("rollback_checkpoint")


def test_normalize_tool_arguments_schema_mirroring():
    # Model mirrors JSON schema structure inside arguments
    mirrored = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "value": "buggy.py"},
            "start_line": 1,
        },
        "required": ["path"],
    }
    norm = LocalLLMClient._normalize_tool_arguments(mirrored)
    assert norm == {"path": "buggy.py", "start_line": 1}

    # Top-level real keys alongside accidental 'properties' echo
    mixed = {
        "path": "real.py",
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }
    norm2 = LocalLLMClient._normalize_tool_arguments(mixed)
    assert norm2 == {"path": "real.py"}


def test_windows_powershell_call_operator_prepended(tmp_path: Path):
    sandbox = TerminalSandbox(working_dir=tmp_path)
    sandbox.is_windows = True

    _, args_quoted = sandbox._prepare_shell('".\\.venv\\Scripts\\pytest.exe" -q')
    assert '& ".\\.venv\\Scripts\\pytest.exe" -q' in args_quoted[-1]

    _, args_single = sandbox._prepare_shell("'.\\.venv\\Scripts\\python.exe' -V")
    assert "& '.\\.venv\\Scripts\\python.exe' -V" in args_single[-1]

    _, args_already = sandbox._prepare_shell('& "python.exe" -V')
    assert args_already[-1].count("&") == 1


class _StubMalformedThenFixLLM:
    """Stub LLM that emits a malformed <tool_call> on step 1, then a valid tool call on step 2, then finishes."""

    def __init__(self):
        self.calls = 0

    async def chat(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[ToolDefinition]] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
    ) -> ChatMessage:
        self.calls += 1
        if self.calls == 1:
            # Malformed XML tool call (broken JSON) + empty tool call for required-arg tool
            return ChatMessage(
                role=ChatRole.ASSISTANT,
                content='<tool_call>{"name": "read_file", "arguments": {BROKEN_JSON}}</tool_call>',
                tool_calls=[ToolCall(id="empty1", name="read_file", arguments={})],
            )
        if self.calls == 2:
            # Valid tool call after receiving retry prompt
            return ChatMessage(
                role=ChatRole.ASSISTANT,
                content="Reading target.py now.",
                tool_calls=[ToolCall(id="valid1", name="read_file", arguments={"path": "target.py"})],
            )
        return ChatMessage(
            role=ChatRole.ASSISTANT,
            content="Inspected target.py and verified everything.",
        )


@pytest.mark.asyncio
async def test_react_loop_filters_empty_calls_and_retries_malformed(tmp_path: Path):
    (tmp_path / "target.py").write_text("x = 42\n", encoding="utf-8")
    stub_llm = _StubMalformedThenFixLLM()

    loop = AgentReActLoop(
        llm_client=stub_llm,  # type: ignore[arg-type]
        workspace_dir=tmp_path,
        max_steps=5,
        allowed_tools=["read_file", "edit_file"],
    )
    try:
        traj = await loop.run("Inspect target.py")
        assert traj.status == AgentStatus.COMPLETED
        assert stub_llm.calls == 3
        assert len(traj.steps) == 3
        # Step 1 filtered out the empty read_file call, Step 2 executed the valid read_file call
        assert len(traj.steps[0].tool_calls) == 0
        assert len(traj.steps[1].tool_calls) == 1
        assert "x = 42" in traj.steps[1].tool_results["valid1"]
    finally:
        loop.close()


def test_indentation_aware_multiline_patcher(tmp_path: Path):
    fs = FileSystemManager(tmp_path)
    patcher = MultiLinePatcher(fs)
    code = (
        "def paginate(items, page, page_size):\n"
        "    start = page * page_size\n"
        "    end = start + page_size\n"
        "    return items[start:end]\n"
    )
    (tmp_path / "paginator.py").write_text(code, encoding="utf-8")

    # Target omits the 4-space indentation present in paginator.py
    diff, updated = patcher.apply_patch(
        "paginator.py",
        target_block="start = page * page_size\nend = start + page_size",
        replacement_block="start = (page - 1) * page_size\nend = start + page_size",
    )
    assert "    start = (page - 1) * page_size\n    end = start + page_size" in updated
    assert "-    start = page * page_size" in diff
    assert "+    start = (page - 1) * page_size" in diff


def test_balanced_json_extraction_and_deduplication(tmp_path: Path):
    registry = build_default_tool_registry(tmp_path)
    defs = registry.get_definitions(allowed_tools=["read_file", "edit_file"])

    # Bare JSON with nested braces + duplicate call
    raw_output = (
        '{"name": "read_file", "arguments": {"path": "a.py"}}\n'
        '{"name": "read_file", "arguments": {"path": "a.py"}}\n'
        '{"name": "edit_file", "arguments": {"path": "a.py", "target": "x = {1: 2}", "replacement": "x = {1: 3}"}}'
    )
    cleaned, tcs = ToolCallExtractor.extract_from_text(raw_output, available_tools=defs)
    assert cleaned == ""
    assert len(tcs) == 2
    assert tcs[0].name == "read_file"
    assert tcs[1].name == "edit_file"
    assert tcs[1].arguments["replacement"] == "x = {1: 3}"

