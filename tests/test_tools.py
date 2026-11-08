"""Comprehensive Test Suite for Track 2: Sandboxed Tooling & Deterministic Workspace."""

import asyncio
import tempfile
import pytest
from pathlib import Path

from antimatter.tools.registry import ToolRegistry, RegisteredTool
from antimatter.tools.terminal import TerminalSandbox, CommandResult
from antimatter.tools.fs_manager import FileSystemManager, SecurityException
from antimatter.tools.patcher import MultiLinePatcher, PatchError
from antimatter.tools import build_default_tool_registry
from antimatter.brain.types import ToolCall


# ============================================================================
# 1. TOOL REGISTRY TESTS
# ============================================================================

def test_tool_registry_schema_generation():
    registry = ToolRegistry()

    @registry.register(name="calculate_sum", description="Add two numbers together")
    def calculate_sum(a: int, b: int = 10) -> int:
        return a + b

    definitions = registry.get_definitions()
    assert len(definitions) == 1
    d = definitions[0].function
    assert d.name == "calculate_sum"
    assert d.description == "Add two numbers together"
    assert "a" in d.parameters["properties"]
    assert "b" in d.parameters["properties"]
    assert d.parameters["required"] == ["a"]


@pytest.mark.asyncio
async def test_tool_registry_dispatch():
    registry = ToolRegistry()

    @registry.register(name="echo_message")
    async def echo_message(msg: str) -> str:
        return f"ECHO: {msg}"

    call = ToolCall(id="call_1", name="echo_message", arguments={"msg": "Antimatter"})
    res = await registry.dispatch(call)
    assert res == "ECHO: Antimatter"


# ============================================================================
# 2. TERMINAL SANDBOX TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_terminal_basic_execution(tmp_path):
    sandbox = TerminalSandbox(working_dir=tmp_path, max_timeout_sec=5)
    result = await sandbox.execute("echo 'Sovereign Antimatter'")
    assert result.exit_code == 0
    assert "Sovereign Antimatter" in result.stdout
    assert not result.timed_out
    assert result.duration_ms > 0


@pytest.mark.asyncio
async def test_terminal_timeout_enforcement(tmp_path):
    sandbox = TerminalSandbox(working_dir=tmp_path, max_timeout_sec=1)
    # 5 second sleep on PowerShell vs 1s timeout
    result = await sandbox.execute("Start-Sleep -Seconds 5", timeout_sec=1)
    assert result.timed_out is True
    assert result.exit_code == -1
    assert "timed out" in result.stderr


@pytest.mark.asyncio
async def test_terminal_streaming(tmp_path):
    sandbox = TerminalSandbox(working_dir=tmp_path, max_timeout_sec=5)
    lines = []
    async for line in sandbox.stream("echo line1; echo line2"):
        lines.append(line.strip())
    combined = " ".join(lines)
    assert "line1" in combined
    assert "line2" in combined


# ============================================================================
# 3. FILE SYSTEM & ROLLBACK SNAPSHOT TESTS
# ============================================================================

def test_fs_security_sandbox(tmp_path):
    fs = FileSystemManager(tmp_path)

    # Valid internal path
    safe = fs.resolve_safe_path("src/core.py")
    assert safe == (tmp_path / "src" / "core.py").resolve()

    # Path traversal attack attempt
    with pytest.raises(SecurityException):
        fs.resolve_safe_path("../../outside_secret.txt")


def test_fs_atomic_write_and_read(tmp_path):
    fs = FileSystemManager(tmp_path)
    file_rel = "app/config.json"
    content = '{"mode": "offline", "version": 1}'

    bytes_written = fs.write_atomic(file_rel, content)
    assert bytes_written == len(content)

    read_back = fs.read_file(file_rel)
    assert read_back == content


def test_fs_checkpoint_and_rollback(tmp_path):
    fs = FileSystemManager(tmp_path)
    file_rel = "code.py"
    original_code = "def start():\n    return 'v1'\n"
    fs.write_atomic(file_rel, original_code)

    # Start checkpoint transaction
    cp_id = fs.create_checkpoint("Update to v2 and add new file")

    # Modify existing file and create a new file under this checkpoint
    modified_code = "def start():\n    return 'v2_modified'\n"
    fs.write_atomic(file_rel, modified_code, checkpoint_id=cp_id)

    new_file_rel = "new_module.py"
    fs.write_atomic(new_file_rel, "# Brand new", checkpoint_id=cp_id)

    # Verify mutations are active
    assert fs.read_file(file_rel) == modified_code
    assert (tmp_path / new_file_rel).exists()

    # Rollback checkpoint
    success, msg = fs.rollback(cp_id)
    assert success is True

    # Assert original state restored and new file removed
    assert fs.read_file(file_rel) == original_code
    assert not (tmp_path / new_file_rel).exists()


# ============================================================================
# 4. MULTI-LINE PATCHER TESTS
# ============================================================================

def test_patcher_exact_application(tmp_path):
    fs = FileSystemManager(tmp_path)
    patcher = MultiLinePatcher(fs)

    file_rel = "service.py"
    content = (
        "class Service:\n"
        "    def run(self):\n"
        "        print('old logic')\n"
        "        return False\n"
    )
    fs.write_atomic(file_rel, content)

    target = (
        "        print('old logic')\n"
        "        return False"
    )
    replacement = (
        "        print('new logic')\n"
        "        return True"
    )

    diff, updated = patcher.apply_patch(file_rel, target, replacement)

    assert "old logic" not in updated
    assert "new logic" in updated
    assert "-        print('old logic')" in diff
    assert "+        print('new logic')" in diff


def test_patcher_crlf_lf_tolerance(tmp_path):
    fs = FileSystemManager(tmp_path)
    patcher = MultiLinePatcher(fs)

    file_rel = "crlf_file.py"
    # Write with Windows CRLF
    with open(tmp_path / file_rel, "wb") as f:
        f.write(b"def test():\r\n    line1\r\n    line2\r\n")

    # Target specified with Unix LF
    target = "    line1\n    line2"
    replacement = "    line1_patched\n    line2_patched"

    diff, updated = patcher.apply_patch(file_rel, target, replacement)
    assert "line1_patched" in updated
    assert "+    line1_patched" in diff


def test_patcher_ambiguity_rejection(tmp_path):
    fs = FileSystemManager(tmp_path)
    patcher = MultiLinePatcher(fs)

    file_rel = "duplicate.py"
    content = "item = 1\nitem = 1\n"
    fs.write_atomic(file_rel, content)

    # Ambiguous without replace_all=True
    with pytest.raises(PatchError) as exc_info:
        patcher.apply_patch(file_rel, "item = 1", "item = 2", replace_all=False)
    assert "occurs 2 times" in str(exc_info.value)


# ============================================================================
# 5. INTEGRATED DEFAULT TOOL REGISTRY
# ============================================================================

@pytest.mark.asyncio
async def test_integrated_default_tool_registry(tmp_path):
    registry = build_default_tool_registry(workspace_dir=tmp_path)

    # 1. Write file
    call_write = ToolCall(
        id="c1",
        name="write_file",
        arguments={"path": "main.py", "content": "print('hello from default registry')"}
    )
    res_write = await registry.dispatch(call_write)
    assert "Successfully wrote" in res_write

    # 2. Read file
    call_read = ToolCall(
        id="c2",
        name="read_file",
        arguments={"path": "main.py"}
    )
    res_read = await registry.dispatch(call_read)
    assert res_read == "print('hello from default registry')"

    # 3. Edit file
    call_edit = ToolCall(
        id="c3",
        name="edit_file",
        arguments={
            "path": "main.py",
            "target": "hello from default registry",
            "replacement": "hello sovereign antimatter"
        }
    )
    res_edit = await registry.dispatch(call_edit)
    assert "Patch successfully applied" in res_edit

    # Verify final content
    assert (tmp_path / "main.py").read_text(encoding="utf-8") == "print('hello sovereign antimatter')"
