"""Deterministic Sovereign Tools for Project Antimatter."""

from pathlib import Path
from typing import Optional

from antimatter.core.config import get_settings
from antimatter.tools.registry import ToolRegistry
from antimatter.tools.terminal import TerminalSandbox, CommandResult
from antimatter.tools.fs_manager import FileSystemManager, SecurityException
from antimatter.tools.patcher import MultiLinePatcher, PatchError

__all__ = [
    "ToolRegistry",
    "TerminalSandbox",
    "CommandResult",
    "FileSystemManager",
    "SecurityException",
    "MultiLinePatcher",
    "PatchError",
    "build_default_tool_registry",
]


def build_default_tool_registry(workspace_dir: Optional[Path] = None) -> ToolRegistry:
    """Build and wire the default sovereign tool suite."""
    settings = get_settings()
    ws = (workspace_dir or settings.workspace_dir).resolve()

    fs = FileSystemManager(ws)
    patcher = MultiLinePatcher(fs)
    terminal = TerminalSandbox(ws, max_timeout_sec=settings.max_exec_timeout_sec)

    registry = ToolRegistry()

    @registry.register(name="read_file", description="Read contents of a file in the workspace.")
    def read_file(path: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> str:
        return fs.read_file(path, start_line=start_line, end_line=end_line)

    @registry.register(name="write_file", description="Write content to a file in the workspace atomically.")
    def write_file(path: str, content: str, checkpoint_id: Optional[str] = None) -> str:
        bytes_written = fs.write_atomic(path, content, checkpoint_id=checkpoint_id)
        return f"Successfully wrote {bytes_written} bytes to '{path}'."

    @registry.register(name="edit_file", description="Apply an exact multi-line patch to an existing file.")
    def edit_file(
        path: str,
        target: str,
        replacement: str,
        replace_all: bool = False,
        checkpoint_id: Optional[str] = None,
    ) -> str:
        diff, _ = patcher.apply_patch(
            file_path=path,
            target_block=target,
            replacement_block=replacement,
            replace_all=replace_all,
            checkpoint_id=checkpoint_id,
        )
        return f"Patch successfully applied to '{path}'.\nUnified Diff:\n{diff}"

    @registry.register(name="run_command", description="Execute a shell command inside the workspace sandbox.")
    async def run_command(command: str, timeout_sec: Optional[int] = None) -> str:
        res = await terminal.execute(command, timeout_sec=timeout_sec)
        out = []
        if res.stdout:
            out.append(f"[STDOUT]\n{res.stdout}")
        if res.stderr:
            out.append(f"[STDERR]\n{res.stderr}")
        out.append(f"[EXIT CODE: {res.exit_code} | DURATION: {res.duration_ms:.1f}ms]")
        if res.timed_out:
            out.append("[STATUS: TIMED OUT]")
        return "\n".join(out)

    @registry.register(name="list_directory", description="List files and directories in a workspace subfolder.")
    def list_directory(subpath: str = "") -> str:
        entries = fs.list_dir(subpath)
        if not entries:
            return f"Directory '{subpath}' is empty."
        lines = [f"{'DIR ' if e['type'] == 'directory' else 'FILE'} {e['rel_path']}" for e in entries]
        return "\n".join(lines)

    @registry.register(name="create_checkpoint", description="Create a rollback checkpoint for transactional file changes.")
    def create_checkpoint(description: str = "") -> str:
        cp_id = fs.create_checkpoint(description)
        return f"Checkpoint created: {cp_id}"

    @registry.register(name="rollback_checkpoint", description="Rollback all file modifications recorded in a checkpoint.")
    def rollback_checkpoint(checkpoint_id: str) -> str:
        success, msg = fs.rollback(checkpoint_id)
        return msg

    # Code Intelligence & Deep Context Tools
    _code_intel = None

    def _get_intel():
        nonlocal _code_intel
        if _code_intel is None:
            from antimatter.context.service import CodeIntelligenceService
            _code_intel = CodeIntelligenceService(workspace_dir=ws)
        return _code_intel

    @registry.register(name="search_code", description="Semantically search the codebase using local vector embeddings.")
    def search_code(query: str, limit: int = 5, file_filter: Optional[str] = None) -> str:
        intel = _get_intel()
        results = intel.semantic_search(query, limit=limit, file_filter=file_filter)
        if not results:
            return f"No code snippets matched query '{query}'."
        out = []
        for i, r in enumerate(results, 1):
            out.append(f"--- [Match #{i} (Score: {r.score:.3f})] {r.chunk.file_path}:{r.chunk.start_line}-{r.chunk.end_line} ({r.chunk.name}) ---")
            out.append(r.chunk.content)
        return "\n".join(out)

    @registry.register(name="get_symbol_outline", description="Extract class and function symbol hierarchy for a file.")
    def get_symbol_outline(path: str) -> str:
        intel = _get_intel()
        hierarchy = intel.get_symbol_hierarchy(path)
        if not hierarchy:
            return f"No symbols found in '{path}'."
        def _format_sym(s, depth=0):
            indent = "  " * depth
            lines = [f"{indent}- [{s.kind.value}] {s.name} (L{s.start_line}-L{s.end_line})"]
            for c in s.children:
                lines.extend(_format_sym(c, depth + 1))
            return lines
        res = []
        for h in hierarchy:
            res.extend(_format_sym(h, 0))
        return "\n".join(res)

    @registry.register(name="find_symbol", description="Locate definitions of a symbol across the workspace.")
    def find_symbol(name: str) -> str:
        intel = _get_intel()
        symbols = intel.find_symbols(name)
        if not symbols:
            return f"No definitions found matching '{name}'."
        lines = [f"- [{s.kind.value}] {s.name} in {s.file_path}:{s.start_line}" for s in symbols]
        return "\n".join(lines)

    @registry.register(name="get_diagnostics", description="Get syntax and type diagnostics for a file.")
    async def get_diagnostics(path: str) -> str:
        intel = _get_intel()
        diags = await intel.get_diagnostics(path)
        if not diags:
            return f"No diagnostics or errors for '{path}'."
        return "\n".join(f"- [Severity {d.severity.name}] L{d.line}: {d.message} ({d.source or 'lsp'})" for d in diags)

    return registry
