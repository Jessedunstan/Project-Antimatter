"""Test Suite for Track 3: Code Intelligence & Deep Context Layer (Consolidated Context Package)."""

import pytest
from pathlib import Path

from antimatter.context.ast_parser import CodeASTParser
from antimatter.context.types import SymbolKind
from antimatter.context.service import CodeIntelligenceService


# ============================================================================
# 1. AST PARSER TESTS
# ============================================================================

def test_ast_python_symbol_extraction():
    parser = CodeASTParser()
    py_code = (
        "class AgentOrchestrator:\n"
        "    def __init__(self, name: str):\n"
        "        self.name = name\n"
        "\n"
        "    def run_cycle(self) -> bool:\n"
        "        return True\n"
        "\n"
        "def standalone_worker(task: str) -> None:\n"
        "    pass\n"
    )

    symbols = parser.extract_symbols(py_code, file_path="agent.py")
    assert len(symbols) == 4

    names = {s.name: s for s in symbols}
    assert "AgentOrchestrator" in names
    assert names["AgentOrchestrator"].kind == SymbolKind.CLASS
    assert names["AgentOrchestrator"].start_line == 1

    assert "run_cycle" in names
    assert names["run_cycle"].kind == SymbolKind.METHOD
    assert names["run_cycle"].parent_name == "AgentOrchestrator"

    assert "standalone_worker" in names
    assert names["standalone_worker"].kind == SymbolKind.FUNCTION


def test_ast_js_symbol_extraction():
    parser = CodeASTParser()
    js_code = (
        "class EditorView {\n"
        "    render() {\n"
        "        return 1;\n"
        "    }\n"
        "}\n"
        "function formatText(s) {\n"
        "    return s;\n"
        "}\n"
    )

    symbols = parser.extract_symbols(js_code, file_path="editor.js")
    assert len(symbols) == 3
    names = {s.name: s for s in symbols}
    assert "EditorView" in names
    assert "render" in names
    assert "formatText" in names


def test_ast_semantic_chunking():
    parser = CodeASTParser()
    code = (
        "import sys\n"
        "\n"
        "def compute(a, b):\n"
        "    return a + b\n"
        "\n"
        "class MathEngine:\n"
        "    def multiply(self, x, y):\n"
        "        return x * y\n"
    )
    chunks = parser.chunk_code(code, file_path="math_utils.py")
    assert len(chunks) >= 2

    func_chunk = next((c for c in chunks if c.name == "compute"), None)
    assert func_chunk is not None
    assert func_chunk.chunk_type == "function"
    assert "compute(a, b)" in func_chunk.content


# ============================================================================
# 2. CODEBASE VECTOR STORE (SQLITE-VEC) TESTS
# ============================================================================

def test_vector_store_index_and_search(tmp_path):
    code_file1 = (
        "class TerminalSandbox:\n"
        "    def run_command(self, cmd: str):\n"
        "        # Execute a shell command inside a restricted subprocess\n"
        "        return subprocess.run(cmd)\n"
    )
    code_file2 = (
        "class FileSystemManager:\n"
        "    def write_atomic(self, path: str, content: str):\n"
        "        # Atomically write data to disk with sync and rename\n"
        "        with open(path, 'w') as f: f.write(content)\n"
    )

    tools_dir = tmp_path / "tools"
    tools_dir.mkdir(parents=True, exist_ok=True)
    f1 = tools_dir / "terminal.py"
    f2 = tools_dir / "fs.py"
    f1.write_text(code_file1, encoding="utf-8")
    f2.write_text(code_file2, encoding="utf-8")

    service = CodeIntelligenceService(workspace_dir=tmp_path, db_path=":memory:")
    indexed1 = service.index_file("tools/terminal.py")
    indexed2 = service.index_file("tools/fs.py")
    assert indexed1 > 0
    assert indexed2 > 0
    assert service.vector_store.count_chunks() >= 2

    results = service.semantic_search("execute shell subprocess", limit=1)
    assert len(results) == 1
    top_match = results[0]
    assert "terminal.py" in top_match.chunk.file_path
    assert top_match.distance >= 0.0

    # Delete file and ensure removal
    f1.unlink()
    service.index_file("tools/terminal.py")
    assert service.vector_store.count_chunks() == indexed2
    service.vector_store.close()


# ============================================================================
# 3. HEADLESS LSP / HOVER TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_lsp_client_ast_fallback(tmp_path):
    code = (
        "class CompilerEngine:\n"
        "    def compile_source(self, code: str) -> bool:\n"
        "        '''Compile source into AST.'''\n"
        "        return True\n"
    )
    test_file = tmp_path / "compiler.py"
    test_file.write_text(code, encoding="utf-8")

    service = CodeIntelligenceService(workspace_dir=tmp_path, db_path=":memory:")
    hover = await service.get_hover(test_file, line=2, character=8)
    assert hover is not None
    assert "compile_source" in hover.contents

    await service.close()
