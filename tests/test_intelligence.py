"""Test Suite for Track 3: Code Intelligence & Deep Context Layer."""

import pytest
from pathlib import Path

from antimatter.intelligence.ast_parser import ASTParser, SymbolType
from antimatter.intelligence.vector_store import CodebaseVectorStore
from antimatter.intelligence.lsp_client import LSPClient


# ============================================================================
# 1. AST PARSER TESTS
# ============================================================================

def test_ast_python_symbol_extraction():
    parser = ASTParser()
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
    assert names["AgentOrchestrator"].symbol_type == SymbolType.CLASS
    assert names["AgentOrchestrator"].start_line == 1

    assert "run_cycle" in names
    assert names["run_cycle"].symbol_type == SymbolType.METHOD
    assert names["run_cycle"].parent_symbol == "AgentOrchestrator"

    assert "standalone_worker" in names
    assert names["standalone_worker"].symbol_type == SymbolType.FUNCTION


def test_ast_js_symbol_extraction():
    parser = ASTParser()
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
    parser = ASTParser()
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
    chunks = parser.chunk_source_code(code, file_path="math_utils.py")
    assert len(chunks) >= 2

    # Check that compute function chunk exists
    func_chunk = next((c for c in chunks if c.symbol_name == "compute"), None)
    assert func_chunk is not None
    assert func_chunk.symbol_type == SymbolType.FUNCTION
    assert "compute(a, b)" in func_chunk.content


# ============================================================================
# 2. CODEBASE VECTOR STORE (SQLITE-VEC) TESTS
# ============================================================================

def test_vector_store_index_and_search():
    store = CodebaseVectorStore(db_path=":memory:")

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

    indexed1 = store.index_file("tools/terminal.py", code_file1)
    indexed2 = store.index_file("tools/fs.py", code_file2)
    assert indexed1 > 0
    assert indexed2 > 0

    stats = store.get_stats()
    assert stats["total_files"] == 2
    assert stats["total_chunks"] >= 2

    # Semantic search query for subprocess / terminal command execution
    results = store.search_similar("execute shell subprocess", limit=1)
    assert len(results) == 1
    top_match = results[0]
    assert "terminal.py" in top_match.file_path
    assert top_match.distance >= 0.0

    # Delete file and ensure removal
    deleted_count = store.delete_file("tools/terminal.py")
    assert deleted_count > 0
    assert store.get_stats()["total_files"] == 1


# ============================================================================
# 3. HEADLESS LSP CLIENT TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_lsp_client_ast_fallback(tmp_path):
    lsp = LSPClient(workspace_root=tmp_path)
    started = await lsp.start()
    assert started is True

    code = (
        "class CompilerEngine:\n"
        "    def compile_source(self, code: str) -> bool:\n"
        "        '''Compile source into AST.'''\n"
        "        return True\n"
    )
    test_file = tmp_path / "compiler.py"
    test_file.write_text(code, encoding="utf-8")

    # Hover over method at line 2
    hover = await lsp.get_hover(test_file, line=2, character=8)
    assert hover is not None
    assert hover.symbol_name == "compile_source"
    assert "compile_source" in hover.signature

    # Jump to definition of CompilerEngine
    defs = await lsp.get_definition(test_file, line=1, character=5, symbol_name="CompilerEngine")
    assert len(defs) == 1
    assert defs[0].start_line == 1

    await lsp.stop()
