"""Comprehensive Test Suite for Track 3: Code Intelligence & Deep Context Layer.

Validates:
1. Tree-sitter AST parser extracting definitions, references, and symbol hierarchies.
2. Async LSP client connecting to local language server for hover/diagnostics.
3. sqlite-vec persistent vector store chunking code intelligently by AST nodes.
4. Sub-second semantic code retrieval and incremental index synchronization.
"""

from __future__ import annotations
import asyncio
from pathlib import Path
import tempfile
import time
import pytest

from antimatter.context.ast_parser import CodeASTParser, detect_language
from antimatter.context.lsp_client import AsyncLSPClient
from antimatter.context.types import (
    CodeChunk,
    DiagnosticSeverity,
    SymbolKind,
)
from antimatter.context.vector_store import SqliteVecStore
from antimatter.context.service import CodeIntelligenceService
from antimatter.brain.embeddings import LocalEmbeddingEngine


# ============================================================================
# 1. TREE-SITTER AST PARSER TESTS
# ============================================================================

def test_ast_language_detection():
    assert detect_language("service.py") == "python"
    assert detect_language("app.ts") == "typescript"
    assert detect_language("component.tsx") == "tsx"
    assert detect_language("index.js") == "javascript"
    assert detect_language("readme.md") is None


def test_python_symbol_extraction_and_hierarchy():
    parser = CodeASTParser()
    code = """
import os
import sys

GLOBAL_CONFIG = {"debug": True}

class DatabaseManager:
    \"\"\"Manages local database connections and migrations.\"\"\"
    timeout = 30

    def __init__(self, db_url: str):
        self.db_url = db_url

    def execute_query(self, query: str) -> list:
        \"\"\"Execute a SQL query safely.\"\"\"
        return [query]

def health_check() -> bool:
    \"\"\"Verify system health.\"\"\"
    return True
"""
    # 1. Extract symbols
    symbols = parser.extract_symbols(code, "db.py")
    sym_names = {s.name: s for s in symbols}

    assert "DatabaseManager" in sym_names
    assert sym_names["DatabaseManager"].kind == SymbolKind.CLASS
    assert "Manages local database" in (sym_names["DatabaseManager"].docstring or "")

    assert "execute_query" in sym_names
    assert sym_names["execute_query"].kind == SymbolKind.METHOD
    assert sym_names["execute_query"].parent_name == "DatabaseManager"
    assert "Execute a SQL query" in (sym_names["execute_query"].docstring or "")

    assert "health_check" in sym_names
    assert sym_names["health_check"].kind == SymbolKind.FUNCTION
    assert sym_names["health_check"].parent_name is None

    # 2. Extract symbol hierarchy
    hierarchy = parser.extract_symbol_hierarchy(code, "db.py")
    assert len(hierarchy) >= 2  # DatabaseManager and health_check (plus GLOBAL_CONFIG)

    db_class = next(s for s in hierarchy if s.name == "DatabaseManager")
    child_names = [c.name for c in db_class.children]
    assert "__init__" in child_names
    assert "execute_query" in child_names


def test_python_references_extraction():
    parser = CodeASTParser()
    code = """
def calculate(a, b):
    return a + b

def main():
    res = calculate(10, 20)
    print(res)
"""
    refs = parser.extract_references(code, "calc.py")
    ref_names = [r.name for r in refs]

    assert "calculate" in ref_names
    assert "print" in ref_names
    # Check enclosing symbol
    calc_call = next(r for r in refs if r.name == "calculate")
    assert calc_call.enclosing_symbol == "main"


def test_typescript_ast_parsing():
    parser = CodeASTParser()
    ts_code = """
export interface UserProfile {
    id: number;
    username: string;
}

export class AuthService {
    private token: string;

    constructor(token: string) {
        this.token = token;
    }

    public validate(): boolean {
        return this.token.length > 0;
    }
}
"""
    symbols = parser.extract_symbols(ts_code, "auth.ts")
    sym_dict = {s.name: s for s in symbols}

    assert "UserProfile" in sym_dict
    assert sym_dict["UserProfile"].kind == SymbolKind.INTERFACE

    assert "AuthService" in sym_dict
    assert sym_dict["AuthService"].kind == SymbolKind.CLASS

    assert "validate" in sym_dict
    assert sym_dict["validate"].kind == SymbolKind.METHOD
    assert sym_dict["validate"].parent_name == "AuthService"


def test_ast_intelligent_chunking():
    parser = CodeASTParser()
    code = """
\"\"\"Database helper module.\"\"\"
import sqlite3

DEFAULT_PORT = 5432

class CompactClass:
    def small_method(self):
        return 1

class LargeClass:
    \"\"\"Large class spanning multiple chunks.\"\"\"
    
    def method_one(self, x):
        \"\"\"First method docstring.\"\"\"
        result = x * 2
        return result

    def method_two(self, y):
        \"\"\"Second method docstring.\"\"\"
        return y + 10
"""
    # Chunk with small threshold to force method splitting
    chunks = parser.chunk_code(code, "helper.py", max_chunk_lines=10)
    chunk_names = [c.name for c in chunks]

    # Verify that class and method chunks exist
    assert any("CompactClass" in n for n in chunk_names)
    assert any("LargeClass.method_one" in n for n in chunk_names)
    assert any("LargeClass.method_two" in n for n in chunk_names)

    # Verify parent context is maintained in method chunk
    m1_chunk = next(c for c in chunks if "method_one" in c.name)
    assert m1_chunk.parent_context == "class LargeClass"
    assert "class LargeClass" in m1_chunk.content


# ============================================================================
# 2. SQLITE-VEC PERSISTENT VECTOR STORE TESTS
# ============================================================================

def test_sqlite_vec_crud_and_persistence():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "index.db"
        dim = 4

        # 1. Initialize store and index test chunks
        store = SqliteVecStore(db_path=db_path, dimension=dim)

        chunks = [
            CodeChunk(
                id="c1",
                file_path="src/calc.py",
                chunk_type="function",
                name="add_numbers",
                content="def add_numbers(a, b): return a + b",
                start_line=1,
                end_line=1,
                symbols=["add_numbers"],
                docstring="Add two numbers.",
            ),
            CodeChunk(
                id="c2",
                file_path="src/calc.py",
                chunk_type="function",
                name="multiply_numbers",
                content="def multiply_numbers(a, b): return a * b",
                start_line=3,
                end_line=3,
                symbols=["multiply_numbers"],
            ),
        ]
        embeddings = [
            [0.1, 0.2, 0.3, 0.4],
            [0.9, 0.8, 0.7, 0.6],
        ]

        indexed_count = store.index_chunks(chunks, embeddings)
        assert indexed_count == 2
        assert store.count_chunks() == 2

        # 2. Sub-second search retrieval
        t0 = time.perf_counter()
        query = [0.11, 0.21, 0.29, 0.39]
        results = store.search(query, limit=2)
        elapsed = time.perf_counter() - t0

        assert elapsed < 0.1  # Retrieval must be sub-100ms
        assert len(results) == 2
        assert results[0].chunk.id == "c1"  # Closest vector
        assert results[0].score > results[1].score

        # 3. Test persistence across reopen
        store.close()
        store_reopened = SqliteVecStore(db_path=db_path, dimension=dim)
        assert store_reopened.count_chunks() == 2
        results_reopened = store_reopened.search(query, limit=1)
        assert len(results_reopened) == 1
        assert results_reopened[0].chunk.name == "add_numbers"

        # 4. Delete file
        store_reopened.delete_file("src/calc.py")
        assert store_reopened.count_chunks() == 0
        store_reopened.close()


def test_sqlite_vec_with_fastembed_384d():
    """Verify sqlite-vec indexing with actual 384-dimensional local FastEmbed model."""
    embed_engine = LocalEmbeddingEngine()
    dim = 384
    store = SqliteVecStore(db_path=":memory:", dimension=dim)

    texts = [
        "def connect_postgres(db_url: str): return psycopg2.connect(db_url)",
        "def render_html_template(template_name: str): return jinja2.render(template_name)",
        "def encrypt_password(password: str): return bcrypt.hashpw(password)",
    ]
    chunks = [
        CodeChunk(
            id=f"c_{i}",
            file_path="app.py",
            chunk_type="function",
            name=f"func_{i}",
            content=t,
            start_line=i * 2 + 1,
            end_line=i * 2 + 2,
        )
        for i, t in enumerate(texts)
    ]

    embeddings = embed_engine.embed_documents(texts)
    store.index_chunks(chunks, embeddings)

    # Search for database connection
    q_vec = embed_engine.embed_query("connect to SQL relational database")
    t0 = time.perf_counter()
    matches = store.search(q_vec, limit=1)
    retrieval_ms = (time.perf_counter() - t0) * 1000

    assert len(matches) == 1
    assert matches[0].chunk.id == "c_0"  # connect_postgres
    assert retrieval_ms < 50.0  # Under 50ms vector search
    print(f"\n[OK] 384d sqlite-vec retrieval latency: {retrieval_ms:.2f}ms")


# ============================================================================
# 3. ASYNC LSP CLIENT TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_async_lsp_client_lifecycle_and_diagnostics():
    """Verify async LSP client connecting to local server, publishing diagnostics and hover."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        client = AsyncLSPClient(cwd=tmp_path)

        # 1. Initialize
        init_res = await client.initialize(tmp_path)
        assert client.is_running
        assert init_res["capabilities"]["hoverProvider"] is True

        # 2. Test document opening with syntax error -> expect diagnostic
        bad_file = tmp_path / "syntax_error.py"
        bad_code = "def broken_func(\n    return 42\n"  # Missing closing paren
        bad_file.write_text(bad_code, encoding="utf-8")

        await client.open_document(bad_file, bad_code)
        # Give async publishDiagnostics notification a moment to arrive
        await asyncio.sleep(0.1)

        diags = client.get_diagnostics(bad_file)
        assert len(diags) > 0
        assert diags[0].severity == DiagnosticSeverity.ERROR

        # 3. Test document opening with valid code -> expect clean diagnostics and hover
        valid_file = tmp_path / "valid.py"
        valid_code = (
            "def calculate_tax(amount: float) -> float:\n"
            "    \"\"\"Compute standard tax rate.\"\"\"\n"
            "    return amount * 0.15\n"
        )
        valid_file.write_text(valid_code, encoding="utf-8")

        await client.open_document(valid_file, valid_code)
        await asyncio.sleep(0.1)

        valid_diags = client.get_diagnostics(valid_file)
        assert len(valid_diags) == 0

        # 4. Test Hover
        hover_info = await client.hover(valid_file, line=1, character=6)
        assert hover_info is not None
        assert "calculate_tax" in hover_info.contents
        assert "Compute standard tax rate" in hover_info.contents

        # 5. Test Definition
        def_locs = await client.definition(valid_file, line=1, character=6)
        assert len(def_locs) > 0
        assert def_locs[0].line == 1

        # 6. Stop client
        await client.stop()
        assert not client.is_running


# ============================================================================
# 4. CODE INTELLIGENCE SERVICE END-TO-END
# ============================================================================

@pytest.mark.asyncio
async def test_code_intelligence_service_end_to_end():
    """Verify full end-to-end workspace indexing, semantic search, and symbols."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ws = Path(tmpdir)

        # Create simulated project files
        file1 = ws / "math_ops.py"
        file1.write_text(
            """
def add(a: int, b: int) -> int:
    \"\"\"Sum two integers.\"\"\"
    return a + b

def subtract(a: int, b: int) -> int:
    \"\"\"Subtract b from a.\"\"\"
    return a - b
""",
            encoding="utf-8",
        )

        file2 = ws / "user_service.py"
        file2.write_text(
            """
class UserService:
    \"\"\"User account and authentication management.\"\"\"
    
    def authenticate(self, username: str, password_hash: str) -> bool:
        \"\"\"Verify user password hash.\"\"\"
        return len(password_hash) > 10
""",
            encoding="utf-8",
        )

        service = CodeIntelligenceService(workspace_dir=ws, db_path=":memory:")

        # 1. Index entire workspace
        stats = service.index_workspace()
        assert stats["indexed_files"] == 2
        assert stats["total_chunks_indexed"] >= 2
        assert stats["total_symbols_indexed"] >= 3

        # 2. Semantic Search: Find authentication logic
        t0 = time.perf_counter()
        auth_matches = service.semantic_search("verify user password or log in", limit=1)
        search_ms = (time.perf_counter() - t0) * 1000

        assert len(auth_matches) == 1
        assert "UserService" in auth_matches[0].chunk.content or "authenticate" in auth_matches[0].chunk.content
        assert search_ms < 100.0  # Sub-second guarantee
        print(f"\n[OK] Semantic search latency: {search_ms:.2f}ms")

        # 3. Symbol Search
        symbols = service.find_symbols("add")
        assert len(symbols) == 1
        assert symbols[0].name == "add"
        assert symbols[0].kind == SymbolKind.FUNCTION

        # 4. Symbol Hierarchy
        hierarchy = service.get_symbol_hierarchy(file2)
        assert len(hierarchy) >= 1
        user_svc = next(s for s in hierarchy if s.name == "UserService")
        assert any(c.name == "authenticate" for c in user_svc.children)

        # 5. Clean up
        await service.close()
