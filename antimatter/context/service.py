"""Code Intelligence & Deep Context Layer Coordinator.

Integrates Tree-sitter AST parsing, FastEmbed local neural embeddings,
sqlite-vec persistent vector store, and Async LSP diagnostics.
"""

from __future__ import annotations
import hashlib
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Set, Union

from antimatter.context.ast_parser import CodeASTParser, detect_language
from antimatter.context.lsp_client import AsyncLSPClient
from antimatter.context.types import (
    CodeChunk,
    Diagnostic,
    HoverInfo,
    SearchResult,
    Symbol,
    SymbolKind,
)
from antimatter.context.vector_store import SqliteVecStore
from antimatter.brain.embeddings import LocalEmbeddingEngine
from antimatter.core.config import get_settings

logger = logging.getLogger(__name__)

DEFAULT_INDEX_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".json",
    ".md",
    ".toml",
    ".yaml",
    ".yml",
}


def _file_sha256(path: Path) -> str:
    """Compute sha256 checksum of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class CodeIntelligenceService:
    """High-level sovereign code intelligence and context service."""

    def __init__(
        self,
        workspace_dir: Optional[Union[str, Path]] = None,
        db_path: Optional[Union[str, Path]] = None,
        embedding_engine: Optional[LocalEmbeddingEngine] = None,
        lsp_client: Optional[AsyncLSPClient] = None,
    ):
        settings = get_settings()
        self.workspace_dir = Path(workspace_dir or settings.workspace_dir).resolve()

        # Database path defaults to .antimatter/code_index.db in workspace
        if db_path is None:
            if settings.code_index_path:
                self.db_path = settings.code_index_path
            else:
                self.db_path = self.workspace_dir / ".antimatter" / "code_index.db"
        else:
            self.db_path = Path(db_path) if db_path != ":memory:" else ":memory:"

        self.ast_parser = CodeASTParser()
        self.embedding_engine = embedding_engine or LocalEmbeddingEngine(model_name=settings.embedding_model)
        self.vector_store = SqliteVecStore(db_path=self.db_path, dimension=settings.embedding_dimension)
        self.lsp_client = lsp_client or AsyncLSPClient(cwd=self.workspace_dir)

    def index_workspace(
        self,
        target_dir: Optional[Union[str, Path]] = None,
        force: bool = False,
        extensions: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """Scan workspace, chunk files via AST, compute embeddings, and update vector store."""
        t0 = time.perf_counter()
        root = Path(target_dir or self.workspace_dir).resolve()
        allowed_exts = extensions or DEFAULT_INDEX_EXTENSIONS

        indexed_files = 0
        skipped_files = 0
        total_chunks = 0
        total_symbols = 0

        # Discover all candidate source files, ignoring .git, venv, and cache
        ignored_dirs = {".git", ".venv", "node_modules", "__pycache__", ".pytest_cache", ".antimatter"}

        candidate_files: List[Path] = []
        for p in root.rglob("*"):
            if p.is_file():
                # Check directory exclusion
                parts = set(p.parts)
                if any(ignored in parts for ignored in ignored_dirs):
                    continue
                if p.suffix.lower() in allowed_exts:
                    candidate_files.append(p)

        logger.info("Found %d candidate files to index in '%s'", len(candidate_files), root)

        all_new_chunks: List[CodeChunk] = []
        for file_p in candidate_files:
            rel_path = str(file_p.relative_to(root)).replace("\\", "/")
            current_hash = _file_sha256(file_p)
            stored_hash = self.vector_store.get_file_hash(rel_path)

            if not force and stored_hash == current_hash:
                skipped_files += 1
                continue

            try:
                content = file_p.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                logger.warning("Could not read file %s: %s", rel_path, e)
                continue

            # Clear old records for this file
            self.vector_store.delete_file(rel_path)

            # Extract AST symbols
            symbols = self.ast_parser.extract_symbols(content, rel_path)
            if symbols:
                self.vector_store.index_symbols(symbols)
                total_symbols += len(symbols)

            # AST-aware chunking
            chunks = self.ast_parser.chunk_code(content, rel_path)
            if chunks:
                all_new_chunks.extend(chunks)

            self.vector_store.update_file_hash(rel_path, current_hash)
            indexed_files += 1

        # Batch embed and index all new chunks
        if all_new_chunks:
            # Batch embedding generation
            texts_to_embed = [
                f"# File: {c.file_path}\n# Symbol: {c.name}\n{c.content}" for c in all_new_chunks
            ]
            embeddings = self.embedding_engine.embed_documents(texts_to_embed)
            self.vector_store.index_chunks(all_new_chunks, embeddings)
            total_chunks = len(all_new_chunks)

        duration = time.perf_counter() - t0
        summary = {
            "indexed_files": indexed_files,
            "skipped_files": skipped_files,
            "total_chunks_indexed": total_chunks,
            "total_symbols_indexed": total_symbols,
            "duration_sec": duration,
        }
        logger.info("Indexing complete: %s", summary)
        return summary

    def index_file(self, file_path: Union[str, Path]) -> int:
        """Incrementally index or re-index a single file."""
        abs_p = (self.workspace_dir / file_path if not Path(file_path).is_absolute() else Path(file_path)).resolve()
        if not abs_p.exists():
            rel_p = str(file_path).replace("\\", "/")
            self.vector_store.delete_file(rel_p)
            return 0

        rel_p = str(abs_p.relative_to(self.workspace_dir)).replace("\\", "/")
        content = abs_p.read_text(encoding="utf-8", errors="replace")

        self.vector_store.delete_file(rel_p)

        symbols = self.ast_parser.extract_symbols(content, rel_p)
        if symbols:
            self.vector_store.index_symbols(symbols)

        chunks = self.ast_parser.chunk_code(content, rel_p)
        if chunks:
            texts = [f"# File: {c.file_path}\n# Symbol: {c.name}\n{c.content}" for c in chunks]
            embeddings = self.embedding_engine.embed_documents(texts)
            self.vector_store.index_chunks(chunks, embeddings)

        self.vector_store.update_file_hash(rel_p, _file_sha256(abs_p))
        return len(chunks)

    def semantic_search(
        self,
        query: str,
        limit: int = 5,
        file_filter: Optional[str] = None,
    ) -> List[SearchResult]:
        """Execute semantic search against the local vector store."""
        t0 = time.perf_counter()
        q_emb = self.embedding_engine.embed_query(query)
        results = self.vector_store.search(q_emb, limit=limit, file_filter=file_filter)
        logger.debug("Semantic search for '%s' took %.3fs (retrieved %d)", query, time.perf_counter() - t0, len(results))
        return results

    def find_symbols(
        self,
        name_query: str,
        kind: Optional[Union[SymbolKind, str]] = None,
        file_path: Optional[str] = None,
        limit: int = 20,
    ) -> List[Symbol]:
        """Find symbol definitions matching the query."""
        return self.vector_store.find_symbols(name_query, kind=kind, file_path=file_path, limit=limit)

    def get_symbol_hierarchy(self, file_path: Union[str, Path]) -> List[Symbol]:
        """Get full nested symbol hierarchy for a file."""
        abs_p = (self.workspace_dir / file_path if not Path(file_path).is_absolute() else Path(file_path)).resolve()
        if not abs_p.exists():
            return []
        content = abs_p.read_text(encoding="utf-8", errors="replace")
        try:
            rel_p = str(abs_p.relative_to(self.workspace_dir)).replace("\\", "/")
        except ValueError:
            rel_p = str(abs_p.name)
        return self.ast_parser.extract_symbol_hierarchy(content, rel_p)

    def extract_references(self, file_path: Union[str, Path]) -> List[Any]:
        """Extract symbol usages/references for a file."""
        abs_p = (self.workspace_dir / file_path if not Path(file_path).is_absolute() else Path(file_path)).resolve()
        if not abs_p.exists():
            return []
        content = abs_p.read_text(encoding="utf-8", errors="replace")
        try:
            rel_p = str(abs_p.relative_to(self.workspace_dir)).replace("\\", "/")
        except ValueError:
            rel_p = str(abs_p.name)
        return self.ast_parser.extract_references(content, rel_p)

    async def get_hover(
        self,
        file_path: Union[str, Path],
        line: int,
        character: int,
    ) -> Optional[HoverInfo]:
        """Query hover information via LSP or AST fallback."""
        abs_p = (self.workspace_dir / file_path if not Path(file_path).is_absolute() else Path(file_path)).resolve()
        if not abs_p.exists():
            return None

        # 1. Attempt LSP hover if running or startable
        try:
            if not self.lsp_client.is_running:
                await self.lsp_client.start()
                await self.lsp_client.initialize(self.workspace_dir)

            content = abs_p.read_text(encoding="utf-8", errors="replace")
            await self.lsp_client.open_document(abs_p, content)
            info = await self.lsp_client.hover(abs_p, line, character)
            if info:
                return info
        except Exception as e:
            logger.debug("LSP hover query failed: %s. Using AST fallback.", e)

        # 2. AST Fallback hover
        symbols = self.ast_parser.extract_symbols(abs_p.read_text(encoding="utf-8", errors="replace"), str(file_path))
        for s in symbols:
            if s.start_line <= line <= s.end_line:
                doc = f"**Symbol:** `{s.name}` ({s.kind.value})\n"
                if s.signature:
                    doc += f"\n```python\n{s.signature}\n```\n"
                if s.docstring:
                    doc += f"\n{s.docstring}"
                return HoverInfo(file_path=str(file_path), contents=doc, line=line, column=character)

        return None

    async def get_diagnostics(self, file_path: Union[str, Path]) -> List[Diagnostic]:
        """Query file diagnostics via LSP."""
        abs_p = (self.workspace_dir / file_path if not Path(file_path).is_absolute() else Path(file_path)).resolve()
        if not abs_p.exists():
            return []

        try:
            if not self.lsp_client.is_running:
                await self.lsp_client.start()
                await self.lsp_client.initialize(self.workspace_dir)

            content = abs_p.read_text(encoding="utf-8", errors="replace")
            await self.lsp_client.open_document(abs_p, content)
            # Short wait for async diagnostic publication from server
            await asyncio.sleep(0.05)
            return self.lsp_client.get_diagnostics(abs_p)
        except Exception as e:
            logger.debug("LSP diagnostics failed: %s", e)
            return []

    async def close(self) -> None:
        """Close vector store and shutdown LSP client."""
        self.vector_store.close()
        if self.lsp_client.is_running:
            await self.lsp_client.stop()
