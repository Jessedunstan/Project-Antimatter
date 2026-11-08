"""Local Codebase Vector Store using SQLite and sqlite-vec extension.

Performs sub-second KNN semantic code search over AST chunks with 100% offline FastEmbed.
"""

import logging
import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import sqlite_vec

from antimatter.brain.embeddings import LocalEmbeddingEngine
from antimatter.intelligence.ast_parser import ASTParser, CodeChunk, SymbolType

logger = logging.getLogger(__name__)


@dataclass
class CodeMatch:
    """Semantic search retrieval result."""
    chunk_id: str
    file_path: str
    symbol_name: Optional[str]
    symbol_type: str
    start_line: int
    end_line: int
    content: str
    distance: float  # Lower distance = higher semantic similarity


class CodebaseVectorStore:
    """Persistent vector index over codebase using sqlite-vec."""

    VECTOR_DIM = 384

    def __init__(
        self,
        db_path: Path | str = ":memory:",
        embedding_engine: Optional[LocalEmbeddingEngine] = None,
    ):
        self.db_path = str(db_path)
        self.embedding_engine = embedding_engine or LocalEmbeddingEngine()
        self.ast_parser = ASTParser()
        self._conn = self._init_database()

    def _init_database(self) -> sqlite3.Connection:
        """Initialize SQLite database with sqlite-vec extension and schemas."""
        conn = sqlite3.connect(self.db_path)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)

        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunk_metadata (
                    chunk_id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    symbol_name TEXT,
                    symbol_type TEXT NOT NULL,
                    start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    content TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunk_file_path ON chunk_metadata(file_path)
            """)
            conn.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                    chunk_id TEXT PRIMARY KEY,
                    embedding float[{self.VECTOR_DIM}]
                )
            """)
        return conn

    def _serialize_vector(self, vec: List[float]) -> bytes:
        """Pack float array into binary blob for sqlite-vec."""
        return struct.pack(f"{len(vec)}f", *vec)

    def delete_file(self, file_path: str) -> int:
        """Remove all indexed chunks for a file."""
        norm_path = file_path.replace("\\", "/")
        with self._conn:
            cur = self._conn.execute(
                "SELECT chunk_id FROM chunk_metadata WHERE file_path = ?",
                (norm_path,)
            )
            chunk_ids = [row[0] for row in cur.fetchall()]
            if not chunk_ids:
                return 0

            # Delete from both relational and vector virtual table
            self._conn.executemany(
                "DELETE FROM chunk_metadata WHERE chunk_id = ?",
                [(cid,) for cid in chunk_ids]
            )
            self._conn.executemany(
                "DELETE FROM vec_chunks WHERE chunk_id = ?",
                [(cid,) for cid in chunk_ids]
            )
            return len(chunk_ids)

    def index_file(self, file_path: str, code: str) -> int:
        """Parse code into AST chunks, generate embeddings, and store in index."""
        norm_path = file_path.replace("\\", "/")
        chunks = self.ast_parser.chunk_source_code(code, norm_path)
        if not chunks:
            return 0

        # Purge existing chunks for this file before re-indexing
        self.delete_file(norm_path)

        # Batch embed chunk texts
        texts = [c.full_text_for_embedding() for c in chunks]
        embeddings = self.embedding_engine.embed_documents(texts)

        with self._conn:
            for chunk, emb in zip(chunks, embeddings):
                # Insert metadata
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO chunk_metadata 
                    (chunk_id, file_path, symbol_name, symbol_type, start_line, end_line, content)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        chunk.file_path,
                        chunk.symbol_name,
                        chunk.symbol_type.value,
                        chunk.start_line,
                        chunk.end_line,
                        chunk.content,
                    )
                )

                # Insert vector
                vec_blob = self._serialize_vector(emb)
                self._conn.execute(
                    "INSERT OR REPLACE INTO vec_chunks (chunk_id, embedding) VALUES (?, ?)",
                    (chunk.chunk_id, vec_blob)
                )

        return len(chunks)

    def search_similar(
        self,
        query: str,
        limit: int = 5,
        file_filter: Optional[str] = None,
    ) -> List[CodeMatch]:
        """Perform semantic KNN search using sqlite-vec."""
        query_vector = self.embedding_engine.embed_query(query)
        query_blob = self._serialize_vector(query_vector)

        # Query KNN from vec0 virtual table
        # Retrieve more candidates if file_filter is active to allow filtering
        search_k = limit * 3 if file_filter else limit

        cur = self._conn.execute(
            """
            SELECT 
                v.chunk_id,
                v.distance,
                m.file_path,
                m.symbol_name,
                m.symbol_type,
                m.start_line,
                m.end_line,
                m.content
            FROM vec_chunks v
            JOIN chunk_metadata m ON v.chunk_id = m.chunk_id
            WHERE v.embedding MATCH ? AND k = ?
            ORDER BY v.distance ASC
            """,
            (query_blob, search_k)
        )

        matches: List[CodeMatch] = []
        for row in cur.fetchall():
            chunk_id, distance, fpath, sname, stype, sline, eline, content = row
            if file_filter and file_filter not in fpath:
                continue

            matches.append(
                CodeMatch(
                    chunk_id=chunk_id,
                    distance=float(distance),
                    file_path=fpath,
                    symbol_name=sname,
                    symbol_type=stype,
                    start_line=sline,
                    end_line=eline,
                    content=content,
                )
            )
            if len(matches) >= limit:
                break

        return matches

    def reindex_workspace(self, workspace_dir: Path) -> Dict[str, int]:
        """Recursively scan and index code files in workspace."""
        stats = {"files_indexed": 0, "chunks_indexed": 0}
        supported_exts = {".py", ".js", ".jsx", ".ts", ".tsx"}

        for p in workspace_dir.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in supported_exts:
                continue
            if ".venv" in p.parts or ".antimatter" in p.parts or "__pycache__" in p.parts:
                continue

            try:
                rel_path = str(p.relative_to(workspace_dir)).replace("\\", "/")
                code = p.read_text(encoding="utf-8", errors="replace")
                chunks_count = self.index_file(rel_path, code)
                stats["files_indexed"] += 1
                stats["chunks_indexed"] += chunks_count
            except Exception as err:
                logger.warning("Failed indexing file %s: %s", p, err)

        return stats

    def get_stats(self) -> Dict[str, Any]:
        """Return index statistics."""
        cur = self._conn.execute("SELECT COUNT(DISTINCT file_path), COUNT(*) FROM chunk_metadata")
        files_count, chunks_count = cur.fetchone()
        return {
            "total_files": files_count,
            "total_chunks": chunks_count,
            "vector_dimension": self.VECTOR_DIM,
        }
