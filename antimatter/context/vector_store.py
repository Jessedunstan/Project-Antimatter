"""Persistent local vector store powered by sqlite-vec.

Provides sub-second nearest-neighbor semantic search over code chunks
and deterministic symbol indexing without cloud services or external databases.
"""

from __future__ import annotations
import json
import logging
import math
from pathlib import Path
import sqlite3
import struct
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from antimatter.context.types import (
    CodeChunk,
    SearchResult,
    Symbol,
    SymbolKind,
)

logger = logging.getLogger(__name__)


def _serialize_f32(vec: List[float]) -> bytes:
    """Pack Python float array into binary 32-bit little-endian floats for sqlite-vec."""
    return struct.pack(f"{len(vec)}f", *vec)


def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


class SqliteVecStore:
    """Persistent local vector and symbol store backed by SQLite + sqlite-vec."""

    def __init__(
        self,
        db_path: Union[str, Path] = ":memory:",
        dimension: int = 384,
    ):
        self.db_path = str(db_path) if db_path == ":memory:" else str(Path(db_path).resolve())
        self.dimension = dimension
        self._has_sqlite_vec = False
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if self._conn is None:
            if self.db_path != ":memory:":
                Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row

            # Attempt to load sqlite-vec extension
            try:
                import sqlite_vec
                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
                conn.enable_load_extension(False)
                self._has_sqlite_vec = True
            except Exception as e:
                logger.warning("Could not load sqlite-vec extension: %s. Using fallback vector search.", e)
                self._has_sqlite_vec = False

            self._conn = conn
        return self._conn
        return self._conn

    def _init_db(self) -> None:
        conn = self._get_connection()
        with conn:
            # Metadata table for code chunks
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS code_chunks (
                    id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    chunk_type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    parent_context TEXT,
                    symbols_json TEXT,
                    docstring TEXT,
                    content TEXT NOT NULL,
                    raw_vector BLOB,
                    updated_at REAL NOT NULL
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_file_path ON code_chunks(file_path);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_name ON code_chunks(name);")

            # File registry table for incremental indexing tracking
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS file_registry (
                    file_path TEXT PRIMARY KEY,
                    file_hash TEXT NOT NULL,
                    indexed_at REAL NOT NULL
                );
                """
            )

            # Code symbols table for direct definition and hierarchy lookup
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS code_symbols (
                    name TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    start_column INTEGER NOT NULL,
                    end_column INTEGER NOT NULL,
                    signature TEXT,
                    docstring TEXT,
                    parent_name TEXT,
                    PRIMARY KEY (file_path, name, start_line)
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbols_name ON code_symbols(name);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbols_kind ON code_symbols(kind);")

            # Create sqlite-vec virtual table if extension is active
            if self._has_sqlite_vec:
                try:
                    conn.execute(
                        f"""
                        CREATE VIRTUAL TABLE IF NOT EXISTS vec_code_chunks USING vec0(
                            chunk_id text primary key,
                            embedding float[{self.dimension}]
                        );
                        """
                    )
                except Exception as e:
                    logger.warning("Failed to create vec0 virtual table: %s. Falling back to blob storage.", e)
                    self._has_sqlite_vec = False

    def index_chunks(self, chunks: List[CodeChunk], embeddings: List[List[float]]) -> int:
        """Batch insert or replace code chunks with their embedding vectors."""
        if not chunks:
            return 0
        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatched count: {len(chunks)} chunks vs {len(embeddings)} embeddings")

        conn = self._get_connection()
        now = time.time()

        with conn:
            for chunk, emb in zip(chunks, embeddings):
                if len(emb) != self.dimension:
                    raise ValueError(f"Vector dimension {len(emb)} does not match store dimension {self.dimension}")

                packed_vec = _serialize_f32(emb)
                symbols_json = json.dumps(chunk.symbols)

                # Upsert into metadata table
                conn.execute(
                    """
                    INSERT INTO code_chunks (
                        id, file_path, chunk_type, name, start_line, end_line,
                        parent_context, symbols_json, docstring, content, raw_vector, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        file_path = excluded.file_path,
                        chunk_type = excluded.chunk_type,
                        name = excluded.name,
                        start_line = excluded.start_line,
                        end_line = excluded.end_line,
                        parent_context = excluded.parent_context,
                        symbols_json = excluded.symbols_json,
                        docstring = excluded.docstring,
                        content = excluded.content,
                        raw_vector = excluded.raw_vector,
                        updated_at = excluded.updated_at;
                    """,
                    (
                        chunk.id,
                        chunk.file_path,
                        chunk.chunk_type,
                        chunk.name,
                        chunk.start_line,
                        chunk.end_line,
                        chunk.parent_context,
                        symbols_json,
                        chunk.docstring,
                        chunk.content,
                        packed_vec,
                        now,
                    ),
                )

                # Upsert into sqlite-vec virtual table
                if self._has_sqlite_vec:
                    conn.execute("DELETE FROM vec_code_chunks WHERE chunk_id = ?;", (chunk.id,))
                    conn.execute(
                        """
                        INSERT INTO vec_code_chunks (chunk_id, embedding)
                        VALUES (?, ?);
                        """,
                        (chunk.id, packed_vec),
                    )

        return len(chunks)

    def index_symbols(self, symbols: List[Symbol]) -> int:
        """Store code definitions and symbols in the persistent symbol table."""
        if not symbols:
            return 0

        conn = self._get_connection()
        with conn:
            for s in symbols:
                conn.execute(
                    """
                    INSERT INTO code_symbols (
                        name, kind, file_path, start_line, end_line,
                        start_column, end_column, signature, docstring, parent_name
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(file_path, name, start_line) DO UPDATE SET
                        kind = excluded.kind,
                        end_line = excluded.end_line,
                        start_column = excluded.start_column,
                        end_column = excluded.end_column,
                        signature = excluded.signature,
                        docstring = excluded.docstring,
                        parent_name = excluded.parent_name;
                    """,
                    (
                        s.name,
                        s.kind.value,
                        s.file_path,
                        s.start_line,
                        s.end_line,
                        s.start_column,
                        s.end_column,
                        s.signature,
                        s.docstring,
                        s.parent_name,
                    ),
                )
        return len(symbols)

    def delete_file(self, file_path: str) -> None:
        """Purge all chunks, vectors, and symbols associated with a file."""
        conn = self._get_connection()
        with conn:
            if self._has_sqlite_vec:
                conn.execute(
                    """
                    DELETE FROM vec_code_chunks
                    WHERE chunk_id IN (SELECT id FROM code_chunks WHERE file_path = ?);
                    """,
                    (file_path,),
                )
            conn.execute("DELETE FROM code_chunks WHERE file_path = ?;", (file_path,))
            conn.execute("DELETE FROM code_symbols WHERE file_path = ?;", (file_path,))
            conn.execute("DELETE FROM file_registry WHERE file_path = ?;", (file_path,))

    def search(
        self,
        query_vector: List[float],
        limit: int = 5,
        file_filter: Optional[str] = None,
    ) -> List[SearchResult]:
        """Execute sub-second vector search for nearest neighbor code chunks."""
        if len(query_vector) != self.dimension:
            raise ValueError(f"Query vector dimension {len(query_vector)} != {self.dimension}")

        conn = self._get_connection()
        packed_query = _serialize_f32(query_vector)
        results: List[SearchResult] = []

        if self._has_sqlite_vec:
            # Query via sqlite-vec ANN index
            k_val = limit * 2 if file_filter else limit
            query_sql = """
                SELECT
                    c.id, c.file_path, c.chunk_type, c.name, c.start_line, c.end_line,
                    c.parent_context, c.symbols_json, c.docstring, c.content,
                    v.distance
                FROM vec_code_chunks v
                JOIN code_chunks c ON c.id = v.chunk_id
                WHERE v.embedding MATCH ? AND k = ?
            """
            params: List[Any] = [packed_query, k_val]
            if file_filter:
                query_sql += " AND c.file_path LIKE ?"
                params.append(f"%{file_filter}%")
            query_sql += " ORDER BY v.distance ASC LIMIT ?"
            params.append(limit)

            cursor = conn.execute(query_sql, params)
            for row in cursor.fetchall():
                symbols = json.loads(row["symbols_json"]) if row["symbols_json"] else []
                chunk = CodeChunk(
                    id=row["id"],
                    file_path=row["file_path"],
                    chunk_type=row["chunk_type"],
                    name=row["name"],
                    start_line=row["start_line"],
                    end_line=row["end_line"],
                    parent_context=row["parent_context"],
                    symbols=symbols,
                    docstring=row["docstring"],
                    content=row["content"],
                )
                dist = float(row["distance"])
                # L2 or cosine distance conversion to similarity score (higher = closer)
                score = 1.0 / (1.0 + max(0.0, dist))
                results.append(SearchResult(chunk=chunk, score=score, distance=dist))

        else:
            # Fallback in-memory cosine search over raw vectors
            query_sql = "SELECT id, file_path, chunk_type, name, start_line, end_line, parent_context, symbols_json, docstring, content, raw_vector FROM code_chunks"
            params = []
            if file_filter:
                query_sql += " WHERE file_path LIKE ?"
                params.append(f"%{file_filter}%")

            candidates = []
            for row in conn.execute(query_sql, params).fetchall():
                raw_blob = row["raw_vector"]
                if not raw_blob:
                    continue
                v = list(struct.unpack(f"{self.dimension}f", raw_blob))
                sim = _cosine_similarity(query_vector, v)
                dist = max(0.0, 1.0 - sim)
                candidates.append((sim, dist, row))

            candidates.sort(key=lambda x: x[0], reverse=True)
            for sim, dist, row in candidates[:limit]:
                symbols = json.loads(row["symbols_json"]) if row["symbols_json"] else []
                chunk = CodeChunk(
                    id=row["id"],
                    file_path=row["file_path"],
                    chunk_type=row["chunk_type"],
                    name=row["name"],
                    start_line=row["start_line"],
                    end_line=row["end_line"],
                    parent_context=row["parent_context"],
                    symbols=symbols,
                    docstring=row["docstring"],
                    content=row["content"],
                )
                results.append(SearchResult(chunk=chunk, score=sim, distance=dist))

        return results

    def find_symbols(
        self,
        name_query: str,
        kind: Optional[Union[SymbolKind, str]] = None,
        file_path: Optional[str] = None,
        limit: int = 20,
    ) -> List[Symbol]:
        """Locate symbol definitions by name prefix or exact match."""
        conn = self._get_connection()
        sql = "SELECT name, kind, file_path, start_line, end_line, start_column, end_column, signature, docstring, parent_name FROM code_symbols WHERE name LIKE ?"
        params: List[Any] = [f"%{name_query}%"]

        if kind:
            k_val = kind.value if isinstance(kind, SymbolKind) else str(kind)
            sql += " AND kind = ?"
            params.append(k_val)

        if file_path:
            sql += " AND file_path LIKE ?"
            params.append(f"%{file_path}%")

        sql += " ORDER BY name ASC LIMIT ?"
        params.append(limit)

        symbols: List[Symbol] = []
        for row in conn.execute(sql, params).fetchall():
            try:
                sk = SymbolKind(row["kind"])
            except ValueError:
                sk = SymbolKind.VARIABLE
            symbols.append(
                Symbol(
                    name=row["name"],
                    kind=sk,
                    file_path=row["file_path"],
                    start_line=row["start_line"],
                    end_line=row["end_line"],
                    start_column=row["start_column"],
                    end_column=row["end_column"],
                    signature=row["signature"],
                    docstring=row["docstring"],
                    parent_name=row["parent_name"],
                )
            )
        return symbols

    def get_file_hash(self, file_path: str) -> Optional[str]:
        """Check recorded hash for a file to skip re-indexing if unchanged."""
        conn = self._get_connection()
        row = conn.execute("SELECT file_hash FROM file_registry WHERE file_path = ?", (file_path,)).fetchone()
        return row["file_hash"] if row else None

    def update_file_hash(self, file_path: str, file_hash: str) -> None:
        """Update indexed hash and timestamp for file."""
        conn = self._get_connection()
        with conn:
            conn.execute(
                """
                INSERT INTO file_registry (file_path, file_hash, indexed_at)
                VALUES (?, ?, ?)
                ON CONFLICT(file_path) DO UPDATE SET
                    file_hash = excluded.file_hash,
                    indexed_at = excluded.indexed_at;
                """,
                (file_path, file_hash, time.time()),
            )

    def count_chunks(self) -> int:
        """Return total number of indexed code chunks."""
        conn = self._get_connection()
        row = conn.execute("SELECT count(*) FROM code_chunks").fetchone()
        return row[0] if row else 0

    def close(self) -> None:
        """Close connection."""
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
