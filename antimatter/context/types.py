"""Data types and contracts for Code Intelligence & Deep Context Layer."""

from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SymbolKind(str, Enum):
    """Classification of code symbols."""
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    VARIABLE = "variable"
    CONSTANT = "constant"
    INTERFACE = "interface"
    ENUM = "enum"
    STRUCT = "struct"
    IMPORT = "import"


class Symbol(BaseModel):
    """An extracted symbol definition (e.g., class, function, method)."""
    name: str = Field(description="Symbol identifier name")
    kind: SymbolKind = Field(description="Symbol kind")
    file_path: str = Field(description="Path to the file containing this symbol")
    start_line: int = Field(description="1-based start line")
    end_line: int = Field(description="1-based end line")
    start_column: int = Field(default=0, description="0-based start column")
    end_column: int = Field(default=0, description="0-based end column")
    signature: Optional[str] = Field(default=None, description="Full signature or header")
    docstring: Optional[str] = Field(default=None, description="Docstring or comment if any")
    parent_name: Optional[str] = Field(default=None, description="Parent symbol name if nested")
    children: List[Symbol] = Field(default_factory=list, description="Nested child symbols")


class Reference(BaseModel):
    """Usage or reference to a symbol identifier."""
    name: str = Field(description="Referenced symbol identifier")
    file_path: str = Field(description="Path to referencing file")
    line: int = Field(description="1-based line of reference")
    column: int = Field(description="0-based column of reference")
    enclosing_symbol: Optional[str] = Field(default=None, description="Enclosing symbol if any")


class CodeChunk(BaseModel):
    """Semantically chunked code unit derived from AST boundaries."""
    id: str = Field(description="Unique deterministic chunk ID")
    file_path: str = Field(description="Relative or absolute file path")
    chunk_type: str = Field(description="Type: 'class', 'function', 'method', 'block', etc.")
    name: str = Field(description="Symbol or block identifier")
    content: str = Field(description="Raw source text of the chunk")
    start_line: int = Field(description="1-based start line")
    end_line: int = Field(description="1-based end line")
    parent_context: Optional[str] = Field(default=None, description="Contextual parent line (e.g. class header)")
    symbols: List[str] = Field(default_factory=list, description="Symbols defined or referenced in chunk")
    docstring: Optional[str] = Field(default=None, description="Extracted docstring")


class SearchResult(BaseModel):
    """Vector search retrieval result."""
    chunk: CodeChunk
    score: float = Field(description="Similarity score (higher is more relevant)")
    distance: float = Field(description="Raw vector distance metric")


class DiagnosticSeverity(int, Enum):
    """LSP Diagnostic severity matching protocol specification."""
    ERROR = 1
    WARNING = 2
    INFORMATION = 3
    HINT = 4


class Diagnostic(BaseModel):
    """Language server or parser diagnostic warning/error."""
    file_path: str
    message: str
    severity: DiagnosticSeverity
    line: int = Field(description="1-based line number")
    column: int = Field(default=0, description="0-based column")
    end_line: Optional[int] = None
    end_column: Optional[int] = None
    source: Optional[str] = None
    code: Optional[str] = None


class HoverInfo(BaseModel):
    """Type information and documentation on hover."""
    file_path: str
    contents: str
    line: int
    column: int
    range_info: Optional[Dict[str, Any]] = None


class DefinitionLocation(BaseModel):
    """Go-to-definition target location."""
    file_path: str
    line: int = Field(description="1-based line number")
    column: int = Field(default=0, description="0-based column")
    end_line: Optional[int] = None
    end_column: Optional[int] = None
