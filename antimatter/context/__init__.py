"""Context & Code Intelligence layer for Project Antimatter."""

from antimatter.context.types import (
    CodeChunk,
    DefinitionLocation,
    Diagnostic,
    DiagnosticSeverity,
    HoverInfo,
    Reference,
    SearchResult,
    Symbol,
    SymbolKind,
)
from antimatter.context.ast_parser import CodeASTParser, detect_language
from antimatter.context.vector_store import SqliteVecStore
from antimatter.context.lsp_client import AsyncLSPClient
from antimatter.context.service import CodeIntelligenceService

__all__ = [
    "CodeChunk",
    "DefinitionLocation",
    "Diagnostic",
    "DiagnosticSeverity",
    "HoverInfo",
    "Reference",
    "SearchResult",
    "Symbol",
    "SymbolKind",
    "CodeASTParser",
    "detect_language",
    "SqliteVecStore",
    "AsyncLSPClient",
    "CodeIntelligenceService",
]
