"""Tree-sitter AST Code Parser and Semantic Chunker for Project Antimatter.

Extracts symbols, function signatures, class hierarchies, and performs
AST-aware semantic chunking for local codebase intelligence.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional
import tree_sitter
import tree_sitter_python
import tree_sitter_javascript


class SymbolType(str, Enum):
    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    IMPORT = "import"
    VARIABLE = "variable"
    MODULE = "module"


@dataclass
class CodeSymbol:
    """Represents a code symbol extracted from AST."""
    name: str
    symbol_type: SymbolType
    start_line: int
    end_line: int
    file_path: str = ""
    signature: Optional[str] = None
    docstring: Optional[str] = None
    parent_symbol: Optional[str] = None


@dataclass
class CodeChunk:
    """AST-aligned source code chunk with semantic metadata."""
    chunk_id: str
    file_path: str
    symbol_name: Optional[str]
    symbol_type: SymbolType
    start_line: int
    end_line: int
    content: str
    context_prefix: str = ""  # e.g., enclosing class header

    def full_text_for_embedding(self) -> str:
        """Text payload passed to embedding model."""
        header = f"# File: {self.file_path}"
        if self.symbol_name:
            header += f" | {self.symbol_type.value}: {self.symbol_name}"
        if self.context_prefix:
            header += f"\n{self.context_prefix}"
        return f"{header}\n{self.content}"


class ASTParser:
    """Multi-language Tree-sitter parser with AST chunking."""

    def __init__(self):
        self._languages: Dict[str, tree_sitter.Language] = {}
        self._parsers: Dict[str, tree_sitter.Parser] = {}
        self._init_languages()

    def _init_languages(self) -> None:
        """Initialize language grammars."""
        # Python
        py_lang = tree_sitter.Language(tree_sitter_python.language())
        self._languages["python"] = py_lang
        self._parsers["python"] = tree_sitter.Parser(py_lang)

        # JavaScript / TypeScript
        js_lang = tree_sitter.Language(tree_sitter_javascript.language())
        self._languages["javascript"] = js_lang
        self._parsers["javascript"] = tree_sitter.Parser(js_lang)

    def detect_language(self, file_path: str | Path) -> Optional[str]:
        """Detect language identifier from file extension."""
        ext = Path(file_path).suffix.lower()
        if ext in (".py", ".pyw"):
            return "python"
        if ext in (".js", ".jsx", ".ts", ".tsx", ".mjs"):
            return "javascript"
        return None

    def parse(self, code: str, language: str) -> tree_sitter.Tree:
        """Parse source code into Tree-sitter Tree."""
        parser = self._parsers.get(language)
        if not parser:
            raise ValueError(f"Language '{language}' not supported by ASTParser.")
        return parser.parse(code.encode("utf-8"))

    def extract_symbols(self, code: str, file_path: str = "") -> List[CodeSymbol]:
        """Extract all top-level and nested symbols from source code."""
        lang = self.detect_language(file_path)
        if not lang or lang not in self._parsers:
            return []

        tree = self.parse(code, lang)
        symbols: List[CodeSymbol] = []
        code_lines = code.splitlines()

        if lang == "python":
            self._extract_python_symbols(tree.root_node, code_lines, file_path, symbols)
        elif lang == "javascript":
            self._extract_js_symbols(tree.root_node, code_lines, file_path, symbols)

        return symbols

    def _extract_python_symbols(
        self,
        node: tree_sitter.Node,
        code_lines: List[str],
        file_path: str,
        symbols: List[CodeSymbol],
        parent_name: Optional[str] = None,
    ) -> None:
        """Traverse Python AST nodes to extract functions and classes."""
        for child in node.children:
            if child.type == "class_definition":
                name_node = child.child_by_field_name("name")
                class_name = name_node.text.decode("utf-8") if name_node and name_node.text else "AnonymousClass"
                start_line = child.start_point.row + 1
                end_line = child.end_point.row + 1

                symbols.append(
                    CodeSymbol(
                        name=class_name,
                        symbol_type=SymbolType.CLASS,
                        start_line=start_line,
                        end_line=end_line,
                        file_path=file_path,
                        parent_symbol=parent_name,
                    )
                )

                # Traverse class body for methods
                body_node = child.child_by_field_name("body")
                if body_node:
                    self._extract_python_symbols(body_node, code_lines, file_path, symbols, parent_name=class_name)

            elif child.type == "function_definition":
                name_node = child.child_by_field_name("name")
                func_name = name_node.text.decode("utf-8") if name_node and name_node.text else "anonymous"
                params_node = child.child_by_field_name("parameters")
                sig = f"{func_name}{params_node.text.decode('utf-8')}" if params_node and params_node.text else func_name

                start_line = child.start_point.row + 1
                end_line = child.end_point.row + 1

                symbols.append(
                    CodeSymbol(
                        name=func_name,
                        symbol_type=SymbolType.METHOD if parent_name else SymbolType.FUNCTION,
                        start_line=start_line,
                        end_line=end_line,
                        file_path=file_path,
                        signature=sig,
                        parent_symbol=parent_name,
                    )
                )

    def _extract_js_symbols(
        self,
        node: tree_sitter.Node,
        code_lines: List[str],
        file_path: str,
        symbols: List[CodeSymbol],
        parent_name: Optional[str] = None,
    ) -> None:
        """Traverse JS/TS AST nodes to extract functions and classes."""
        for child in node.children:
            if child.type in ("function_declaration", "method_definition"):
                name_node = child.child_by_field_name("name")
                func_name = name_node.text.decode("utf-8") if name_node and name_node.text else "anonymous"
                start_line = child.start_point.row + 1
                end_line = child.end_point.row + 1

                symbols.append(
                    CodeSymbol(
                        name=func_name,
                        symbol_type=SymbolType.METHOD if parent_name else SymbolType.FUNCTION,
                        start_line=start_line,
                        end_line=end_line,
                        file_path=file_path,
                        parent_symbol=parent_name,
                    )
                )
            elif child.type == "class_declaration":
                name_node = child.child_by_field_name("name")
                class_name = name_node.text.decode("utf-8") if name_node and name_node.text else "AnonymousClass"
                start_line = child.start_point.row + 1
                end_line = child.end_point.row + 1

                symbols.append(
                    CodeSymbol(
                        name=class_name,
                        symbol_type=SymbolType.CLASS,
                        start_line=start_line,
                        end_line=end_line,
                        file_path=file_path,
                        parent_symbol=parent_name,
                    )
                )
                body_node = child.child_by_field_name("body")
                if body_node:
                    self._extract_js_symbols(body_node, code_lines, file_path, symbols, parent_name=class_name)

    def chunk_source_code(
        self,
        code: str,
        file_path: str,
        max_chunk_lines: int = 60,
    ) -> List[CodeChunk]:
        """
        Split source code into AST-aware semantic chunks.
        Guarantees that functions and classes are kept intact when possible.
        """
        lang = self.detect_language(file_path)
        code_lines = code.splitlines()
        total_lines = len(code_lines)
        if total_lines == 0:
            return []

        if not lang or lang not in self._parsers:
            # Fallback to rolling line chunking for non-AST languages
            chunks: List[CodeChunk] = []
            for i in range(0, total_lines, max_chunk_lines):
                end_idx = min(i + max_chunk_lines, total_lines)
                content = "\n".join(code_lines[i:end_idx])
                chunks.append(
                    CodeChunk(
                        chunk_id=f"{file_path}#L{i+1}-L{end_idx}",
                        file_path=file_path,
                        symbol_name=None,
                        symbol_type=SymbolType.MODULE,
                        start_line=i + 1,
                        end_line=end_idx,
                        content=content,
                    )
                )
            return chunks

        # AST-driven chunking: extract symbols and generate focused chunks
        symbols = self.extract_symbols(code, file_path)
        chunks: List[CodeChunk] = []
        covered_lines = set()

        for sym in symbols:
            if sym.symbol_type in (SymbolType.FUNCTION, SymbolType.METHOD, SymbolType.CLASS):
                chunk_lines = code_lines[sym.start_line - 1 : sym.end_line]
                content = "\n".join(chunk_lines)

                prefix = f"class {sym.parent_symbol}:" if sym.parent_symbol else ""
                chunks.append(
                    CodeChunk(
                        chunk_id=f"{file_path}#{sym.name}#L{sym.start_line}-L{sym.end_line}",
                        file_path=file_path,
                        symbol_name=sym.name,
                        symbol_type=sym.symbol_type,
                        start_line=sym.start_line,
                        end_line=sym.end_line,
                        content=content,
                        context_prefix=prefix,
                    )
                )
                for line_num in range(sym.start_line, sym.end_line + 1):
                    covered_lines.add(line_num)

        # Catch remaining top-level code (imports, globals, script logic)
        uncovered = sorted(set(range(1, total_lines + 1)) - covered_lines)
        if uncovered:
            current_block = [uncovered[0]]
            for ln in uncovered[1:]:
                if ln == current_block[-1] + 1:
                    current_block.append(ln)
                else:
                    self._create_uncovered_chunk(current_block, code_lines, file_path, chunks)
                    current_block = [ln]
            if current_block:
                self._create_uncovered_chunk(current_block, code_lines, file_path, chunks)

        return sorted(chunks, key=lambda c: c.start_line)

    def _create_uncovered_chunk(
        self,
        block_lines: List[int],
        code_lines: List[str],
        file_path: str,
        chunks: List[CodeChunk],
    ) -> None:
        """Create a module-level chunk for lines outside explicit functions/classes."""
        s = block_lines[0]
        e = block_lines[-1]
        raw = "\n".join(code_lines[s - 1 : e]).strip()
        if raw:
            chunks.append(
                CodeChunk(
                    chunk_id=f"{file_path}#module#L{s}-L{e}",
                    file_path=file_path,
                    symbol_name="<module_scope>",
                    symbol_type=SymbolType.MODULE,
                    start_line=s,
                    end_line=e,
                    content=raw,
                )
            )
