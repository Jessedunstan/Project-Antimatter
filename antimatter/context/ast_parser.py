"""Tree-sitter AST parser for definitions, references, symbol hierarchies, and AST chunking."""

from __future__ import annotations
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import tree_sitter
from antimatter.context.types import (
    CodeChunk,
    Reference,
    Symbol,
    SymbolKind,
)

logger = logging.getLogger(__name__)

# Cache language instances to avoid reloading
_PARSERS: Dict[str, tree_sitter.Parser] = {}
_LANGUAGES: Dict[str, tree_sitter.Language] = {}


def _get_language_and_parser(lang_key: str) -> Tuple[Optional[tree_sitter.Language], Optional[tree_sitter.Parser]]:
    """Retrieve or initialize Tree-sitter Language and Parser for given language key."""
    if lang_key in _PARSERS:
        return _LANGUAGES[lang_key], _PARSERS[lang_key]

    lang: Optional[tree_sitter.Language] = None
    try:
        if lang_key == "python":
            import tree_sitter_python
            lang = tree_sitter.Language(tree_sitter_python.language())
        elif lang_key == "javascript":
            import tree_sitter_javascript
            lang = tree_sitter.Language(tree_sitter_javascript.language())
        elif lang_key == "typescript":
            import tree_sitter_typescript
            lang = tree_sitter.Language(tree_sitter_typescript.language_typescript())
        elif lang_key == "tsx":
            import tree_sitter_typescript
            lang = tree_sitter.Language(tree_sitter_typescript.language_tsx())
    except Exception as e:
        logger.warning("Could not load Tree-sitter grammar for '%s': %s", lang_key, e)
        return None, None

    if lang:
        parser = tree_sitter.Parser(lang)
        _LANGUAGES[lang_key] = lang
        _PARSERS[lang_key] = parser
        return lang, parser

    return None, None


def detect_language(file_path: Union[str, Path]) -> Optional[str]:
    """Detect language key from file extension."""
    suffix = Path(file_path).suffix.lower()
    if suffix == ".py":
        return "python"
    elif suffix in (".js", ".mjs", ".cjs", ".jsx"):
        return "javascript"
    elif suffix == ".ts":
        return "typescript"
    elif suffix == ".tsx":
        return "tsx"
    return None


class CodeASTParser:
    """Multi-language AST Parser extracting symbols, references, hierarchies, and AST chunks."""

    def __init__(self):
        pass

    def parse(self, code: str, language: str) -> Optional[tree_sitter.Tree]:
        """Parse source code into a Tree-sitter Tree."""
        _, parser = _get_language_and_parser(language)
        if not parser:
            return None
        return parser.parse(code.encode("utf-8"))

    def extract_symbols(
        self,
        code: str,
        file_path: str,
        language: Optional[str] = None,
    ) -> List[Symbol]:
        """Extract definitions (classes, functions, methods, variables) from source code."""
        lang = language or detect_language(file_path)
        if not lang:
            return []

        tree = self.parse(code, lang)
        if not tree:
            return []

        lines = code.splitlines()
        symbols: List[Symbol] = []

        if lang == "python":
            self._extract_python_symbols(tree.root_node, lines, file_path, symbols)
        elif lang in ("javascript", "typescript", "tsx"):
            self._extract_js_ts_symbols(tree.root_node, lines, file_path, symbols)

        return symbols

    def extract_symbol_hierarchy(
        self,
        code: str,
        file_path: str,
        language: Optional[str] = None,
    ) -> List[Symbol]:
        """Extract nested symbol hierarchy representing parent-child tree."""
        flat_symbols = self.extract_symbols(code, file_path, language)
        return self._build_hierarchy(flat_symbols)

    def extract_references(
        self,
        code: str,
        file_path: str,
        language: Optional[str] = None,
    ) -> List[Reference]:
        """Extract identifier references and usage points throughout the AST."""
        lang = language or detect_language(file_path)
        if not lang:
            return []

        tree = self.parse(code, lang)
        if not tree:
            return []

        symbols = self.extract_symbols(code, file_path, lang)
        definition_spans: Set[Tuple[int, int]] = {
            (s.start_line, s.start_column) for s in symbols
        }

        references: List[Reference] = []
        self._find_references_recursive(tree.root_node, file_path, definition_spans, references, None)
        return references

    def chunk_code(
        self,
        code: str,
        file_path: str,
        language: Optional[str] = None,
        max_chunk_lines: int = 50,
    ) -> List[CodeChunk]:
        """Intelligently chunk code using AST boundaries.
        
        Preserves logical blocks: class bodies, methods (with parent context),
        functions, and module headers.
        """
        lang = language or detect_language(file_path)
        lines = code.splitlines()
        total_lines = len(lines)

        if not lang or total_lines == 0:
            # Fallback to line-based chunking for unsupported languages
            return self._fallback_line_chunks(code, file_path, max_chunk_lines)

        tree = self.parse(code, lang)
        if not tree:
            return self._fallback_line_chunks(code, file_path, max_chunk_lines)

        chunks: List[CodeChunk] = []
        covered_line_ranges: List[Tuple[int, int]] = []

        # 1. AST-based chunking for Python
        if lang == "python":
            self._chunk_python_ast(tree.root_node, lines, file_path, max_chunk_lines, chunks, covered_line_ranges)
        elif lang in ("javascript", "typescript", "tsx"):
            self._chunk_js_ts_ast(tree.root_node, lines, file_path, max_chunk_lines, chunks, covered_line_ranges)
        else:
            return self._fallback_line_chunks(code, file_path, max_chunk_lines)

        # 2. Add module header or unchunked top-level code (imports, constants)
        self._add_uncovered_chunks(lines, file_path, covered_line_ranges, chunks)

        # Sort chunks deterministically by start_line
        chunks.sort(key=lambda c: c.start_line)
        return chunks

    # =========================================================================
    # Python Symbol Extraction
    # =========================================================================

    def _extract_python_symbols(
        self,
        node: tree_sitter.Node,
        lines: List[str],
        file_path: str,
        symbols: List[Symbol],
        parent_name: Optional[str] = None,
    ) -> None:
        for child in node.children:
            curr = child
            decorators: List[str] = []

            # Handle decorated definitions
            if curr.type == "decorated_definition":
                for sub in curr.children:
                    if sub.type == "decorator":
                        decorators.append(self._node_text(sub))
                    elif sub.type in ("function_definition", "class_definition"):
                        curr = sub
                        break

            if curr.type == "class_definition":
                name_node = curr.child_by_field_name("name")
                if name_node:
                    name = self._node_text(name_node)
                    start_l = curr.start_point.row + 1
                    end_l = curr.end_point.row + 1
                    sig = lines[start_l - 1].strip() if start_l <= len(lines) else f"class {name}:"
                    doc = self._get_python_docstring(curr)
                    sym = Symbol(
                        name=name,
                        kind=SymbolKind.CLASS,
                        file_path=file_path,
                        start_line=start_l,
                        end_line=end_l,
                        start_column=curr.start_point.column,
                        end_column=curr.end_point.column,
                        signature=sig,
                        docstring=doc,
                        parent_name=parent_name,
                    )
                    symbols.append(sym)

                    # Inspect class body for methods and attributes
                    body = curr.child_by_field_name("body")
                    if body:
                        self._extract_python_symbols(body, lines, file_path, symbols, parent_name=name)

            elif curr.type == "function_definition":
                name_node = curr.child_by_field_name("name")
                if name_node:
                    name = self._node_text(name_node)
                    start_l = curr.start_point.row + 1
                    end_l = curr.end_point.row + 1
                    sig = lines[start_l - 1].strip() if start_l <= len(lines) else f"def {name}():"
                    doc = self._get_python_docstring(curr)
                    kind = SymbolKind.METHOD if parent_name else SymbolKind.FUNCTION
                    sym = Symbol(
                        name=name,
                        kind=kind,
                        file_path=file_path,
                        start_line=start_l,
                        end_line=end_l,
                        start_column=curr.start_point.column,
                        end_column=curr.end_point.column,
                        signature=sig,
                        docstring=doc,
                        parent_name=parent_name,
                    )
                    symbols.append(sym)

                    # Look for nested functions
                    body = curr.child_by_field_name("body")
                    if body:
                        self._extract_python_symbols(body, lines, file_path, symbols, parent_name=name)

            elif curr.type == "expression_statement" and parent_name is None:
                # Top level assignments (e.g. CONSTANT = 123)
                for expr in curr.children:
                    if expr.type == "assignment":
                        left = expr.child_by_field_name("left")
                        if left and left.type == "identifier":
                            var_name = self._node_text(left)
                            start_l = curr.start_point.row + 1
                            end_l = curr.end_point.row + 1
                            sym = Symbol(
                                name=var_name,
                                kind=SymbolKind.CONSTANT if var_name.isupper() else SymbolKind.VARIABLE,
                                file_path=file_path,
                                start_line=start_l,
                                end_line=end_l,
                                start_column=curr.start_point.column,
                                end_column=curr.end_point.column,
                                signature=lines[start_l - 1].strip() if start_l <= len(lines) else var_name,
                                parent_name=None,
                            )
                            symbols.append(sym)

    def _get_python_docstring(self, def_node: tree_sitter.Node) -> Optional[str]:
        """Extract docstring string node from function/class body."""
        body = def_node.child_by_field_name("body")
        if not body:
            return None
        for child in body.children:
            if child.type == "expression_statement":
                for sub in child.children:
                    if sub.type == "string":
                        raw = self._node_text(sub)
                        # Strip outer quotes (triple or single)
                        for q in ('"""', "'''", '"', "'"):
                            if raw.startswith(q) and raw.endswith(q) and len(raw) >= 2 * len(q):
                                return raw[len(q):-len(q)].strip()
                        return raw.strip()
            elif child.type not in ("comment",):
                break
        return None

    # =========================================================================
    # JS/TS Symbol Extraction
    # =========================================================================

    def _extract_js_ts_symbols(
        self,
        node: tree_sitter.Node,
        lines: List[str],
        file_path: str,
        symbols: List[Symbol],
        parent_name: Optional[str] = None,
    ) -> None:
        for child in node.children:
            curr = child
            if curr.type == "export_statement":
                for sub in curr.children:
                    if sub.type in ("class_declaration", "function_declaration", "interface_declaration", "type_alias_declaration", "lexical_declaration"):
                        curr = sub
                        break

            if curr.type == "class_declaration":
                name_node = curr.child_by_field_name("name")
                if name_node:
                    name = self._node_text(name_node)
                    start_l = curr.start_point.row + 1
                    end_l = curr.end_point.row + 1
                    sig = lines[start_l - 1].strip() if start_l <= len(lines) else f"class {name}"
                    sym = Symbol(
                        name=name,
                        kind=SymbolKind.CLASS,
                        file_path=file_path,
                        start_line=start_l,
                        end_line=end_l,
                        start_column=curr.start_point.column,
                        end_column=curr.end_point.column,
                        signature=sig,
                        parent_name=parent_name,
                    )
                    symbols.append(sym)
                    body = curr.child_by_field_name("body")
                    if body:
                        self._extract_js_ts_symbols(body, lines, file_path, symbols, parent_name=name)

            elif curr.type == "function_declaration":
                name_node = curr.child_by_field_name("name")
                if name_node:
                    name = self._node_text(name_node)
                    start_l = curr.start_point.row + 1
                    end_l = curr.end_point.row + 1
                    sig = lines[start_l - 1].strip() if start_l <= len(lines) else f"function {name}()"
                    sym = Symbol(
                        name=name,
                        kind=SymbolKind.FUNCTION,
                        file_path=file_path,
                        start_line=start_l,
                        end_line=end_l,
                        start_column=curr.start_point.column,
                        end_column=curr.end_point.column,
                        signature=sig,
                        parent_name=parent_name,
                    )
                    symbols.append(sym)

            elif curr.type == "method_definition":
                name_node = curr.child_by_field_name("name")
                if name_node:
                    name = self._node_text(name_node)
                    start_l = curr.start_point.row + 1
                    end_l = curr.end_point.row + 1
                    sig = lines[start_l - 1].strip() if start_l <= len(lines) else f"{name}()"
                    sym = Symbol(
                        name=name,
                        kind=SymbolKind.METHOD,
                        file_path=file_path,
                        start_line=start_l,
                        end_line=end_l,
                        start_column=curr.start_point.column,
                        end_column=curr.end_point.column,
                        signature=sig,
                        parent_name=parent_name,
                    )
                    symbols.append(sym)

            elif curr.type == "interface_declaration":
                name_node = curr.child_by_field_name("name")
                if name_node:
                    name = self._node_text(name_node)
                    start_l = curr.start_point.row + 1
                    end_l = curr.end_point.row + 1
                    sym = Symbol(
                        name=name,
                        kind=SymbolKind.INTERFACE,
                        file_path=file_path,
                        start_line=start_l,
                        end_line=end_l,
                        start_column=curr.start_point.column,
                        end_column=curr.end_point.column,
                        signature=lines[start_l - 1].strip() if start_l <= len(lines) else f"interface {name}",
                        parent_name=parent_name,
                    )
                    symbols.append(sym)

    # =========================================================================
    # References & Hierarchy
    # =========================================================================

    def _find_references_recursive(
        self,
        node: tree_sitter.Node,
        file_path: str,
        def_spans: Set[Tuple[int, int]],
        references: List[Reference],
        enclosing: Optional[str] = None,
    ) -> None:
        curr_enclosing = enclosing
        if node.type in ("function_definition", "class_definition", "method_definition"):
            name_node = node.child_by_field_name("name")
            if name_node:
                nm = self._node_text(name_node)
                curr_enclosing = f"{enclosing}.{nm}" if enclosing else nm

        # Check for identifiers (usages)
        if node.type in ("identifier", "property_identifier"):
            # Exclude if this node is the name identifier of a definition
            is_def_name = False
            if node.parent:
                try:
                    if node.parent.child_by_field_name("name") == node:
                        is_def_name = True
                except Exception:
                    pass

            if not is_def_name:
                line = node.start_point.row + 1
                col = node.start_point.column
                if (line, col) not in def_spans:
                    name = self._node_text(node)
                    # Ignore language keywords or single-character noise
                    if len(name) > 1 and not name.startswith("__"):
                        references.append(
                            Reference(
                                name=name,
                                file_path=file_path,
                                line=line,
                                column=col,
                                enclosing_symbol=curr_enclosing,
                            )
                        )

        for child in node.children:
            self._find_references_recursive(child, file_path, def_spans, references, curr_enclosing)

    def _build_hierarchy(self, flat_symbols: List[Symbol]) -> List[Symbol]:
        """Convert flat list of symbols with parent_name into tree."""
        by_name: Dict[str, Symbol] = {}
        for s in flat_symbols:
            by_name[s.name] = s.model_copy()

        root_symbols: List[Symbol] = []
        for s in flat_symbols:
            copied = by_name[s.name]
            if s.parent_name and s.parent_name in by_name:
                by_name[s.parent_name].children.append(copied)
            else:
                root_symbols.append(copied)

        return root_symbols

    # =========================================================================
    # AST Chunking
    # =========================================================================

    def _chunk_python_ast(
        self,
        root_node: tree_sitter.Node,
        lines: List[str],
        file_path: str,
        max_chunk_lines: int,
        chunks: List[CodeChunk],
        covered_ranges: List[Tuple[int, int]],
    ) -> None:
        for child in root_node.children:
            node = child
            if node.type == "decorated_definition":
                for sub in node.children:
                    if sub.type in ("function_definition", "class_definition"):
                        # Keep the entire decorated node range
                        break

            start_l = node.start_point.row + 1
            end_l = node.end_point.row + 1
            length = end_l - start_l + 1

            if node.type in ("class_definition", "decorated_definition") and "class_definition" in (node.type, [c.type for c in node.children]):
                actual_class = node if node.type == "class_definition" else next(c for c in node.children if c.type == "class_definition")
                class_name_node = actual_class.child_by_field_name("name")
                class_name = self._node_text(class_name_node) if class_name_node else "AnonymousClass"

                if length <= max_chunk_lines:
                    # Class is compact: store as single chunk
                    content = "\n".join(lines[start_l - 1:end_l])
                    cid = self._chunk_id(file_path, "class", class_name, start_l)
                    chunks.append(
                        CodeChunk(
                            id=cid,
                            file_path=file_path,
                            chunk_type="class",
                            name=class_name,
                            content=content,
                            start_line=start_l,
                            end_line=end_l,
                            parent_context=None,
                            symbols=[class_name],
                            docstring=self._get_python_docstring(actual_class),
                        )
                    )
                    covered_ranges.append((start_l, end_l))
                else:
                    # Class is large: chunk methods separately while maintaining class context
                    body = actual_class.child_by_field_name("body")
                    class_header_sig = lines[start_l - 1].strip()
                    parent_ctx = f"class {class_name}"

                    # 1. Class header chunk
                    doc = self._get_python_docstring(actual_class)
                    header_end = min(start_l + 10, end_l)
                    header_content = "\n".join(lines[start_l - 1:header_end])
                    chunks.append(
                        CodeChunk(
                            id=self._chunk_id(file_path, "class_header", class_name, start_l),
                            file_path=file_path,
                            chunk_type="class_header",
                            name=class_name,
                            content=f"# Parent Context: {parent_ctx}\n{header_content}",
                            start_line=start_l,
                            end_line=header_end,
                            parent_context=parent_ctx,
                            symbols=[class_name],
                            docstring=doc,
                        )
                    )

                    # 2. Method chunks
                    if body:
                        for bchild in body.children:
                            meth_node = bchild
                            if meth_node.type == "decorated_definition":
                                sub_meth = next((sc for sc in meth_node.children if sc.type == "function_definition"), None)
                                if sub_meth:
                                    m_name_node = sub_meth.child_by_field_name("name")
                                    m_name = self._node_text(m_name_node) if m_name_node else "anon"
                                    m_start = meth_node.start_point.row + 1
                                    m_end = meth_node.end_point.row + 1
                                    m_content = "\n".join(lines[m_start - 1:m_end])
                                    chunks.append(
                                        CodeChunk(
                                            id=self._chunk_id(file_path, "method", f"{class_name}.{m_name}", m_start),
                                            file_path=file_path,
                                            chunk_type="method",
                                            name=f"{class_name}.{m_name}",
                                            content=f"# Context: {parent_ctx}\n{m_content}",
                                            start_line=m_start,
                                            end_line=m_end,
                                            parent_context=parent_ctx,
                                            symbols=[class_name, m_name],
                                            docstring=self._get_python_docstring(sub_meth),
                                        )
                                    )
                            elif meth_node.type == "function_definition":
                                m_name_node = meth_node.child_by_field_name("name")
                                m_name = self._node_text(m_name_node) if m_name_node else "anon"
                                m_start = meth_node.start_point.row + 1
                                m_end = meth_node.end_point.row + 1
                                m_content = "\n".join(lines[m_start - 1:m_end])
                                chunks.append(
                                    CodeChunk(
                                        id=self._chunk_id(file_path, "method", f"{class_name}.{m_name}", m_start),
                                        file_path=file_path,
                                        chunk_type="method",
                                        name=f"{class_name}.{m_name}",
                                        content=f"# Context: {parent_ctx}\n{m_content}",
                                        start_line=m_start,
                                        end_line=m_end,
                                        parent_context=parent_ctx,
                                        symbols=[class_name, m_name],
                                        docstring=self._get_python_docstring(meth_node),
                                    )
                                )
                    covered_ranges.append((start_l, end_l))

            elif node.type in ("function_definition", "decorated_definition"):
                actual_fn = node if node.type == "function_definition" else next(c for c in node.children if c.type == "function_definition")
                fn_name_node = actual_fn.child_by_field_name("name")
                fn_name = self._node_text(fn_name_node) if fn_name_node else "AnonymousFunction"
                content = "\n".join(lines[start_l - 1:end_l])
                cid = self._chunk_id(file_path, "function", fn_name, start_l)
                chunks.append(
                    CodeChunk(
                        id=cid,
                        file_path=file_path,
                        chunk_type="function",
                        name=fn_name,
                        content=content,
                        start_line=start_l,
                        end_line=end_l,
                        parent_context=None,
                        symbols=[fn_name],
                        docstring=self._get_python_docstring(actual_fn),
                    )
                )
                covered_ranges.append((start_l, end_l))

    def _chunk_js_ts_ast(
        self,
        root_node: tree_sitter.Node,
        lines: List[str],
        file_path: str,
        max_chunk_lines: int,
        chunks: List[CodeChunk],
        covered_ranges: List[Tuple[int, int]],
    ) -> None:
        for child in root_node.children:
            curr = child
            if curr.type == "export_statement":
                for sub in curr.children:
                    if sub.type in ("class_declaration", "function_declaration", "interface_declaration"):
                        curr = sub
                        break

            start_l = child.start_point.row + 1
            end_l = child.end_point.row + 1

            if curr.type == "class_declaration":
                name_node = curr.child_by_field_name("name")
                c_name = self._node_text(name_node) if name_node else "Class"
                content = "\n".join(lines[start_l - 1:end_l])
                chunks.append(
                    CodeChunk(
                        id=self._chunk_id(file_path, "class", c_name, start_l),
                        file_path=file_path,
                        chunk_type="class",
                        name=c_name,
                        content=content,
                        start_line=start_l,
                        end_line=end_l,
                        parent_context=None,
                        symbols=[c_name],
                    )
                )
                covered_ranges.append((start_l, end_l))

            elif curr.type == "function_declaration":
                name_node = curr.child_by_field_name("name")
                f_name = self._node_text(name_node) if name_node else "Function"
                content = "\n".join(lines[start_l - 1:end_l])
                chunks.append(
                    CodeChunk(
                        id=self._chunk_id(file_path, "function", f_name, start_l),
                        file_path=file_path,
                        chunk_type="function",
                        name=f_name,
                        content=content,
                        start_line=start_l,
                        end_line=end_l,
                        symbols=[f_name],
                    )
                )
                covered_ranges.append((start_l, end_l))

            elif curr.type == "interface_declaration":
                name_node = curr.child_by_field_name("name")
                i_name = self._node_text(name_node) if name_node else "Interface"
                content = "\n".join(lines[start_l - 1:end_l])
                chunks.append(
                    CodeChunk(
                        id=self._chunk_id(file_path, "interface", i_name, start_l),
                        file_path=file_path,
                        chunk_type="interface",
                        name=i_name,
                        content=content,
                        start_line=start_l,
                        end_line=end_l,
                        symbols=[i_name],
                    )
                )
                covered_ranges.append((start_l, end_l))

    def _add_uncovered_chunks(
        self,
        lines: List[str],
        file_path: str,
        covered_ranges: List[Tuple[int, int]],
        chunks: List[CodeChunk],
    ) -> None:
        """Collect top-level imports and module code not inside classes/functions."""
        total_lines = len(lines)
        if total_lines == 0:
            return

        # Build boolean array of covered lines
        covered = [False] * (total_lines + 1)
        for s, e in covered_ranges:
            for l in range(max(1, s), min(total_lines, e) + 1):
                covered[l] = True

        idx = 1
        while idx <= total_lines:
            # Skip empty lines or covered lines
            if covered[idx] or not lines[idx - 1].strip():
                idx += 1
                continue

            block_start = idx
            while idx <= total_lines and not covered[idx] and (idx - block_start < 40):
                idx += 1
            block_end = idx - 1

            block_text = "\n".join(lines[block_start - 1:block_end]).strip()
            if block_text:
                chunk_type = "module_header" if block_start == 1 else "top_level"
                cid = self._chunk_id(file_path, chunk_type, f"lines_{block_start}_{block_end}", block_start)
                chunks.append(
                    CodeChunk(
                        id=cid,
                        file_path=file_path,
                        chunk_type=chunk_type,
                        name=f"{Path(file_path).stem}:{chunk_type}",
                        content=block_text,
                        start_line=block_start,
                        end_line=block_end,
                        parent_context=None,
                        symbols=[],
                    )
                )

    def _fallback_line_chunks(
        self,
        code: str,
        file_path: str,
        chunk_size: int = 50,
        overlap: int = 10,
    ) -> List[CodeChunk]:
        """Fallback chunker for plain text or unsupported formats."""
        lines = code.splitlines()
        chunks: List[CodeChunk] = []
        if not lines:
            return chunks

        i = 0
        step = max(1, chunk_size - overlap)
        while i < len(lines):
            end = min(i + chunk_size, len(lines))
            chunk_lines = lines[i:end]
            content = "\n".join(chunk_lines)
            start_l = i + 1
            end_l = end
            cid = self._chunk_id(file_path, "block", f"lines_{start_l}_{end_l}", start_l)
            chunks.append(
                CodeChunk(
                    id=cid,
                    file_path=file_path,
                    chunk_type="block",
                    name=f"{Path(file_path).name}:{start_l}-{end_l}",
                    content=content,
                    start_line=start_l,
                    end_line=end_l,
                )
            )
            i += step

        return chunks

    @staticmethod
    def _node_text(node: tree_sitter.Node) -> str:
        return node.text.decode("utf-8")

    @staticmethod
    def _chunk_id(file_path: str, chunk_type: str, name: str, start_line: int) -> str:
        raw = f"{file_path}:{chunk_type}:{name}:{start_line}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
