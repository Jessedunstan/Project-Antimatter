"""Local Python LSP Server communicating over standard I/O (JSON-RPC 2.0).

Provides local offline hover, diagnostics, and definition resolution
without requiring external cloud telemetry or unverified binaries.
"""

from __future__ import annotations
import ast
import json
import logging
import sys
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def write_lsp_message(msg: Dict[str, Any]) -> None:
    """Encode and write JSON-RPC message with Content-Length header to stdout."""
    body = json.dumps(msg).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("latin1")
    sys.stdout.buffer.write(header + body)
    sys.stdout.buffer.flush()


class LocalPythonLspServer:
    """Lightweight in-process / subprocess LSP server for Python."""

    def __init__(self):
        self._documents: Dict[str, str] = {}
        self._is_shutdown = False

    def run(self) -> None:
        """Run the main server loop reading from stdin buffer."""
        buffer = bytearray()
        while not self._is_shutdown:
            line = sys.stdin.buffer.readline()
            if not line:
                break
            if line.startswith(b"Content-Length:"):
                length_str = line.split(b":")[1].strip()
                content_length = int(length_str)
                # Read until empty header line \r\n
                while True:
                    empty_line = sys.stdin.buffer.readline()
                    if empty_line in (b"\r\n", b"\n", b""):
                        break
                # Read exact content bytes
                content = sys.stdin.buffer.read(content_length)
                try:
                    payload = json.loads(content.decode("utf-8"))
                    self._handle_message(payload)
                except Exception as e:
                    logger.error("Failed handling LSP message: %s", e)

    def _handle_message(self, msg: Dict[str, Any]) -> None:
        msg_id = msg.get("id")
        method = msg.get("method")
        params = msg.get("params", {})

        # Handle Responses / Requests
        if method == "initialize":
            res = {
                "capabilities": {
                    "textDocumentSync": 1,
                    "hoverProvider": True,
                    "definitionProvider": True,
                    "documentSymbolProvider": True,
                },
                "serverInfo": {
                    "name": "AntimatterLocalPythonLSP",
                    "version": "0.1.0",
                },
            }
            write_lsp_message({"jsonrpc": "2.0", "id": msg_id, "result": res})

        elif method == "initialized":
            # Notification from client after initialize
            pass

        elif method == "textDocument/didOpen":
            doc = params.get("textDocument", {})
            uri = doc.get("uri", "")
            text = doc.get("text", "")
            self._documents[uri] = text
            self._publish_diagnostics(uri, text)

        elif method == "textDocument/didChange":
            doc = params.get("textDocument", {})
            uri = doc.get("uri", "")
            changes = params.get("contentChanges", [])
            if changes:
                text = changes[-1].get("text", "")
                self._documents[uri] = text
                self._publish_diagnostics(uri, text)

        elif method == "textDocument/hover":
            uri = params.get("textDocument", {}).get("uri", "")
            pos = params.get("position", {})
            line = pos.get("line", 0)  # 0-based
            col = pos.get("character", 0)
            hover_res = self._compute_hover(uri, line, col)
            write_lsp_message({"jsonrpc": "2.0", "id": msg_id, "result": hover_res})

        elif method == "textDocument/definition":
            uri = params.get("textDocument", {}).get("uri", "")
            pos = params.get("position", {})
            line = pos.get("line", 0)
            col = pos.get("character", 0)
            def_res = self._compute_definition(uri, line, col)
            write_lsp_message({"jsonrpc": "2.0", "id": msg_id, "result": def_res})

        elif method == "textDocument/documentSymbol":
            uri = params.get("textDocument", {}).get("uri", "")
            syms = self._compute_symbols(uri)
            write_lsp_message({"jsonrpc": "2.0", "id": msg_id, "result": syms})

        elif method == "shutdown":
            self._is_shutdown = True
            write_lsp_message({"jsonrpc": "2.0", "id": msg_id, "result": None})

        elif method == "exit":
            sys.exit(0)

        elif msg_id is not None:
            # Unknown request
            write_lsp_message({"jsonrpc": "2.0", "id": msg_id, "result": None})

    def _publish_diagnostics(self, uri: str, code: str) -> None:
        """Parse Python AST and emit syntax errors as LSP diagnostics."""
        diagnostics = []
        try:
            ast.parse(code)
        except SyntaxError as e:
            line_0 = max(0, (e.lineno or 1) - 1)
            col_0 = max(0, (e.offset or 1) - 1)
            diagnostics.append(
                {
                    "range": {
                        "start": {"line": line_0, "character": col_0},
                        "end": {"line": line_0, "character": col_0 + 5},
                    },
                    "severity": 1,  # Error
                    "source": "AntimatterSyntaxChecker",
                    "message": str(e.msg),
                }
            )

        write_lsp_message(
            {
                "jsonrpc": "2.0",
                "method": "textDocument/publishDiagnostics",
                "params": {"uri": uri, "diagnostics": diagnostics},
            }
        )

    def _compute_hover(self, uri: str, line_0: int, col_0: int) -> Optional[Dict[str, Any]]:
        """Resolve symbol docstring or signature under cursor."""
        code = self._documents.get(uri, "")
        if not code:
            return None

        lines = code.splitlines()
        if line_0 >= len(lines):
            return None

        current_line = lines[line_0]
        # Find word under cursor
        word = self._extract_word(current_line, col_0)
        if not word:
            return None

        # Look for definition in AST
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name == word:
                        doc = ast.get_docstring(node) or "Function definition"
                        args = [a.arg for a in node.args.args]
                        sig = f"def {node.name}({', '.join(args)}):"
                        return {
                            "contents": {
                                "kind": "markdown",
                                "value": f"```python\n{sig}\n```\n\n{doc}",
                            }
                        }
                elif isinstance(node, ast.ClassDef):
                    if node.name == word:
                        doc = ast.get_docstring(node) or "Class definition"
                        return {
                            "contents": {
                                "kind": "markdown",
                                "value": f"```python\nclass {node.name}\n```\n\n{doc}",
                            }
                        }
        except Exception:
            pass

        return {
            "contents": {
                "kind": "markdown",
                "value": f"**Symbol:** `{word}` (line {line_0 + 1})",
            }
        }

    def _compute_definition(self, uri: str, line_0: int, col_0: int) -> List[Dict[str, Any]]:
        """Locate definition of identifier in document."""
        code = self._documents.get(uri, "")
        if not code:
            return []

        lines = code.splitlines()
        if line_0 >= len(lines):
            return []

        word = self._extract_word(lines[line_0], col_0)
        if not word:
            return []

        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if node.name == word:
                        def_line_0 = max(0, node.lineno - 1)
                        def_col_0 = max(0, node.col_offset)
                        return [
                            {
                                "uri": uri,
                                "range": {
                                    "start": {"line": def_line_0, "character": def_col_0},
                                    "end": {"line": def_line_0, "character": def_col_0 + len(word)},
                                },
                            }
                        ]
        except Exception:
            pass

        return []

    def _compute_symbols(self, uri: str) -> List[Dict[str, Any]]:
        """Return document symbols."""
        code = self._documents.get(uri, "")
        if not code:
            return []

        symbols = []
        try:
            tree = ast.parse(code)
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbols.append(
                        {
                            "name": node.name,
                            "kind": 12,  # Function
                            "range": {
                                "start": {"line": node.lineno - 1, "character": node.col_offset},
                                "end": {"line": node.end_lineno - 1 if node.end_lineno else node.lineno, "character": 0},
                            },
                        }
                    )
                elif isinstance(node, ast.ClassDef):
                    symbols.append(
                        {
                            "name": node.name,
                            "kind": 5,  # Class
                            "range": {
                                "start": {"line": node.lineno - 1, "character": node.col_offset},
                                "end": {"line": node.end_lineno - 1 if node.end_lineno else node.lineno, "character": 0},
                            },
                        }
                    )
        except Exception:
            pass
        return symbols

    def _extract_word(self, line: str, col: int) -> str:
        """Find identifier word around column index."""
        if col < 0 or col > len(line):
            return ""
        # Search backwards
        start = col
        while start > 0 and (line[start - 1].isalnum() or line[start - 1] == "_"):
            start -= 1
        # Search forwards
        end = col
        while end < len(line) and (line[end].isalnum() or line[end] == "_"):
            end += 1
        return line[start:end]


if __name__ == "__main__":
    server = LocalPythonLspServer()
    server.run()
