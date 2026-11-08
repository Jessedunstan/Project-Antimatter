"""Async Language Server Protocol (LSP) Client over JSON-RPC 2.0.

Provides sub-second async symbol hover, definitions, and diagnostics
by communicating over asynchronous pipes with local language servers.
"""

from __future__ import annotations
import asyncio
import json
import logging
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Optional, Union

from antimatter.context.types import (
    DefinitionLocation,
    Diagnostic,
    DiagnosticSeverity,
    HoverInfo,
)

logger = logging.getLogger(__name__)


def _to_uri(file_path: Union[str, Path]) -> str:
    """Convert file path to file:// URI format."""
    p = Path(file_path).resolve()
    return p.as_uri()


def _from_uri(uri: str) -> str:
    """Convert file:// URI back to file path."""
    if uri.startswith("file:///"):
        # Windows / posix path
        raw = uri[7:]
        # On Windows: /C:/foo -> C:/foo
        if len(raw) > 2 and raw[0] == "/" and raw[2] == ":":
            return raw[1:]
        return raw
    elif uri.startswith("file://"):
        return uri[7:]
    return uri


class AsyncLSPClient:
    """Asynchronous client connecting to local Language Server via JSON-RPC 2.0 stdio."""

    def __init__(self, server_cmd: Optional[List[str]] = None, cwd: Optional[Path] = None):
        self.server_cmd = server_cmd or [sys.executable, "-m", "antimatter.context.lsp_server"]
        self.cwd = cwd or Path.cwd()
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._request_counter = 0
        self._pending_requests: Dict[int, asyncio.Future] = {}
        self._diagnostics: Dict[str, List[Diagnostic]] = {}
        self._diagnostics_listeners: List[Callable[[str, List[Diagnostic]], None]] = []
        self._is_initialized = False

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    async def start(self) -> None:
        """Start language server process and background message listener."""
        if self.is_running:
            return

        logger.info("Spawning LSP server process: %s", self.server_cmd)
        self._proc = await asyncio.create_subprocess_exec(
            *self.server_cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.cwd),
        )
        self._reader_task = asyncio.create_task(self._read_stdout_loop())

    async def stop(self) -> None:
        """Gracefully shutdown and terminate language server."""
        if not self.is_running:
            return

        try:
            if self._is_initialized:
                # Send shutdown request and wait
                try:
                    await self._send_request("shutdown", {}, timeout_sec=2.0)
                except Exception as e:
                    logger.debug("Shutdown request error: %s", e)
                # Send exit notification
                await self._send_notification("exit", {})
        except Exception as e:
            logger.debug("Error during LSP shutdown: %s", e)
        finally:
            if self._proc:
                try:
                    self._proc.terminate()
                    await asyncio.wait_for(self._proc.wait(), timeout=1.0)
                except Exception:
                    try:
                        self._proc.kill()
                    except Exception:
                        pass
                self._proc = None

            if self._reader_task and not self._reader_task.done():
                self._reader_task.cancel()
                try:
                    await self._reader_task
                except asyncio.CancelledError:
                    pass

            # Fail any remaining futures
            for fut in self._pending_requests.values():
                if not fut.done():
                    fut.set_exception(ConnectionResetError("LSP server stopped"))
            self._pending_requests.clear()
            self._is_initialized = False

    async def initialize(self, root_path: Union[str, Path]) -> Dict[str, Any]:
        """Perform LSP handshake (initialize request + initialized notification)."""
        await self.start()
        root_p = Path(root_path).resolve()
        params = {
            "processId": None,
            "rootUri": root_p.as_uri(),
            "rootPath": str(root_p),
            "capabilities": {
                "textDocument": {
                    "hover": {"contentFormat": ["markdown", "plaintext"]},
                    "definition": {"dynamicRegistration": True},
                    "documentSymbol": {"hierarchicalDocumentSymbolSupport": True},
                    "publishDiagnostics": {"relatedInformation": True},
                },
                "workspace": {"applyEdit": False},
            },
            "workspaceFolders": [
                {"name": root_p.name, "uri": root_p.as_uri()}
            ],
        }

        res = await self._send_request("initialize", params)
        await self._send_notification("initialized", {})
        self._is_initialized = True
        return res

    async def open_document(
        self,
        file_path: Union[str, Path],
        content: str,
        language_id: str = "python",
        version: int = 1,
    ) -> None:
        """Notify server that document has been opened."""
        uri = _to_uri(file_path)
        params = {
            "textDocument": {
                "uri": uri,
                "languageId": language_id,
                "version": version,
                "text": content,
            }
        }
        await self._send_notification("textDocument/didOpen", params)

    async def change_document(
        self,
        file_path: Union[str, Path],
        content: str,
        version: int,
    ) -> None:
        """Notify server that document contents have changed."""
        uri = _to_uri(file_path)
        params = {
            "textDocument": {"uri": uri, "version": version},
            "contentChanges": [{"text": content}],
        }
        await self._send_notification("textDocument/didChange", params)

    async def hover(
        self,
        file_path: Union[str, Path],
        line: int,
        character: int,
        timeout_sec: float = 3.0,
    ) -> Optional[HoverInfo]:
        """Request hover documentation at 1-based line and 0-based character."""
        uri = _to_uri(file_path)
        # LSP positions are 0-based
        line_0 = max(0, line - 1)
        params = {
            "textDocument": {"uri": uri},
            "position": {"line": line_0, "character": character},
        }

        result = await self._send_request("textDocument/hover", params, timeout_sec=timeout_sec)
        if not result or not result.get("contents"):
            return None

        contents = result["contents"]
        text_content = ""
        if isinstance(contents, dict):
            text_content = contents.get("value", "")
        elif isinstance(contents, list):
            text_content = "\n\n".join(
                c.get("value", "") if isinstance(c, dict) else str(c) for c in contents
            )
        else:
            text_content = str(contents)

        return HoverInfo(
            file_path=str(file_path),
            contents=text_content,
            line=line,
            column=character,
            range_info=result.get("range"),
        )

    async def definition(
        self,
        file_path: Union[str, Path],
        line: int,
        character: int,
        timeout_sec: float = 3.0,
    ) -> List[DefinitionLocation]:
        """Request definition location for symbol at given line and column."""
        uri = _to_uri(file_path)
        line_0 = max(0, line - 1)
        params = {
            "textDocument": {"uri": uri},
            "position": {"line": line_0, "character": character},
        }

        result = await self._send_request("textDocument/definition", params, timeout_sec=timeout_sec)
        if not result:
            return []

        locations: List[DefinitionLocation] = []
        raw_items = [result] if isinstance(result, dict) else (result if isinstance(result, list) else [])

        for item in raw_items:
            target_uri = item.get("uri") or item.get("targetUri")
            r = item.get("range") or item.get("targetSelectionRange") or item.get("targetRange")
            if target_uri and r:
                start = r.get("start", {})
                end = r.get("end", {})
                locations.append(
                    DefinitionLocation(
                        file_path=_from_uri(target_uri),
                        line=start.get("line", 0) + 1,
                        column=start.get("character", 0),
                        end_line=end.get("line", 0) + 1,
                        end_column=end.get("character", 0),
                    )
                )

        return locations

    async def document_symbols(
        self,
        file_path: Union[str, Path],
        timeout_sec: float = 3.0,
    ) -> List[Dict[str, Any]]:
        """Request document symbols from language server."""
        uri = _to_uri(file_path)
        params = {"textDocument": {"uri": uri}}
        result = await self._send_request("textDocument/documentSymbol", params, timeout_sec=timeout_sec)
        return result if isinstance(result, list) else []

    def get_diagnostics(self, file_path: Union[str, Path]) -> List[Diagnostic]:
        """Retrieve stored diagnostics for a given file."""
        uri = _to_uri(file_path)
        return self._diagnostics.get(uri, [])

    def add_diagnostics_listener(self, listener: Callable[[str, List[Diagnostic]], None]) -> None:
        """Register callback for when new diagnostics are published."""
        self._diagnostics_listeners.append(listener)

    # =========================================================================
    # Internal JSON-RPC 2.0 Protocol Handling
    # =========================================================================

    async def _send_request(
        self,
        method: str,
        params: Dict[str, Any],
        timeout_sec: float = 5.0,
    ) -> Any:
        if not self.is_running:
            raise RuntimeError("LSP client is not running")

        self._request_counter += 1
        req_id = self._request_counter
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending_requests[req_id] = future

        msg = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        }
        await self._write_message(msg)

        try:
            return await asyncio.wait_for(future, timeout=timeout_sec)
        except asyncio.TimeoutError:
            self._pending_requests.pop(req_id, None)
            raise TimeoutError(f"LSP request '{method}' (id={req_id}) timed out after {timeout_sec}s")

    async def _send_notification(self, method: str, params: Dict[str, Any]) -> None:
        if not self.is_running:
            return
        msg = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        await self._write_message(msg)

    async def _write_message(self, msg: Dict[str, Any]) -> None:
        if not self._proc or not self._proc.stdin:
            return
        body = json.dumps(msg).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("latin1")
        self._proc.stdin.write(header + body)
        await self._proc.stdin.drain()

    async def _read_stdout_loop(self) -> None:
        """Background reader loop parsing LSP frames from server stdout."""
        reader = self._proc.stdout if self._proc else None
        if not reader:
            return

        while True:
            try:
                line = await reader.readline()
                if not line:
                    break

                if line.startswith(b"Content-Length:"):
                    length_str = line.split(b":")[1].strip()
                    content_length = int(length_str)

                    # Consume remaining headers until empty line
                    while True:
                        header_line = await reader.readline()
                        if header_line in (b"\r\n", b"\n", b""):
                            break

                    # Read payload bytes
                    content_bytes = await reader.readexactly(content_length)
                    payload = json.loads(content_bytes.decode("utf-8"))
                    self._dispatch_message(payload)

            except (asyncio.CancelledError, asyncio.IncompleteReadError):
                break
            except Exception as e:
                logger.error("Error in LSP reader loop: %s", e)

    def _dispatch_message(self, msg: Dict[str, Any]) -> None:
        msg_id = msg.get("id")
        method = msg.get("method")

        # Response to pending request
        if msg_id is not None and msg_id in self._pending_requests:
            fut = self._pending_requests.pop(msg_id)
            if not fut.done():
                if "error" in msg:
                    fut.set_exception(RuntimeError(f"LSP Error: {msg['error']}"))
                else:
                    fut.set_result(msg.get("result"))
            return

        # Notification from server
        if method == "textDocument/publishDiagnostics":
            params = msg.get("params", {})
            uri = params.get("uri", "")
            raw_diags = params.get("diagnostics", [])
            parsed_diags: List[Diagnostic] = []

            for rd in raw_diags:
                r = rd.get("range", {})
                start = r.get("start", {})
                end = r.get("end", {})
                sev_num = rd.get("severity", 1)
                try:
                    sev = DiagnosticSeverity(sev_num)
                except ValueError:
                    sev = DiagnosticSeverity.ERROR

                parsed_diags.append(
                    Diagnostic(
                        file_path=_from_uri(uri),
                        message=rd.get("message", ""),
                        severity=sev,
                        line=start.get("line", 0) + 1,
                        column=start.get("character", 0),
                        end_line=end.get("line", 0) + 1 if "line" in end else None,
                        end_column=end.get("character", 0) if "character" in end else None,
                        source=rd.get("source"),
                        code=str(rd.get("code")) if rd.get("code") is not None else None,
                    )
                )

            self._diagnostics[uri] = parsed_diags
            for listener in self._diagnostics_listeners:
                try:
                    listener(uri, parsed_diags)
                except Exception as e:
                    logger.debug("Diagnostics listener error: %s", e)
