"""Headless Language Server Protocol (LSP) Client and Fallback Intelligence Engine.

Communicates with local language servers (Pyright, TS-server, gopls) via JSON-RPC 2.0
over stdio, with an in-process AST-powered fallback for sovereign zero-dependency operation.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from antimatter.intelligence.ast_parser import ASTParser, CodeSymbol, SymbolType

logger = logging.getLogger(__name__)


@dataclass
class Diagnostic:
    """Compiler / linter diagnostic message."""
    file_path: str
    message: str
    severity: int  # 1: Error, 2: Warning, 3: Info, 4: Hint
    start_line: int
    start_char: int
    end_line: int
    end_char: int


@dataclass
class HoverResult:
    """Hover documentation and type information."""
    symbol_name: str
    signature: Optional[str]
    documentation: Optional[str]
    range_lines: tuple[int, int]


@dataclass
class LocationResult:
    """Definition or reference code location."""
    file_path: str
    start_line: int
    start_char: int
    end_line: int
    end_char: int


class LSPClient:
    """Headless LSP client bridge with sovereign AST fallback."""

    def __init__(self, workspace_root: Path, server_cmd: Optional[List[str]] = None):
        self.workspace_root = workspace_root.resolve()
        self.server_cmd = server_cmd
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._request_id = 0
        self._pending_requests: Dict[int, asyncio.Future] = {}
        self._diagnostics: Dict[str, List[Diagnostic]] = {}
        self._ast_fallback = ASTParser()
        self._is_running = False

    async def start(self) -> bool:
        """Attempt to spawn local language server process, or activate AST fallback."""
        if not self.server_cmd:
            logger.info("No external LSP binary provided; running with sovereign AST engine.")
            self._is_running = True
            return True

        try:
            self._proc = await asyncio.create_subprocess_exec(
                *self.server_cmd,
                cwd=str(self.workspace_root),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            self._is_running = True
            asyncio.create_task(self._listen_stdout())

            # Perform LSP initialize handshake
            await self._send_request(
                "initialize",
                {
                    "processId": None,
                    "rootUri": self.workspace_root.as_uri(),
                    "capabilities": {},
                }
            )
            await self._send_notification("initialized", {})
            return True
        except Exception as e:
            logger.warning("Failed to start external LSP server %s: %s. Using AST engine.", self.server_cmd, e)
            self._proc = None
            self._is_running = True
            return True

    async def stop(self) -> None:
        """Terminate LSP server process if active."""
        if self._proc:
            try:
                await self._send_request("shutdown", {})
                await self._send_notification("exit", {})
                self._proc.terminate()
            except Exception:
                pass
        self._is_running = False

    async def _send_request(self, method: str, params: Any) -> Any:
        """Send JSON-RPC request and await response."""
        if not self._proc or not self._proc.stdin:
            return None

        self._request_id += 1
        req_id = self._request_id
        payload = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        }
        body = json.dumps(payload).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")

        future = asyncio.get_event_loop().create_future()
        self._pending_requests[req_id] = future

        self._proc.stdin.write(header + body)
        await self._proc.stdin.drain()

        return await asyncio.wait_for(future, timeout=5.0)

    async def _send_notification(self, method: str, params: Any) -> None:
        """Send JSON-RPC notification (no response expected)."""
        if not self._proc or not self._proc.stdin:
            return

        payload = {"jsonrpc": "2.0", "method": method, "params": params}
        body = json.dumps(payload).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        self._proc.stdin.write(header + body)
        await self._proc.stdin.drain()

    async def _listen_stdout(self) -> None:
        """Read and dispatch JSON-RPC messages from LSP server stdout."""
        assert self._proc and self._proc.stdout
        buffer = b""

        while self._is_running:
            line = await self._proc.stdout.readline()
            if not line:
                break

            if line.startswith(b"Content-Length:"):
                length = int(line.split(b":")[1].strip())
                # Consume empty separator line
                await self._proc.stdout.readline()
                content = await self._proc.stdout.readexactly(length)

                try:
                    msg = json.loads(content.decode("utf-8"))
                    # Response to request
                    if "id" in msg and msg["id"] in self._pending_requests:
                        self._pending_requests.pop(msg["id"]).set_result(msg.get("result"))
                    # Diagnostic notification
                    elif msg.get("method") == "textDocument/publishDiagnostics":
                        params = msg.get("params", {})
                        uri = params.get("uri", "")
                        raw_diags = params.get("diagnostics", [])
                        self._diagnostics[uri] = [
                            Diagnostic(
                                file_path=uri,
                                message=d.get("message", ""),
                                severity=d.get("severity", 1),
                                start_line=d.get("range", {}).get("start", {}).get("line", 0) + 1,
                                start_char=d.get("range", {}).get("start", {}).get("character", 0),
                                end_line=d.get("range", {}).get("end", {}).get("line", 0) + 1,
                                end_char=d.get("range", {}).get("end", {}).get("character", 0),
                            )
                            for d in raw_diags
                        ]
                except Exception as err:
                    logger.debug("Error parsing LSP packet: %s", err)

    async def get_hover(
        self,
        file_path: Path | str,
        line: int,
        character: int,
        code_override: Optional[str] = None,
    ) -> Optional[HoverResult]:
        """Query hover symbol information (delegates to server or AST engine)."""
        target_path = Path(file_path)
        code = code_override or (target_path.read_text(encoding="utf-8", errors="replace") if target_path.exists() else "")

        # Use sovereign AST resolution: pick innermost/narrowest symbol covering the line
        symbols = self._ast_fallback.extract_symbols(code, str(file_path))
        matching_symbols = [
            sym for sym in symbols if sym.start_line <= line <= sym.end_line
        ]
        if not matching_symbols:
            return None

        # Sort by narrowest line range (innermost scope)
        narrowest = min(matching_symbols, key=lambda s: s.end_line - s.start_line)
        return HoverResult(
            symbol_name=narrowest.name,
            signature=narrowest.signature or f"{narrowest.symbol_type.value} {narrowest.name}",
            documentation=narrowest.docstring or f"Symbol '{narrowest.name}' declared at line {narrowest.start_line}",
            range_lines=(narrowest.start_line, narrowest.end_line),
        )

    async def get_definition(
        self,
        file_path: Path | str,
        line: int,
        character: int,
        symbol_name: Optional[str] = None,
        code_override: Optional[str] = None,
    ) -> List[LocationResult]:
        """Jump to definition for a symbol."""
        target_path = Path(file_path)
        code = code_override or (target_path.read_text(encoding="utf-8", errors="replace") if target_path.exists() else "")

        symbols = self._ast_fallback.extract_symbols(code, str(file_path))
        results: List[LocationResult] = []

        for sym in symbols:
            if symbol_name and sym.name == symbol_name:
                results.append(
                    LocationResult(
                        file_path=str(file_path),
                        start_line=sym.start_line,
                        start_char=0,
                        end_line=sym.end_line,
                        end_char=0,
                    )
                )
            elif not symbol_name and sym.start_line <= line <= sym.end_line:
                results.append(
                    LocationResult(
                        file_path=str(file_path),
                        start_line=sym.start_line,
                        start_char=0,
                        end_line=sym.end_line,
                        end_char=0,
                    )
                )

        return results

    def get_diagnostics(self, file_path: str) -> List[Diagnostic]:
        """Retrieve cached diagnostics for a file."""
        return self._diagnostics.get(file_path, [])
