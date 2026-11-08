"""Model Context Protocol (MCP) Stdio Client Bridge for Project Antimatter.

Allows local MCP servers to expose tools dynamically to the sovereign agent loop
via JSON-RPC 2.0 over standard I/O.
"""

from __future__ import annotations
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from antimatter.tools.registry import ToolRegistry, RegisteredTool
from antimatter.brain.types import ToolDefinition, ToolFunctionSchema

logger = logging.getLogger(__name__)


class MCPClient:
    """Client for a single stdio-based Model Context Protocol (MCP) server."""

    def __init__(self, server_cmd: List[str], server_name: str = "mcp_server"):
        self.server_cmd = server_cmd
        self.server_name = server_name
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._request_id = 0
        self._pending: Dict[int, asyncio.Future] = {}
        self._is_running = False
        self._discovered_tools: List[Dict[str, Any]] = []

    async def connect(self) -> bool:
        """Spawn MCP server process and perform initialize handshake."""
        try:
            self._proc = await asyncio.create_subprocess_exec(
                *self.server_cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            self._is_running = True
            asyncio.create_task(self._listen_stdout())

            # 1. Initialize
            init_res = await self._send_request(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "clientInfo": {"name": "Antimatter", "version": "0.1.0"},
                }
            )
            # 2. Initialized notification
            await self._send_notification("notifications/initialized", {})

            # 3. Discover tools
            tools_res = await self._send_request("tools/list", {})
            self._discovered_tools = tools_res.get("tools", []) if tools_res else []
            logger.info(
                "MCP Server '%s' connected. Discovered %d tools.",
                self.server_name,
                len(self._discovered_tools)
            )
            return True

        except Exception as e:
            logger.error("Failed connecting to MCP server '%s': %s", self.server_name, e)
            self._is_running = False
            return False

    async def disconnect(self) -> None:
        """Close stdio streams and terminate server process."""
        self._is_running = False
        if self._proc:
            try:
                self._proc.terminate()
            except Exception:
                pass

    async def _send_request(self, method: str, params: Any) -> Any:
        """Send JSON-RPC request and wait for result."""
        if not self._proc or not self._proc.stdin:
            raise RuntimeError("MCP process stdin not available.")

        self._request_id += 1
        req_id = self._request_id
        payload = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        }
        line = json.dumps(payload) + "\n"

        future = asyncio.get_event_loop().create_future()
        self._pending[req_id] = future

        self._proc.stdin.write(line.encode("utf-8"))
        await self._proc.stdin.drain()

        return await asyncio.wait_for(future, timeout=15.0)

    async def _send_notification(self, method: str, params: Any) -> None:
        """Send JSON-RPC notification."""
        if not self._proc or not self._proc.stdin:
            return
        payload = {"jsonrpc": "2.0", "method": method, "params": params}
        line = json.dumps(payload) + "\n"
        self._proc.stdin.write(line.encode("utf-8"))
        await self._proc.stdin.drain()

    async def _listen_stdout(self) -> None:
        """Read and dispatch newline-delimited JSON-RPC messages."""
        assert self._proc and self._proc.stdout
        while self._is_running:
            line_bytes = await self._proc.stdout.readline()
            if not line_bytes:
                break

            line_str = line_bytes.decode("utf-8", errors="replace").strip()
            if not line_str:
                continue

            try:
                msg = json.loads(line_str)
                req_id = msg.get("id")
                if req_id in self._pending:
                    fut = self._pending.pop(req_id)
                    if "error" in msg:
                        fut.set_exception(RuntimeError(f"MCP Error: {msg['error']}"))
                    else:
                        fut.set_result(msg.get("result"))
            except Exception as err:
                logger.debug("Failed parsing MCP response: %s", err)

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool call on the MCP server."""
        res = await self._send_request(
            "tools/call",
            {"name": tool_name, "arguments": arguments}
        )
        if not res:
            return "No output returned from MCP tool."

        content_blocks = res.get("content", [])
        outputs = []
        for block in content_blocks:
            if block.get("type") == "text":
                outputs.append(block.get("text", ""))
            elif block.get("type") == "json":
                outputs.append(json.dumps(block.get("json", {}), indent=2))
        return "\n".join(outputs) if outputs else str(res)

    def register_tools_into_registry(self, registry: ToolRegistry) -> None:
        """Register all discovered MCP tools into Antimatter's ToolRegistry."""
        for tool_meta in self._discovered_tools:
            t_name = f"{self.server_name}_{tool_meta.get('name')}"
            t_desc = tool_meta.get("description", f"MCP Tool: {tool_meta.get('name')}")
            input_schema = tool_meta.get("inputSchema", {"type": "object", "properties": {}})

            # Create closure for execution
            original_name = tool_meta.get("name")

            async def _executor(**kwargs) -> str:
                return await self.call_tool(original_name, kwargs)

            # Build RegisteredTool directly
            reg_tool = RegisteredTool(_executor, name=t_name, description=t_desc)
            # Override schema with exact MCP inputSchema
            reg_tool.schema = ToolDefinition(
                function=ToolFunctionSchema(
                    name=t_name,
                    description=t_desc,
                    parameters=input_schema,
                )
            )
            registry._tools[t_name] = reg_tool
            logger.info("Registered MCP tool '%s' into agent registry.", t_name)
