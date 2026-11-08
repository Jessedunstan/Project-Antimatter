"""Tool Registry and schema generator for Project Antimatter.

Converts typed Python functions into LLM-ready JSON schemas and handles
safe tool invocation and dispatching.
"""

import inspect
import json
import logging
from typing import Any, Callable, Dict, List, Optional, get_type_hints
from pydantic import BaseModel
from antimatter.brain.types import ToolCall, ToolDefinition, ToolFunctionSchema

logger = logging.getLogger(__name__)


def _python_type_to_json_schema(py_type: Any) -> Dict[str, Any]:
    """Map basic Python type annotations to JSON Schema types."""
    if py_type in (int, float):
        return {"type": "integer" if py_type is int else "number"}
    if py_type is str:
        return {"type": "string"}
    if py_type is bool:
        return {"type": "boolean"}
    if py_type in (list, List) or getattr(py_type, "__origin__", None) in (list, List):
        return {"type": "array"}
    if py_type in (dict, Dict) or getattr(py_type, "__origin__", None) in (dict, Dict):
        return {"type": "object"}
    if inspect.isclass(py_type) and issubclass(py_type, BaseModel):
        return py_type.model_json_schema()
    return {"type": "string"}


class RegisteredTool:
    """Wrapper holding executable function and its generated ToolDefinition."""

    def __init__(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        self.func = func
        self.name = name or func.__name__
        self.description = description or (func.__doc__ or "").strip() or f"Execute {self.name}"
        self.schema = self._build_schema()

    def _build_schema(self) -> ToolDefinition:
        """Derive JSON schema from function signature and type annotations."""
        sig = inspect.signature(self.func)
        try:
            type_hints = get_type_hints(self.func)
        except Exception:
            type_hints = {}

        properties: Dict[str, Any] = {}
        required: List[str] = []

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            param_type = type_hints.get(param_name, str)
            schema_type = _python_type_to_json_schema(param_type)

            # Check for default value
            if param.default is inspect.Parameter.empty:
                required.append(param_name)
            else:
                schema_type["default"] = param.default

            properties[param_name] = schema_type

        return ToolDefinition(
            function=ToolFunctionSchema(
                name=self.name,
                description=self.description,
                parameters={
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            )
        )

    async def execute(self, arguments: Dict[str, Any]) -> Any:
        """Execute the wrapped function with provided arguments."""
        if inspect.iscoroutinefunction(self.func):
            return await self.func(**arguments)
        return self.func(**arguments)


class ToolRegistry:
    """Central registry for sovereign agent tools."""

    def __init__(self):
        self._tools: Dict[str, RegisteredTool] = {}

    def register(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Callable[[Callable], Callable]:
        """Decorator to register a tool function."""
        def decorator(func: Callable) -> Callable:
            reg_tool = RegisteredTool(func, name=name, description=description)
            self._tools[reg_tool.name] = reg_tool
            return func
        return decorator

    def get_definitions(self) -> List[ToolDefinition]:
        """Return all registered tool definitions formatted for LLM consumption."""
        return [tool.schema for tool in self._tools.values()]

    def has_tool(self, name: str) -> bool:
        """Check if a tool exists in the registry."""
        return name in self._tools

    async def dispatch(self, tool_call: ToolCall) -> str:
        """Execute a tool call and format output as a string response."""
        tool = self._tools.get(tool_call.name)
        if not tool:
            return f"Error: Tool '{tool_call.name}' not found in registry."

        try:
            result = await tool.execute(tool_call.arguments)
            if isinstance(result, (dict, list)):
                return json.dumps(result, indent=2)
            return str(result)
        except Exception as e:
            logger.exception("Execution error in tool %s", tool_call.name)
            return f"Error executing '{tool_call.name}': {str(e)}"
