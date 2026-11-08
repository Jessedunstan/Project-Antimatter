"""Type contracts for LLM message passing, tool declarations, and streaming."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChatRole(str, Enum):
    """Message author role."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCall(BaseModel):
    """Specification of an invocation initiated by the model."""
    id: str = Field(description="Unique tool call execution identifier")
    name: str = Field(description="Tool function name to execute")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Parsed JSON arguments")


class ChatMessage(BaseModel):
    """Single chat turn in conversation trajectory."""
    role: ChatRole
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None


class ToolFunctionSchema(BaseModel):
    """JSON Schema definition for an individual tool function."""
    name: str
    description: str
    parameters: Dict[str, Any]


class ToolDefinition(BaseModel):
    """Standard tool container matching OpenAI / Ollama tool structure."""
    type: str = "function"
    function: ToolFunctionSchema


class CompletionChunk(BaseModel):
    """Individual streamed token or event chunk from the local inference engine."""
    delta_text: str = ""
    tool_call_delta: Optional[ToolCall] = None
    finish_reason: Optional[str] = None
