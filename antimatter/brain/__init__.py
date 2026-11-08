"""Brain: Sovereign local LLM, embeddings, and multimodal sensory engine."""

from antimatter.brain.types import (
    ChatMessage,
    ChatRole,
    CompletionChunk,
    ToolDefinition,
    ToolCall,
    ToolFunctionSchema,
)
from antimatter.brain.client import LocalLLMClient
from antimatter.brain.embeddings import LocalEmbeddingEngine
from antimatter.brain.tool_extractor import ToolCallExtractor

__all__ = [
    "ChatMessage",
    "ChatRole",
    "CompletionChunk",
    "ToolDefinition",
    "ToolCall",
    "ToolFunctionSchema",
    "LocalLLMClient",
    "LocalEmbeddingEngine",
    "ToolCallExtractor",
]
