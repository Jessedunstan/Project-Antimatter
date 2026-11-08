"""Deterministic Tool Call Extraction Engine for Sovereign Local LLMs.

Supports native Ollama/vLLM tool calls and extracts inline tool calls
from model text output (XML <tool_call>, Markdown JSON blocks, ReAct Action/Input).
"""

import json
import re
import uuid
import logging
from typing import List, Optional, Tuple, Dict, Any
from antimatter.brain.types import ToolCall, ToolDefinition

logger = logging.getLogger(__name__)

# Pattern for XML style: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
XML_TOOL_PATTERN = re.compile(
    r"<tool_call>\s*(?P<payload>\{.*?\})\s*</tool_call>",
    re.DOTALL
)

# Pattern for fenced code blocks with JSON
JSON_BLOCK_PATTERN = re.compile(
    r"```(?:json)?\s*(?P<payload>\{\s*\"(?:name|tool|function|action)\"\s*:.*?\})\s*```",
    re.DOTALL
)

# Pattern for ReAct format:
# Action: tool_name
# Action Input: {...}
REACT_PATTERN = re.compile(
    r"Action:\s*(?P<name>[a-zA-Z0-9_\-\.]+)\s*\nAction Input:\s*(?P<args>\{.*?\}|\[.*?\]|[^\n]+)",
    re.DOTALL
)


class ToolCallExtractor:
    """Parses and sanitizes tool calls from both structured APIs and raw LLM text streams."""

    @classmethod
    def extract_from_text(
        cls,
        text: str,
        available_tools: Optional[List[ToolDefinition]] = None
    ) -> Tuple[str, List[ToolCall]]:
        """
        Extract tool calls embedded in freeform text.
        Returns cleaned text (with tool markup stripped) and a list of parsed ToolCall objects.
        """
        tool_calls: List[ToolCall] = []
        cleaned_text = text
        known_tool_names = {t.function.name for t in available_tools} if available_tools else set()

        # 1. Check for XML format: <tool_call>{...}</tool_call>
        for match in XML_TOOL_PATTERN.finditer(text):
            payload_raw = match.group("payload")
            tc = cls._parse_tool_payload(payload_raw, known_tool_names)
            if tc:
                tool_calls.append(tc)
                cleaned_text = cleaned_text.replace(match.group(0), "").strip()

        if tool_calls:
            return cleaned_text, tool_calls

        # 2. Check for JSON markdown blocks containing tool invocations
        for match in JSON_BLOCK_PATTERN.finditer(text):
            payload_raw = match.group("payload")
            tc = cls._parse_tool_payload(payload_raw, known_tool_names)
            if tc:
                tool_calls.append(tc)
                cleaned_text = cleaned_text.replace(match.group(0), "").strip()

        if tool_calls:
            return cleaned_text, tool_calls

        # 3. Check for ReAct Action / Action Input format
        for match in REACT_PATTERN.finditer(text):
            tool_name = match.group("name").strip()
            args_raw = match.group("args").strip()
            args = cls._parse_arguments(args_raw)
            if not known_tool_names or tool_name in known_tool_names:
                tool_calls.append(
                    ToolCall(
                        id=f"call_{uuid.uuid4().hex[:8]}",
                        name=tool_name,
                        arguments=args
                    )
                )
                cleaned_text = cleaned_text.replace(match.group(0), "").strip()

        return cleaned_text, tool_calls

    @classmethod
    def _parse_tool_payload(cls, raw_payload: str, known_tools: set) -> Optional[ToolCall]:
        """Safely parse a JSON dictionary into a ToolCall."""
        try:
            data = json.loads(raw_payload)
            if not isinstance(data, dict):
                return None

            # Handle different common tool call JSON formats
            name = data.get("name") or data.get("tool") or data.get("function") or data.get("action")
            if not name or not isinstance(name, str):
                return None

            # Check if name is within known tools if specified
            if known_tools and name not in known_tools:
                return None

            raw_args = data.get("arguments") or data.get("parameters") or data.get("input") or {}
            args = cls._parse_arguments(raw_args)

            call_id = data.get("id") or f"call_{uuid.uuid4().hex[:8]}"
            return ToolCall(id=str(call_id), name=name, arguments=args)

        except Exception as err:
            logger.debug("Failed parsing tool payload '%s': %s", raw_payload, err)
            return None

    @classmethod
    def _parse_arguments(cls, raw_args: Any) -> Dict[str, Any]:
        """Ensure tool arguments are parsed into a dictionary."""
        if isinstance(raw_args, dict):
            return raw_args
        if isinstance(raw_args, str):
            try:
                parsed = json.loads(raw_args)
                if isinstance(parsed, dict):
                    return parsed
                return {"input": parsed}
            except json.JSONDecodeError:
                return {"input": raw_args}
        return {"input": str(raw_args)}

    @classmethod
    def format_tools_for_system_prompt(cls, tools: List[ToolDefinition]) -> str:
        """
        Generate instructions for local models that don't support native tool APIs,
        guiding them to emit structured XML tool calls.
        """
        lines = [
            "\n## AVAILABLE TOOLS",
            "You have access to the following sovereign tools. If you need to execute a tool, "
            "respond ONLY with a tool call in the following format:",
            "<tool_call>",
            '{"name": "tool_name", "arguments": {"arg_key": "arg_value"}}',
            "</tool_call>",
            "\nTool Specifications:"
        ]
        for t in tools:
            fn = t.function
            lines.append(f"- **{fn.name}**: {fn.description}")
            lines.append(f"  Parameters: {json.dumps(fn.parameters)}")

        lines.append("\nWhen executing a tool, do not add conversational pleasantries—emit the <tool_call> directly.\n")
        return "\n".join(lines)
