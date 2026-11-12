"""Deterministic Tool Call Extraction Engine for Sovereign Local LLMs.

Supports native Ollama/vLLM tool calls and extracts inline tool calls
from model text output:
- Bare JSON objects (`{"name": "...", "arguments": {...}}`)
- XML `<tool_call>{...}</tool_call>` blocks
- Markdown ```` ```json ... ``` ```` blocks
- ReAct `Action: ... / Action Input: ...` blocks
"""

import json
import re
import uuid
import logging
from typing import List, Optional, Tuple, Dict, Any
from antimatter.brain.types import ToolCall, ToolDefinition

logger = logging.getLogger(__name__)

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
        Extract tool calls embedded in freeform text using balanced JSON decoding
        (`json.JSONDecoder.raw_decode`) and ReAct pattern matching.
        Returns cleaned text (with tool markup stripped) and a list of parsed ToolCall objects.
        """
        if not text:
            return "", []

        tool_calls: List[ToolCall] = []
        known_tool_names = {t.function.name for t in available_tools} if available_tools else set()

        # 1. Scan for balanced JSON tool-call objects anywhere in text (handles XML, fenced, and bare JSON)
        decoder = json.JSONDecoder()
        spans_to_remove: List[Tuple[int, int]] = []
        idx = 0
        n = len(text)

        while idx < n:
            brace_pos = text.find("{", idx)
            if brace_pos == -1:
                break

            try:
                obj, end_pos = decoder.raw_decode(text, brace_pos)
            except json.JSONDecodeError:
                idx = brace_pos + 1
                continue

            tc = cls._dict_to_tool_call(obj, known_tool_names)
            if tc is not None:
                # Deduplicate identical tool calls within the same turn and cap per-turn extraction
                is_dup = any(
                    existing.name == tc.name and existing.arguments == tc.arguments
                    for existing in tool_calls
                )
                if not is_dup and len(tool_calls) < 4:
                    tool_calls.append(tc)
                spans_to_remove.append((brace_pos, end_pos))
                idx = end_pos
            else:
                idx = brace_pos + 1

        if tool_calls:
            # Remove extracted JSON payloads from text (in reverse order to preserve indices)
            cleaned = text
            for start_i, end_i in reversed(spans_to_remove):
                cleaned = cleaned[:start_i] + cleaned[end_i:]
            # Strip leftover wrapper tags/fences
            cleaned = re.sub(r"<tool_call>\s*</tool_call>", "", cleaned, flags=re.DOTALL)
            cleaned = re.sub(r"```(?:json|tool_call)?\s*```", "", cleaned, flags=re.DOTALL)
            return cleaned.strip(), tool_calls

        # 2. Fallback: Check for ReAct Action / Action Input format
        cleaned_text = text
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
    def _dict_to_tool_call(cls, data: Any, known_tools: set) -> Optional[ToolCall]:
        """Convert a decoded JSON dict into a ToolCall if it matches a tool invocation schema."""
        if not isinstance(data, dict):
            return None

        # Unwrap OpenAI/Ollama style {"type": "function", "function": {"name": ..., "arguments": ...}}
        if isinstance(data.get("function"), dict):
            fn_dict = data["function"]
            name = fn_dict.get("name")
            raw_args = fn_dict.get("arguments") or fn_dict.get("parameters") or {}
        else:
            name = data.get("name") or data.get("tool") or data.get("function") or data.get("action")
            raw_args = (
                data.get("arguments")
                if "arguments" in data
                else (
                    data.get("parameters")
                    if "parameters" in data
                    else data.get("input", {})
                )
            )

        if not name or not isinstance(name, str):
            return None

        if known_tools and name not in known_tools:
            return None

        args = cls._parse_arguments(raw_args)
        call_id = data.get("id") or f"call_{uuid.uuid4().hex[:8]}"
        return ToolCall(id=str(call_id), name=name, arguments=args)

    @classmethod
    def _parse_tool_payload(cls, raw_payload: str, known_tools: set) -> Optional[ToolCall]:
        """Safely parse a JSON dictionary string into a ToolCall."""
        try:
            data = json.loads(raw_payload)
            return cls._dict_to_tool_call(data, known_tools)
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
        if raw_args is None:
            return {}
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
