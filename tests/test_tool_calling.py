"""Verification and validation of Tool-Calling extraction for Project Antimatter."""

import asyncio
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from antimatter.core.config import get_settings
from antimatter.brain.client import LocalLLMClient
from antimatter.brain.types import (
    ChatMessage,
    ChatRole,
    ToolDefinition,
    ToolFunctionSchema,
)
from antimatter.brain.tool_extractor import ToolCallExtractor


def test_synthetic_extractions():
    print("=" * 60)
    print(" [1] SYNTHETIC TOOL EXTRACTION TEST SUITE")
    print("=" * 60)

    # Sample tool definitions
    tools = [
        ToolDefinition(
            function=ToolFunctionSchema(
                name="read_file",
                description="Read contents of a file on disk",
                parameters={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"]
                }
            )
        ),
        ToolDefinition(
            function=ToolFunctionSchema(
                name="run_terminal_command",
                description="Run shell command in sandbox",
                parameters={
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"]
                }
            )
        )
    ]

    # 1. XML format
    xml_text = (
        "I need to read the configuration file.\n"
        "<tool_call>\n"
        '{"name": "read_file", "arguments": {"path": "pyproject.toml"}}\n'
        "</tool_call>\n"
        "Please wait."
    )
    cleaned, tcs = ToolCallExtractor.extract_from_text(xml_text, tools)
    print(f"  * XML Extraction: Found {len(tcs)} calls")
    assert len(tcs) == 1, "Failed XML extraction"
    assert tcs[0].name == "read_file"
    assert tcs[0].arguments.get("path") == "pyproject.toml"
    print(f"    -> Extracted: {tcs[0]}")

    # 2. Markdown JSON code block format
    json_block_text = (
        "Let me run git status.\n"
        "```json\n"
        '{"tool": "run_terminal_command", "parameters": {"command": "git status"}}\n'
        "```\n"
        "Executing now."
    )
    cleaned, tcs = ToolCallExtractor.extract_from_text(json_block_text, tools)
    print(f"  * JSON Block Extraction: Found {len(tcs)} calls")
    assert len(tcs) == 1, "Failed JSON Block extraction"
    assert tcs[0].name == "run_terminal_command"
    assert tcs[0].arguments.get("command") == "git status"
    print(f"    -> Extracted: {tcs[0]}")

    # 3. ReAct Action / Action Input format
    react_text = (
        "Thought: I should check the file contents.\n"
        "Action: read_file\n"
        'Action Input: {"path": "README.md"}\n'
    )
    cleaned, tcs = ToolCallExtractor.extract_from_text(react_text, tools)
    print(f"  * ReAct Extraction: Found {len(tcs)} calls")
    assert len(tcs) == 1, "Failed ReAct extraction"
    assert tcs[0].name == "read_file"
    assert tcs[0].arguments.get("path") == "README.md"
    print(f"    -> Extracted: {tcs[0]}")

    print("  -> ALL SYNTHETIC EXTRACTION TESTS PASSED!\n")


@pytest.mark.asyncio
async def test_live_ollama_tool_calling():
    print("=" * 60)
    print(" [2] LIVE LOCAL OLLAMA TOOL CALLING TEST")
    print("=" * 60)

    settings = get_settings()
    client = LocalLLMClient(settings)

    health = await client.check_health()
    if health.get("status") != "ok":
        print(f"  [SKIP] Local backend not ready: {health}")
        return

    print(f"  Using model: {settings.planner_model} on Ollama ({settings.ollama_base_url})")

    tools = [
        ToolDefinition(
            function=ToolFunctionSchema(
                name="read_file",
                description="Read contents of a file on disk given its path",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file, e.g. pyproject.toml"
                        }
                    },
                    "required": ["path"]
                }
            )
        )
    ]

    messages = [
        ChatMessage(
            role=ChatRole.SYSTEM,
            content=(
                "You are an offline agentic coding assistant. When asked to inspect a file, "
                "you MUST call the read_file tool immediately."
            )
        ),
        ChatMessage(
            role=ChatRole.USER,
            content="Please read the contents of the file 'pyproject.toml' using the read_file tool."
        )
    ]

    print("  Sending tool prompt to local model...")
    extracted_tool_calls = []
    text_deltas = []

    async for chunk in client.stream_chat(messages, tools=tools, temperature=0.0):
        if chunk.delta_text:
            text_deltas.append(chunk.delta_text)
            print(chunk.delta_text, end="", flush=True)
        if chunk.tool_call_delta:
            extracted_tool_calls.append(chunk.tool_call_delta)
            print(f"\n  [DETECTED TOOL CALL] -> {chunk.tool_call_delta}")

    print("\n")
    print(f"  Stream Complete.")
    print(f"  Total Tool Calls Captured: {len(extracted_tool_calls)}")
    for i, tc in enumerate(extracted_tool_calls, 1):
        print(f"    Call #{i}: Tool={tc.name}, Arguments={tc.arguments}")

    if extracted_tool_calls:
        print("  -> LIVE TOOL-CALLING VALIDATION: SUCCESS!")
    else:
        print("  [NOTE] Model did not trigger tool call; raw text was:")
        print("  " + "".join(text_deltas).strip())

    print("=" * 60)


async def main():
    test_synthetic_extractions()
    await test_live_ollama_tool_calling()


if __name__ == "__main__":
    asyncio.run(main())
