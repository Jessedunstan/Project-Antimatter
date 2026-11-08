"""Pytest test suite for Project Antimatter Track 1: Sovereign Brain."""

import pytest
from antimatter.core.config import get_settings, LLMBackendType
from antimatter.brain.types import (
    ChatMessage,
    ChatRole,
    ToolDefinition,
    ToolFunctionSchema,
)
from antimatter.brain.client import LocalLLMClient
from antimatter.brain.embeddings import LocalEmbeddingEngine
from antimatter.brain.tool_extractor import ToolCallExtractor


def test_config_loading():
    settings = get_settings()
    assert settings.app_name == "Antimatter Sovereign Engine"
    assert settings.workspace_dir.exists()


def test_tool_call_extractor_xml():
    xml_text = '<tool_call>{"name": "read_file", "arguments": {"path": "src/main.py"}}</tool_call>'
    cleaned, tcs = ToolCallExtractor.extract_from_text(xml_text)
    assert len(tcs) == 1
    assert tcs[0].name == "read_file"
    assert tcs[0].arguments == {"path": "src/main.py"}
    assert cleaned == ""


def test_tool_call_extractor_json_block():
    block_text = '```json\n{"tool": "write_file", "parameters": {"path": "out.txt", "content": "hi"}}\n```'
    cleaned, tcs = ToolCallExtractor.extract_from_text(block_text)
    assert len(tcs) == 1
    assert tcs[0].name == "write_file"
    assert tcs[0].arguments["path"] == "out.txt"


def test_tool_call_extractor_react():
    react_text = 'Action: execute_command\nAction Input: {"cmd": "pytest"}'
    cleaned, tcs = ToolCallExtractor.extract_from_text(react_text)
    assert len(tcs) == 1
    assert tcs[0].name == "execute_command"
    assert tcs[0].arguments["cmd"] == "pytest"


def test_fastembed_local_dimensions():
    engine = LocalEmbeddingEngine("BAAI/bge-small-en-v1.5")
    vec = engine.embed_query("test query")
    assert len(vec) == 384
    assert isinstance(vec[0], float)


@pytest.mark.asyncio
async def test_local_llm_client_mock():
    settings = get_settings()
    settings.llm_backend = LLMBackendType.MOCK
    client = LocalLLMClient(settings)

    chunks = []
    async for chunk in client.stream_chat([ChatMessage(role=ChatRole.USER, content="hello")]):
        chunks.append(chunk)

    text = "".join(c.delta_text for c in chunks)
    assert "Mock Engine" in text
