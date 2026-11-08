"""Track 1 Comprehensive Verification Suite: Sovereign Brain & Multi-Backend Inference Runtime.

Validates:
1. Local Ollama inference streaming and token generation
2. Tool-calling extraction and structured argument parsing
3. FastEmbed ONNX local CPU embedding speed and throughput benchmark
"""

import asyncio
import time
from typing import List
import pytest

from antimatter.core.config import get_settings, LLMBackendType
from antimatter.brain.client import LocalLLMClient
from antimatter.brain.embeddings import LocalEmbeddingEngine
from antimatter.brain.types import (
    ChatMessage,
    ChatRole,
    ToolDefinition,
    ToolFunctionSchema,
)


@pytest.mark.asyncio
async def test_ollama_health_and_model_presence():
    """Verify local Ollama daemon connectivity and model availability."""
    settings = get_settings()
    client = LocalLLMClient(settings)
    health = await client.check_health()

    assert health["status"] == "ok", f"Health check failed: {health}"
    assert health["backend"] == "ollama"
    assert len(health.get("models", [])) > 0, "No models found in Ollama"
    print(f"\n[OK] Ollama connected. Available models: {health['models']}")


@pytest.mark.asyncio
async def test_ollama_streaming_inference():
    """Verify local Ollama streaming inference with token latency calculation."""
    settings = get_settings()
    client = LocalLLMClient(settings)

    prompt = "In 2 sentences, explain why sovereign offline AI is vital for confidential codebases."
    messages = [
        ChatMessage(role=ChatRole.SYSTEM, content="You are a helpful software architecture assistant."),
        ChatMessage(role=ChatRole.USER, content=prompt),
    ]

    t0 = time.perf_counter()
    tokens_received = 0
    first_token_time = None
    streamed_text = []

    async for chunk in client.stream_chat(messages=messages, temperature=0.1):
        if chunk.delta_text:
            if first_token_time is None:
                first_token_time = time.perf_counter() - t0
            streamed_text.append(chunk.delta_text)
            tokens_received += 1

    total_time = time.perf_counter() - t0
    full_response = "".join(streamed_text).strip()

    assert len(full_response) > 0, "Empty response received from Ollama"
    assert tokens_received > 0

    tps = tokens_received / total_time if total_time > 0 else 0
    print(f"\n[OK] Streaming Inference Verified:")
    print(f"     Time to first token (TTFT): {first_token_time * 1000:.1f}ms" if first_token_time else "N/A")
    print(f"     Total time:                 {total_time:.2f}s")
    print(f"     Tokens (chunks):            {tokens_received}")
    print(f"     Throughput:                 {tps:.1f} chunks/sec")
    print(f"     Generated snippet:          {full_response[:100]}...")


def test_tool_call_argument_normalization():
    """Verify robust argument normalization across schema artifacts and JSON formats."""
    # Direct dictionary
    assert LocalLLMClient._normalize_tool_arguments({"filepath": "foo.py"}) == {"filepath": "foo.py"}
    
    # JSON string
    assert LocalLLMClient._normalize_tool_arguments('{"filepath": "foo.py"}') == {"filepath": "foo.py"}
    
    # Model mirroring properties with real values
    nested = {"type": "object", "properties": {"filepath": "foo.py", "line": 10}}
    assert LocalLLMClient._normalize_tool_arguments(nested) == {"filepath": "foo.py", "line": 10}

    # Model providing top-level key alongside accidental properties dict
    mixed = {"filepath": "foo.py", "properties": {"filepath": "foo.py"}}
    assert LocalLLMClient._normalize_tool_arguments(mixed) == {"filepath": "foo.py"}


@pytest.mark.asyncio
async def test_tool_calling_extraction():
    """Validate live tool-calling detection and structured argument extraction with Ollama."""
    settings = get_settings()
    client = LocalLLMClient(settings)

    file_tool = ToolDefinition(
        function=ToolFunctionSchema(
            name="read_source_file",
            description="Reads contents of a local repository file given an absolute or relative path",
            parameters={
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "The path of the source code file to inspect",
                    }
                },
                "required": ["filepath"],
            },
        )
    )

    messages = [
        ChatMessage(
            role=ChatRole.SYSTEM,
            content="You are an autonomous AI software engineer. Use designated tools to read files.",
        ),
        ChatMessage(
            role=ChatRole.USER,
            content="Read the file antimatter/brain/client.py using read_source_file.",
        ),
    ]

    detected_tools = []
    async for chunk in client.stream_chat(messages=messages, tools=[file_tool], temperature=0.0):
        if chunk.tool_call_delta:
            detected_tools.append(chunk.tool_call_delta)

    assert len(detected_tools) > 0, "No tool calls detected by the local model"
    tool_call = detected_tools[0]
    print(f"\n[OK] Tool-calling extracted successfully:")
    print(f"     Call ID:   {tool_call.id}")
    print(f"     Tool Name: {tool_call.name}")
    print(f"     Arguments: {tool_call.arguments}")

    assert tool_call.name == "read_source_file"
    assert "filepath" in tool_call.arguments
    assert "client.py" in tool_call.arguments["filepath"]


def test_fastembed_onnx_benchmark():
    """Benchmark FastEmbed ONNX embedding speed across varying batch sizes."""
    engine = LocalEmbeddingEngine(model_name="BAAI/bge-small-en-v1.5")

    corpus = [
        "Project Antimatter: Autonomous offline software engineering platform.",
        "Deterministic local execution sandbox with zero external telemetry.",
        "FastEmbed ONNX runtime eliminates heavy PyTorch dependencies for embeddings.",
        "Multi-backend inference orchestrator routing between Ollama and vLLM.",
        "High-performance vector memory indexing semantic code search chunks.",
        "Local Tree-Sitter AST parser extracting code symbols and dependencies.",
        "Sub-millisecond token streaming from sovereign quantized models.",
        "Context window optimization with automatic compaction and caching.",
    ]

    print("\n[OK] Benchmarking FastEmbed ONNX CPU embeddings:")
    batch_sizes = [1, 4, 8, 32]
    
    for batch_size in batch_sizes:
        # Generate batch
        batch = (corpus * ((batch_size // len(corpus)) + 1))[:batch_size]

        t0 = time.perf_counter()
        embeddings = engine.embed_documents(batch)
        elapsed = time.perf_counter() - t0

        docs_per_sec = len(batch) / elapsed
        ms_per_doc = (elapsed / len(batch)) * 1000

        assert len(embeddings) == batch_size
        assert len(embeddings[0]) == 384  # bge-small-en-v1.5 dimension

        print(
            f"     Batch Size: {batch_size:2d} | "
            f"Total: {elapsed * 1000:6.1f}ms | "
            f"Latency: {ms_per_doc:5.2f}ms/doc | "
            f"Throughput: {docs_per_sec:6.1f} docs/sec"
        )
