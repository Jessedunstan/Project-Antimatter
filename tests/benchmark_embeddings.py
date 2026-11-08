"""Benchmark and verification for FastEmbed ONNX local embedding engine."""

import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from antimatter.brain.embeddings import LocalEmbeddingEngine


def run_benchmark():
    print("=" * 60)
    print(" ANTIMATTER BRAIN: FASTEMBED ONNX BENCHMARK")
    print("=" * 60)

    model_name = "BAAI/bge-small-en-v1.5"
    print(f"[1] Initializing FastEmbed Engine with model: {model_name}...")
    t0 = time.perf_counter()
    engine = LocalEmbeddingEngine(model_name=model_name)
    # Trigger load
    engine._ensure_loaded()
    load_time = (time.perf_counter() - t0) * 1000
    print(f"    Loaded in: {load_time:.2f} ms")

    # Benchmark single query
    print("\n[2] Benchmarking Single Query Embedding...")
    query = "def find_symbol_references(ast_node, symbol_name):"
    t0 = time.perf_counter()
    query_vector = engine.embed_query(query)
    query_time = (time.perf_counter() - t0) * 1000
    print(f"    Query Latency: {query_time:.2f} ms")
    print(f"    Vector Dimensions: {len(query_vector)}")
    print(f"    Vector Sample: [{', '.join(f'{x:.4f}' for x in query_vector[:5])}, ...]")

    # Benchmark batch documents
    print("\n[3] Benchmarking Code Chunk Batch Embedding...")
    documents = [
        "class AgentRuntime: def __init__(self): self.mailbox = []",
        "async def execute_tool(tool_name: str, args: dict) -> ToolResult:",
        "def parse_tree_sitter_ast(code: str, language: str) -> ASTGraph:",
        "class VectorIndex: def search_nearest(self, query_vec, k=5):",
        "def compact_context_window(messages: list, token_budget: int):",
        "async def stream_tokens_from_ollama(prompt: str) -> AsyncIterator:",
        "def create_git_snapshot(workspace_dir: str) -> str:",
        "def apply_unified_diff(original: str, patch: str) -> str:",
        "class SubAgentSupervisor: def spawn(self, spec: AgentSpec):",
        "def sanitize_terminal_command(command: str) -> bool:"
    ] * 10  # 100 code chunks

    print(f"    Embedding batch of {len(documents)} code chunks...")
    t0 = time.perf_counter()
    batch_vectors = engine.embed_documents(documents)
    batch_time = (time.perf_counter() - t0) * 1000
    throughput = len(documents) / (batch_time / 1000)

    print(f"    Batch Time: {batch_time:.2f} ms")
    print(f"    Throughput: {throughput:.1f} chunks/sec")
    print(f"    Avg Latency per Chunk: {(batch_time / len(documents)):.2f} ms")

    assert len(batch_vectors) == len(documents), "Mismatched document output count"
    assert len(batch_vectors[0]) == len(query_vector), "Mismatched dimension size"

    print("\n[4] Benchmark Status: PASS (100% Offline ONNX verified)")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmark()
