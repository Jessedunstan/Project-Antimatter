# Project Antimatter 🚀

> **The Sovereign, 100% Offline-Capable Agentic AI Software Engineering Platform**  
> Synthesizing Google Antigravity, Cursor, Devin Desktop, and OpenAI Codex.

---

## Key Pillars

1. **Zero Cloud Telemetry**: 100% offline-first. Models, embeddings, AST parsing, linting, audio, and tools run on localhost.
2. **Multi-Backend Inference**: Seamless abstraction over Ollama HTTP, `llama-cpp-python`, and vLLM.
3. **Deterministic Agent Tooling**: ReAct cycle with atomic multi-line patching, sandbox timeouts, and checkpoint rollbacks.
4. **Code Intelligence**: Tree-sitter AST parsing + sqlite-vec semantic codebase index + LSP bridge.
5. **Lightweight Surface**: Native desktop shell with Tauri v2 (<30MB binary, <100MB RAM).

---

## Quickstart

```bash
# Clone and setup Python environment
python -m venv .venv
source .venv/bin/activate  # Or on Windows: .venv\Scripts\Activate.ps1
pip install -e .

# Copy environment config
cp .env.example .env

# Verify inference connectivity (Ollama must be running on localhost:11434)
python tests/test_inference.py

# Run complete test suite (Track 1, Track 2, Track 3, Agent & API)
pytest

# Code Intelligence & CLI Commands
python -m antimatter.cli info
python -m antimatter.cli index
python -m antimatter.cli search "relational database"
python -m antimatter.cli symbols antimatter/context/types.py
python -m antimatter.cli check antimatter/context/types.py
python -m antimatter.cli run "Inspect project and summarize tool capabilities"
```

---

## Architectural Tracks

- **Track 1: Sovereign Brain & Multi-Backend Inference**
  - Offline Ollama, vLLM, and llama-cpp serving.
  - Streaming tool extraction (XML, Markdown JSON, ReAct blocks).
  - FastEmbed CPU ONNX 384d neural embeddings.
- **Track 2: Sandboxed Tooling & Deterministic Workspace**
  - Atomic filesystem manager with transactional checkpoint/rollback.
  - Multi-line patcher with CRLF/LF tolerance and strict ambiguity rejection.
  - Async terminal sandbox with wall-clock timeout enforcement.
- **Track 3: Code Intelligence & Deep Context Layer**
  - Tree-sitter AST parser (Python, JS, TS) extracting definitions, references, and symbol hierarchies.
  - Async LSP client (JSON-RPC 2.0 stdio) with bundled local Python language server for hover and diagnostics.
  - Persistent `sqlite-vec` vector store (`vec0` virtual table) with AST-aware contextual chunking and sub-millisecond retrieval.
- **Track 4: Autonomous ReAct Agent Loop & Surface**
  - Autonomous ReAct execution loop connecting Brain, Tools, and Context.
  - Local FastAPI REST & Streaming server for desktop shell (Tauri v2).
  - Rich CLI for terminal workflows (`index`, `search`, `symbols`, `check`, `run`, `serve`).
