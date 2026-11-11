"""FastAPI REST & Streaming Server for Project Antimatter.

Exposes sovereign local APIs for Desktop Shell (Tauri v2), Web GUI, and CLI.
"""

from __future__ import annotations
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel

from antimatter.agent.loop import AgentReActLoop
from antimatter.agent.supervisor import SubagentSupervisor
from antimatter.agent.subagent import SubagentSpec, WorkspaceMode
from antimatter.artifacts.manager import ArtifactManager
from antimatter.artifacts.renderer import ArtifactRenderer
from antimatter.artifacts.types import ArtifactType
from antimatter.context.service import CodeIntelligenceService
from antimatter.core.config import get_settings
from antimatter.brain.client import LocalLLMClient
from antimatter.brain.types import ChatMessage, ChatRole
from antimatter.tools.fs_manager import FileSystemManager
from antimatter.tools.terminal import TerminalSandbox


class ChatRequest(BaseModel):
    task: str
    workspace_dir: Optional[str] = None
    max_steps: int = 15


class IndexRequest(BaseModel):
    workspace_dir: Optional[str] = None
    force: bool = False


class AutocompleteRequest(BaseModel):
    prefix: str
    suffix: str = ""
    file_path: Optional[str] = None
    max_tokens: int = 64


class FileWriteRequest(BaseModel):
    path: str
    content: str
    checkpoint_id: Optional[str] = None


class TerminalRunRequest(BaseModel):
    command: str
    timeout_sec: Optional[int] = None


class SpawnSubagentRequest(BaseModel):
    name: str
    role: str
    prompt: str
    workspace_mode: str = "inherit"


def create_app() -> FastAPI:
    """Build and configure Antimatter FastAPI application."""
    settings = get_settings()
    workspace_root = settings.workspace_dir.resolve()

    fs_manager = FileSystemManager(workspace_root)
    terminal_sandbox = TerminalSandbox(workspace_root, max_timeout_sec=settings.max_exec_timeout_sec)
    artifact_manager = ArtifactManager(workspace_root / ".antimatter" / "artifacts")
    supervisor = SubagentSupervisor(workspace_root)
    llm_client = LocalLLMClient(settings)

    app = FastAPI(
        title="Antimatter Sovereign Engine API",
        version=settings.version,
        description="Local offline API for Code Intelligence, Inference, and Autonomous Agent Runtime",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------------
    # 1. SYSTEM & HEALTH
    # ------------------------------------------------------------------------
    @app.get("/health")
    async def health_check():
        llm_health = await llm_client.check_health()
        return {
            "status": "ok",
            "app_name": settings.app_name,
            "version": settings.version,
            "profile": settings.profile.value,
            "backend": settings.llm_backend.value,
            "offline": True,
            "llm": llm_health,
        }

    # ------------------------------------------------------------------------
    # 2. CODE INTELLIGENCE & AST
    # ------------------------------------------------------------------------
    @app.post("/index")
    async def index_workspace(req: IndexRequest):
        ws = Path(req.workspace_dir or settings.workspace_dir).resolve()
        service = CodeIntelligenceService(workspace_dir=ws)
        stats = service.index_workspace(force=req.force)
        return {"status": "indexed", "stats": stats}

    @app.get("/search")
    async def search_code(q: str = Query(..., description="Semantic search query"), limit: int = 5):
        service = CodeIntelligenceService()
        results = service.semantic_search(query=q, limit=limit)
        return {
            "query": q,
            "count": len(results),
            "results": [
                {
                    "file_path": r.chunk.file_path,
                    "name": r.chunk.name,
                    "start_line": r.chunk.start_line,
                    "end_line": r.chunk.end_line,
                    "content": r.chunk.content,
                    "score": r.score,
                    "distance": r.distance,
                }
                for r in results
            ],
        }

    @app.get("/symbols")
    async def get_symbols(file_path: str = Query(..., description="File path relative to workspace")):
        service = CodeIntelligenceService()
        hierarchy = service.get_symbol_hierarchy(file_path)
        return {"file_path": file_path, "symbols": [s.model_dump() for s in hierarchy]}

    @app.get("/diagnostics")
    async def get_diagnostics(file_path: str = Query(..., description="File path")):
        service = CodeIntelligenceService()
        diags = await service.get_diagnostics(file_path)
        return {"file_path": file_path, "diagnostics": [d.model_dump() for d in diags]}

    # ------------------------------------------------------------------------
    # 3. CURSOR-STYLE INLINE AUTOCOMPLETE (FIM)
    # ------------------------------------------------------------------------
    @app.post("/autocomplete")
    async def inline_autocomplete(req: AutocompleteRequest):
        """Perform Fill-In-the-Middle (FIM) tab completion using local model."""
        prompt = (
            f"<｜fim begin｜>{req.prefix}<｜fim hole｜>{req.suffix}<｜fim end｜>"
            if "qwen" in settings.autocomplete_model.lower()
            else f"<PRE> {req.prefix} <SUF>{req.suffix} <MID>"
        )

        messages = [
            ChatMessage(
                role=ChatRole.USER,
                content=f"Complete the code at the cursor position without extra conversation:\n{prompt}",
            )
        ]

        text_parts = []
        async for chunk in llm_client.stream_chat(
            messages=messages,
            model=settings.autocomplete_model,
            temperature=0.1,
        ):
            if chunk.delta_text:
                text_parts.append(chunk.delta_text)

        completion = "".join(text_parts).strip("\n\r`")
        return {"completion": completion}

    # ------------------------------------------------------------------------
    # 4. WORKSPACE & FILE OPERATIONS
    # ------------------------------------------------------------------------
    @app.get("/files")
    def list_files(subpath: str = ""):
        return {"entries": fs_manager.list_dir(subpath)}

    @app.get("/file/read")
    def read_file(path: str, start_line: Optional[int] = None, end_line: Optional[int] = None):
        content = fs_manager.read_file(path, start_line=start_line, end_line=end_line)
        return {"path": path, "content": content}

    @app.post("/file/write")
    def write_file(req: FileWriteRequest):
        bytes_written = fs_manager.write_atomic(req.path, req.content, checkpoint_id=req.checkpoint_id)
        return {"status": "ok", "path": req.path, "bytes_written": bytes_written}

    # ------------------------------------------------------------------------
    # 5. SANDBOXED TERMINAL RUNNER
    # ------------------------------------------------------------------------
    @app.post("/terminal/execute")
    async def execute_terminal(req: TerminalRunRequest):
        result = await terminal_sandbox.execute(req.command, timeout_sec=req.timeout_sec)
        return result.model_dump()

    # ------------------------------------------------------------------------
    # 6. AUTONOMOUS AGENT REACT LOOP
    # ------------------------------------------------------------------------
    @app.post("/agent/run")
    async def run_agent(req: ChatRequest):
        ws = Path(req.workspace_dir or settings.workspace_dir).resolve()
        agent = AgentReActLoop(workspace_dir=ws, max_steps=req.max_steps)
        trajectory = await agent.run(task=req.task)
        return trajectory.model_dump()

    # ------------------------------------------------------------------------
    # 7. SUBAGENT SUPERVISOR & TELEMETRY
    # ------------------------------------------------------------------------
    @app.get("/subagents")
    async def list_subagents():
        return {"subagents": [s.model_dump() for s in supervisor.list_subagents()]}

    @app.post("/subagents/spawn")
    async def spawn_subagent(req: SpawnSubagentRequest):
        mode = WorkspaceMode.BRANCH if req.workspace_mode == "branch" else WorkspaceMode.INHERIT
        spec = SubagentSpec(
            name=req.name,
            role=req.role,
            prompt=req.prompt,
            workspace_mode=mode,
        )
        worker = supervisor.spawn(spec)
        return {"status": "spawned", "id": worker.id, "workspace_mode": mode.value}

    @app.post("/subagents/kill")
    async def kill_subagent(subagent_id: str = Query(...)):
        success, msg = supervisor.kill_subagent(subagent_id)
        return {"status": "killed" if success else "failed", "message": msg}

    # ------------------------------------------------------------------------
    # 8. ARTIFACTS & CANVAS
    # ------------------------------------------------------------------------
    @app.get("/artifacts")
    def list_artifacts():
        return {"artifacts": [a.model_dump() for a in artifact_manager.list_artifacts()]}

    @app.get("/artifacts/{artifact_id}")
    def get_artifact(artifact_id: str):
        art = artifact_manager.get_artifact(artifact_id)
        if not art:
            raise HTTPException(status_code=404, detail="Artifact not found")
        return art.model_dump()

    @app.get("/artifacts/{artifact_id}/html", response_class=HTMLResponse)
    def view_artifact_html(artifact_id: str):
        art = artifact_manager.get_artifact(artifact_id)
        if not art:
            raise HTTPException(status_code=404, detail="Artifact not found")
        return HTMLResponse(content=ArtifactRenderer.render_to_html(art))

    return app


app = create_app()
