"""FastAPI REST & Streaming Server for Project Antimatter.

Exposes sovereign local APIs for Desktop Shell (Tauri v2), Web GUI, and CLI.
"""

from __future__ import annotations
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from antimatter.agent.loop import AgentReActLoop
from antimatter.context.service import CodeIntelligenceService
from antimatter.core.config import get_settings


class ChatRequest(BaseModel):
    task: str
    workspace_dir: Optional[str] = None
    max_steps: int = 15


class IndexRequest(BaseModel):
    workspace_dir: Optional[str] = None
    force: bool = False


def create_app() -> FastAPI:
    """Build and configure Antimatter FastAPI application."""
    settings = get_settings()

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

    @app.get("/health")
    async def health_check():
        return {
            "status": "ok",
            "app_name": settings.app_name,
            "version": settings.version,
            "profile": settings.profile.value,
            "backend": settings.llm_backend.value,
            "offline": True,
        }

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

    @app.post("/agent/run")
    async def run_agent(req: ChatRequest):
        ws = Path(req.workspace_dir or settings.workspace_dir).resolve()
        agent = AgentReActLoop(workspace_dir=ws, max_steps=req.max_steps)
        trajectory = await agent.run(task=req.task)
        return trajectory.model_dump()

    return app


app = create_app()
