"""Local HTTP and WebSocket Preview Server for Artifacts Canvas.

Enables instant offline viewing of interactive web views, Mermaid diagrams,
and code diffs inside desktop webviews or local browsers.
"""

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse, PlainTextResponse
from typing import List, Optional
from pydantic import BaseModel

from antimatter.artifacts.manager import ArtifactManager
from antimatter.artifacts.renderer import ArtifactRenderer
from antimatter.artifacts.types import Artifact, ArtifactMetadata, ArtifactType


class CreateArtifactRequest(BaseModel):
    title: str
    content: str
    artifact_type: ArtifactType = ArtifactType.MARKDOWN
    summary: str = ""
    request_feedback: bool = False


def create_artifact_app(manager: ArtifactManager) -> FastAPI:
    """Create FastAPI application wired to the ArtifactManager."""
    app = FastAPI(title="Antimatter Artifact Canvas Server")

    @app.get("/", response_class=HTMLResponse)
    def index_dashboard():
        artifacts = manager.list_artifacts()
        cards_html = ""
        for a in artifacts:
            cards_html += f"""
            <div style="background:#1e2025; border:1px solid #2e323b; border-radius:8px; padding:16px; margin-bottom:12px;">
              <h3 style="margin:0 0 8px 0;"><a href="/artifacts/{a.id}" style="color:#60a5fa; text-decoration:none;">{a.title}</a></h3>
              <p style="color:#9da3ae; font-size:0.9rem; margin-bottom:8px;">{a.summary or 'No summary'}</p>
              <span style="background:#282b33; color:#a78bfa; padding:2px 8px; border-radius:4px; font-size:0.75rem;">{a.artifact_type.value}</span>
              <span style="color:#64748b; font-size:0.8rem; margin-left:12px;">v{a.version}</span>
            </div>
            """

        if not cards_html:
            cards_html = "<p style='color:#64748b;'>No artifacts created yet.</p>"

        return f"""<!DOCTYPE html>
        <html>
        <head>
          <title>Antimatter Artifacts Canvas</title>
          <style>
            body {{ font-family: -apple-system, sans-serif; background: #121316; color: #e6e8eb; padding: 32px; max-width: 800px; margin: auto; }}
            h1 {{ border-bottom: 1px solid #2e323b; padding-bottom: 16px; margin-bottom: 24px; }}
          </style>
        </head>
        <body>
          <h1>Antimatter Artifacts Canvas</h1>
          {cards_html}
        </body>
        </html>"""

    @app.get("/artifacts", response_model=List[ArtifactMetadata])
    def list_artifacts():
        return manager.list_artifacts()

    @app.get("/artifacts/{artifact_id}", response_class=HTMLResponse)
    def view_artifact(artifact_id: str):
        artifact = manager.get_artifact(artifact_id)
        if not artifact:
            raise HTTPException(status_code=404, detail="Artifact not found")
        html_page = ArtifactRenderer.render_to_html(artifact)
        return HTMLResponse(content=html_page)

    @app.get("/artifacts/{artifact_id}/raw", response_class=PlainTextResponse)
    def get_raw_content(artifact_id: str):
        artifact = manager.get_artifact(artifact_id)
        if not artifact:
            raise HTTPException(status_code=404, detail="Artifact not found")
        return PlainTextResponse(content=artifact.content)

    @app.post("/artifacts", response_model=ArtifactMetadata)
    def create_artifact(req: CreateArtifactRequest):
        artifact = manager.create_artifact(
            title=req.title,
            content=req.content,
            artifact_type=req.artifact_type,
            summary=req.summary,
            request_feedback=req.request_feedback,
        )
        return artifact.metadata

    @app.delete("/artifacts/{artifact_id}")
    def delete_artifact(artifact_id: str):
        success = manager.delete_artifact(artifact_id)
        if not success:
            raise HTTPException(status_code=404, detail="Artifact not found")
        return {"status": "deleted", "id": artifact_id}

    return app
