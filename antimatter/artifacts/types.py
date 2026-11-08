"""Data contracts for Project Antimatter Artifacts and Canvas Engine."""

from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class ArtifactType(str, Enum):
    MARKDOWN = "markdown"
    HTML = "html"
    MERMAID = "mermaid"
    DIFF = "diff"
    SVG = "svg"


class ArtifactMetadata(BaseModel):
    """Metadata describing an artifact document or canvas item."""
    id: str = Field(description="Unique artifact identifier, e.g. art_20261109_abc123")
    title: str = Field(description="Short human-readable title")
    summary: str = Field(default="", description="Multi-line summary of contents and purpose")
    artifact_type: ArtifactType = Field(default=ArtifactType.MARKDOWN)
    user_facing: bool = Field(default=True, description="Whether to display in the main user UI canvas")
    request_feedback: bool = Field(default=False, description="Whether to request explicit user feedback")
    created_at: float = Field(default=0.0)
    updated_at: float = Field(default=0.0)
    version: int = Field(default=1)


class Artifact(BaseModel):
    """Full artifact container including metadata, content, and local path."""
    metadata: ArtifactMetadata
    content: str
    file_path: Optional[str] = None
