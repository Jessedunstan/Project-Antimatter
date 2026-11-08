"""Persistent Artifact Manager for Project Antimatter.

Creates, versions, and manages artifacts on local disk with atomic updates.
"""

import json
import logging
from pathlib import Path
import time
import uuid
from typing import Dict, List, Optional

from antimatter.artifacts.types import Artifact, ArtifactMetadata, ArtifactType

logger = logging.getLogger(__name__)


class ArtifactManager:
    """Manages creation, retrieval, and disk persistence of artifacts."""

    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir.resolve()
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, Artifact] = {}
        self._load_existing()

    def _load_existing(self) -> None:
        """Scan directory and index existing artifacts into memory."""
        self._cache.clear()
        for art_folder in self.storage_dir.iterdir():
            if not art_folder.is_dir():
                continue
            meta_file = art_folder / "meta.json"
            content_file = art_folder / "content.txt"
            if meta_file.exists() and content_file.exists():
                try:
                    meta_dict = json.loads(meta_file.read_text(encoding="utf-8"))
                    content = content_file.read_text(encoding="utf-8", errors="replace")
                    meta = ArtifactMetadata(**meta_dict)
                    self._cache[meta.id] = Artifact(
                        metadata=meta,
                        content=content,
                        file_path=str(content_file)
                    )
                except Exception as err:
                    logger.debug("Failed loading artifact in %s: %s", art_folder, err)

    def create_artifact(
        self,
        title: str,
        content: str,
        artifact_type: ArtifactType = ArtifactType.MARKDOWN,
        summary: str = "",
        request_feedback: bool = False,
        artifact_id: Optional[str] = None,
    ) -> Artifact:
        """Create and persist a new artifact."""
        art_id = artifact_id or f"art_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        now = time.time()

        meta = ArtifactMetadata(
            id=art_id,
            title=title,
            summary=summary,
            artifact_type=artifact_type,
            user_facing=True,
            request_feedback=request_feedback,
            created_at=now,
            updated_at=now,
            version=1,
        )

        folder = self.storage_dir / art_id
        folder.mkdir(parents=True, exist_ok=True)

        meta_path = folder / "meta.json"
        content_path = folder / "content.txt"

        meta_path.write_text(json.dumps(meta.model_dump(), indent=2), encoding="utf-8")
        content_path.write_text(content, encoding="utf-8")

        artifact = Artifact(metadata=meta, content=content, file_path=str(content_path))
        self._cache[art_id] = artifact
        return artifact

    def get_artifact(self, artifact_id: str) -> Optional[Artifact]:
        """Retrieve artifact by ID."""
        return self._cache.get(artifact_id)

    def update_artifact(
        self,
        artifact_id: str,
        content: str,
        summary: Optional[str] = None,
    ) -> Optional[Artifact]:
        """Update an existing artifact's content and increment version."""
        existing = self._cache.get(artifact_id)
        if not existing:
            return None

        folder = self.storage_dir / artifact_id
        now = time.time()

        existing.content = content
        existing.metadata.version += 1
        existing.metadata.updated_at = now
        if summary:
            existing.metadata.summary = summary

        meta_path = folder / "meta.json"
        content_path = folder / "content.txt"

        meta_path.write_text(json.dumps(existing.metadata.model_dump(), indent=2), encoding="utf-8")
        content_path.write_text(content, encoding="utf-8")

        return existing

    def list_artifacts(self) -> List[ArtifactMetadata]:
        """List metadata of all stored artifacts."""
        return sorted(
            [a.metadata for a in self._cache.values()],
            key=lambda m: m.updated_at,
            reverse=True
        )

    def delete_artifact(self, artifact_id: str) -> bool:
        """Delete artifact from disk and memory."""
        if artifact_id not in self._cache:
            return False

        folder = self.storage_dir / artifact_id
        if folder.exists():
            import shutil
            shutil.rmtree(folder, ignore_errors=True)

        del self._cache[artifact_id]
        return True
