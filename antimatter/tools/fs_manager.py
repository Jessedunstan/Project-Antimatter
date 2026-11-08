"""Deterministic File System and Snapshot Manager for Project Antimatter.

Enforces workspace boundary sandboxing, atomic file operations, and
instantaneous rollback checkpoints.
"""

import json
import logging
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SecurityException(Exception):
    """Raised when an operation attempts to breach workspace boundaries."""
    pass


class Checkpoint(Dict):
    """Container tracking snapshot file records for a rollback transaction."""
    pass


class FileSystemManager:
    """Sandbox-guarded file system manager with atomic writes and checkpoints."""

    def __init__(self, workspace_dir: Path):
        self.workspace_dir = workspace_dir.resolve()
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir = self.workspace_dir / ".antimatter" / "snapshots"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def resolve_safe_path(self, relative_or_abs_path: str | Path) -> Path:
        """
        Resolve path and ensure it does not escape workspace boundaries.
        Raises SecurityException on path traversal attacks.
        """
        p = Path(relative_or_abs_path)
        if not p.is_absolute():
            resolved = (self.workspace_dir / p).resolve()
        else:
            resolved = p.resolve()

        # Check boundary containment
        try:
            resolved.relative_to(self.workspace_dir)
        except ValueError:
            raise SecurityException(
                f"Path '{relative_or_abs_path}' resolves to '{resolved}', "
                f"which is outside sovereign workspace '{self.workspace_dir}'."
            )
        return resolved

    def create_checkpoint(self, description: str = "") -> str:
        """Create a new rollback checkpoint ID."""
        checkpoint_id = f"chk_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        cp_dir = self.snapshot_dir / checkpoint_id
        cp_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "checkpoint_id": checkpoint_id,
            "created_at": time.time(),
            "description": description,
            "files": []
        }
        with open(cp_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return checkpoint_id

    def snapshot_file(self, target_path: Path, checkpoint_id: str) -> None:
        """Record the current state of a file before modification into a checkpoint."""
        safe_path = self.resolve_safe_path(target_path)
        cp_dir = self.snapshot_dir / checkpoint_id
        if not cp_dir.exists():
            cp_dir.mkdir(parents=True, exist_ok=True)

        meta_file = cp_dir / "meta.json"
        meta = {"files": []}
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)

        rel_path = str(safe_path.relative_to(self.workspace_dir))
        # Only snapshot once per checkpoint to preserve original baseline
        if any(item["rel_path"] == rel_path for item in meta.get("files", [])):
            return

        snapshot_record = {
            "rel_path": rel_path,
            "existed": safe_path.exists(),
        }

        if safe_path.exists():
            # Copy original content to snapshot storage
            snap_name = f"{uuid.uuid4().hex}_{safe_path.name}"
            shutil.copy2(safe_path, cp_dir / snap_name)
            snapshot_record["snapshot_file"] = snap_name

        meta.setdefault("files", []).append(snapshot_record)
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def rollback(self, checkpoint_id: str) -> Tuple[bool, str]:
        """Restore all files modified in a checkpoint to their original states."""
        cp_dir = self.snapshot_dir / checkpoint_id
        meta_file = cp_dir / "meta.json"
        if not meta_file.exists():
            return False, f"Checkpoint '{checkpoint_id}' not found."

        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        restored_count = 0
        deleted_count = 0

        for file_record in reversed(meta.get("files", [])):
            target_path = self.workspace_dir / file_record["rel_path"]
            if file_record["existed"]:
                snap_source = cp_dir / file_record["snapshot_file"]
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(snap_source, target_path)
                restored_count += 1
            else:
                # File was created during transaction, remove it
                if target_path.exists():
                    target_path.unlink()
                    deleted_count += 1

        return True, f"Rollback successful: {restored_count} restored, {deleted_count} newly-created deleted."

    def read_file(
        self,
        path: str | Path,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> str:
        """Read file contents with line range filtering."""
        safe_path = self.resolve_safe_path(path)
        if not safe_path.exists() or not safe_path.is_file():
            raise FileNotFoundError(f"File '{path}' does not exist.")

        with open(safe_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        if start_line is not None or end_line is not None:
            s = (start_line - 1) if (start_line and start_line > 0) else 0
            e = end_line if end_line is not None else len(lines)
            return "".join(lines[s:e])

        return "".join(lines)

    def write_atomic(
        self,
        path: str | Path,
        content: str,
        checkpoint_id: Optional[str] = None,
    ) -> int:
        """Atomically write content to file, creating snapshots if checkpoint is active."""
        safe_path = self.resolve_safe_path(path)
        safe_path.parent.mkdir(parents=True, exist_ok=True)

        if checkpoint_id:
            self.snapshot_file(safe_path, checkpoint_id)

        # Write to temporary file in same directory then atomically rename
        tmp_path = safe_path.with_suffix(f".tmp_{uuid.uuid4().hex[:6]}")
        with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
            bytes_written = f.write(content)
            f.flush()
            os.fsync(f.fileno())

        tmp_path.replace(safe_path)
        return bytes_written

    def list_dir(self, subpath: str = "") -> List[Dict[str, str]]:
        """List files and folders within a workspace subpath."""
        safe_dir = self.resolve_safe_path(subpath)
        if not safe_dir.exists() or not safe_dir.is_dir():
            raise NotADirectoryError(f"Directory '{subpath}' does not exist.")

        entries = []
        for item in sorted(safe_dir.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            if item.name == ".antimatter" or item.name == "__pycache__":
                continue
            entries.append({
                "name": item.name,
                "type": "directory" if item.is_dir() else "file",
                "rel_path": str(item.relative_to(self.workspace_dir)).replace("\\", "/")
            })
        return entries
