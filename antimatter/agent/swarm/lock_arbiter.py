"""Swarm File Lock Arbiter inspired by Project Chimera.

Guarantees strict concurrency isolation across parallel swarm micro-agents (Scout, Architect, Coder, Verifier).
Prevents race conditions, concurrent write collisions, and deadlocks through deterministic timed leases.
"""

from __future__ import annotations
import asyncio
import time
from typing import Dict, List, Optional, Set
from pydantic import BaseModel, Field


class FileLease(BaseModel):
    file_path: str
    owner_agent: str
    acquired_at: float = Field(default_factory=time.time)
    ttl_sec: float = 30.0
    reason: str = "patch_execution"

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.acquired_at) > self.ttl_sec


class LockArbiter:
    """Manages atomic file locks across concurrent swarm subagents."""

    def __init__(self, default_ttl_sec: float = 30.0):
        self.default_ttl_sec = default_ttl_sec
        self._locks: Dict[str, FileLease] = {}
        self._lock = asyncio.Lock()

    def _cleanup_expired(self) -> None:
        """Removes expired leases."""
        expired_paths = [path for path, lease in self._locks.items() if lease.is_expired]
        for path in expired_paths:
            del self._locks[path]

    async def acquire(
        self,
        file_path: str,
        owner_agent: str,
        ttl_sec: Optional[float] = None,
        reason: str = "patch_execution",
    ) -> bool:
        """Attempt to acquire an exclusive lock on a file path."""
        normalized_path = file_path.replace("\\", "/").lower()
        ttl = ttl_sec or self.default_ttl_sec

        async with self._lock:
            self._cleanup_expired()
            existing = self._locks.get(normalized_path)

            if existing:
                if existing.owner_agent == owner_agent:
                    # Refresh lease
                    existing.acquired_at = time.time()
                    existing.ttl_sec = ttl
                    return True
                # Locked by another agent
                return False

            self._locks[normalized_path] = FileLease(
                file_path=normalized_path,
                owner_agent=owner_agent,
                acquired_at=time.time(),
                ttl_sec=ttl,
                reason=reason,
            )
            return True

    async def release(self, file_path: str, owner_agent: str) -> bool:
        """Release a previously acquired lock."""
        normalized_path = file_path.replace("\\", "/").lower()

        async with self._lock:
            existing = self._locks.get(normalized_path)
            if existing and existing.owner_agent == owner_agent:
                del self._locks[normalized_path]
                return True
            return False

    async def release_all(self, owner_agent: str) -> int:
        """Release all locks held by a specific agent (e.g. on exit or failure)."""
        async with self._lock:
            to_remove = [p for p, l in self._locks.items() if l.owner_agent == owner_agent]
            for p in to_remove:
                del self._locks[p]
            return len(to_remove)

    def is_locked(self, file_path: str) -> bool:
        """Check if a file is currently under an active non-expired lock."""
        normalized_path = file_path.replace("\\", "/").lower()
        existing = self._locks.get(normalized_path)
        if existing and not existing.is_expired:
            return True
        return False

    def get_holder(self, file_path: str) -> Optional[str]:
        """Returns the agent currently holding the lock, if any."""
        normalized_path = file_path.replace("\\", "/").lower()
        existing = self._locks.get(normalized_path)
        if existing and not existing.is_expired:
            return existing.owner_agent
        return None
