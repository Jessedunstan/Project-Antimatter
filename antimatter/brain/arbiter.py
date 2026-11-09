"""VRAM Arbiter & Local Model Lifecycle Manager for Project Antimatter.

Ensures seamless co-existence and deterministic swapping of multi-modal models
(LLMs, VLMs, Diffusion Image/Video) within constrained consumer GPUs (e.g., 8GB RTX 3060 Ti).
"""

from __future__ import annotations
import asyncio
import logging
import time
from enum import Enum
from typing import Any, Dict, List, Optional
import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ModelTier(str, Enum):
    """Execution priority and latency profile for models."""
    STEERING = "steering"      # Real-time game/robot control, sub-50ms latency (SmolLM2, Qwen2.5-3B)
    REASONING = "reasoning"    # Complex code/logic synthesis, 1-3s latency (Qwen2.5-Coder, DeepSeek-R1-Distill)
    VISION = "vision"          # Visual inspection, frame QA (Qwen2.5-VL, Moondream2)
    MEDIA = "media"            # Diffusion image / video generation (SDXL Lightning, Wan2.1, LTX)


class ModelLease(BaseModel):
    """Active VRAM lease allocation tracking."""
    lease_id: str
    model_name: str
    tier: ModelTier
    vram_mb: int
    acquired_at: float = Field(default_factory=time.time)


class VRAMArbiter:
    """Arbiter managing GPU memory allocations and model swapping across local backends."""

    def __init__(
        self,
        total_vram_mb: int = 8192,
        vram_headroom_mb: int = 1024,
        ollama_base_url: str = "http://127.0.0.1:11434",
    ):
        self.total_vram_mb = total_vram_mb
        self.vram_headroom_mb = vram_headroom_mb
        self.usable_vram_mb = max(1024, total_vram_mb - vram_headroom_mb)
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self._active_leases: Dict[str, ModelLease] = {}
        self._lock = asyncio.Lock()

    @property
    def allocated_vram_mb(self) -> int:
        """Total VRAM actively held by leases."""
        return sum(l.vram_mb for l in self._active_leases.values())

    @property
    def free_vram_mb(self) -> int:
        """Available VRAM before hitting maximum usable capacity."""
        return max(0, self.usable_vram_mb - self.allocated_vram_mb)

    async def evict_ollama_model(self, model_name: str) -> bool:
        """Explicitly evict an Ollama model from VRAM by setting keep_alive to 0."""
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.post(
                    f"{self.ollama_base_url}/api/generate",
                    json={"model": model_name, "keep_alive": 0},
                )
                if resp.status_code == 200:
                    logger.info("Evicted model '%s' from Ollama VRAM.", model_name)
                    return True
        except Exception as e:
            logger.debug("Could not reach Ollama to evict '%s': %s", model_name, e)
        return False

    async def acquire_lease(
        self,
        lease_id: str,
        model_name: str,
        tier: ModelTier,
        required_vram_mb: int,
    ) -> ModelLease:
        """Acquire a VRAM lease, evicting lower-priority or idle models if needed."""
        async with self._lock:
            # Check if lease already exists
            if lease_id in self._active_leases:
                return self._active_leases[lease_id]

            # If not enough VRAM, evict eligible leases based on tier priority
            while self.free_vram_mb < required_vram_mb and self._active_leases:
                # Evict lowest priority tier (MEDIA -> VISION -> REASONING -> STEERING)
                evictable = sorted(
                    self._active_leases.values(),
                    key=lambda l: (
                        0 if l.tier == ModelTier.MEDIA else
                        1 if l.tier == ModelTier.VISION else
                        2 if l.tier == ModelTier.REASONING else 3,
                        -l.acquired_at
                    )
                )
                target = evictable[0]
                logger.info(
                    "Evicting lease '%s' (Model: %s, %d MB) to satisfy request for %d MB",
                    target.lease_id, target.model_name, target.vram_mb, required_vram_mb
                )
                await self.evict_ollama_model(target.model_name)
                del self._active_leases[target.lease_id]

            lease = ModelLease(
                lease_id=lease_id,
                model_name=model_name,
                tier=tier,
                vram_mb=required_vram_mb,
            )
            self._active_leases[lease_id] = lease
            return lease

    async def release_lease(self, lease_id: str, unload_from_gpu: bool = False) -> None:
        """Release a VRAM lease and optionally purge the model from GPU memory."""
        async with self._lock:
            lease = self._active_leases.pop(lease_id, None)
            if lease and unload_from_gpu:
                await self.evict_ollama_model(lease.model_name)

    def get_status(self) -> Dict[str, Any]:
        """Telemetry on current VRAM utilization."""
        return {
            "total_vram_mb": self.total_vram_mb,
            "usable_vram_mb": self.usable_vram_mb,
            "allocated_vram_mb": self.allocated_vram_mb,
            "free_vram_mb": self.free_vram_mb,
            "active_leases": [l.model_dump() for l in self._active_leases.values()],
        }
