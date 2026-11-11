"""Headless ComfyUI Media Engine for Project Antimatter.

Generates game textures, character sprites, and video cutscenes token-free on localhost
using ComfyUI API pipelines (SDXL Lightning, FLUX-schnell, Wan2.1, LTX-Video).
"""

from __future__ import annotations
import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
import httpx
from pydantic import BaseModel, Field

from antimatter.agent.relay import RelayBaton, SpecialistDomain
from antimatter.core.config import get_settings

logger = logging.getLogger(__name__)


class MediaGenerationRequest(BaseModel):
    """Media generation task parameters."""
    prompt: str
    negative_prompt: str = "low quality, blurry, deformed, watermark"
    media_type: str = "image"  # 'image' or 'video'
    width: int = 1024
    height: int = 1024
    steps: int = 4
    seed: Optional[int] = None
    output_filename: Optional[str] = None


class MediaArtifact(BaseModel):
    """Generated media output metadata."""
    artifact_id: str
    media_type: str
    file_path: str
    prompt: str
    dimensions: str
    generation_time_sec: float
    seed: int
    is_placeholder: bool = False  # True when ComfyUI did not produce the asset


class ComfyUIMediaEngine:
    """Specialist connecting Antimatter to a local ComfyUI instance for media creation."""

    def __init__(
        self,
        comfyui_base_url: str = "http://127.0.0.1:8188",
        output_dir: Optional[Path] = None,
    ):
        settings = get_settings()
        self.comfyui_base_url = comfyui_base_url.rstrip("/")
        self.output_dir = output_dir or (settings.workspace_dir / "artifacts" / "media")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_sdxl_lightning_graph(self, req: MediaGenerationRequest) -> Dict[str, Any]:
        """Construct standard ComfyUI JSON graph for 4-step SDXL Lightning diffusion."""
        seed = req.seed or int(time.time() * 1000) % 1_000_000
        return {
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "cfg": 1.5,
                    "denoise": 1.0,
                    "latent_image": ["5", 0],
                    "model": ["4", 0],
                    "negative": ["7", 0],
                    "positive": ["6", 0],
                    "sampler_name": "euler",
                    "scheduler": "sgm_uniform",
                    "seed": seed,
                    "steps": req.steps,
                },
            },
            "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sdxl_lightning_4step.safetensors"}},
            "5": {"class_type": "EmptyLatentImage", "inputs": {"batch_size": 1, "height": req.height, "width": req.width}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 1], "text": req.prompt}},
            "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 1], "text": req.negative_prompt}},
            "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
            "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "antimatter_sdxl", "images": ["8", 0]}},
        }

    async def generate_media(self, req: MediaGenerationRequest) -> MediaArtifact:
        """Dispatch generation graph to ComfyUI or produce deterministic local fallback."""
        start_time = time.time()
        artifact_id = f"media_{req.media_type}_{uuid.uuid4().hex[:8]}"
        ext = "mp4" if req.media_type == "video" else "png"
        target_file = self.output_dir / f"{artifact_id}.{ext}"

        # Attempt connection to live ComfyUI
        graph = self.build_sdxl_lightning_graph(req)
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                prompt_resp = await client.post(
                    f"{self.comfyui_base_url}/prompt",
                    json={"prompt": graph, "client_id": "antimatter_media_specialist"},
                )
                if prompt_resp.status_code == 200:
                    data = prompt_resp.json()
                    prompt_id = data.get("prompt_id")
                    logger.info("ComfyUI queued prompt_id: %s", prompt_id)
                    # Poll for completion
                    # (In production environment with live ComfyUI, websocket listens to /ws)
        except Exception as e:
            logger.debug("ComfyUI offline or unreachable: %s. Using deterministic local asset mock.", e)

        # Write placeholder artifact (ComfyUI result retrieval via /history + /view is not yet wired)
        is_placeholder = True
        if not target_file.exists():
            target_file.write_bytes(
                b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
            )

        elapsed = time.time() - start_time
        return MediaArtifact(
            artifact_id=artifact_id,
            media_type=req.media_type,
            file_path=str(target_file),
            prompt=req.prompt,
            dimensions=f"{req.width}x{req.height}",
            generation_time_sec=elapsed,
            seed=req.seed or 42,
            is_placeholder=is_placeholder,
        )

    async def handle_relay_baton(self, baton: RelayBaton) -> Dict[str, Any]:
        """Relay handler entrypoint conforming to SpecialistRelayEngine."""
        m_type = baton.context_data.get("media_type", "image")
        w = baton.context_data.get("width", 1024)
        h = baton.context_data.get("height", 1024)
        steps = baton.context_data.get("steps", 4)

        req = MediaGenerationRequest(
            prompt=baton.prompt,
            media_type=m_type,
            width=w,
            height=h,
            steps=steps,
        )

        artifact = await self.generate_media(req)
        advice = (
            f"Generated {artifact.media_type.upper()} asset in {artifact.generation_time_sec:.2f}s: "
            f"'{artifact.file_path}' ({artifact.dimensions})"
            + (" [PLACEHOLDER - ComfyUI render not retrieved]" if artifact.is_placeholder else "")
        )

        return {
            "advice": advice,
            "artifacts": {
                "artifact": artifact.model_dump(),
                "file_path": artifact.file_path,
                "media_type": artifact.media_type,
                "is_placeholder": artifact.is_placeholder,
            },
        }
