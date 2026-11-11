"""Vision Frame Inspector Specialist for Project Antimatter.

Ingests game frames, application screenshots, and rendering buffers to detect visual glitches,
clipping, texture corruption, and UI alignment defects using local Vision-Language Models
(Qwen2.5-VL / Moondream2).
"""

from __future__ import annotations
import base64
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import httpx
from pydantic import BaseModel, Field

from antimatter.agent.relay import RelayBaton, SpecialistDomain
from antimatter.core.config import get_settings

logger = logging.getLogger(__name__)


class VisualAnomaly(BaseModel):
    """Structured report of a detected visual defect or anomaly."""
    label: str
    severity: str = "medium"  # low, medium, high, critical
    bounding_box: Optional[List[int]] = None  # [ymin, xmin, ymax, xmax] normalized 0-1000
    description: str


class VisionInspectionReport(BaseModel):
    """Comprehensive visual quality assurance report."""
    image_source: str
    pass_qa: bool
    summary: str
    anomalies: List[VisualAnomaly] = Field(default_factory=list)
    hud_state: Dict[str, Any] = Field(default_factory=dict)
    raw_response: Optional[str] = None
    inspected: bool = True  # False when no VLM actually looked at the frame (offline fallback)


class VisionFrameInspector:
    """Specialist agent analyzing frames and screenshots with local VLMs."""

    def __init__(
        self,
        model_name: str = "qwen2.5-vl:7b",
        ollama_base_url: Optional[str] = None,
    ):
        settings = get_settings()
        self.model_name = model_name
        self.ollama_base_url = (ollama_base_url or settings.ollama_base_url).rstrip("/")

    @staticmethod
    def encode_image(image_path: Path) -> str:
        """Read and encode an image file to a base64 string."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    async def inspect_frame(
        self,
        image_path: Optional[Path] = None,
        image_base64: Optional[str] = None,
        query: str = "Inspect this game frame for visual glitches, rendering bugs, or UI clipping.",
    ) -> VisionInspectionReport:
        """Execute visual QA inspection on an image using the local VLM."""
        source_label = str(image_path) if image_path else "base64_stream"

        if image_path and image_path.exists():
            b64_img = self.encode_image(image_path)
        elif image_base64:
            b64_img = image_base64
        else:
            return VisionInspectionReport(
                image_source=source_label,
                pass_qa=False,
                summary="Image source not provided or file not found.",
                anomalies=[VisualAnomaly(label="missing_input", severity="critical", description="Image file missing")],
            )

        system_prompt = (
            "You are an expert game engine and UI visual QA tester. "
            "Analyze the image precisely. Report any visual glitches, clipping, missing textures, "
            "or HUD layout errors. Format your assessment with clear bullet points."
        )

        payload = {
            "model": self.model_name,
            "prompt": f"{system_prompt}\n\nTask: {query}",
            "images": [b64_img],
            "stream": False,
            "options": {"temperature": 0.1},
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(f"{self.ollama_base_url}/api/generate", json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    response_text = data.get("response", "")
                    return self._parse_vlm_output(source_label, response_text)
                else:
                    logger.warning("Ollama VLM returned HTTP %d: %s", resp.status_code, resp.text)
        except Exception as e:
            logger.debug("Ollama VLM unreachable or offline: %s. Using deterministic local inspection.", e)

        # Deterministic offline inspection fallback
        return self._fallback_inspection(source_label, query)

    def _parse_vlm_output(self, source_label: str, text: str) -> VisionInspectionReport:
        """Parse free-form VLM text into structured inspection data."""
        anomalies: List[VisualAnomaly] = []
        lower = text.lower()

        # Simple semantic extraction of defect keywords
        glitch_keywords = ["glitch", "clipping", "corrupt", "missing texture", "artifact", "overlap", "tear"]
        detected = any(k in lower for k in glitch_keywords)

        if "clip" in lower:
            anomalies.append(VisualAnomaly(
                label="mesh_clipping",
                severity="medium",
                description="Potential geometry or mesh interpenetration detected.",
            ))
        if "texture" in lower and ("missing" in lower or "placeholder" in lower or "magenta" in lower):
            anomalies.append(VisualAnomaly(
                label="texture_missing",
                severity="high",
                description="Unbound shader or missing texture placeholder detected.",
            ))

        return VisionInspectionReport(
            image_source=source_label,
            pass_qa=(not detected or len(anomalies) == 0),
            summary=text.strip(),
            anomalies=anomalies,
            raw_response=text,
        )

    def _fallback_inspection(self, source_label: str, query: str) -> VisionInspectionReport:
        """Structured fallback when the VLM is offline.

        A tester must never report PASS for a frame it did not actually see,
        so this returns pass_qa=False with inspected=False.
        """
        return VisionInspectionReport(
            image_source=source_label,
            pass_qa=False,
            inspected=False,
            summary=f"VLM '{self.model_name}' unavailable; frame NOT inspected for query: '{query}'.",
            anomalies=[],
            raw_response=None,
        )

    async def handle_relay_baton(self, baton: RelayBaton) -> Dict[str, Any]:
        """Relay handler entrypoint conforming to SpecialistRelayEngine."""
        img_path_str = baton.context_data.get("image_path")
        img_path = Path(img_path_str) if img_path_str else None
        b64_data = baton.context_data.get("image_base64")

        report = await self.inspect_frame(
            image_path=img_path,
            image_base64=b64_data,
            query=baton.prompt,
        )

        status = "UNINSPECTED" if not report.inspected else ("PASS" if report.pass_qa else "FAIL")
        advice = (
            f"QA Status: {status}\n"
            f"Summary: {report.summary}\n"
            f"Anomalies Found: {len(report.anomalies)}"
        )

        return {
            "advice": advice,
            "artifacts": {
                "report": report.model_dump(),
                "pass_qa": report.pass_qa,
                "inspected": report.inspected,
                "anomalies_count": len(report.anomalies),
            },
        }
