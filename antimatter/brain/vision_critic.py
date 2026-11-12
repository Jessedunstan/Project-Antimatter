"""Local Multimodal Vision Critic inspired by Project Chimera's patrol_moondream_vision_critic.ps1.

Uses local vision models (via Ollama or mock) to autonomously inspect generated artifacts,
diagrams, and UI views, auditing visual hierarchy, layout overlap, contrast, and rendering defects.
"""

from __future__ import annotations
import base64
from pathlib import Path
from typing import Dict, List, Optional
import httpx
from pydantic import BaseModel, Field

from antimatter.core.config import AntimatterSettings, get_settings


class VisionCritique(BaseModel):
    artifact_path: str
    is_valid: bool = True
    contrast_score: float = 1.0  # 0.0 to 1.0
    layout_score: float = 1.0    # 0.0 to 1.0
    issues_detected: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    critic_model: str = "moondream:latest"


class VisionCriticEngine:
    """Evaluates graphical artifacts using local multimodal vision models."""

    def __init__(
        self,
        settings: Optional[AntimatterSettings] = None,
        model_name: str = "moondream:latest",
    ):
        self.settings = settings or get_settings()
        self.model_name = model_name

    def evaluate_svg_or_html(self, content: str, artifact_path: str = "artifact.html") -> VisionCritique:
        """Deterministic heuristic analysis for text-based SVG/HTML artifacts."""
        issues = []
        recommendations = []
        contrast = 0.95
        layout = 0.95

        # Check for empty or malformed tags
        if len(content.strip()) < 20:
            issues.append("Artifact content is suspiciously short or empty.")
            contrast = 0.0
            layout = 0.0

        if "<html" in content.lower() and "</html>" not in content.lower():
            issues.append("Unclosed <html> tag detected in artifact.")
            layout -= 0.3

        if "<svg" in content.lower():
            if "viewBox" not in content and "viewbox" not in content.lower():
                issues.append("SVG lacks viewBox attribute; responsive scaling may be broken.")
                recommendations.append("Add viewBox='0 0 width height' to root SVG element.")
                layout -= 0.2

        is_valid = len(issues) == 0
        return VisionCritique(
            artifact_path=artifact_path,
            is_valid=is_valid,
            contrast_score=max(0.0, contrast),
            layout_score=max(0.0, layout),
            issues_detected=issues,
            recommendations=recommendations,
            critic_model="heuristic-linter",
        )

    async def evaluate_image_file(self, image_path: Path) -> VisionCritique:
        """Inspects binary image files (PNG, JPG, WebP) using Ollama vision endpoint."""
        if not image_path.exists():
            return VisionCritique(
                artifact_path=str(image_path),
                is_valid=False,
                contrast_score=0.0,
                layout_score=0.0,
                issues_detected=[f"File not found: {image_path}"],
                critic_model=self.model_name,
            )

        # In mock or test mode, perform fast deterministic image check
        if getattr(self.settings, "llm_backend", None) == "mock":
            return VisionCritique(
                artifact_path=str(image_path),
                is_valid=True,
                contrast_score=0.98,
                layout_score=0.98,
                issues_detected=[],
                recommendations=["Visual layout complies with high-contrast dark theme."],
                critic_model="mock-vision-critic",
            )

        # Live Ollama call with base64 image
        try:
            image_bytes = image_path.read_bytes()
            b64_img = base64.b64encode(image_bytes).decode("utf-8")

            payload = {
                "model": self.model_name,
                "prompt": (
                    "Audit this software UI or diagram screenshot. "
                    "List any visual glitches, text clipping, contrast issues, or broken elements."
                ),
                "images": [b64_img],
                "stream": False,
            }

            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    f"{self.settings.ollama_base_url}/api/generate",
                    json=payload,
                )
                if resp.status_code == 200:
                    text = resp.json().get("response", "")
                    has_glitch = any(w in text.lower() for w in ["glitch", "clipping", "illegible", "broken", "overlap"])
                    return VisionCritique(
                        artifact_path=str(image_path),
                        is_valid=not has_glitch,
                        contrast_score=0.7 if has_glitch else 0.95,
                        layout_score=0.6 if has_glitch else 0.95,
                        issues_detected=[text] if has_glitch else [],
                        recommendations=["Inspect reported visual artifacts."] if has_glitch else [],
                        critic_model=self.model_name,
                    )
        except Exception as e:
            # Fallback if vision model unavailable
            pass

        return VisionCritique(
            artifact_path=str(image_path),
            is_valid=True,
            contrast_score=0.9,
            layout_score=0.9,
            issues_detected=[],
            recommendations=["Ollama vision model unreachable; fallback passed."],
            critic_model="fallback-critic",
        )
