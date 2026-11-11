# Staging 3D Blender VLM Package
from .generator import AnatomicalParams, Mesh3D, ProceduralAnatomicalGenerator
from .renderer import SoftwareCameraRenderer
from .critic_loop import HillClimbingLoop, VLMCritic, CritiqueResult

__all__ = [
    "AnatomicalParams",
    "Mesh3D",
    "ProceduralAnatomicalGenerator",
    "SoftwareCameraRenderer",
    "HillClimbingLoop",
    "VLMCritic",
    "CritiqueResult",
]
