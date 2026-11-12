"""Microsoft TRELLIS Structured Latents (SLAT) Package for Project Antimatter."""

from staging.lab_3d_trellis_latents.slat_engine import (
    SLATEngine,
    SLATRepresentation,
    GaussianSplat,
)
from staging.lab_3d_trellis_latents.mesh_exporter import MeshExporter
from staging.lab_3d_trellis_latents.pipeline import TrellisPipeline, TrellisGenerationResult

__all__ = [
    "SLATEngine",
    "SLATRepresentation",
    "GaussianSplat",
    "MeshExporter",
    "TrellisPipeline",
    "TrellisGenerationResult",
]
