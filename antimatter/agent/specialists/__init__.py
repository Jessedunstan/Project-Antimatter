"""Domain-specific local AI specialists for Project Antimatter."""

from antimatter.agent.specialists.vision import VisionFrameInspector
from antimatter.agent.specialists.steering import (
    SteeringSpecialist,
    GameObservation,
    SteeringAction,
    ActionType,
)
from antimatter.agent.specialists.media import (
    ComfyUIMediaEngine,
    MediaGenerationRequest,
    MediaArtifact,
)
from antimatter.agent.specialists.critic import (
    CodeCriticSpecialist,
    CriticVerdict,
    CodeCritiqueReport,
    CodeIssue,
)
from antimatter.agent.specialists.fuzzer import (
    GameQAFuzzer,
    FuzzingSessionReport,
)

__all__ = [
    "VisionFrameInspector",
    "SteeringSpecialist",
    "GameObservation",
    "SteeringAction",
    "ActionType",
    "ComfyUIMediaEngine",
    "MediaGenerationRequest",
    "MediaArtifact",
    "CodeCriticSpecialist",
    "CriticVerdict",
    "CodeCritiqueReport",
    "CodeIssue",
    "GameQAFuzzer",
    "FuzzingSessionReport",
]
