"""Sovereign Offline Multimodal Audio Engine for Project Antimatter."""

from antimatter.brain.audio.tts import PiperTTSEngine, AudioSynthesisResult
from antimatter.brain.audio.stt import (
    FasterWhisperSTTEngine,
    TranscriptionResult,
    TranscriptionSegment,
)

__all__ = [
    "PiperTTSEngine",
    "AudioSynthesisResult",
    "FasterWhisperSTTEngine",
    "TranscriptionResult",
    "TranscriptionSegment",
]
