"""Piper ONNX Text-to-Speech (TTS) Engine for Project Antimatter.

Generates ultra-low latency, sovereign offline speech synthesis using Piper ONNX models.
"""

from dataclasses import dataclass
import io
import logging
from pathlib import Path
import wave
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class AudioSynthesisResult:
    """Result of speech synthesis."""
    audio_bytes: bytes
    sample_rate: int
    duration_sec: float
    file_path: Optional[Path] = None


class PiperTTSEngine:
    """Offline speech synthesizer powered by Piper ONNX."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        config_path: Optional[Path] = None,
        sample_rate: int = 22050,
    ):
        self.model_path = model_path
        self.config_path = config_path
        self.sample_rate = sample_rate
        self._voice = None

    def _ensure_loaded(self) -> None:
        """Load Piper voice if model path exists, else prepare mock fallback."""
        if self._voice is None and self.model_path and Path(self.model_path).exists():
            try:
                from piper.voice import PiperVoice
                logger.info("Loading Piper ONNX model from: %s", self.model_path)
                self._voice = PiperVoice.load(
                    model_path=str(self.model_path),
                    config_path=str(self.config_path) if self.config_path else None
                )
                self.sample_rate = self._voice.config.sample_rate
            except Exception as e:
                logger.warning("Failed loading Piper voice (%s). Fallback active.", e)
                self._voice = None

    def synthesize(self, text: str, output_path: Optional[Path] = None) -> AudioSynthesisResult:
        """Synthesize text to PCM WAV audio."""
        self._ensure_loaded()
        clean_text = text.strip()

        if self._voice is not None:
            # Native Piper ONNX synthesis
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sample_rate)
                self._voice.synthesize(clean_text, wav_file)
            audio_data = buffer.getvalue()
        else:
            # Deterministic offline fallback (produces valid silent/beep WAV container)
            buffer = io.BytesIO()
            duration_frames = int(self.sample_rate * min(len(clean_text) * 0.05, 1.0))
            with wave.open(buffer, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sample_rate)
                # 16-bit silence
                wav_file.writeframes(b"\x00\x00" * max(duration_frames, 100))
            audio_data = buffer.getvalue()

        duration = len(audio_data) / (self.sample_rate * 2)

        if output_path:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(audio_data)

        return AudioSynthesisResult(
            audio_bytes=audio_data,
            sample_rate=self.sample_rate,
            duration_sec=duration,
            file_path=output_path,
        )
