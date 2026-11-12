"""Configuration management for Project Antimatter.

Detects local hardware capabilities, loads environmental settings,
and enforces sovereign offline execution boundaries.
"""

from enum import Enum
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class HardwareProfile(str, Enum):
    """Target hardware profiles for model sizing."""
    EDGE = "edge"               # 8GB - 12GB VRAM or 16GB RAM
    PRO = "pro"                 # 16GB - 24GB VRAM or 32GB RAM
    WORKSTATION = "workstation" # 48GB+ VRAM / Unified Mac
    CPU_ONLY = "cpu_only"       # No GPU acceleration


class LLMBackendType(str, Enum):
    """Supported local LLM serving backends."""
    OLLAMA = "ollama"
    LLAMA_CPP = "llama_cpp"
    VLLM = "vllm"
    MOCK = "mock"               # Offline deterministic test stub


class AntimatterSettings(BaseSettings):
    """Global configuration settings for Antimatter runtime."""

    model_config = SettingsConfigDict(
        env_prefix="ANTIMATTER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Runtime Identity
    app_name: str = "Antimatter Sovereign Engine"
    version: str = "0.1.0"
    log_level: str = "INFO"

    # Hardware Profile
    profile: HardwareProfile = HardwareProfile.PRO

    # LLM Inference Backends
    llm_backend: LLMBackendType = LLMBackendType.OLLAMA
    ollama_base_url: str = "http://127.0.0.1:11434"
    vllm_base_url: str = "http://127.0.0.1:8000/v1"
    llama_cpp_model_path: Optional[Path] = None

    # Model Assignments
    planner_model: str = "qwen2.5-coder:7b"
    autocomplete_model: str = "llama3.2:1b"
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # Execution & Sandbox
    workspace_dir: Path = Field(default_factory=lambda: Path("./workspace").resolve())
    max_exec_timeout_sec: int = 120
    stream_chunk_timeout_sec: float = 30.0

    # Audio Engine
    audio_enabled: bool = False
    whisper_model: str = "base.en"
    piper_voice: str = "en_US-lessac-medium"

    # Code Intelligence & Context
    code_index_path: Optional[Path] = None
    embedding_dimension: int = 384

    def ensure_directories(self) -> None:
        """Create required local directories if not present."""
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        if self.code_index_path:
            self.code_index_path.parent.mkdir(parents=True, exist_ok=True)


_settings_instance: Optional[AntimatterSettings] = None


def get_settings() -> AntimatterSettings:
    """Retrieve or initialize singleton configuration settings."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = AntimatterSettings()
        _settings_instance.ensure_directories()
    return _settings_instance
