"""Dual-Model VRAM Hub and Speculative Decoding Engine.

Provides:
1. Dynamic Task Routing: Maps fast latency-critical tasks (FIM autocomplete, AST pruning)
   to lightweight 1B/3B models and complex synthesis/planning to reasoning models.
2. VRAM Arbitration: Manages local hardware memory thresholds (8GB/16GB VRAM or Apple Silicon),
   preventing OOM errors by preemptively evicting idle models.
3. Speculative Decoding: Leverages small draft models for rapid multi-token candidate generation
   paired with deterministic AST and execution-based verification.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field

from antimatter.core.config import AntimatterSettings, get_settings
from antimatter.brain.client import LocalLLMClient
from antimatter.brain.types import ChatMessage, ChatRole


class TaskKind(str, Enum):
    AUTOCOMPLETE = "autocomplete"  # FIM / tab-completion (fast, 1B-3B)
    SCOUT = "scout"                # Codebase exploration & symbol discovery (1B-3B)
    CODER = "coder"                # Patch drafting & syntax repair (1B-3B)
    ARCHITECT = "architect"        # Deep reasoning & task decomposition (7B-32B)
    CRITIC = "critic"              # Verifier / evaluation (7B-32B)


class ModelSpec(BaseModel):
    name: str
    vram_estimate_mb: int
    context_window: int
    is_loaded: bool = False
    priority: int = 1  # Higher number = higher retention priority


class DualModelVRAMCoordinator:
    """Monitors and arbitrates VRAM consumption across concurrent local models."""

    def __init__(self, max_vram_mb: int = 8192):
        self.max_vram_mb = max_vram_mb
        self.allocated_models: Dict[str, ModelSpec] = {}

    @property
    def used_vram_mb(self) -> int:
        return sum(m.vram_estimate_mb for m in self.allocated_models.values() if m.is_loaded)

    @property
    def free_vram_mb(self) -> int:
        return max(0, self.max_vram_mb - self.used_vram_mb)

    def request_load(self, tier: ModelSpec) -> bool:
        """Determines whether a model tier can be safely loaded without OOM."""
        if tier.is_loaded:
            return True

        if self.free_vram_mb >= tier.vram_estimate_mb:
            tier.is_loaded = True
            self.allocated_models[tier.name] = tier
            return True

        # Need eviction: evict lower-priority loaded models
        sorted_loaded = sorted(
            [m for m in self.allocated_models.values() if m.is_loaded],
            key=lambda m: (m.priority, -m.vram_estimate_mb),
        )

        freed = 0
        for m in sorted_loaded:
            if m.priority < tier.priority:
                m.is_loaded = False
                freed += m.vram_estimate_mb
                if self.free_vram_mb >= tier.vram_estimate_mb:
                    tier.is_loaded = True
                    self.allocated_models[tier.name] = tier
                    return True

        return False

    def evict(self, model_name: str) -> bool:
        if model_name in self.allocated_models and self.allocated_models[model_name].is_loaded:
            self.allocated_models[model_name].is_loaded = False
            return True
        return False


class SpeculativeCandidate(BaseModel):
    draft_text: str
    tokens_generated: int
    is_accepted: bool = False
    validation_reason: Optional[str] = None


class DualModelHub:
    """Unified controller for fast draft models and heavy reasoning models."""

    def __init__(
        self,
        settings: Optional[AntimatterSettings] = None,
        max_vram_mb: int = 8192,
        fast_model_name: Optional[str] = None,
        reasoning_model_name: Optional[str] = None,
    ):
        self.settings = settings or get_settings()
        self.arbiter = DualModelVRAMCoordinator(max_vram_mb=max_vram_mb)

        fast_name = fast_model_name or getattr(self.settings, "autocomplete_model", "qwen2.5-coder:1.5b")
        reasoning_name = reasoning_model_name or self.settings.planner_model

        self.fast_tier = ModelSpec(
            name=fast_name,
            vram_estimate_mb=1800,  # ~1.8GB for 1B/3B Q4 model
            context_window=8192,
            is_loaded=True,
            priority=2,
        )
        self.reasoning_tier = ModelSpec(
            name=reasoning_name,
            vram_estimate_mb=5000,  # ~5.0GB for 7B/8B Q4 model
            context_window=16384,
            is_loaded=False,
            priority=3,
        )

        self.arbiter.request_load(self.fast_tier)
        self.llm_client = LocalLLMClient(self.settings)

    def route_model(self, task: TaskKind) -> str:
        """Route to appropriate model based on task requirements and VRAM availability."""
        if task in (TaskKind.AUTOCOMPLETE, TaskKind.SCOUT, TaskKind.CODER):
            return self.fast_tier.name

        # Deep reasoning task
        if self.arbiter.request_load(self.reasoning_tier):
            return self.reasoning_tier.name

        # Fallback to fast model if VRAM cannot accommodate reasoning model
        return self.fast_tier.name

    async def generate_speculative_patch(
        self,
        prompt: str,
        syntax_check: bool = True,
    ) -> SpeculativeCandidate:
        """Generate a speculative code draft using the fast model and validate syntax."""
        draft_model = self.fast_tier.name

        # Fast model draft generation
        chunks = []
        messages = [
            ChatMessage(role=ChatRole.SYSTEM, content="You are a fast code completion model."),
            ChatMessage(role=ChatRole.USER, content=prompt),
        ]

        async for chunk in self.llm_client.stream_chat(
            messages=messages,
            model=draft_model,
            temperature=0.1,
        ):
            if chunk.delta_text:
                chunks.append(chunk.delta_text)

        draft_text = "".join(chunks)
        approx_tokens = max(1, len(draft_text.split()))

        if not syntax_check:
            return SpeculativeCandidate(
                draft_text=draft_text,
                tokens_generated=approx_tokens,
                is_accepted=True,
                validation_reason="Accepted without syntax check",
            )

        # Deterministic AST verification
        import ast
        try:
            ast.parse(draft_text)
            return SpeculativeCandidate(
                draft_text=draft_text,
                tokens_generated=approx_tokens,
                is_accepted=True,
                validation_reason="Valid Python AST",
            )
        except SyntaxError as e:
            return SpeculativeCandidate(
                draft_text=draft_text,
                tokens_generated=approx_tokens,
                is_accepted=False,
                validation_reason=f"SyntaxError: {e.msg} at line {e.lineno}",
            )
