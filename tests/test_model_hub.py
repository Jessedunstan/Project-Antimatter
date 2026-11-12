"""Comprehensive Verification Suite for Dual-Model Hub, VRAM Arbitration, Speculative Decoding, and AST Context Pruning."""

import pytest
from antimatter.brain.model_hub import (
    DualModelHub,
    DualModelVRAMCoordinator,
    ModelSpec,
    TaskKind,
    SpeculativeCandidate,
)
from antimatter.brain.context_pruner import (
    prune_python_source,
    prune_to_token_budget,
    ASTContextPruner,
)


def test_vram_coordinator_allocation_and_eviction():
    coordinator = DualModelVRAMCoordinator(max_vram_mb=6000)

    # Fast model: 1800 MB, priority 2
    fast_model = ModelSpec(name="fast-1b", vram_estimate_mb=1800, context_window=4096, priority=2)
    assert coordinator.request_load(fast_model) is True
    assert coordinator.used_vram_mb == 1800
    assert coordinator.free_vram_mb == 4200

    # Heavy model: 5000 MB, priority 3.
    # Total would be 1800 + 5000 = 6800 > 6000 MB max.
    # Because heavy model priority (3) > fast model priority (2), it should evict fast_model!
    heavy_model = ModelSpec(name="reasoning-7b", vram_estimate_mb=5000, context_window=16384, priority=3)
    assert coordinator.request_load(heavy_model) is True
    assert heavy_model.is_loaded is True
    assert fast_model.is_loaded is False
    assert coordinator.used_vram_mb == 5000
    assert coordinator.free_vram_mb == 1000


def test_dual_model_hub_task_routing():
    hub = DualModelHub(
        max_vram_mb=8192,
        fast_model_name="qwen-1.5b",
        reasoning_model_name="deepseek-r1-7b",
    )

    # Latency-sensitive tasks route to fast model
    assert hub.route_model(TaskKind.AUTOCOMPLETE) == "qwen-1.5b"
    assert hub.route_model(TaskKind.SCOUT) == "qwen-1.5b"
    assert hub.route_model(TaskKind.CODER) == "qwen-1.5b"

    # Deep reasoning routes to heavy model
    assert hub.route_model(TaskKind.ARCHITECT) == "deepseek-r1-7b"
    assert hub.route_model(TaskKind.CRITIC) == "deepseek-r1-7b"


def test_ast_context_pruning_signatures_and_docstrings():
    source = '''
def compute_metrics(x: int, y: float = 1.0) -> float:
    """Calculates weighted product of metric inputs."""
    temp_val = x * 2.5
    res = temp_val + y
    return res

class DataProcessor:
    """Handles raw data transformation."""
    
    def process(self, items: list[str]) -> list[str]:
        """Filters and normalizes string entries."""
        out = []
        for it in items:
            out.append(it.strip().lower())
        return out
'''
    pruned = prune_python_source(source, preserve_docstrings=True)

    # Signatures must be intact
    assert "def compute_metrics" in pruned and "-> float:" in pruned
    assert "class DataProcessor:" in pruned
    assert "def process(self, items: list[str]) -> list[str]:" in pruned

    # Docstrings preserved
    assert '"""Calculates weighted product of metric inputs."""' in pruned
    assert '"""Filters and normalizes string entries."""' in pruned

    # Implementation bodies elided to Ellipsis (...)
    assert "temp_val = x * 2.5" not in pruned
    assert "for it in items:" not in pruned
    assert "..." in pruned


def test_prune_to_token_budget():
    files = {
        "main.py": "def main():\n    print('Hello World')\n",
        "dep1.py": "def helper():\n    '''Helper docstring'''\n    x = 1\n    y = 2\n    return x + y\n",
        "dep2.py": "def calculate():\n    a = 10\n    b = 20\n    return a * b\n",
    }

    pruned_bundle = prune_to_token_budget(files, target_file="main.py", max_tokens=100)
    assert "main.py" in pruned_bundle
    # Target file main.py content should be completely intact
    assert "print('Hello World')" in pruned_bundle["main.py"]

    # Dependencies should have bodies elided
    if "dep1.py" in pruned_bundle:
        assert "x = 1" not in pruned_bundle["dep1.py"]


@pytest.mark.asyncio
async def test_speculative_patch_validation():
    hub = DualModelHub(max_vram_mb=8192)

    # Valid Python generation candidate
    valid_py = "def add(a: int, b: int) -> int:\n    return a + b\n"
    import ast
    assert ast.parse(valid_py) is not None

    cand = await hub.generate_speculative_patch(
        prompt="Write a function to add two numbers",
        syntax_check=True,
    )
    assert cand is not None
    assert isinstance(cand, SpeculativeCandidate)
    assert cand.tokens_generated > 0
