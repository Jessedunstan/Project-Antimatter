"""Verification Suite for Chimera Systems Synthesis in Project Antimatter.

Validates:
1. LockArbiter concurrency and lease expiration
2. TranscriptBlunderHarvester pattern detection & ledger generation
3. VisionCriticEngine heuristic and mock image inspection
4. WikiManager AST extraction & living documentation generation
5. GapAnalysisSentinel audit and track verification
"""

import pytest
import time
from pathlib import Path
from antimatter.agent.swarm.lock_arbiter import LockArbiter
from antimatter.agent.blunder_harvester import TranscriptBlunderHarvester
from antimatter.brain.vision_critic import VisionCriticEngine
from antimatter.core.wiki_manager import WikiManager
from antimatter.core.gap_analysis import GapAnalysisSentinel


@pytest.mark.asyncio
async def test_lock_arbiter_concurrency():
    arbiter = LockArbiter(default_ttl_sec=1.0)
    file_path = "antimatter/core/config.py"

    # Agent 1 acquires lock
    ok1 = await arbiter.acquire(file_path, owner_agent="agent-coder-1", ttl_sec=1.0)
    assert ok1 is True
    assert arbiter.is_locked(file_path) is True
    assert arbiter.get_holder(file_path) == "agent-coder-1"

    # Agent 2 fails to acquire same lock
    ok2 = await arbiter.acquire(file_path, owner_agent="agent-coder-2")
    assert ok2 is False

    # Agent 1 refreshes lock
    ok1_refresh = await arbiter.acquire(file_path, owner_agent="agent-coder-1", ttl_sec=1.0)
    assert ok1_refresh is True

    # Agent 1 releases lock
    rel = await arbiter.release(file_path, owner_agent="agent-coder-1")
    assert rel is True
    assert arbiter.is_locked(file_path) is False

    # Now Agent 2 can acquire it
    ok2_after = await arbiter.acquire(file_path, owner_agent="agent-coder-2")
    assert ok2_after is True


@pytest.mark.asyncio
async def test_lock_arbiter_expiration():
    arbiter = LockArbiter(default_ttl_sec=0.1)
    file_path = "antimatter/tools/terminal.py"

    # Acquire short lease
    await arbiter.acquire(file_path, owner_agent="scout", ttl_sec=0.05)
    assert arbiter.is_locked(file_path) is True

    # Wait for expiration
    time.sleep(0.08)
    assert arbiter.is_locked(file_path) is False

    # Another agent can now acquire it cleanly
    acquired = await arbiter.acquire(file_path, owner_agent="architect", ttl_sec=1.0)
    assert acquired is True


def test_blunder_harvester_parsing(tmp_path):
    sample_transcript = [
        '{"type": "USER_INPUT", "content": "build the project", "source": "USER"}',
        '{"type": "PLANNER_RESPONSE", "content": "building now...", "source": "MODEL"}',
        '{"type": "USER_INPUT", "content": "wait that is the wrong command! stop!", "source": "USER"}',
        '{"type": "USER_INPUT", "content": "sorry i meant to run tests instead", "source": "USER"}',
    ]

    ledger_file = tmp_path / "BLUNDER_LEDGER.md"
    harvester = TranscriptBlunderHarvester(ledger_path=ledger_file)
    blunders = harvester.parse_transcript_lines(sample_transcript)

    assert len(blunders) >= 2
    assert any("wait" in b.trigger_message.lower() for b in blunders)
    assert any("sorry" in b.trigger_message.lower() for b in blunders)

    written_path = harvester.write_ledger(blunders)
    assert written_path.exists()
    content = written_path.read_text(encoding="utf-8")
    assert "# Project Antimatter — Continuous Blunder Ledger" in content
    assert "USER_CORRECTION" in content


def test_vision_critic_heuristics():
    engine = VisionCriticEngine()

    # Valid SVG
    good_svg = '<svg viewBox="0 0 100 100"><circle cx="50" cy="50" r="40" fill="purple" /></svg>'
    res1 = engine.evaluate_svg_or_html(good_svg)
    assert res1.is_valid is True
    assert res1.layout_score > 0.8

    # Broken SVG missing viewBox
    bad_svg = '<svg width="100" height="100"><circle cx="50" cy="50" r="40" fill="purple" /></svg>'
    res2 = engine.evaluate_svg_or_html(bad_svg)
    assert any("viewBox" in issue for issue in res2.issues_detected)


def test_wiki_manager_generation(tmp_path):
    # Create mock package
    mock_src = tmp_path / "src"
    mock_src.mkdir()
    mod_file = mock_src / "sample_module.py"
    mod_file.write_text(
        '"""Sample module docstring."""\n\n'
        'class EngineCore:\n'
        '    """Core engine class."""\n'
        '    def start(self):\n'
        '        pass\n\n'
        'def helper_function():\n'
        '    pass\n',
        encoding="utf-8",
    )

    wiki_out = tmp_path / "wiki"
    manager = WikiManager(root_dir=mock_src, output_dir=wiki_out)
    index_file = manager.build_wiki()

    assert index_file.exists()
    content = index_file.read_text(encoding="utf-8")
    assert "# Project Antimatter — Living Technical Wiki" in content
    assert "EngineCore" in content
    assert "helper_function" in content
    assert "Sample module docstring" in content


def test_gap_analysis_sentinel():
    sentinel = GapAnalysisSentinel()
    reports = sentinel.audit_tracks()

    assert len(reports) >= 6
    # Core foundational tracks must be verified
    brain_track = next((r for r in reports if "Track 1" in r.component), None)
    assert brain_track is not None
    assert brain_track.status == "VERIFIED"
