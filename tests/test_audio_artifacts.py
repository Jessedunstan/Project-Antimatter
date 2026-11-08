"""Comprehensive Test Suite for Track 5: Sovereign Audio & Artifact Canvas Engine."""

import pytest
from pathlib import Path
from starlette.testclient import TestClient

from antimatter.brain.audio.tts import PiperTTSEngine
from antimatter.brain.audio.stt import FasterWhisperSTTEngine
from antimatter.artifacts.types import ArtifactType, ArtifactMetadata
from antimatter.artifacts.manager import ArtifactManager
from antimatter.artifacts.renderer import ArtifactRenderer
from antimatter.artifacts.server import create_artifact_app


# ============================================================================
# 1. AUDIO ENGINE TESTS (TTS & STT)
# ============================================================================

def test_tts_synthesis(tmp_path):
    tts = PiperTTSEngine()
    out_wav = tmp_path / "speech.wav"
    res = tts.synthesize("Project Antimatter audio output online.", output_path=out_wav)

    assert len(res.audio_bytes) > 44  # Valid WAV file is larger than 44-byte header
    assert res.sample_rate == 22050
    assert res.duration_sec > 0
    assert out_wav.exists()
    assert out_wav.stat().st_size == len(res.audio_bytes)


def test_stt_transcription(tmp_path):
    stt = FasterWhisperSTTEngine(model_size_or_path="mock")
    # Generate dummy WAV audio to feed to STT
    tts = PiperTTSEngine()
    dummy_audio = tmp_path / "input.wav"
    tts.synthesize("Unit test speech sample", output_path=dummy_audio)

    res = stt.transcribe(dummy_audio)
    assert res.text is not None
    assert len(res.text) > 0
    assert res.duration_sec > 0


# ============================================================================
# 2. ARTIFACT MANAGER & RENDERER TESTS
# ============================================================================

def test_artifact_manager_crud(tmp_path):
    manager = ArtifactManager(storage_dir=tmp_path / "artifacts")

    # 1. Create Markdown Artifact
    art1 = manager.create_artifact(
        title="System Architecture",
        content="# Architecture\n- Core\n- Engine",
        artifact_type=ArtifactType.MARKDOWN,
        summary="Master system design",
    )
    assert art1.metadata.id.startswith("art_")
    assert art1.metadata.version == 1

    # 2. Retrieve
    retrieved = manager.get_artifact(art1.metadata.id)
    assert retrieved is not None
    assert retrieved.content == art1.content

    # 3. Update Artifact
    updated = manager.update_artifact(
        art1.metadata.id,
        content="# Architecture v2\n- Enhanced Engine",
        summary="Updated design"
    )
    assert updated.metadata.version == 2
    assert "v2" in updated.content

    # 4. List Artifacts
    arts = manager.list_artifacts()
    assert len(arts) == 1
    assert arts[0].title == "System Architecture"

    # 5. Delete Artifact
    deleted = manager.delete_artifact(art1.metadata.id)
    assert deleted is True
    assert manager.get_artifact(art1.metadata.id) is None


def test_artifact_renderer(tmp_path):
    manager = ArtifactManager(storage_dir=tmp_path / "artifacts")

    # Mermaid
    mermaid_art = manager.create_artifact(
        title="Agent Workflow",
        content="graph TD; Think-->Plan; Plan-->Execute;",
        artifact_type=ArtifactType.MERMAID,
    )
    html_mermaid = ArtifactRenderer.render_to_html(mermaid_art)
    assert "mermaid.initialize" in html_mermaid
    assert "graph TD;" in html_mermaid

    # Diff
    diff_art = manager.create_artifact(
        title="Code Diff",
        content="--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-old\n+new",
        artifact_type=ArtifactType.DIFF,
    )
    html_diff = ArtifactRenderer.render_to_html(diff_art)
    assert "diff-line-del" in html_diff
    assert "diff-line-add" in html_diff


# ============================================================================
# 3. ARTIFACT SERVER (FASTAPI) TESTS
# ============================================================================

def test_artifact_fastapi_server(tmp_path):
    manager = ArtifactManager(storage_dir=tmp_path / "artifacts")
    app = create_artifact_app(manager)
    client = TestClient(app)

    # 1. Index dashboard
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Antimatter Artifacts Canvas" in resp.text

    # 2. Create artifact via REST
    payload = {
        "title": "Interactive Widget",
        "content": "<button>Click Me</button>",
        "artifact_type": "html",
        "summary": "Sample button",
    }
    create_resp = client.post("/artifacts", json=payload)
    assert create_resp.status_code == 200
    created_meta = create_resp.json()
    art_id = created_meta["id"]

    # 3. View HTML rendered artifact
    view_resp = client.get(f"/artifacts/{art_id}")
    assert view_resp.status_code == 200
    assert "<button>Click Me</button>" in view_resp.text
    assert "Interactive Widget" in view_resp.text

    # 4. View raw content
    raw_resp = client.get(f"/artifacts/{art_id}/raw")
    assert raw_resp.status_code == 200
    assert raw_resp.text == "<button>Click Me</button>"

    # 5. Delete artifact via REST
    del_resp = client.delete(f"/artifacts/{art_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "deleted"
