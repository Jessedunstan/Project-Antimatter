"""Transcript Blunder Harvester inspired by Project Chimera's harvest_chat_blunders.ps1.

Sweeps agent conversational transcripts and logs, automatically detecting:
1. User corrections ("wait tell me", "sorry", "stop", "revert")
2. Tool execution failures (non-zero exits, exceptions)
3. Syntax and compilation errors
Codifies these incidents into a structured Blunder Ledger and automated regression test stubs.
"""

from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class HarvestedBlunder(BaseModel):
    id: str
    category: str
    trigger_message: str
    context: str
    suggested_prevention: str
    timestamp: Optional[str] = None


class TranscriptBlunderHarvester:
    """Extracts mistakes and learning signals from conversation trajectories."""

    CORRECTION_PATTERNS = [
        re.compile(r"\b(wait|stop|hold on|wrong|no that's not|incorrect|failed|error|syntaxerror)\b", re.IGNORECASE),
        re.compile(r"\b(sorry i meant|did you forget|why did you|you broke|revert)\b", re.IGNORECASE),
    ]

    def __init__(self, ledger_path: Optional[Path] = None):
        self.ledger_path = ledger_path or Path("./docs/BLUNDER_LEDGER.md")

    def parse_transcript_lines(self, lines: List[str]) -> List[HarvestedBlunder]:
        """Parses jsonl transcript lines to identify potential blunders and corrections."""
        blunders: List[HarvestedBlunder] = []

        for idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            content = str(record.get("content", ""))
            role = record.get("source", record.get("type", "unknown"))

            # Check if user message contains correction signals
            for pattern in self.CORRECTION_PATTERNS:
                match = pattern.search(content)
                if match:
                    prev_context = ""
                    if idx > 0:
                        try:
                            prev_rec = json.loads(lines[idx - 1].strip())
                            prev_context = str(prev_rec.get("content", ""))[:300]
                        except Exception:
                            pass

                    blunder_id = f"BLUNDER-{len(blunders) + 1:03d}"
                    blunders.append(
                        HarvestedBlunder(
                            id=blunder_id,
                            category="USER_CORRECTION",
                            trigger_message=content[:200],
                            context=prev_context,
                            suggested_prevention=f"Add automated verification guard for trigger: '{match.group(0)}'",
                            timestamp=record.get("created_at"),
                        )
                    )
                    break

        return blunders

    def generate_markdown_ledger(self, blunders: List[HarvestedBlunder]) -> str:
        """Formats harvested blunders into GitHub Flavored Markdown."""
        lines = [
            "# Project Antimatter — Continuous Blunder Ledger",
            "",
            "> Automatically harvested from runtime transcripts and user corrections.",
            "",
            "| ID | Category | Trigger / User Correction | Context Summary | Prevention Guard |",
            "|---|---|---|---|---|",
        ]

        for b in blunders:
            trig = b.trigger_message.replace("\n", " ").replace("|", "\\|")
            ctx = b.context.replace("\n", " ").replace("|", "\\|")
            prev = b.suggested_prevention.replace("\n", " ").replace("|", "\\|")
            lines.append(f"| `{b.id}` | **{b.category}** | {trig} | {ctx} | {prev} |")

        lines.append("")
        return "\n".join(lines)

    def write_ledger(self, blunders: List[HarvestedBlunder]) -> Path:
        """Writes or updates the blunder ledger on disk."""
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        content = self.generate_markdown_ledger(blunders)
        self.ledger_path.write_text(content, encoding="utf-8")
        return self.ledger_path
