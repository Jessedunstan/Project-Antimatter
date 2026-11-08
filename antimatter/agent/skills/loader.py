"""Dynamic SKILL.md Parser and Customization Loader for Project Antimatter.

Discovers, parses YAML frontmatter, and dynamically registers skills
into agent prompts following the Antigravity skill architecture.
"""

from dataclasses import dataclass, field
import logging
from pathlib import Path
import re
from typing import Dict, List, Optional
import yaml

logger = logging.getLogger(__name__)

FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


@dataclass
class Skill:
    """Represents an agent capability loaded from a SKILL.md specification."""
    name: str
    description: str
    instructions: str
    path: Path
    version: str = "0.1.0"
    tags: List[str] = field(default_factory=list)
    scripts: Dict[str, Path] = field(default_factory=dict)

    def format_for_prompt(self) -> str:
        """Render skill instruction block for injection into agent context."""
        return (
            f"### Skill: {self.name}\n"
            f"**Description**: {self.description}\n\n"
            f"{self.instructions.strip()}\n"
        )


class SkillLoader:
    """Discovers and parses SKILL.md files from workspace and configuration roots."""

    def __init__(self, search_paths: Optional[List[Path]] = None):
        self.search_paths = search_paths or []
        self._skills: Dict[str, Skill] = {}

    def add_search_path(self, path: Path) -> None:
        """Add a directory to scan for skills."""
        if path.exists() and path not in self.search_paths:
            self.search_paths.append(path)

    def parse_skill_file(self, skill_file: Path) -> Optional[Skill]:
        """Parse a single SKILL.md file with YAML frontmatter."""
        try:
            content = skill_file.read_text(encoding="utf-8", errors="replace")
            match = FRONTMATTER_PATTERN.match(content)

            if match:
                raw_yaml = match.group(1)
                instructions = match.group(2)
                meta = yaml.safe_load(raw_yaml) or {}
            else:
                meta = {}
                instructions = content

            name = meta.get("name") or skill_file.parent.name
            description = meta.get("description") or f"Custom skill from {skill_file.name}"
            version = str(meta.get("version", "0.1.0"))
            tags = meta.get("tags", [])

            # Discover associated scripts in scripts/ subfolder if present
            scripts_dir = skill_file.parent / "scripts"
            scripts = {}
            if scripts_dir.exists() and scripts_dir.is_dir():
                for sf in scripts_dir.glob("*"):
                    if sf.is_file():
                        scripts[sf.name] = sf

            return Skill(
                name=name,
                description=description,
                instructions=instructions,
                path=skill_file,
                version=version,
                tags=tags if isinstance(tags, list) else [str(tags)],
                scripts=scripts,
            )

        except Exception as e:
            logger.error("Failed parsing skill file '%s': %s", skill_file, e)
            return None

    def scan_and_load(self) -> Dict[str, Skill]:
        """Scan all registered paths for SKILL.md files."""
        self._skills.clear()

        for search_path in self.search_paths:
            if not search_path.exists():
                continue

            for skill_file in search_path.rglob("SKILL.md"):
                skill = self.parse_skill_file(skill_file)
                if skill:
                    self._skills[skill.name] = skill
                    logger.info("Loaded skill '%s' from %s", skill.name, skill_file)

        return self._skills

    def get_skill(self, name: str) -> Optional[Skill]:
        """Retrieve loaded skill by name."""
        return self._skills.get(name)

    def get_all_skills(self) -> List[Skill]:
        """Return list of all loaded skills."""
        return list(self._skills.values())

    def format_all_for_system_prompt(self) -> str:
        """Format all loaded skills into a system prompt section."""
        if not self._skills:
            return ""

        sections = ["\n## AVAILABLE CUSTOM SKILLS"]
        for skill in self._skills.values():
            sections.append(skill.format_for_prompt())

        return "\n\n".join(sections) + "\n"
