"""Exact Multi-Line Patcher and Unified Diff Engine for Project Antimatter.

Ensures deterministic, surgical file edits with newline normalization,
unified diff generation, and checkpoint snapshot integration.
"""

import difflib
import logging
from pathlib import Path
from typing import Optional, Tuple
from antimatter.tools.fs_manager import FileSystemManager

logger = logging.getLogger(__name__)


class PatchError(Exception):
    """Raised when a patch cannot be cleanly applied."""
    pass


class MultiLinePatcher:
    """Surgical multi-line file patcher."""

    def __init__(self, fs_manager: FileSystemManager):
        self.fs = fs_manager

    @staticmethod
    def normalize_newlines(text: str) -> str:
        """Normalize all newlines to standard UNIX '\n'."""
        return text.replace("\r\n", "\n").replace("\r", "\n")

    @staticmethod
    def generate_diff(original: str, modified: str, file_name: str = "target_file") -> str:
        """Produce unified diff string between original and modified text."""
        orig_lines = original.splitlines(keepends=True)
        mod_lines = modified.splitlines(keepends=True)
        diff_lines = list(
            difflib.unified_diff(
                orig_lines,
                mod_lines,
                fromfile=f"a/{file_name}",
                tofile=f"b/{file_name}",
                lineterm="",
            )
        )
        return "\n".join(diff_lines)

    def apply_patch(
        self,
        file_path: str | Path,
        target_block: str,
        replacement_block: str,
        replace_all: bool = False,
        checkpoint_id: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Apply exact multi-line patch to a file.
        Returns (unified_diff, updated_content).
        Raises PatchError if target block is not found or ambiguous.
        """
        safe_path = self.fs.resolve_safe_path(file_path)
        if not safe_path.exists():
            raise PatchError(f"Cannot patch non-existent file: {file_path}")

        original_content = self.fs.read_file(safe_path)

        # Normalize newlines for deterministic matching
        norm_original = self.normalize_newlines(original_content)
        norm_target = self.normalize_newlines(target_block)
        norm_replacement = self.normalize_newlines(replacement_block)

        occurrences = norm_original.count(norm_target)

        if occurrences == 0:
            # Fallback: try indentation-insensitive line matching while preserving file indentation
            indented_result = self._try_indentation_aware_patch(
                norm_original, norm_target, norm_replacement, replace_all=replace_all
            )
            if indented_result is not None:
                updated_content = indented_result
                diff = self.generate_diff(norm_original, updated_content, file_name=safe_path.name)
                self.fs.write_atomic(safe_path, updated_content, checkpoint_id=checkpoint_id)
                return diff, updated_content

            # Check for close match to provide helpful diagnostic error
            preview = norm_target[:120].strip()
            raise PatchError(
                f"Target block not found in '{file_path}'.\n"
                f"Target excerpt:\n'''\n{preview}\n'''"
            )

        if occurrences > 1 and not replace_all:
            raise PatchError(
                f"Target block occurs {occurrences} times in '{file_path}'. "
                f"Provide more surrounding context to disambiguate, or pass replace_all=True."
            )

        if replace_all:
            updated_content = norm_original.replace(norm_target, norm_replacement)
        else:
            updated_content = norm_original.replace(norm_target, norm_replacement, 1)

        # Generate unified diff before saving
        diff = self.generate_diff(norm_original, updated_content, file_name=safe_path.name)

        # Write atomically with checkpoint tracking
        self.fs.write_atomic(safe_path, updated_content, checkpoint_id=checkpoint_id)

        return diff, updated_content

    @staticmethod
    def _try_indentation_aware_patch(
        norm_original: str,
        norm_target: str,
        norm_replacement: str,
        replace_all: bool = False,
    ) -> Optional[str]:
        """Match target lines ignoring leading whitespace differences and preserve file indentation."""
        target_lines = norm_target.strip("\n").split("\n")
        target_stripped = [l.strip() for l in target_lines]
        if not target_stripped or not any(target_stripped):
            return None

        orig_lines = norm_original.split("\n")
        k = len(target_lines)
        if k > len(orig_lines):
            return None

        match_indices = []
        for i in range(len(orig_lines) - k + 1):
            window = orig_lines[i : i + k]
            if [w.strip() for w in window] == target_stripped:
                match_indices.append(i)

        if not match_indices:
            return None
        if len(match_indices) > 1 and not replace_all:
            return None

        replacement_lines = norm_replacement.strip("\n").split("\n")
        target_first_nonempty = next((l for l in target_lines if l.strip()), "")
        target_base_indent = len(target_first_nonempty) - len(target_first_nonempty.lstrip(" \t"))

        updated_lines = list(orig_lines)
        for idx in reversed(match_indices if replace_all else match_indices[:1]):
            orig_first_nonempty = next((l for l in orig_lines[idx : idx + k] if l.strip()), "")
            orig_indent_str = orig_first_nonempty[: len(orig_first_nonempty) - len(orig_first_nonempty.lstrip(" \t"))]

            reindented = []
            for r_line in replacement_lines:
                if not r_line.strip():
                    reindented.append("")
                else:
                    r_indent = len(r_line) - len(r_line.lstrip(" \t"))
                    rel_indent = max(0, r_indent - target_base_indent)
                    reindented.append(orig_indent_str + (" " * rel_indent) + r_line.lstrip(" \t"))

            updated_lines[idx : idx + k] = reindented

        return "\n".join(updated_lines)
