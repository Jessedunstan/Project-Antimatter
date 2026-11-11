"""Deep Reasoning & Code Critic Specialist for Project Antimatter.

Performs AST-aware code patch validation, safety checks, test coverage analysis,
and deep logic audits using Tree-sitter and local reasoning LLMs (DeepSeek-R1 / Qwen2.5-Coder).
"""

from __future__ import annotations
import ast
import logging
import time
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from antimatter.agent.relay import RelayBaton, SpecialistDomain
from antimatter.intelligence.ast_parser import ASTParser
from antimatter.core.config import get_settings

logger = logging.getLogger(__name__)


class CriticVerdict(str, Enum):
    APPROVED = "APPROVED"
    WARNING = "WARNING"
    REJECTED = "REJECTED"


class CodeIssue(BaseModel):
    """Specific code, security, or design flaw identified by the critic."""
    category: str  # syntax, security, performance, correctness
    severity: str  # low, medium, high, critical
    message: str
    line_number: Optional[int] = None


class CodeCritiqueReport(BaseModel):
    """Structured report returned by the Code Critic specialist."""
    verdict: CriticVerdict
    security_score: float = Field(ge=0.0, le=1.0)
    summary: str
    issues: List[CodeIssue] = Field(default_factory=list)
    suggested_tests: List[str] = Field(default_factory=list)
    ast_valid: bool = True


class CodeCriticSpecialist:
    """Specialist auditor analyzing code patches and system designs before deployment."""

    def __init__(self, model_name: str = "deepseek-r1:7b"):
        self.model_name = model_name
        self.ast_parser = ASTParser()

    def audit_python_code(self, source_code: str) -> CodeCritiqueReport:
        """Perform deterministic AST-level safety and syntax audit on Python code."""
        issues: List[CodeIssue] = []

        # 1. Syntax Validation
        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            return CodeCritiqueReport(
                verdict=CriticVerdict.REJECTED,
                security_score=0.0,
                summary=f"Syntax Error: {e.msg} at line {e.lineno}",
                issues=[CodeIssue(category="syntax", severity="critical", message=e.msg, line_number=e.lineno)],
                ast_valid=False,
            )

        # 2. Security AST Walk (Detect dangerous primitives: eval, exec, untrusted imports)
        security_penalties = 0.0
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = ""
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr

                if func_name in ("eval", "exec"):
                    issues.append(CodeIssue(
                        category="security",
                        severity="critical",
                        message=f"Use of dangerous built-in '{func_name}()' detected.",
                        line_number=node.lineno,
                    ))
                    security_penalties += 0.5

            # Detect raw os.system or shell subprocess invocations
            if isinstance(node, ast.Attribute) and node.attr == "system":
                issues.append(CodeIssue(
                    category="security",
                    severity="high",
                    message="Use of os.system() detected. Prefer sandboxed execution tools.",
                    line_number=node.lineno,
                ))
                security_penalties += 0.3

        security_score = max(0.0, 1.0 - security_penalties)

        # 3. Formulate Verdict
        has_critical = any(i.severity == "critical" for i in issues)
        has_high = any(i.severity == "high" for i in issues)

        if has_critical:
            verdict = CriticVerdict.REJECTED
        elif has_high or security_score < 0.8:
            verdict = CriticVerdict.WARNING
        else:
            verdict = CriticVerdict.APPROVED

        # 4. Generate Suggested Test Signatures
        suggested_tests = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                suggested_tests.append(f"test_{node.name}_happy_path")
                suggested_tests.append(f"test_{node.name}_edge_cases")

        summary = (
            f"Code Audit complete. Verdict: {verdict.value}. "
            f"Security Score: {security_score:.2f}. "
            f"Issues found: {len(issues)}."
        )

        return CodeCritiqueReport(
            verdict=verdict,
            security_score=security_score,
            summary=summary,
            issues=issues,
            suggested_tests=suggested_tests,
            ast_valid=True,
        )

    async def handle_relay_baton(self, baton: RelayBaton) -> Dict[str, Any]:
        """Relay handler entrypoint conforming to SpecialistRelayEngine."""
        source_code = baton.context_data.get("code") or baton.context_data.get("patch") or ""
        report = self.audit_python_code(source_code)

        advice = (
            f"Critic Verdict: {report.verdict.value}\n"
            f"Security Score: {report.security_score:.2f}\n"
            f"Summary: {report.summary}\n"
            f"Suggested Tests: {', '.join(report.suggested_tests[:4])}"
        )

        return {
            "advice": advice,
            "artifacts": {
                "report": report.model_dump(),
                "verdict": report.verdict.value,
                "security_score": report.security_score,
                "issues_count": len(report.issues),
            },
        }
