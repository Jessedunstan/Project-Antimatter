"""Autonomous Bug-Repair Benchmark Runner for Project Antimatter.

Executes the genuine AgentReActLoop (zero scripted patches) across the 10 seeded
bug-repair tasks in isolated temporary workspaces and compares local models
(e.g., `qwen2.5-coder:7b` vs `llama3.2:1b`).

Usage:
    python -m evals.run
    python -m evals.run --models qwen2.5-coder:7b llama3.2:1b --max-steps 6
"""

from __future__ import annotations
import argparse
import asyncio
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

from antimatter.agent.loop import AgentReActLoop
from antimatter.context.service import CodeIntelligenceService
from evals.tasks import BugRepairTask, get_benchmark_tasks


CORE_REPAIR_TOOLS = [
    "read_file",
    "edit_file",
    "write_file",
    "run_command",
    "list_directory",
]


@dataclass
class TaskEvalResult:
    """Result of running a single BugRepairTask on a single model."""
    model: str
    task_id: str
    title: str
    pre_test_failed_as_expected: bool
    passed: bool
    steps_taken: int
    tool_calls_count: int
    tool_errors_count: int
    wall_time_sec: float
    agent_status: str
    error_message: Optional[str] = None
    tools_invoked: List[str] = field(default_factory=list)


def _run_pytest_in_workspace(workspace_dir: Path, test_filename: str, timeout_sec: int = 20) -> tuple[bool, str]:
    """Run pytest on test_filename inside workspace_dir and return (passed, output)."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", test_filename, "-q", "--tb=short", "-o", "addopts="],
        cwd=str(workspace_dir),
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=False,
    )
    output = (proc.stdout + "\n" + proc.stderr).strip()
    return proc.returncode == 0, output


async def evaluate_task_on_model(
    task: BugRepairTask,
    model_name: str,
    max_steps: int = 6,
) -> TaskEvalResult:
    """Run a single seeded bug task in an isolated workspace using AgentReActLoop."""
    with tempfile.TemporaryDirectory(prefix=f"antimatter_eval_{task.task_id}_") as tmp_dir:
        ws_path = Path(tmp_dir).resolve()
        src_path = ws_path / task.source_filename
        test_path = ws_path / task.test_filename

        src_path.write_text(task.buggy_code, encoding="utf-8")
        test_path.write_text(task.test_code, encoding="utf-8")

        # 1. Verify the seeded bug actually fails pytest prior to agent intervention
        pre_passed, _ = _run_pytest_in_workspace(ws_path, task.test_filename)
        pre_failed_as_expected = not pre_passed

        # 2. Initialize genuine AgentReActLoop with in-memory code intelligence
        code_intel = CodeIntelligenceService(workspace_dir=ws_path, db_path=":memory:")
        code_intel.index_file(task.source_filename)

        loop = AgentReActLoop(
            code_intel=code_intel,
            workspace_dir=ws_path,
            max_steps=max_steps,
            allowed_tools=CORE_REPAIR_TOOLS,
            model=model_name,
        )

        t0 = time.perf_counter()
        try:
            trajectory = await loop.run(task.prompt)
        finally:
            loop.close()
        wall_time = round(time.perf_counter() - t0, 2)

        # 3. Restore canonical test file to prevent test-tampering false positives
        test_path.write_text(task.test_code, encoding="utf-8")

        # 4. Run pytest to verify whether the bug in source_filename was genuinely repaired
        post_passed, _ = _run_pytest_in_workspace(ws_path, task.test_filename)

        # 5. Collect telemetry from trajectory steps
        steps_taken = len(trajectory.steps)
        tool_calls_count = 0
        tool_errors_count = 0
        tools_invoked: List[str] = []

        for step in trajectory.steps:
            for tc in step.tool_calls:
                tool_calls_count += 1
                tools_invoked.append(tc.name)
            for res_str in step.tool_results.values():
                if str(res_str).startswith("Error") or "PatchError" in str(res_str):
                    tool_errors_count += 1

        return TaskEvalResult(
            model=model_name,
            task_id=task.task_id,
            title=task.title,
            pre_test_failed_as_expected=pre_failed_as_expected,
            passed=post_passed,
            steps_taken=steps_taken,
            tool_calls_count=tool_calls_count,
            tool_errors_count=tool_errors_count,
            wall_time_sec=wall_time,
            agent_status=trajectory.status.value,
            error_message=trajectory.error_message,
            tools_invoked=tools_invoked,
        )


def format_markdown_report(
    results_by_model: Dict[str, List[TaskEvalResult]],
    tasks: List[BugRepairTask],
) -> str:
    """Format comparative benchmark scoreboard and per-task breakdown as Markdown."""
    lines: List[str] = [
        "# Project Antimatter — Autonomous Bug-Repair Benchmark Report",
        "",
        "## 1. Aggregate Model Scoreboard",
        "",
        "| Model | Tasks Passed | Pass Rate | Avg Steps | Total Tool Calls | Tool Call Errors | Avg Wall Time (s) | Total Time (s) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for model_name, res_list in results_by_model.items():
        total = len(res_list)
        passed = sum(1 for r in res_list if r.passed)
        pass_rate = (passed / total * 100.0) if total else 0.0
        avg_steps = sum(r.steps_taken for r in res_list) / total if total else 0.0
        total_tcs = sum(r.tool_calls_count for r in res_list)
        total_errs = sum(r.tool_errors_count for r in res_list)
        total_time = sum(r.wall_time_sec for r in res_list)
        avg_time = total_time / total if total else 0.0

        lines.append(
            f"| `{model_name}` | **{passed} / {total}** | **{pass_rate:.1f}%** | "
            f"{avg_steps:.2f} | {total_tcs} | {total_errs} | {avg_time:.2f}s | {total_time:.1f}s |"
        )

    lines.extend([
        "",
        "## 2. Per-Task Breakdown",
        "",
        "| Task ID | Bug Description | Model | Result | Steps | Tool Calls | Tool Errors | Time (s) | Tools Used |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |",
    ])

    for task in tasks:
        for model_name, res_list in results_by_model.items():
            match = next((r for r in res_list if r.task_id == task.task_id), None)
            if not match:
                continue
            badge = "PASS" if match.passed else "FAIL"
            tools_str = ", ".join(match.tools_invoked) if match.tools_invoked else "*(none)*"
            lines.append(
                f"| `{task.task_id}` | {task.title} | `{model_name}` | **{badge}** | "
                f"{match.steps_taken} | {match.tool_calls_count} | {match.tool_errors_count} | "
                f"{match.wall_time_sec:.2f}s | `{tools_str}` |"
            )

    lines.append("")
    return "\n".join(lines)


async def run_benchmark(
    models: List[str],
    max_steps: int = 6,
    task_filter: Optional[str] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run the full autonomous bug-repair benchmark across all specified models."""
    all_tasks = get_benchmark_tasks()
    if task_filter:
        all_tasks = [t for t in all_tasks if task_filter in t.task_id]

    results_by_model: Dict[str, List[TaskEvalResult]] = {m: [] for m in models}

    print("=" * 84)
    print(" PROJECT ANTIMATTER — AUTONOMOUS BUG-REPAIR BENCHMARK HARNESS")
    print(f" Models: {', '.join(models)} | Tasks: {len(all_tasks)} | Max Steps/Task: {max_steps}")
    print("=" * 84)

    for model_name in models:
        print(f"\n>>> Evaluating Model: {model_name}")
        for idx, task in enumerate(all_tasks, 1):
            print(f"  [{idx:02d}/{len(all_tasks):02d}] {task.task_id} ({task.title}) ... ", end="", flush=True)
            res = await evaluate_task_on_model(task, model_name=model_name, max_steps=max_steps)
            results_by_model[model_name].append(res)
            status_label = "PASS" if res.passed else "FAIL"
            print(
                f"{status_label} (steps={res.steps_taken}, calls={res.tool_calls_count}, "
                f"errors={res.tool_errors_count}, time={res.wall_time_sec:.2f}s)"
            )

    md_report = format_markdown_report(results_by_model, all_tasks)
    print("\n" + md_report)

    out_dir = output_dir or (Path(__file__).resolve().parent / "results")
    out_dir.mkdir(parents=True, exist_ok=True)

    json_payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "models": models,
        "max_steps": max_steps,
        "results": {
            m: [asdict(r) for r in r_list]
            for m, r_list in results_by_model.items()
        },
    }

    (out_dir / "latest_benchmark.json").write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
    (out_dir / "latest_benchmark.md").write_text(md_report, encoding="utf-8")
    return json_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Project Antimatter autonomous bug-repair benchmark.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["qwen2.5-coder:7b", "llama3.2:1b"],
        help="List of Ollama model names to benchmark.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=6,
        help="Maximum ReAct steps per task.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Optional substring filter on task_id.",
    )
    args = parser.parse_args()
    asyncio.run(run_benchmark(models=args.models, max_steps=args.max_steps, task_filter=args.task))


if __name__ == "__main__":
    main()
