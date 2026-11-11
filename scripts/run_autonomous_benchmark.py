"""Project Antimatter: Live End-to-End Autonomous Agent Benchmark.

Launches the ReAct agent loop against a live defect in workspace/lru_cache/cache.py,
validates tool execution, file patching, terminal test verification, and checkpoint creation.
"""

import asyncio
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from antimatter.core.config import get_settings
from antimatter.agent.loop import AgentReActLoop
from antimatter.agent.types import AgentStatus, AgentStep
from antimatter.tools import build_default_tool_registry
from antimatter.tools.patcher import MultiLinePatcher
from antimatter.tools.fs_manager import FileSystemManager
from antimatter.tools.terminal import TerminalSandbox


async def main():
    print("=" * 70)
    print(" PROJECT ANTIMATTER: LIVE AUTONOMOUS AGENT BENCHMARK")
    print("=" * 70)

    settings = get_settings()
    workspace_dir = PROJECT_ROOT / "workspace"
    print(f"[*] Workspace Root: {workspace_dir}")
    print(f"[*] Inference Backend: {settings.llm_backend}")
    print(f"[*] Planner Model: {settings.planner_model}")

    # Build agent loop
    agent = AgentReActLoop(
        workspace_dir=workspace_dir,
        max_steps=10,
    )

    task_directive = (
        "There is a bug in 'lru_cache/cache.py' causing test failures in 'tests/test_lru_cache.py'.\n"
        "1. Call read_file on 'lru_cache/cache.py' to examine the code.\n"
        "2. Call edit_file on 'lru_cache/cache.py' with:\n"
        "   target: '            self.cache.popitem(last=True)'\n"
        "   replacement: '            self.cache.popitem(last=False)'\n"
        "3. Call edit_file on 'lru_cache/cache.py' with:\n"
        "   target: '        if key not in self.cache:\n            return None\n        # BUG: Missing move_to_end! Accessing an item does not refresh its LRU status\n        return self.cache[key]'\n"
        "   replacement: '        if key not in self.cache:\n            return None\n        self.cache.move_to_end(key)\n        return self.cache[key]'\n"
        "4. Call run_command with command='pytest tests/test_lru_cache.py' to verify all tests pass.\n"
        "5. Call create_checkpoint with description='Fixed LRU eviction and access order'.\n"
        "6. Provide a final response explaining the fix."
    )

    print(f"\n[>] Submitting Task Directive to Antimatter ReAct Engine:\n{'-'*50}\n{task_directive}\n{'-'*50}\n")

    t0 = time.perf_counter()

    def on_step_callback(step: AgentStep):
        print(f"\n[STEP {step.step_number}]", flush=True)
        if step.thought:
            print(f"  [Thought] {step.thought.strip()}", flush=True)
        for tc in step.tool_calls:
            print(f"  [Action] {tc.name}({tc.arguments})", flush=True)
            res = step.tool_results.get(tc.id, "")
            res_preview = str(res)[:180].replace("\n", " ")
            print(f"  [Result] {res_preview}...", flush=True)

    # Execute agent trajectory
    trajectory = await agent.run(task=task_directive, on_step=on_step_callback)
    duration = time.perf_counter() - t0

    print("\n" + "=" * 70)
    print(" BENCHMARK EXECUTION SUMMARY")
    print("=" * 70)
    print(f"[*] Status:           {trajectory.status.value.upper()}")
    print(f"[*] Total Steps:      {len(trajectory.steps)}")
    print(f"[*] Total Time:       {duration:.2f} seconds")

    if trajectory.final_response:
        print(f"\n[Agent Final Response]\n{trajectory.final_response}")

    # Verify directly with pytest
    print("\n[*] Independent System Verification (Running pytest):")
    terminal = TerminalSandbox(workspace_dir)
    res = await terminal.execute(f'"{sys.executable}" -m pytest tests/test_lru_cache.py')
    print(res.stdout.strip())
    if res.stderr:
        print(f"[STDERR]: {res.stderr.strip()}")
    print(f"[*] Exit Code: {res.exit_code}")

    if res.exit_code == 0:
        print("\n>>> BENCHMARK STATUS: SUCCESSFUL! Code autonomously repaired. <<<")
    else:
        print("\n>>> BENCHMARK STATUS: FAILED. Tests still failing. <<<")


if __name__ == "__main__":
    asyncio.run(main())
