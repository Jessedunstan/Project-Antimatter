"""Demonstration and verification of Antimatter Tool Suite repairing LRUCache."""

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from antimatter.tools.fs_manager import FileSystemManager
from antimatter.tools.patcher import MultiLinePatcher
from antimatter.tools.terminal import TerminalSandbox


async def main():
    print("=" * 70)
    print(" ANTIMATTER DETERMINISTIC TOOL WORKSPACE REPAIR VERIFICATION")
    print("=" * 70)

    workspace_dir = PROJECT_ROOT / "workspace"
    fs = FileSystemManager(workspace_dir)
    patcher = MultiLinePatcher(fs)
    terminal = TerminalSandbox(workspace_dir)

    print("[1] Creating Checkpoint before repair...")
    cp_id = fs.create_checkpoint("Pre-repair baseline for LRU cache")
    print(f"    Checkpoint created: {cp_id}")

    print("\n[2] Applying Surgical Patch #1 to lru_cache/cache.py (Fix Eviction Direction)...")
    target1 = "            self.cache.popitem(last=True)"
    replacement1 = "            self.cache.popitem(last=False)  # Evict oldest (least recently used)"
    diff1, _ = patcher.apply_patch("lru_cache/cache.py", target1, replacement1, checkpoint_id=cp_id)
    print("    Patch #1 Applied:\n" + diff1)

    print("\n[3] Applying Surgical Patch #2 to lru_cache/cache.py (Update Access Order in get)...")
    target2 = (
        "        if key not in self.cache:\n"
        "            return None\n"
        "        # BUG: Missing move_to_end! Accessing an item does not refresh its LRU status\n"
        "        return self.cache[key]"
    )
    replacement2 = (
        "        if key not in self.cache:\n"
        "            return None\n"
        "        self.cache.move_to_end(key)  # Mark key as most recently used\n"
        "        return self.cache[key]"
    )
    diff2, _ = patcher.apply_patch("lru_cache/cache.py", target2, replacement2, checkpoint_id=cp_id)
    print("    Patch #2 Applied:\n" + diff2)

    print("\n[4] Running pytest in Terminal Sandbox to verify fix...")
    res = await terminal.execute(f'"{sys.executable}" -m pytest tests/test_lru_cache.py')
    print(res.stdout.strip())
    print(f"    Exit Code: {res.exit_code} | Duration: {res.duration_ms:.1f}ms")

    assert res.exit_code == 0, f"Tests failed: {res.stderr}"
    print("\n[5] All 4/4 LRU Cache Unit Tests Passed with Exit Code 0!")

    print("\n[6] Testing Checkpoint Rollback (Reverting fix to confirm rollback reliability)...")
    success, msg = fs.rollback(cp_id)
    print(f"    {msg}")

    # Re-run pytest to prove it reverted back to failing state
    res_reverted = await terminal.execute(f'"{sys.executable}" -m pytest tests/test_lru_cache.py')
    assert res_reverted.exit_code != 0
    print("    Verified: Rollback restored original pre-fix state (tests fail again as expected).")

    print("\n[7] Re-applying permanent fix...")
    patcher.apply_patch("lru_cache/cache.py", target1, replacement1)
    patcher.apply_patch("lru_cache/cache.py", target2, replacement2)
    res_final = await terminal.execute(f'"{sys.executable}" -m pytest tests/test_lru_cache.py')
    assert res_final.exit_code == 0
    print("    Permanent fix applied and verified 100% passing.")

    print("\n" + "=" * 70)
    print(" VERIFICATION COMPLETE: ALL TOOLS & ROLLBACKS FUNCTIONING PERFECTLY")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
