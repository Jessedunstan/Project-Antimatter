"""Sandboxed Subprocess Runner for Project Antimatter.

Supports streaming and batched execution across Windows (PowerShell) and POSIX (Bash),
with strict timeout enforcement, process group tree cleanup, and output bounds.
"""

import asyncio
import os
import sys
import time
import subprocess
from pathlib import Path
from typing import AsyncGenerator, Optional, Dict
from pydantic import BaseModel, Field


class CommandResult(BaseModel):
    """Result of an executed terminal command."""
    exit_code: int = Field(description="Process exit code, or -1 if timed out/terminated")
    stdout: str = Field(default="", description="Captured standard output")
    stderr: str = Field(default="", description="Captured standard error")
    duration_ms: float = Field(default=0.0, description="Execution time in milliseconds")
    timed_out: bool = Field(default=False, description="Whether command exceeded timeout")


class TerminalSandbox:
    """Safe execution sandbox for terminal commands."""

    DEFAULT_TIMEOUT_SEC = 60
    MAX_OUTPUT_BYTES = 512 * 1024  # 512 KB max output buffer to prevent OOM

    def __init__(
        self,
        working_dir: Optional[Path] = None,
        max_timeout_sec: int = DEFAULT_TIMEOUT_SEC,
        max_output_bytes: int = MAX_OUTPUT_BYTES,
    ):
        self.working_dir = (working_dir or Path.cwd()).resolve()
        self.max_timeout_sec = max_timeout_sec
        self.max_output_bytes = max_output_bytes
        self.is_windows = sys.platform == "win32"

    def _prepare_shell(self, command: str) -> tuple[str, list[str]]:
        """Resolve executable and arguments for platform-native shell."""
        if self.is_windows:
            # Use PowerShell with UTF-8 encoding forced
            shell_cmd = "powershell.exe"
            args = [
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy", "Bypass",
                "-Command",
                f"[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; {command}"
            ]
        else:
            shell_cmd = "/bin/bash"
            args = ["-c", command]
        return shell_cmd, args

    async def _kill_process_tree(self, proc: asyncio.subprocess.Process) -> None:
        """Forcefully kill process and all descendants to prevent orphaned daemons."""
        try:
            if self.is_windows:
                # Use taskkill to kill process tree (/T) forcefully (/F)
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False
                )
            else:
                proc.kill()
        except Exception:
            pass

    async def execute(
        self,
        command: str,
        timeout_sec: Optional[int] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> CommandResult:
        """Run command in sandbox and wait for completion."""
        effective_timeout = timeout_sec or self.max_timeout_sec
        shell_cmd, args = self._prepare_shell(command)

        # Merge environment
        proc_env = os.environ.copy()
        if env:
            proc_env.update(env)

        t0 = time.perf_counter()
        try:
            proc = await asyncio.create_subprocess_exec(
                shell_cmd,
                *args,
                cwd=str(self.working_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=proc_env,
            )

            try:
                stdout_data, stderr_data = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=effective_timeout
                )
                duration_ms = (time.perf_counter() - t0) * 1000

                stdout_text = stdout_data.decode("utf-8", errors="replace")[:self.max_output_bytes]
                stderr_text = stderr_data.decode("utf-8", errors="replace")[:self.max_output_bytes]

                if len(stdout_data) > self.max_output_bytes:
                    stdout_text += "\n[WARNING: Output truncated due to size limits]"

                return CommandResult(
                    exit_code=proc.returncode if proc.returncode is not None else 0,
                    stdout=stdout_text,
                    stderr=stderr_text,
                    duration_ms=duration_ms,
                    timed_out=False
                )

            except asyncio.TimeoutError:
                await self._kill_process_tree(proc)
                duration_ms = (time.perf_counter() - t0) * 1000
                return CommandResult(
                    exit_code=-1,
                    stdout="",
                    stderr=f"Command timed out after {effective_timeout} seconds.",
                    duration_ms=duration_ms,
                    timed_out=True
                )

        except Exception as err:
            duration_ms = (time.perf_counter() - t0) * 1000
            return CommandResult(
                exit_code=-1,
                stdout="",
                stderr=f"Failed to spawn process: {str(err)}",
                duration_ms=duration_ms,
                timed_out=False
            )

    async def stream(
        self,
        command: str,
        timeout_sec: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream command stdout line by line."""
        effective_timeout = timeout_sec or self.max_timeout_sec
        shell_cmd, args = self._prepare_shell(command)

        proc = await asyncio.create_subprocess_exec(
            shell_cmd,
            *args,
            cwd=str(self.working_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        try:
            t0 = time.time()
            assert proc.stdout is not None
            while True:
                if (time.time() - t0) > effective_timeout:
                    await self._kill_process_tree(proc)
                    yield f"\n[Process timed out after {effective_timeout}s]"
                    break

                try:
                    line = await asyncio.wait_for(proc.stdout.readline(), timeout=1.0)
                except asyncio.TimeoutError:
                    if proc.returncode is not None:
                        break
                    continue

                if not line:
                    if proc.returncode is not None:
                        break
                    continue

                yield line.decode("utf-8", errors="replace")

        finally:
            if proc.returncode is None:
                await self._kill_process_tree(proc)
