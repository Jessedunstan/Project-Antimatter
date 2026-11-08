"""Verification test harness for Antimatter Sovereign Brain.

Can run with Ollama running or in fallback Mock mode.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path for direct test execution
sys.path.insert(0, str(Path(__file__).parent.parent))

from antimatter.core.config import get_settings, LLMBackendType
from antimatter.brain.client import LocalLLMClient
from antimatter.brain.types import ChatMessage, ChatRole


async def main():
    print("=" * 60)
    print(" PROJECT ANTIMATTER: SOVEREIGN INFERENCE VERIFICATION")
    print("=" * 60)

    settings = get_settings()
    print(f"[1] Configuration Loaded:")
    print(f"    - Hardware Profile: {settings.profile}")
    print(f"    - Target Backend:   {settings.llm_backend}")
    print(f"    - Planner Model:    {settings.planner_model}")
    print(f"    - Autocomplete:     {settings.autocomplete_model}")

    client = LocalLLMClient(settings)
    
    print("\n[2] Probing Local Backend Health...")
    health = await client.check_health()
    print(f"    Health Status: {health}")

    # Fallback to mock for testing if Ollama isn't currently alive on localhost
    if health.get("status") in ["unreachable", "error"]:
        print("\n[!] Local daemon unreachable. Switching to deterministic Mock backend for test suite...")
        settings.llm_backend = LLMBackendType.MOCK
        client = LocalLLMClient(settings)

    print("\n[3] Testing Streaming LLM Response...")
    messages = [
        ChatMessage(
            role=ChatRole.SYSTEM,
            content="You are Antimatter, a sovereign offline AI software engineer.",
        ),
        ChatMessage(
            role=ChatRole.USER,
            content="Confirm initialization and report ready status for offline software engineering tasks.",
        ),
    ]

    print("    Stream Output: ", end="", flush=True)
    async for chunk in client.stream_chat(messages):
        if chunk.delta_text:
            print(chunk.delta_text, end="", flush=True)
        if chunk.finish_reason:
            print(f" [DONE: {chunk.finish_reason}]")

    print("\n[4] Verification Complete: Sovereign Brain foundation ready.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
