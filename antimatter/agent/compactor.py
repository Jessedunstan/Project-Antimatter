"""Context Window Compactor for Project Antimatter.

Maintains conversation trajectories strictly within local model token budgets
via semantic summary compression and rolling tool output compaction.
"""

from typing import List, Optional
import logging
from antimatter.brain.types import ChatMessage, ChatRole

logger = logging.getLogger(__name__)


class ContextCompactor:
    """Sliding-window and semantic memory compactor for local LLM context limits."""

    def __init__(
        self,
        max_token_budget: int = 8192,
        preserve_recent_turns: int = 6,
        chars_per_token_est: float = 3.8,
    ):
        self.max_token_budget = max_token_budget
        self.preserve_recent_turns = preserve_recent_turns
        self.chars_per_token_est = chars_per_token_est

    def estimate_tokens(self, messages: List[ChatMessage]) -> int:
        """Estimate token count across a list of chat messages."""
        total_chars = sum(len(m.content) for m in messages)
        # Account for message framing overhead
        overhead = len(messages) * 4
        return int(total_chars / self.chars_per_token_est) + overhead

    def compact(
        self,
        messages: List[ChatMessage],
        target_budget: Optional[int] = None,
    ) -> List[ChatMessage]:
        """
        Compress message history to fit within target token budget.
        
        Strategy:
        1. Always preserve the System message (index 0).
        2. Always preserve the initial User instruction (index 1).
        3. Always preserve the last N active turns (recent conversation).
        4. For older intermediate tool/assistant messages:
           - Truncate large tool result payloads with concise summaries.
           - If still over budget, summarize intermediate blocks.
        """
        budget = target_budget or self.max_token_budget
        current_tokens = self.estimate_tokens(messages)

        if current_tokens <= budget or len(messages) <= (self.preserve_recent_turns + 2):
            return list(messages)

        logger.info("Context tokens (%d) exceed budget (%d). Compacting...", current_tokens, budget)

        # Split messages: protected head, intermediate body, protected tail
        head = [messages[0]]  # System prompt
        user_start_idx = 1
        if len(messages) > 1 and messages[1].role == ChatRole.USER:
            head.append(messages[1])
            user_start_idx = 2

        split_point = max(user_start_idx, len(messages) - self.preserve_recent_turns)
        intermediate = messages[user_start_idx:split_point]
        tail = messages[split_point:]

        # Step 1: Compact large tool outputs in intermediate messages
        compacted_intermediate: List[ChatMessage] = []
        for msg in intermediate:
            if msg.role == ChatRole.TOOL and len(msg.content) > 300:
                # Condense verbose tool output (e.g. long terminal / file reads)
                preview = msg.content[:150].strip()
                summary_content = (
                    f"{preview}\n... [Tool output compressed: original length {len(msg.content)} chars] ..."
                )
                compacted_intermediate.append(
                    ChatMessage(
                        role=msg.role,
                        content=summary_content,
                        name=msg.name,
                        tool_call_id=msg.tool_call_id,
                    )
                )
            else:
                compacted_intermediate.append(msg)

        candidate = head + compacted_intermediate + tail
        if self.estimate_tokens(candidate) <= budget:
            return candidate

        # Step 2: If still over budget, replace intermediate steps with a consolidated summary milestone
        summary_lines = ["[System Note: Intermediate execution history compacted]"]
        for m in intermediate:
            if m.role == ChatRole.ASSISTANT and m.tool_calls:
                call_names = ", ".join(tc.name for tc in m.tool_calls)
                summary_lines.append(f"- Agent performed actions: {call_names}")
            elif m.role == ChatRole.TOOL:
                summary_lines.append(f"- Tool '{m.name or 'unknown'}' executed.")

        summary_message = ChatMessage(
            role=ChatRole.SYSTEM,
            content="\n".join(summary_lines)
        )

        return head + [summary_message] + tail
