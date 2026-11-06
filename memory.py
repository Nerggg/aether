"""Token estimation, compaction decisions and prompt assembly. Pure logic.

The transcript is append-only. What the model sees is rebuilt every turn:

    system prompt + context + summary ("the player continues:")  +  recent messages

so the four storage stages from the design discussion collapse into one rule.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from config import MemoryConfig
from models import Message, Summary

MESSAGE_OVERHEAD_TOKENS = 4   # role markers and separators per message
MIN_MESSAGES_TO_COMPACT = 4   # never run a summary call for a tiny slice
CONTINUES_MARKER = "The player continues from here:"


def estimate_tokens(text: str) -> int:
    """Rough token count (about 4 characters per token). Good enough for budgeting."""
    return math.ceil(len(text) / 4) if text else 0


def message_tokens(message: Message) -> int:
    return estimate_tokens(message.content) + MESSAGE_OVERHEAD_TOKENS


def total_tokens(messages: Sequence[Message]) -> int:
    return sum(message_tokens(m) for m in messages)


# --- summary text ------------------------------------------------------------------------

def summary_to_text(summary: Optional[Summary]) -> str:
    """The summary as it appears in the prompt. Empty string when there is none."""
    if summary is None or not summary.narrative.strip():
        return ""
    parts = ["STORY SO FAR:", summary.narrative.strip()]
    for label, items in (
        ("Key characters met", summary.key_npcs),
        ("Unresolved threads", summary.open_threads),
        ("Promises and debts", summary.promises_debts),
    ):
        items = [i.strip() for i in items if i and i.strip()]
        if items:
            parts.append(f"{label}: " + "; ".join(items))
    return "\n".join(parts)


# --- compaction -------------------------------------------------------------------------

@dataclass
class CompactionSplit:
    to_compact: List[Message]   # goes into the new summary
    keep: List[Message]         # stays verbatim

    @property
    def covers_through_id(self) -> Optional[int]:
        return self.to_compact[-1].id if self.to_compact else None


def split_for_compaction(messages: Sequence[Message], keep_recent: int) -> CompactionSplit:
    """Split the raw tail into (summarise, keep verbatim).

    At least `keep_recent` messages are kept, and the kept part always starts at
    a player message so the scene is never cut off mid-exchange. If no clean
    split exists, nothing is compacted.
    """
    messages = list(messages)
    boundary = len(messages) - keep_recent
    while boundary > 0 and messages[boundary].role != "player":
        boundary -= 1
    if boundary <= 0:
        return CompactionSplit([], messages)
    return CompactionSplit(messages[:boundary], messages[boundary:])


def should_compact(
    messages: Sequence[Message], cfg: MemoryConfig, summary: Optional[Summary] = None
) -> bool:
    """True when the raw tail is big enough to be worth summarising.

    Triggers when the tail exceeds `compaction_trigger_ratio` of the history
    budget left after the summary, or when it reaches `max_uncompacted_messages`.
    """
    split = split_for_compaction(messages, cfg.keep_recent_messages)
    if len(split.to_compact) < MIN_MESSAGES_TO_COMPACT:
        return False
    if len(messages) >= cfg.max_uncompacted_messages:
        return True
    available = max(1, cfg.history_budget_tokens - estimate_tokens(summary_to_text(summary)))
    return total_tokens(messages) > cfg.compaction_trigger_ratio * available


def format_transcript(messages: Sequence[Message], player_name: str = "Player") -> str:
    """Plain 'Player: ... / DM: ...' text, used when asking the model to summarise."""
    lines = []
    for m in messages:
        if m.role == "player":
            lines.append(f"{player_name}: {m.content}")
        elif m.role == "dm":
            lines.append(f"DM: {m.content}")
        else:
            lines.append(f"[{m.content}]")
    return "\n".join(lines)


# --- prompt assembly --------------------------------------------------------------------

_ROLE_MAP = {"player": "user", "dm": "assistant"}


def build_history_messages(messages: Sequence[Message]) -> List[Dict[str, str]]:
    """Convert transcript rows to chat messages.

    Dice and engine notes (role 'system') are folded into the preceding player
    message, and consecutive same-role messages are merged, so the model always
    sees a clean user/assistant alternation.
    """
    out: List[Dict[str, str]] = []
    for m in messages:
        if m.role == "system":
            note = f"[{m.content}]"
            if out and out[-1]["role"] == "user":
                out[-1]["content"] += "\n" + note
            else:
                out.append({"role": "user", "content": note})
            continue
        role = _ROLE_MAP[m.role]
        if out and out[-1]["role"] == role:
            out[-1]["content"] += "\n\n" + m.content
        else:
            out.append({"role": role, "content": m.content})
    return out


def trim_to_budget(messages: Sequence[Message], max_tokens: int, min_keep: int = 2) -> List[Message]:
    """Safety net: drop the oldest messages until the tail fits `max_tokens`.

    Normal compaction should keep this from ever firing; it only matters if
    compaction is running late or a few replies were very long. The newest
    `min_keep` messages are never dropped.
    """
    kept = list(messages)
    while len(kept) > min_keep and total_tokens(kept) > max_tokens:
        kept.pop(0)
    return kept


def assemble_prompt(
    system_prompt: str,
    summary: Optional[Summary],
    messages: Sequence[Message],
    context_blocks: Sequence[str] = (),
) -> List[Dict[str, str]]:
    """Build the full chat for one DM call.

    `context_blocks` are extra pieces such as the character sheet, the current
    location, NPCs present, the story objective and RAG recall. Empty blocks
    are skipped.
    """
    parts = [system_prompt.strip()]
    parts.extend(b.strip() for b in context_blocks if b and b.strip())
    summary_text = summary_to_text(summary)
    if summary_text:
        parts.append(summary_text + ("\n" + CONTINUES_MARKER if messages else ""))
    chat = [{"role": "system", "content": "\n\n".join(parts)}]
    chat.extend(build_history_messages(messages))
    return chat
