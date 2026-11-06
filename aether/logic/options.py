"""The action menu: which kinds of options to ask for, cleaning what the model
returns, and reading the player's choice. Pure logic, no LLM.

The code picks the *categories* (so a cautious or evasive option can never be
forgotten); the model only writes one short line per category.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from aether.logic.stats import normalize_skill

EXPLORATION = "exploration"
CONVERSATION = "conversation"
SITUATIONS = (EXPLORATION, CONVERSATION)

STORY = "story"
CATEGORY_POOLS: Dict[str, tuple] = {
    EXPLORATION: ("cautious", "bold", "investigative", "resourceful"),
    CONVERSATION: ("friendly", "probing", "confrontational", "leave"),
}

# Instruction the options prompt gives the model for each slot.
CATEGORY_HINTS: Dict[str, str] = {
    STORY: "an action that moves toward the current story objective",
    "cautious": "a careful, low-risk action",
    "bold": "a daring, high-risk action",
    "investigative": "an action that gathers information or examines something",
    "resourceful": "a clever action using an item, ability or the surroundings",
    "friendly": "a friendly or diplomatic reply or action",
    "probing": "asking a question or probing for information",
    "confrontational": "a firm, threatening or confrontational action",
    "leave": "ending the exchange and moving on",
}

# Used when the model fails to produce an option for a slot, so the menu is never short.
FALLBACK_OPTIONS: Dict[str, "ActionOption"] = {}


@dataclass
class ActionOption:
    text: str
    skill: Optional[str] = None      # canonical skill name, or None if no roll is implied
    category: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "skill": self.skill, "category": self.category}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ActionOption":
        return cls(
            text=str(data.get("text", "")),
            skill=normalize_skill(data.get("skill")),
            category=data.get("category"),
        )


FALLBACK_OPTIONS.update({
    STORY: ActionOption("Press on toward your goal", None, STORY),
    "cautious": ActionOption("Proceed carefully", "perception", "cautious"),
    "bold": ActionOption("Act boldly", "athletics", "bold"),
    "investigative": ActionOption("Look around closely", "investigation", "investigative"),
    "resourceful": ActionOption("Try something clever", "survival", "resourceful"),
    "friendly": ActionOption("Speak kindly", "persuasion", "friendly"),
    "probing": ActionOption("Ask probing questions", "insight", "probing"),
    "confrontational": ActionOption("Press them hard", "intimidation", "confrontational"),
    "leave": ActionOption("Walk away", None, "leave"),
})


def detect_situation(npc_names_present: Iterable[str], last_narration: str) -> str:
    """Conversation if a living NPC here is mentioned in the latest narration, else exploration."""
    text = last_narration.lower()
    for name in npc_names_present:
        if name.strip() and name.strip().lower() in text:
            return CONVERSATION
    return EXPLORATION


def select_categories(
    situation: str, num_options: int, turn_number: int = 0, has_story_beat: bool = True
) -> List[str]:
    """Pick `num_options` categories. The story slot comes first when there is a beat.

    The remaining slots rotate with the turn number so menus don't look identical
    every turn, but the same inputs always give the same result.
    """
    if situation not in CATEGORY_POOLS:
        raise ValueError(f"Unknown situation {situation!r}. Expected one of {SITUATIONS}")
    if num_options < 1:
        raise ValueError("num_options must be at least 1")

    chosen: List[str] = [STORY] if has_story_beat else []
    pool = CATEGORY_POOLS[situation]
    shift = turn_number % len(pool)
    for category in pool[shift:] + pool[:shift]:
        if len(chosen) >= num_options:
            break
        chosen.append(category)
    return chosen[:num_options]


# --- cleaning what the model returns -------------------------------------------------

_LEADING = re.compile(r"^\s*(?:\(?\d+[.)\]:]|[-*\u2022])\s*")
_FILLER = {"the", "a", "an", "to", "and", "of", "your", "you", "at", "in", "on", "for"}


def clean_option_text(text: Any, max_words: int) -> str:
    """Strip numbering and quotes, collapse whitespace, cut to `max_words`."""
    t = _LEADING.sub("", str(text or "")).strip(" \t\"'`")
    words = t.split()
    if len(words) > max_words:
        words = words[:max_words]
    return " ".join(words).rstrip(",;:- ")


def _tokens(text: str) -> set:
    return set(re.findall(r"[a-z']+", text.lower())) - _FILLER


def _too_similar(a: str, b: str, threshold: float = 0.7) -> bool:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return a.strip().lower() == b.strip().lower()
    return len(ta & tb) / len(ta | tb) >= threshold


def sanitize_options(
    raw: Sequence[Any],
    max_words: int,
    limit: int,
    categories: Optional[Sequence[str]] = None,
) -> List[ActionOption]:
    """Turn raw model output (strings or dicts) into clean, distinct options.

    Drops empty and near-duplicate lines, normalises skill names (unknown ones
    become None) and fills a missing category from the slot's position.
    """
    result: List[ActionOption] = []
    for i, item in enumerate(raw):
        data = item if isinstance(item, Mapping) else {"text": item}
        text = clean_option_text(data.get("text"), max_words)
        if not text or any(_too_similar(text, o.text) for o in result):
            continue
        category = data.get("category")
        if not category and categories and i < len(categories):
            category = categories[i]
        result.append(ActionOption(text, normalize_skill(data.get("skill")), category))
        if len(result) >= limit:
            break
    return result


def complete_options(
    options: Sequence[ActionOption], categories: Sequence[str], limit: int
) -> List[ActionOption]:
    """Top up a short menu with built-in fallbacks for the categories still missing."""
    result = list(options)[:limit]
    covered = {o.category for o in result}
    for category in categories:
        if len(result) >= limit:
            break
        if category not in covered and category in FALLBACK_OPTIONS:
            fb = FALLBACK_OPTIONS[category]
            result.append(ActionOption(fb.text, fb.skill, fb.category))
            covered.add(category)
    return result


# --- showing the menu and reading the choice -------------------------------------------

CUSTOM_LABEL = "Write your own action"


def format_menu(options: Sequence[ActionOption], custom_label: str = CUSTOM_LABEL) -> str:
    lines = [f"{i}. {o.text}" for i, o in enumerate(options, start=1)]
    lines.append(f"{len(options) + 1}. {custom_label}")
    return "\n".join(lines)


@dataclass(frozen=True)
class Choice:
    kind: str                 # "option" or "custom"
    index: Optional[int]      # zero-based option index; None for custom


_CHOICE = re.compile(r"^\s*(\d+)\s*[.)]?\s*$")


def parse_choice(raw: str, num_options: int) -> Optional[Choice]:
    """Read '2', '2.' or ' 3 ' as a menu choice. None means it was not a valid number,
    so the caller can decide whether to treat it as a typed action."""
    m = _CHOICE.match(raw or "")
    if not m:
        return None
    n = int(m.group(1))
    if 1 <= n <= num_options:
        return Choice("option", n - 1)
    if n == num_options + 1:
        return Choice("custom", None)
    return None
