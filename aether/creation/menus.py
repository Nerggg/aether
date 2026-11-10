"""Reading the player's menu choices ("b", "2") against the guide's last message.

Shared by campaign and character creation. Pure functions, no LLM: when the
player types just a letter or number, the option text is looked up in code, so
the extractor model never has to guess what "b" meant.
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Sequence

CUSTOM_CHOICE = "<custom>"    # returned by resolve_choice for the "Something else" option

_OPTION_LINE = re.compile(r"^[ \t]*(?:\*[ \t]*)?\(?([A-Za-z])[).][ \t]+(.+?)[ \t]*$", re.MULTILINE)
_CHOICE_INPUT = re.compile(r"^\s*\(?([A-Za-z]|\d)\)?[.)]?\s*$")
_CUSTOM_OPTION = re.compile(r"^(something else|other\b|custom|player'?s custom)", re.IGNORECASE)


def parse_menu(text: str) -> Dict[str, str]:
    """The lettered options in a guide message: {'A': 'Human: versatile', ...}."""
    menu: Dict[str, str] = {}
    for m in _OPTION_LINE.finditer(text or ""):
        menu.setdefault(m.group(1).upper(), m.group(2))
    return menu


def resolve_choice(user_text: str, guide_reply: str) -> Optional[str]:
    """If the player typed just a letter or number that matches an option in the
    guide's last message, return that option's text. Returns CUSTOM_CHOICE for the
    'Something else' option and None when this is not a menu choice."""
    m = _CHOICE_INPUT.match(user_text or "")
    if not m:
        return None
    key = m.group(1).upper()
    if key.isdigit():
        if not 1 <= int(key) <= 9:
            return None
        key = chr(ord("A") + int(key) - 1)
    option = parse_menu(guide_reply).get(key)
    if option is None:
        return None
    return CUSTOM_CHOICE if _CUSTOM_OPTION.match(option) else option


def last_assistant_reply(history: Sequence[Dict[str, str]]) -> str:
    return next((m["content"] for m in reversed(history) if m["role"] == "assistant"), "")
