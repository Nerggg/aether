"""Shared menu helper for campaign and character creation.

Code renders every menu, including the fixed slots after the model's options
("Type my own", "New options", "Surprise me"), so no model ever has to guess
what a reply meant. Anything that is not a valid number prints a hint and
shows the list again.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

CUSTOM = "custom"      # "Type my own ..."
IDEA = "idea"          # "... from my idea"
REGEN = "regen"        # "New ..."
SURPRISE = "surprise"  # "Surprise me"

Extra = Tuple[str, str]   # (kind, label)
_NUMBER = re.compile(r"^\s*\(?(\d+)\)?[.)]?\s*$")


class Cancelled(Exception):
    """The player typed 'cancel'."""


@dataclass
class Reply:
    """What a reply meant. `kind` is "option" (with `index`), "text" (with `text`,
    from a shortcut), or one of the extra kinds above."""
    kind: str
    index: int = -1
    text: str = ""


def parse_number(text: str, count: int) -> Optional[int]:
    """'2', '2.' or '(2)' as a zero-based index into `count` menu lines, else None."""
    m = _NUMBER.match(text or "")
    if not m:
        return None
    n = int(m.group(1))
    return n - 1 if 1 <= n <= count else None


def match_closed(text: str, options: Sequence[str], aliases: Optional[Dict[str, str]] = None) -> Optional[str]:
    """The one option a free-text reply names ('i'd go with paladin', 'elven', 'palladin').

    Exact words and aliases match first; longer words may be one typo away.
    Returns None when nothing matches or when two different options are named.
    """
    lowered = {o.lower(): o for o in options}
    fuzzy_pool = [o for o in lowered if len(o) >= 5]
    found = set()
    for word in re.findall(r"[a-z']+", (text or "").lower()):
        if word in lowered:
            found.add(lowered[word])
        elif aliases and aliases.get(word) in options:
            found.add(aliases[word])
        elif len(word) >= 5:
            close = difflib.get_close_matches(word, fuzzy_pool, n=1, cutoff=0.84)
            if close and close[0][0] == word[0]:
                found.add(lowered[close[0]])
    return next(iter(found)) if len(found) == 1 else None


def ask_menu(
    read: Callable[[str], str],
    say: Callable[..., None],
    title: str,
    options: Sequence[str],
    extras: Sequence[Extra] = (),
    *,
    lines: Optional[Sequence[str]] = None,
    closed: bool = False,
    aliases: Optional[Dict[str, str]] = None,
    shortcut: Optional[Callable[[str], Optional[Reply]]] = None,
) -> Reply:
    """Show a numbered menu and loop until the reply is valid.

    Options are numbered first, then the extras. `lines` are what is shown for
    the options when it differs from the value ("Human: +1 to all abilities").
    A reply is read as: a number, then `shortcut`, then (if `closed`) a name from
    the options. Anything else prints a hint and shows the list again.
    `read` should raise `Cancelled` for 'cancel'.
    """
    shown = list(lines) if lines is not None else list(options)
    total = len(options) + len(extras)
    while True:
        say(f"\n{title}")
        for i, line in enumerate(shown, start=1):
            say(f"  {i}. {line}")
        for j, (_, label) in enumerate(extras, start=len(options) + 1):
            say(f"  {j}. {label}")
        raw = read("\nYou: ")

        idx = parse_number(raw, total)
        if idx is not None:
            if idx < len(options):
                return Reply("option", idx)
            return Reply(extras[idx - len(options)][0])
        if shortcut is not None:
            hit = shortcut(raw)
            if hit is not None:
                return hit
        if closed:
            match = match_closed(raw, options, aliases)
            if match:
                return Reply("option", list(options).index(match))
        say(f"[System] Please pick a number from 1 to {total}.")
