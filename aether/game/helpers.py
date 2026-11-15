"""Small rules the game loop needs. They only touch the database through the
public GameDB methods, so each one is tested on its own."""

from __future__ import annotations

import re
from typing import Callable, List, Optional, Sequence

from aether.db import GameDB
from aether.llm.schemas import NewNpc
from aether.logic.tags import Tag, parse_amount
from aether.models import Character, Location

MAX_TAG_AMOUNT = 20       # one tag can never deal or heal more than this
MAX_ITEM_CHARS = 60

DIFFICULTY_BY_CATEGORY = {"bold": "hard", "cautious": "easy"}

_MOVE_VERBS = re.compile(
    r"\b(?:go|goes|going|head|heads|heading|travel|walk|walking|enter|cross|follow|leave|return|"
    r"descend|ascend|move|run|climb|ride|sail|row|journey|proceed|venture|make your way|set off)\b",
    re.IGNORECASE,
)


def difficulty_for(category: Optional[str]) -> str:
    return DIFFICULTY_BY_CATEGORY.get(category or "", "medium")


def _keys(location: Location) -> List[str]:
    name = location.name.lower().strip()
    stripped = name[4:] if name.startswith("the ") else name
    return [k for k in {name, stripped, location.id.replace("_", " ")} if k]


def detect_destination(action: str, connected: Sequence[Location]) -> Optional[Location]:
    """The connected location an action heads for, or None.

    The action needs a movement verb AND the name of a connected location, so
    'Ask about the Rotting Causeway' does not move the player. If several
    locations match, the longest name wins.
    """
    if not _MOVE_VERBS.search(action or ""):
        return None
    low = action.lower()
    best: Optional[Location] = None
    best_len = 0
    for loc in connected:
        for key in _keys(loc):
            if key in low and len(key) > best_len:
                best, best_len = loc, len(key)
    return best


def apply_tags(db: GameDB, player_id: int, tags: Sequence[Tag], say: Callable[..., None]) -> None:
    """Apply the DM's state tags to the player. Bad values are ignored, never raised."""
    for tag in tags:
        value = tag.value.strip()
        if tag.name == "ADD_ITEM":
            item = value[:MAX_ITEM_CHARS].strip()
            if item:
                db.add_item(player_id, item)
                say(f"[Item gained] {item}")
        elif tag.name == "REMOVE_ITEM":
            if value and db.remove_item(player_id, value):
                say(f"[Item lost] {value}")
        elif tag.name in ("TAKE_DAMAGE", "HEAL_HP"):
            amount = parse_amount(value)
            if amount is None:
                continue
            amount = min(abs(amount), MAX_TAG_AMOUNT)
            hp = db.change_hp(player_id, -amount if tag.name == "TAKE_DAMAGE" else amount)
            say(f"[{'Damage' if tag.name == 'TAKE_DAMAGE' else 'Healed'}] {amount}. HP is now {hp}")


def all_character_names(db: GameDB) -> List[str]:
    """The player and every NPC ever registered, alive or not."""
    names: List[str] = []
    player = db.get_player()
    if player:
        names.append(player.name)
    for loc in db.list_locations():
        names += [c.name for c in db.list_characters_at(loc.id, include_inactive=True)]
    return names


def register_new_npcs(db: GameDB, npcs: Sequence[NewNpc], location_id: str) -> List[str]:
    """Save characters the referee spotted. Names that already exist are skipped.
    Returns the names that were created."""
    known = {n.lower() for n in all_character_names(db)}
    created: List[str] = []
    for npc in npcs:
        name = " ".join(npc.name.split())
        if not name or name.lower() in known:
            continue
        known.add(name.lower())
        persistent = bool(npc.is_persistent)
        score = 12 if persistent else 10
        hp = 12 if persistent else 6
        db.add_character(
            Character(
                name=name, type=npc.type, location_id=location_id,
                strength=score, dexterity=score, constitution=score,
                intelligence=score, wisdom=score, charisma=score,
                hp=hp, max_hp=hp, is_persistent=persistent, template_id=npc.template_id,
                backstory=(npc.brief_backstory or "") if persistent else "",
            )
        )
        created.append(name)
    return created
