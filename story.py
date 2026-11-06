"""Story steering: the drift clock and story-beat condition checks.

Pure logic. The database stores the clock (`turns_since_beat_progress`) and
the beats; this module decides what those numbers mean.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Container, Optional, Protocol

from models import StoryBeat


class DriftLevel(IntEnum):
    NONE = 0       # on track
    HINT = 1       # make the objective easier to notice
    PRESSURE = 2   # the world makes a move
    URGENT = 3     # something decisive forces the issue


def drift_level(turns_since_progress: int, threshold: int) -> DriftLevel:
    """Map the drift clock to a level.

    With threshold 6: turns 0-2 NONE, 3-5 HINT, 6-8 PRESSURE, 9+ URGENT.
    """
    if threshold < 1:
        raise ValueError("threshold must be at least 1")
    ratio = turns_since_progress / threshold
    if ratio < 0.5:
        return DriftLevel.NONE
    if ratio < 1:
        return DriftLevel.HINT
    if ratio < 1.5:
        return DriftLevel.PRESSURE
    return DriftLevel.URGENT


def build_story_block(beat: Optional[StoryBeat], level: DriftLevel, primary_threat: str = "") -> str:
    """The objective (and any drift pressure) as text for the DM prompt."""
    if beat is None:
        return ""
    lines = [
        "CURRENT STORY OBJECTIVE:",
        f"{beat.title}: {beat.objective}",
        "Weave natural hooks, clues and opportunities toward this objective. "
        "Never decide the player's actions for them.",
    ]
    threat = f" ({primary_threat})" if primary_threat else ""
    if level == DriftLevel.HINT:
        lines.append("The player has not progressed lately. Make a clue or path toward the objective more noticeable in this scene.")
    elif level == DriftLevel.PRESSURE:
        lines.append(f"The world moves on without the player. Introduce a concrete development{threat} that raises the stakes or points toward the objective.")
    elif level == DriftLevel.URGENT:
        lines.append(f"Something decisive happens now{threat}. It must force the player's attention onto the objective.")
    return "\n".join(lines)


# --- beat conditions -------------------------------------------------------------------

class WorldView(Protocol):
    """The few facts a beat condition can ask about."""

    def player_location(self) -> Optional[str]: ...
    def player_has_item(self, name: str) -> bool: ...
    def is_npc_dead(self, name: str) -> bool: ...


def is_valid_condition(condition_type: str, value: str, known_location_ids: Container[str]) -> bool:
    """Check a condition the LLM proposed during campaign creation.

    Small models invent location ids, so a 'location' condition must name a real one.
    Anything invalid should be downgraded to 'referee' by the caller.
    """
    if condition_type == "referee":
        return True
    value = (value or "").strip()
    if not value:
        return False
    if condition_type == "location":
        return value in known_location_ids
    return condition_type in ("npc_dead", "has_item")


def evaluate_beat_condition(beat: StoryBeat, world: WorldView) -> Optional[bool]:
    """True/False when code can decide, None when a referee call is needed."""
    value = beat.condition_value.strip()
    if beat.condition_type == "location":
        return world.player_location() == value
    if beat.condition_type == "has_item":
        return bool(value) and world.player_has_item(value)
    if beat.condition_type == "npc_dead":
        return bool(value) and world.is_npc_dead(value)
    return None


class DbWorldView:
    """WorldView backed by a GameDB (uses only its public methods)."""

    def __init__(self, db: Any):
        self.db = db

    def player_location(self) -> Optional[str]:
        player = self.db.get_player()
        return player.location_id if player else None

    def player_has_item(self, name: str) -> bool:
        player = self.db.get_player()
        return bool(player) and self.db.has_item(player.id, name)

    def is_npc_dead(self, name: str) -> bool:
        target = name.strip().lower()
        for location in self.db.list_locations():
            for c in self.db.list_characters_at(location.id, include_inactive=True):
                if c.name.lower() == target and c.status == "dead":
                    return True
        return False
