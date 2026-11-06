"""Plain data classes and the allowed value sets for Aether.

These carry no behaviour. The database layer (db.py) converts rows to and
from them, and the game logic passes them around.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Allowed values. db.py builds its CHECK constraints from these, so the
# database and the Python code can never disagree.
CHARACTER_TYPES = ("player", "ally", "enemy", "merchant")
CHARACTER_STATUSES = ("alive", "dead", "gone")
MESSAGE_ROLES = ("player", "dm", "system")
BEAT_CONDITIONS = ("location", "npc_dead", "has_item", "referee")
BEAT_STATUSES = ("pending", "active", "done")


@dataclass
class CampaignInfo:
    name: str
    slug: str
    setting_description: str
    primary_threat: str
    starting_quest_hook: str
    theme_vibe: str = ""
    starting_patron: str = ""
    storyline_beginning: str = ""
    storyline_middle: str = ""
    storyline_end: str = ""
    victory_condition: str = ""
    starting_location_id: Optional[str] = None
    embedding_model: Optional[str] = None
    created_at: Optional[str] = None


@dataclass
class Location:
    id: str
    name: str
    brief_concept: str
    description: Optional[str] = None  # None until the JIT generator writes it


@dataclass
class Character:
    name: str
    type: str
    location_id: str
    id: Optional[int] = None
    race: Optional[str] = None
    char_class: Optional[str] = None
    strength: int = 10
    dexterity: int = 10
    constitution: int = 10
    intelligence: int = 10
    wisdom: int = 10
    charisma: int = 10
    hp: int = 10
    max_hp: int = 10
    status: str = "alive"
    is_persistent: bool = True
    template_id: Optional[str] = None
    backstory: str = ""


@dataclass
class Message:
    id: int
    turn_number: int
    role: str
    content: str
    created_at: Optional[str] = None


@dataclass
class Summary:
    covers_through_message_id: int
    narrative: str
    key_npcs: List[str] = field(default_factory=list)
    open_threads: List[str] = field(default_factory=list)
    promises_debts: List[str] = field(default_factory=list)
    id: Optional[int] = None
    created_at: Optional[str] = None


@dataclass
class StoryBeat:
    position: int
    title: str
    objective: str
    condition_type: str = "referee"
    condition_value: str = ""
    status: str = "pending"
    completed_at_turn: Optional[int] = None
    id: Optional[int] = None


@dataclass
class GameState:
    current_turn: int = 0
    current_beat_id: Optional[int] = None
    turns_since_beat_progress: int = 0  # the drift clock
    pending_options: Optional[List[Dict[str, Any]]] = None
    last_played_at: Optional[str] = None
