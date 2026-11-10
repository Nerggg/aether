"""Character creation.

Flow (the same for the player as before):

    guide chat (name, race, class, backstory) -> checklist complete -> save

The model only chats and extracts. Everything that matters is decided in code:

* race and class must map onto the real lists (`stats.RACES`, `stats.CLASSES`)
* every extracted value must appear in the conversation, so a model that
  guesses cannot fill the checklist
* backstory is accepted only once name, race and class are settled
* stats and HP come from `calculate_character_stats`, the starting kit from a table
* the player and the kit are saved in one transaction

Run it on its own:  python -m aether.creation.character <campaign_slug>
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, field_validator

from aether.db import GameDB, SchemaVersionError
from aether.creation.menus import CUSTOM_CHOICE, last_assistant_reply, resolve_choice
from aether.llm import calls
from aether.llm.client import LLMClient, LLMError, Messages
from aether.logic.stats import (
    CLASSES,
    HIT_DIE,
    RACES,
    RACIAL_BONUSES,
    STAT_PRIORITIES,
    calculate_character_stats,
)
from aether.models import CampaignInfo, Character

CORE_KEYS = ("name", "race", "char_class")
CHECKLIST_KEYS = CORE_KEYS + ("backstory",)
CHECKLIST_LABELS = {"name": "Name", "race": "Race", "char_class": "Class", "backstory": "Backstory"}

MAX_NAME_CHARS = 40
MAX_NAME_WORDS = 4
MIN_BACKSTORY_WORDS = 6
BACKSTORY_MAX_CHARS = 600
CHECKLIST_MAX_TOKENS = 300

RACE_ALIASES = {
    "elves": "Elf", "elven": "Elf", "dwarves": "Dwarf", "dwarven": "Dwarf", "gnomish": "Gnome",
    "humans": "Human", "halflings": "Halfling", "tieflings": "Tiefling", "dragonborns": "Dragonborn",
}

# Name, quantity. Everyone also gets COMMON_ITEMS.
STARTING_KITS: Dict[str, Tuple[Tuple[str, int], ...]] = {
    "Fighter": (("Longsword", 1), ("Chain mail", 1)),
    "Rogue": (("Dagger", 1), ("Leather armor", 1), ("Thieves' tools", 1)),
    "Wizard": (("Quarterstaff", 1), ("Spellbook", 1), ("Robes", 1)),
    "Cleric": (("Mace", 1), ("Scale mail", 1), ("Holy symbol", 1)),
    "Paladin": (("Longsword", 1), ("Chain mail", 1), ("Holy symbol", 1)),
    "Ranger": (("Short bow", 1), ("Arrows", 12), ("Leather armor", 1)),
    "Bard": (("Rapier", 1), ("Lute", 1), ("Leather armor", 1)),
}
COMMON_ITEMS: Tuple[Tuple[str, int], ...] = (("Stale bread", 1),)

_BAD_NAMES = frozenset({"none", "null", "unknown", "n/a", "na", "player", "character", "name", "unnamed"})
_RESERVED = frozenset(
    {r.lower() for r in RACES} | {c.lower() for c in CLASSES} | set(RACE_ALIASES)
)

FOCUS_INSTRUCTIONS = {
    "name": "Offer three character name suggestions that fit the setting.",
    "race": "Offer three races from the approved list.",
    "char_class": "Offer three classes from the approved list.",
    "backstory": (
        "Offer three short backstory options, one or two sentences each (under 30 words), tied to the "
        "campaign and to the chosen race and class. Do NOT write a full backstory and do NOT summarise the character."
    ),
}


# --- schema ------------------------------------------------------------------------------------

def _joined(value):
    """Small models sometimes return a list where a string is expected."""
    if isinstance(value, list):
        return " ".join(str(v).strip() for v in value if str(v).strip())
    return value


class CharacterChecklist(BaseModel):
    """Plain text on purpose: race and class are mapped to the real lists in code,
    so "Orc" costs nothing instead of invalidating the whole reply."""
    name: Optional[str] = None
    race: Optional[str] = None
    character_class: Optional[str] = None
    backstory: Optional[str] = None

    @field_validator("*", mode="before")
    @classmethod
    def _text(cls, value):
        return _joined(value)


# --- prompts -----------------------------------------------------------------------------------

def _clip(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(",;:-. ") + "..."


def rules_text() -> str:
    """The approved races and classes, built from the real tables so it can't drift."""
    lines = ["APPROVED RACES:"]
    for race in RACES:
        bonuses = RACIAL_BONUSES[race]
        if len(bonuses) == 6:
            text = "+1 to all abilities"
        else:
            text = ", ".join(f"+{b} {a.title()}" for a, b in bonuses.items())
        lines.append(f"- {race}: {text}")
    lines += ["", "APPROVED CLASSES:"]
    for cls in CLASSES:
        lines.append(f"- {cls}: hit die d{HIT_DIE[cls]}, key ability {STAT_PRIORITIES[cls][0].title()}")
    return "\n".join(lines)


def campaign_context(info: CampaignInfo) -> str:
    lines = [
        f"Campaign: {info.name}",
        f"Setting: {_clip(info.setting_description, 300)}",
        f"Threat: {_clip(info.primary_threat, 200)}",
        f"Quest hook: {_clip(info.starting_quest_hook, 300)}",
    ]
    if info.starting_patron:
        lines.append(f"Starting patron: {info.starting_patron}")
    return "\n".join(lines)


def guide_messages(info: CampaignInfo, checklist: Dict[str, str], history: Sequence[Dict[str, str]]) -> Messages:
    def status(key: str) -> str:
        value = checklist.get(key)
        if not value:
            return "PENDING"
        return "COMPLETED" if key == "backstory" else f"COMPLETED ({value})"

    missing = pending(checklist)
    focus = missing[0] if missing else None
    focus_lines = []
    if focus:
        focus_lines = [
            f"- Ask ONLY about the player's {CHECKLIST_LABELS[focus].lower()}. "
            "Anything marked COMPLETED is final: never ask about it again.",
            f"- {FOCUS_INSTRUCTIONS[focus]}",
        ]

    system = "\n".join(
        [
            "You are the Character Creation Guide. Help the player build a character for this campaign.",
            "",
            rules_text(),
            "",
            "CAMPAIGN",
            campaign_context(info),
            "",
            "Current checklist:",
            *(f"- {CHECKLIST_LABELS[k].lower()}: {status(k)}" for k in CHECKLIST_KEYS),
            "",
            "INSTRUCTIONS",
            *focus_lines,
            "- Only offer races and classes from the approved lists.",
            "- Do not repeat or summarise what the player already chose.",
            "- Always present your question as a multiple-choice list in exactly this format:",
            "  A) [Suggestion 1]",
            "  B) [Suggestion 2]",
            "  C) [Suggestion 3]",
            "  D) Something else / Player's custom input",
            "- Keep the introduction under 50 words, collaborative and immersive.",
        ]
    )
    return [{"role": "system", "content": system}, *history]


def checklist_messages(history: Sequence[Dict[str, str]]) -> Messages:
    system = (
        "You are a strict data extraction system. Read the recent conversation and extract the character "
        "details into the JSON schema. Only extract values the player has explicitly chosen or agreed on. "
        f"race is one of: {', '.join(RACES)}. character_class is one of: {', '.join(CLASSES)}. "
        "Use null for anything not decided. Do not invent details."
    )
    return [{"role": "system", "content": system}, *history[-4:]]


# --- pure helpers ------------------------------------------------------------------------------

def _words(text: str) -> set:
    return set(re.findall(r"[a-z]+", (text or "").lower()))


def find_choice(text: Optional[str], options: Sequence[str], aliases: Optional[Dict[str, str]] = None) -> Optional[str]:
    """The first option named in `text` ('Elven' -> 'Elf'), or None."""
    aliases = aliases or {}
    lowered = {o.lower(): o for o in options}
    for word in re.findall(r"[a-z]+", (text or "").lower()):
        if word in lowered:
            return lowered[word]
        if aliases.get(word) in options:
            return aliases[word]
    return None


def _mentioned(canonical: str, words: set, aliases: Optional[Dict[str, str]] = None) -> bool:
    if canonical.lower() in words:
        return True
    return any(w in words and c == canonical for w, c in (aliases or {}).items())


def clean_name(text: Optional[str]) -> str:
    """A plausible character name, or '' (placeholders, race/class words and sentences are rejected)."""
    name = " ".join(str(text or "").split()).strip(" \"'`.,")
    if not name or len(name) > MAX_NAME_CHARS or len(name.split()) > MAX_NAME_WORDS:
        return ""
    if not re.search(r"[A-Za-z]", name) or name.lower() in _BAD_NAMES or name.lower() in _RESERVED:
        return ""
    return name


def pending(checklist: Dict[str, str]) -> List[str]:
    return [k for k in CHECKLIST_KEYS if not checklist.get(k)]


def status_line(checklist: Dict[str, str]) -> str:
    parts = [f"{CHECKLIST_LABELS[k]}: {'OK' if checklist.get(k) else 'PENDING'}" for k in CHECKLIST_KEYS]
    return "[Checklist] " + " | ".join(parts)


def merge_checklist(current: Dict[str, str], parsed: Optional[CharacterChecklist], source: str) -> Dict[str, str]:
    """Add newly decided values; never replace one that is already settled.

    Every value must be valid (a real name, a race and class from the lists, a
    backstory of MIN_BACKSTORY_WORDS or more) and must appear in `source`, the
    conversation text. The backstory is accepted only once name, race and class
    are all settled, which matches the order the guide asks in.
    """
    merged = dict(current)
    if parsed is None:
        return merged
    words = _words(source)
    lowered = source.lower()

    if "name" not in merged:
        name = clean_name(parsed.name)
        if name and name.lower() in lowered:
            merged["name"] = name
    if "race" not in merged:
        race = find_choice(parsed.race, RACES, RACE_ALIASES)
        if race and _mentioned(race, words, RACE_ALIASES):
            merged["race"] = race
    if "char_class" not in merged:
        char_class = find_choice(parsed.character_class, CLASSES)
        if char_class and _mentioned(char_class, words):
            merged["char_class"] = char_class

    if "backstory" not in merged and all(k in merged for k in CORE_KEYS):
        text = _clip(parsed.backstory or "", BACKSTORY_MAX_CHARS)
        if len(text.split()) >= MIN_BACKSTORY_WORDS and calls.is_grounded(text, source):
            merged["backstory"] = text
    return merged


def apply_choice(current: Dict[str, str], option: str) -> Dict[str, str]:
    """Fill the next pending item straight from a chosen option, with no LLM call.

    Race and class must name a real one (looked at before the colon). A backstory
    option needs MIN_BACKSTORY_WORDS. Names are left to the extractor.
    """
    merged = dict(current)
    missing = pending(merged)
    if not missing:
        return merged
    key = missing[0]
    head = option.split(":", 1)[0]
    if key == "race":
        race = find_choice(head, RACES, RACE_ALIASES)
        if race:
            merged["race"] = race
    elif key == "char_class":
        char_class = find_choice(head, CLASSES)
        if char_class:
            merged["char_class"] = char_class
    elif key == "backstory":
        text = _clip(option, BACKSTORY_MAX_CHARS)
        if len(text.split()) >= MIN_BACKSTORY_WORDS:
            merged["backstory"] = text
    return merged


def starting_items(char_class: str) -> List[Tuple[str, int]]:
    return [*STARTING_KITS.get(char_class, ()), *COMMON_ITEMS]


def save_character(
    slug: str,
    *,
    name: str,
    race: str,
    char_class: str,
    backstory: str,
    campaigns_dir: Optional[Path] = None,
) -> int:
    """Create the player and the starting kit in one transaction. Returns the character id.

    Raises ValueError for an unknown race or class, or if the campaign already has a player.
    """
    stats = calculate_character_stats(race, char_class)    # fails before touching the database
    with GameDB.open(slug, campaigns_dir) as db:
        if db.has_player():
            raise ValueError(f"Campaign '{slug}' already has a player character.")
        campaign = db.get_campaign()
        start = campaign.starting_location_id if campaign else None
        if start is None:
            locations = db.list_locations()
            if not locations:
                raise ValueError(f"Campaign '{slug}' has no locations to start in.")
            start = locations[0].id
        with db.transaction():
            pid = db.add_character(
                Character(
                    name=name, type="player", location_id=start, race=race, char_class=char_class,
                    hp=stats.hp, max_hp=stats.hp, backstory=backstory, **stats.abilities,
                )
            )
            for item, quantity in starting_items(char_class):
                db.add_item(pid, item, quantity)
        return pid


# --- the interactive creator ---------------------------------------------------------------------

class CharacterCreator:
    def __init__(
        self,
        client: LLMClient,
        slug: str,
        *,
        campaigns_dir: Optional[Path] = None,
        input_fn: Callable[[str], str] = input,
        print_fn: Callable[..., None] = print,
    ):
        self.client = client
        self.slug = slug
        self.campaigns_dir = campaigns_dir
        self.input = input_fn
        self.print = print_fn
        self.history: List[Dict[str, str]] = []
        self.checklist: Dict[str, str] = {}
        self.info: Optional[CampaignInfo] = None

    def run(self) -> Optional[int]:
        """Run creation. Returns the new character's id, or None if cancelled or failed."""
        if not self._load_campaign():
            return None
        self._intro()
        try:
            if not self._converse():
                self.print("\nCharacter creation cancelled.")
                return None
            return self._finalize()
        except (KeyboardInterrupt, EOFError):
            self.print("\nCharacter creation cancelled.")
        except LLMError as e:
            self.print(f"\n[LLM error] {e}")
        except ValueError as e:
            self.print(f"\n[Error] {e}")
        return None

    def _load_campaign(self) -> bool:
        try:
            with GameDB.open(self.slug, self.campaigns_dir) as db:
                self.info = db.get_campaign()
                has_player = db.has_player()
        except (FileNotFoundError, SchemaVersionError) as e:
            self.print(f"\n[Error] No usable campaign '{self.slug}': {e}")
            return False
        if self.info is None:
            self.print(f"\n[Error] Campaign '{self.slug}' has no campaign data.")
            return False
        if has_player:
            self.print(f"\n[System] Campaign '{self.slug}' already has a player character.")
            return False
        return True

    def _intro(self) -> None:
        self.print(rules_text())
        self.print("\n=======================================================")
        self.print("          AETHER: CHARACTER CREATION TERMINAL")
        self.print("=======================================================")
        self.print(f"Campaign: {self.info.name}")
        self.print("Tell me about your character: name, race, class and where they come from.")
        self.print("Once everything is settled the character is saved automatically.")
        self.print("Type 'complete' to finish early (when nothing is pending) or 'cancel' to leave.\n")

    def _converse(self) -> bool:
        while True:
            text = self.input("\nYou: ").strip()
            if not text:
                continue
            command = text.lower()
            if command == "cancel":
                return False
            if command == "complete":
                missing = pending(self.checklist)
                if not missing:
                    return True
                labels = ", ".join(CHECKLIST_LABELS[k] for k in missing)
                self.print(f"[System] Still pending: {labels}. Keep chatting to settle them.")
                continue

            choice = resolve_choice(text, last_assistant_reply(self.history))
            if choice == CUSTOM_CHOICE:
                self.print("[System] Go ahead and type your own answer.")
                continue

            if choice is not None:
                self.history.append({"role": "user", "content": f"I choose: {choice}"})
                before = dict(self.checklist)
                self.checklist = apply_choice(self.checklist, choice)
                if self.checklist == before:          # e.g. a name option: let the extractor read it
                    self._update_checklist()
            else:
                self.history.append({"role": "user", "content": text})
                self._update_checklist()
            self.print(status_line(self.checklist))
            if not pending(self.checklist):
                self.print("\n[System] All core traits are settled.")
                return True
            self._guide_reply()

    def _update_checklist(self) -> None:
        parsed = self.client.structured(
            "referee",
            checklist_messages(self.history),
            CharacterChecklist,
            max_tokens=CHECKLIST_MAX_TOKENS,
            temperature=0.0,
        )
        source = " ".join(m["content"] for m in self.history)
        self.checklist = merge_checklist(self.checklist, parsed, source)

    def _guide_reply(self) -> None:
        self.print("\nGuide: ", end="", flush=True)
        pieces: List[str] = []
        for piece in self.client.stream("creation", guide_messages(self.info, self.checklist, self.history)):
            self.print(piece, end="", flush=True)
            pieces.append(piece)
        self.print()
        reply = "".join(pieces).strip()
        if reply:
            self.history.append({"role": "assistant", "content": reply})
        else:
            self.print("[System] The guide had nothing to say. Keep describing your character.")

    def _finalize(self) -> int:
        c = self.checklist
        self.print("\nSaving your character...")
        pid = save_character(
            self.slug, name=c["name"], race=c["race"], char_class=c["char_class"],
            backstory=c["backstory"], campaigns_dir=self.campaigns_dir,
        )
        stats = calculate_character_stats(c["race"], c["char_class"])
        self.print("\n=======================================================")
        self.print("          SUCCESS: CHARACTER REGISTERED")
        self.print("=======================================================")
        self.print(f"Name:      {c['name']} ({c['race']} {c['char_class']})")
        self.print(f"HP:        {stats.hp}")
        self.print("Abilities: " + " | ".join(f"{a[:3].upper()} {v}" for a, v in stats.abilities.items()))
        self.print("Kit:       " + ", ".join(f"{n} x{q}" if q > 1 else n for n, q in starting_items(c["char_class"])))
        self.print("=======================================================\n")
        return pid


def _pick_campaign() -> Optional[str]:
    slugs = GameDB.list_campaigns()
    if not slugs:
        print("No campaigns found. Create one first: python -m aether.creation.campaign")
        return None
    print("\nCampaigns:")
    for i, slug in enumerate(slugs, start=1):
        print(f"{i}. {slug}")
    raw = input(f"Select (1-{len(slugs)}): ").strip()
    return slugs[int(raw) - 1] if raw.isdigit() and 1 <= int(raw) <= len(slugs) else None


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    slug = args[0] if args else _pick_campaign()
    if not slug:
        return 1
    return 0 if CharacterCreator(LLMClient(), slug).run() is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
