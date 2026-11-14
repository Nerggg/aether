"""Character creation as a form that code drives.

Code owns the four fields (name, race, class, backstory) and renders every menu
itself, so it always knows which question the player is answering. The model
only writes backstory options (with built-in fallbacks if it fails).

A reply is read as a menu number, then (for race and class) a name from the
list, then a "named X" shortcut at the name menu. Anything else prints a hint
and shows the menu again.

Run it on its own:  python -m aether.creation.character <campaign_slug>
"""

from __future__ import annotations

import random
import re
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from aether.creation.names import pick_patron_name
from aether.db import GameDB, SchemaVersionError
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
from aether.creation.forms import CUSTOM, IDEA, REGEN, SURPRISE, Cancelled, Reply, ask_menu, match_closed, parse_number

CHECKLIST_KEYS = ("name", "race", "char_class", "backstory")
CHECKLIST_LABELS = {"name": "Name", "race": "Race", "char_class": "Class", "backstory": "Backstory"}

MAX_NAME_CHARS = 40
MAX_NAME_WORDS = 4
MIN_BACKSTORY_WORDS = 6
BACKSTORY_MAX_CHARS = 600
OPTION_MAX_CHARS = 300
IDEA_MAX_CHARS = 200
NOTES_MAX_CHARS = 300
NAME_COUNT = 3
BACKSTORY_COUNT = 3
BACKSTORY_MAX_TOKENS = 400

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
_NAME_STOP = frozenset({"and", "who", "from", "with", "a", "an", "the", "that", "which", "but", "or"})
_NAME_PHRASE = re.compile(
    r"\b(?:named|called|name is|name's|names)\s+([A-Za-z][A-Za-z'\-]*(?:\s+[A-Za-z][A-Za-z'\-]*){0,3})",
    re.IGNORECASE,
)


# --- schemas -----------------------------------------------------------------------------------


class BackstoryOptions(BaseModel):
    options: List[str] = Field(default_factory=list)


# --- small pure helpers --------------------------------------------------------------------------

def _clip(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(",;:-. ") + "..."


def _clip_tail(text: str, limit: int) -> str:
    """Keep the END of `text` (the newest ideas) when it is longer than `limit`."""
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return "..." + text[-limit:].split(" ", 1)[-1]


def clean_name(text: Optional[str]) -> str:
    """A plausible character name, or '' (placeholders, race/class words and sentences are rejected)."""
    name = " ".join(str(text or "").split()).strip(" \"'`.,")
    if not name or len(name) > MAX_NAME_CHARS or len(name.split()) > MAX_NAME_WORDS:
        return ""
    if not re.search(r"[A-Za-z]", name) or name.lower() in _BAD_NAMES or name.lower() in _RESERVED:
        return ""
    return name


def _tidy_name(name: str) -> str:
    return name.title() if name.islower() else name


def name_from_phrase(text: str) -> str:
    """'i want a human named baron and a ranger' -> 'Baron'. '' if there is no 'named X' phrase."""
    m = _NAME_PHRASE.search(text or "")
    if not m:
        return ""
    words: List[str] = []
    for w in m.group(1).split():
        if w.lower() in _NAME_STOP:
            break
        words.append(w)
    return _tidy_name(clean_name(" ".join(words)))


def extract_name(text: str) -> str:
    """A name from a 'named X' phrase, or the whole text if it is itself a valid name."""
    return name_from_phrase(text) or _tidy_name(clean_name(text))


def valid_backstory(text: str) -> str:
    text = _clip(text, BACKSTORY_MAX_CHARS)
    return text if len(text.split()) >= MIN_BACKSTORY_WORDS else ""


def race_line(race: str) -> str:
    bonuses = RACIAL_BONUSES[race]
    if len(bonuses) == 6:
        return f"{race}: +1 to all abilities"
    return f"{race}: " + ", ".join(f"+{b} {a.title()}" for a, b in bonuses.items())


def class_line(char_class: str) -> str:
    return f"{char_class}: hit die d{HIT_DIE[char_class]}, key ability {STAT_PRIORITIES[char_class][0].title()}"


def starting_items(char_class: str) -> List[Tuple[str, int]]:
    return [*STARTING_KITS.get(char_class, ()), *COMMON_ITEMS]


def suggest_names(rng: random.Random, count: int = NAME_COUNT, exclude: Sequence[str] = ()) -> List[str]:
    names: List[str] = []
    while len(names) < count:
        names.append(pick_patron_name(rng, exclude=[*names, *exclude]))
    return names


# --- prompts and the two model jobs ----------------------------------------------------------------

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


def backstory_messages(
    info: CampaignInfo, name: str, race: str, char_class: str, notes: str, count: int
) -> Messages:
    system = "\n".join(
        [
            "You write short character backstories for a fantasy role-playing game.",
            f"Write exactly {count} different options. Each is one or two sentences, under 35 words, "
            "in third person, uses the character's name and ties into the campaign.",
            "Do not mention game rules or numbers.",
            "",
            "Reply as JSON in this shape (replace the <placeholders>):",
            '{"options":["<backstory 1>","<backstory 2>"]}',
        ]
    )
    lines = [campaign_context(info), f"Character: {name}, a {race} {char_class}"]
    if notes.strip():
        lines.append(f"Player's ideas: {_clip_tail(notes, NOTES_MAX_CHARS)}")
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n".join(lines)}]


def fallback_backstories(name: str, race: str, char_class: str) -> List[str]:
    return [
        f"{name} the {race} {char_class} lost someone dear to the trouble now spreading across the land and seeks answers.",
        f"{name} left a quiet life behind after hearing rumours of danger, and now answers the call to adventure.",
        f"{name} is a wandering {race} {char_class} with a debt to repay and little reason to trust strangers.",
    ]


def generate_backstories(
    client: LLMClient, info: CampaignInfo, name: str, race: str, char_class: str, notes: str, count: int
) -> List[str]:
    """`count` distinct backstory options. Model output is cleaned, and built-in ones fill any gap."""
    response = client.structured(
        "creation", backstory_messages(info, name, race, char_class, notes, count),
        BackstoryOptions, max_tokens=BACKSTORY_MAX_TOKENS,
    )
    result: List[str] = []
    for raw in (response.options if response else []):
        text = _clip(str(raw), OPTION_MAX_CHARS)
        if len(text.split()) >= MIN_BACKSTORY_WORDS and text.lower() not in {r.lower() for r in result}:
            result.append(text)
    for fallback in fallback_backstories(name, race, char_class):
        if len(result) >= count:
            break
        if fallback.lower() not in {r.lower() for r in result}:
            result.append(fallback)
    return result[:count]


# --- saving --------------------------------------------------------------------------------------------

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
        rng: Optional[random.Random] = None,
    ):
        self.client = client
        self.slug = slug
        self.campaigns_dir = campaigns_dir
        self.input = input_fn
        self.print = print_fn
        self.rng = rng or random.Random()
        self.values: Dict[str, str] = {}
        self.ideas: List[str] = []
        self.info: Optional[CampaignInfo] = None

    @property
    def notes(self) -> str:
        return "; ".join(self.ideas)

    def _add_idea(self, text: str) -> None:
        idea = _clip(text, IDEA_MAX_CHARS)
        if idea:
            self.ideas.append(idea)

    def run(self) -> Optional[int]:
        """Run creation. Returns the new character's id, or None if cancelled or failed."""
        if not self._load_campaign():
            return None
        self._intro()
        try:
            self._collect()
            return self._finalize()
        except (Cancelled, KeyboardInterrupt, EOFError):
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
        self.print("\n=======================================================")
        self.print("          AETHER: CHARACTER CREATION TERMINAL")
        self.print("=======================================================")
        self.print(f"Campaign: {self.info.name}")
        self.print("You will choose a name, race, class and backstory. Answer with a number")
        self.print("(for race and class you can also type the name). Type 'cancel' at any time to leave.\n")

    # --- input helpers ---

    def _read(self, prompt: str = "\nYou: ", allow_empty: bool = False) -> str:
        while True:
            text = self.input(prompt).strip()
            if text.lower() == "cancel":
                raise Cancelled()
            if text or allow_empty:
                return text

    def _done(self, key: str, value: str) -> None:
        self.values[key] = value
        self.print(f"[OK] {CHECKLIST_LABELS[key]}: {value}")

    # --- the form ---

    def _collect(self) -> None:
        first = self._read("Any ideas for your character? (optional, press Enter to skip): ", allow_empty=True)
        self._add_idea(first)
        if first:
            self._prefill(first)
        if "name" not in self.values:
            self._done("name", self._ask_name())
        if "race" not in self.values:
            self._done("race", self._ask_closed("race", list(RACES), [race_line(r) for r in RACES], RACE_ALIASES))
        if "char_class" not in self.values:
            self._done("char_class", self._ask_closed("char_class", list(CLASSES), [class_line(c) for c in CLASSES]))
        if "backstory" not in self.values:
            self._done("backstory", self._ask_backstory())

    def _prefill(self, notes: str) -> None:
        """Use the notes only for things they name unambiguously."""
        name = name_from_phrase(notes)
        if name:
            self._done("name", name)
        race = match_closed(notes, RACES, RACE_ALIASES)
        if race:
            self._done("race", race)
        char_class = match_closed(notes, CLASSES)
        if char_class:
            self._done("char_class", char_class)

    def _ask_name(self) -> str:
        options = suggest_names(self.rng)
        extras = [(CUSTOM, "Type my own name"), (REGEN, "New names"), (SURPRISE, "Surprise me")]
        while True:
            rep = ask_menu(
                self._read, self.print, "Choose a name:", options, extras,
                shortcut=lambda r: Reply("text", text=r) if name_from_phrase(r) else None,
            )
            if rep.kind == "option":
                return options[rep.index]
            if rep.kind == SURPRISE:
                return self.rng.choice(options)
            if rep.kind == REGEN:
                options = suggest_names(self.rng, exclude=options)
                continue
            typed = self._read("Name: ") if rep.kind == CUSTOM else rep.text
            name = extract_name(typed)
            if name:
                return name
            self.print("[System] That does not look like a name. Try again.")

    def _ask_closed(
        self, key: str, options: List[str], lines: List[str], aliases: Optional[Dict[str, str]] = None
    ) -> str:
        label = CHECKLIST_LABELS.get(key, key).lower()
        rep = ask_menu(
            self._read, self.print, f"Choose your {label}:", options, [(SURPRISE, "Surprise me")],
            lines=lines, closed=True, aliases=aliases,
        )
        return options[rep.index] if rep.kind == "option" else self.rng.choice(options)

    def _ask_backstory(self) -> str:
        options = self._backstory_options(BACKSTORY_COUNT)
        extras = [
            (CUSTOM, "Write my own"),
            (REGEN, "New backstories"),
            (IDEA, "New backstories from my idea"),
            (SURPRISE, "Surprise me"),
        ]
        while True:
            rep = ask_menu(self._read, self.print, "Choose a backstory:", options, extras)
            if rep.kind == "option":
                return options[rep.index]
            if rep.kind == SURPRISE:
                return self.rng.choice(options)
            if rep.kind == IDEA:
                self._add_idea(self._read("Your idea: "))
                options = self._backstory_options(BACKSTORY_COUNT)
                continue
            if rep.kind == REGEN:
                options = self._backstory_options(BACKSTORY_COUNT)
                continue
            text = valid_backstory(self._read("Backstory: "))
            if text:
                return text
            self.print(f"[System] Write at least {MIN_BACKSTORY_WORDS} words.")

    def _backstory_options(self, count: int) -> List[str]:
        v = self.values
        self.print("\n[System] Writing backstory ideas...")
        return generate_backstories(self.client, self.info, v["name"], v["race"], v["char_class"], self.notes, count)

    def _finalize(self) -> int:
        c = self.values
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
