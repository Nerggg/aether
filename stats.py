"""Character maths and dice. Pure functions, no database, no LLM.

Moved out of the old character creator so it can be tested on its own.
Randomness is injectable (`rng`) so tests can roll exact numbers.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Dict, Mapping, Optional

ABILITIES = ("strength", "dexterity", "constitution", "intelligence", "wisdom", "charisma")

RACES = ("Human", "Elf", "Dwarf", "Halfling", "Dragonborn", "Gnome", "Tiefling")
CLASSES = ("Fighter", "Rogue", "Wizard", "Cleric", "Paladin", "Ranger", "Bard")

STANDARD_ARRAY = (15, 14, 13, 12, 10, 8)

# Best score goes to the first ability, next best to the second, and so on.
STAT_PRIORITIES: Dict[str, tuple] = {
    "Fighter": ("strength", "constitution", "dexterity", "wisdom", "charisma", "intelligence"),
    "Rogue": ("dexterity", "intelligence", "charisma", "constitution", "wisdom", "strength"),
    "Wizard": ("intelligence", "dexterity", "constitution", "wisdom", "charisma", "strength"),
    "Cleric": ("wisdom", "constitution", "strength", "intelligence", "dexterity", "charisma"),
    "Paladin": ("strength", "charisma", "constitution", "wisdom", "dexterity", "intelligence"),
    "Ranger": ("dexterity", "wisdom", "constitution", "strength", "intelligence", "charisma"),
    "Bard": ("charisma", "dexterity", "constitution", "wisdom", "intelligence", "strength"),
}

RACIAL_BONUSES: Dict[str, Dict[str, int]] = {
    "Human": {a: 1 for a in ABILITIES},
    "Elf": {"dexterity": 2},
    "Dwarf": {"constitution": 2},
    "Halfling": {"dexterity": 2},
    "Dragonborn": {"strength": 2, "charisma": 1},
    "Gnome": {"intelligence": 2},
    "Tiefling": {"charisma": 2, "intelligence": 1},
}

HIT_DIE: Dict[str, int] = {
    "Fighter": 10, "Paladin": 10, "Ranger": 10,
    "Rogue": 8, "Cleric": 8, "Bard": 8,
    "Wizard": 6,
}

SKILL_ABILITIES: Dict[str, str] = {
    "athletics": "strength",
    "acrobatics": "dexterity", "sleight of hand": "dexterity", "stealth": "dexterity",
    "arcana": "intelligence", "history": "intelligence", "investigation": "intelligence",
    "nature": "intelligence", "religion": "intelligence",
    "animal handling": "wisdom", "insight": "wisdom", "medicine": "wisdom",
    "perception": "wisdom", "survival": "wisdom",
    "deception": "charisma", "intimidation": "charisma", "performance": "charisma",
    "persuasion": "charisma",
}
SKILLS = tuple(SKILL_ABILITIES)

_ABBREVIATIONS = {"str": "strength", "dex": "dexterity", "con": "constitution",
                  "int": "intelligence", "wis": "wisdom", "cha": "charisma"}

# Difficulty class by how risky the judge or the option says an action is.
DIFFICULTY_DC: Dict[str, int] = {"easy": 10, "medium": 13, "hard": 16}


def ability_modifier(score: int) -> int:
    return (score - 10) // 2


def _lookup(table: Mapping[str, object], key: str, kind: str) -> str:
    wanted = (key or "").strip().lower()
    for name in table:
        if name.lower() == wanted:
            return name
    raise ValueError(f"Unknown {kind} {key!r}. Expected one of: {', '.join(table)}")


@dataclass(frozen=True)
class CharacterStats:
    abilities: Dict[str, int]
    hp: int  # starting and maximum HP


def calculate_character_stats(race: str, char_class: str) -> CharacterStats:
    """Standard array assigned by class priority, plus racial bonuses.

    HP is the class hit die plus the Constitution modifier (never below 1).
    Raises ValueError for a race or class that isn't in RACES / CLASSES.
    """
    race = _lookup(RACIAL_BONUSES, race, "race")
    char_class = _lookup(STAT_PRIORITIES, char_class, "class")

    abilities = dict(zip(STAT_PRIORITIES[char_class], STANDARD_ARRAY))
    for ability, bonus in RACIAL_BONUSES[race].items():
        abilities[ability] += bonus

    hp = max(1, HIT_DIE[char_class] + ability_modifier(abilities["constitution"]))
    return CharacterStats(abilities=abilities, hp=hp)


def normalize_skill(text: Optional[str]) -> Optional[str]:
    """Canonical lowercase skill or ability name, or None if it isn't one.

    Accepts 'Sleight_of_Hand', 'sleight-of-hand', 'DEX', and so on. LLMs are
    inconsistent about this, so everything that touches a skill goes through here.
    """
    if not text:
        return None
    t = re.sub(r"[_\-\s]+", " ", text.strip().lower())
    if t in SKILL_ABILITIES or t in ABILITIES:
        return t
    return _ABBREVIATIONS.get(t)


def ability_for_skill(skill: str) -> str:
    name = normalize_skill(skill)
    if name is None:
        raise ValueError(f"Unknown skill {skill!r}")
    return SKILL_ABILITIES.get(name, name)


def skill_modifier(abilities: Mapping[str, int], skill: str) -> int:
    """Ability modifier for a skill check (no proficiency bonus in this game)."""
    return ability_modifier(abilities[ability_for_skill(skill)])


@dataclass(frozen=True)
class RollResult:
    skill: str
    ability: str
    die: int
    modifier: int
    total: int
    dc: int
    success: bool


def roll_check(
    abilities: Mapping[str, int],
    skill: str,
    difficulty: str = "medium",
    rng: Optional[random.Random] = None,
) -> RollResult:
    """Roll a d20 against a difficulty class."""
    if difficulty not in DIFFICULTY_DC:
        raise ValueError(f"Unknown difficulty {difficulty!r}. Expected one of {tuple(DIFFICULTY_DC)}")
    name = normalize_skill(skill)
    if name is None:
        raise ValueError(f"Unknown skill {skill!r}")
    ability = SKILL_ABILITIES.get(name, name)
    modifier = ability_modifier(abilities[ability])
    die = (rng or random).randint(1, 20)
    dc = DIFFICULTY_DC[difficulty]
    total = die + modifier
    return RollResult(name, ability, die, modifier, total, dc, total >= dc)


def format_roll(result: RollResult) -> str:
    """One-line summary, stored as a `system` message in the transcript."""
    sign = "+" if result.modifier >= 0 else "-"
    outcome = "Success" if result.success else "Failure"
    return (
        f"{result.skill.title()} check ({result.ability[:3].upper()}): "
        f"d20 {result.die} {sign} {abs(result.modifier)} = {result.total} "
        f"vs DC {result.dc} - {outcome}"
    )
