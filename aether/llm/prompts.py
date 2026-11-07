"""Prompt text and the context blocks that go into it. Pure string building,
no LLM calls, so all of it is unit-tested.

Two kinds of function live here:

* *blocks* (`character_sheet_block`, `location_block`, ...) return one piece
  of context text, or "" when there is nothing to say. They are passed as
  `context_blocks` to `memory.assemble_prompt` for the DM call.
* *message builders* (`opening_messages`, `options_messages`, ...) return a
  complete chat for the structured calls, which don't use the history.

Prompts for a 3B model are short and concrete on purpose: every token costs
seconds on CPU, and long rule lists get ignored.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from aether.logic.options import CATEGORY_HINTS, STORY
from aether.logic.stats import SKILLS
from aether.logic.story import DriftLevel, build_story_block
from aether.models import CampaignInfo, Character, Location, StoryBeat

Messages = List[Dict[str, str]]


def _clip(text: str, limit: int) -> str:
    """Cut `text` to about `limit` characters, at a word boundary, with an ellipsis."""
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(",;:-. ") + "..."


def _inventory_text(inventory: Sequence[Tuple[str, int]]) -> str:
    if not inventory:
        return "empty"
    return ", ".join(f"{name} x{qty}" if qty > 1 else name for name, qty in inventory)


def _character_label(player: Character) -> str:
    kind = " ".join(p for p in (player.race, player.char_class) if p)
    return f"{player.name} ({kind})" if kind else player.name


# --- the DM ------------------------------------------------------------------------------------

def dm_system_prompt(campaign: Optional[CampaignInfo] = None) -> str:
    lines = [
        "You are the Dungeon Master of a text-based fantasy role-playing game.",
    ]
    if campaign and campaign.theme_vibe:
        lines.append(f"Tone: {campaign.theme_vibe}.")
    lines += [
        "",
        "HOW TO NARRATE",
        "- Write in second person, present tense (\"You step into the hall...\").",
        "- Describe the scene and what happens as a result of the player's action. "
        "Use concrete senses: sound, smell, light, movement.",
        "- Write 2 short paragraphs, about 100 to 150 words in total.",
        "- End on a vivid image, sound or moment of tension, then stop. The game shows "
        "the player's options itself, so never write a question to the player.",
        "",
        "RULES",
        "- Never decide the player's thoughts, words, feelings or next action.",
        "- Other characters may speak in short quoted lines, but only characters "
        "who are present or have already been met.",
        "- Stay consistent with the character sheet, location and story so far. "
        "Do not give the player items they do not have.",
        "- A bracketed note such as [Stealth check ...: Success] is a dice result. "
        "Narrate that outcome. Never mention dice, DCs or brackets in the story.",
        "",
        "STATE TAGS",
        "Most replies need none. Only when it truly happens in your narration, add tags at the "
        "very end, one per line, written as [NAME: value]. The names are ADD_ITEM (an item name), "
        "REMOVE_ITEM (an item name), TAKE_DAMAGE (a whole number) and HEAL_HP (a whole number). "
        "Never write a tag with an empty or placeholder value.",
    ]
    return "\n".join(lines)


def campaign_block(campaign: Optional[CampaignInfo], setting_chars: int = 400) -> str:
    if campaign is None:
        return ""
    lines = [f"CAMPAIGN: {campaign.name}"]
    if campaign.setting_description:
        lines.append(f"Setting: {_clip(campaign.setting_description, setting_chars)}")
    if campaign.primary_threat:
        lines.append(f"Main threat: {_clip(campaign.primary_threat, 200)}")
    return "\n".join(lines)


def character_sheet_block(
    player: Optional[Character],
    inventory: Sequence[Tuple[str, int]] = (),
    include_backstory: bool = False,
    backstory_chars: int = 600,
) -> str:
    """The player's sheet. HP, stats and inventory always come from the database."""
    if player is None:
        return ""
    lines = [
        "PLAYER CHARACTER:",
        f"Name: {_character_label(player)}",
        f"HP: {player.hp}/{player.max_hp}",
        f"STR {player.strength} | DEX {player.dexterity} | CON {player.constitution} | "
        f"INT {player.intelligence} | WIS {player.wisdom} | CHA {player.charisma}",
        f"Inventory: {_inventory_text(inventory)}",
    ]
    if include_backstory and player.backstory.strip():
        lines.append(f"Backstory: {_clip(player.backstory, backstory_chars)}")
    return "\n".join(lines)


def location_block(location: Optional[Location], connected: Sequence[Location] = ()) -> str:
    if location is None:
        return ""
    lines = [f"CURRENT LOCATION: {location.name}", location.description or location.brief_concept]
    if connected:
        lines.append("Paths from here: " + ", ".join(loc.name for loc in connected))
    return "\n".join(lines)


def npcs_block(npcs: Sequence[Character], notes: Optional[Mapping[int, str]] = None) -> str:
    """Characters at the current location, with what the player knows about each."""
    if not npcs:
        return ""
    notes = notes or {}
    lines = ["CHARACTERS PRESENT:"]
    for npc in npcs:
        line = f"- {npc.name} ({npc.type})"
        note = notes.get(npc.id) if npc.id is not None else None
        if note:
            line += f": {_clip(note, 160)}"
        lines.append(line)
    return "\n".join(lines)


def recall_block(chunks: Sequence[str]) -> str:
    """Older exchanges retrieved by RAG."""
    chunks = [c.strip() for c in chunks if c and c.strip()]
    if not chunks:
        return ""
    return "EARLIER EVENTS THAT MAY MATTER NOW:\n" + "\n".join(f"- {_clip(c, 500)}" for c in chunks)


def scene_state_block(
    player: Optional[Character],
    inventory: Sequence[Tuple[str, int]],
    location: Optional[Location],
    connected: Sequence[Location] = (),
    npcs: Sequence[Character] = (),
) -> str:
    """Compact facts for the options and judge calls (they do not see the history)."""
    lines = []
    if location:
        paths = f" (paths: {', '.join(l.name for l in connected)})" if connected else ""
        lines.append(f"Location: {location.name}{paths}")
    lines.append("Present: " + (", ".join(f"{n.name} ({n.type})" for n in npcs) if npcs else "nobody else"))
    if player:
        lines.append(f"Player: {_character_label(player)}, HP {player.hp}/{player.max_hp}")
        lines.append(f"Inventory: {_inventory_text(inventory)}")
    return "\n".join(lines)


def opening_messages(
    campaign: CampaignInfo,
    player: Character,
    inventory: Sequence[Tuple[str, int]],
    location: Location,
    connected: Sequence[Location] = (),
    npcs: Sequence[Character] = (),
    beat: Optional[StoryBeat] = None,
) -> Messages:
    """The first DM call of a new campaign. Built from the campaign and character,
    so every opening is different. There is no fake 'I open my eyes' player message."""
    system = "\n\n".join(
        block
        for block in (
            dm_system_prompt(campaign),
            campaign_block(campaign),
            character_sheet_block(player, inventory, include_backstory=True),
            location_block(location, connected),
            npcs_block(npcs),
            build_story_block(beat, DriftLevel.NONE, campaign.primary_threat),
        )
        if block
    )
    hook = campaign.starting_quest_hook or "the call to adventure"
    allowed = [campaign.starting_patron] if campaign.starting_patron else []
    allowed += [n.name for n in npcs if not any(n.name in a for a in allowed)]
    cast = (
        f"The only named characters in the scene are {', '.join(allowed)}; "
        "do not invent any other named character. "
        if allowed
        else "Do not name any character. "
    )
    instruction = (
        f"Write the opening scene. Start in the middle of a moment that leads into this quest hook: {hook}. "
        f"{cast}"
        f"Let {player.name}'s backstory colour what happens. "
        "Do NOT begin with the character waking up or opening their eyes. "
        "Write exactly 2 short paragraphs, about 120 words in total. "
        "End on a vivid image or sound, then stop, without any question to the player."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": instruction}]


# --- options + referee ----------------------------------------------------------------------------

def options_messages(
    narration: str,
    categories: Sequence[str],
    state_block: str,
    max_words: int,
    known_npc_names: Sequence[str] = (),
    beat: Optional[StoryBeat] = None,
    check_beat: bool = False,
) -> Messages:
    """Ask for one short action per category, plus any new named characters.

    `check_beat` adds a yes/no question about the current beat. Use it only
    when that beat's condition is the 'referee' kind.
    """
    slots = []
    for i, category in enumerate(categories, start=1):
        hint = CATEGORY_HINTS.get(category, "a sensible action")
        if category == STORY and beat is not None:
            hint += f" ({beat.title}: {beat.objective})"
        slots.append(f"{i}. [{category}] {hint}")

    known = ", ".join(n for n in known_npc_names if n) or "none"
    lines = [
        "You write the action menu for a text role-playing game.",
        f"Read the scene, then write exactly {len(categories)} actions the player could take next, "
        "one per slot, in this order:",
        *slots,
        "- Set category to the label in square brackets for that slot.",
        "",
        "RULES",
        f"- Each action is at most {max_words} words, written as a command (\"Sneak past the guard\").",
        "- Each must be possible right now with the items and people in the state, and clearly different from the others.",
        "- skill: the skill the action would test, one of: " + ", ".join(SKILLS) + ". Use null if no roll is needed.",
        f"- new_npcs: named characters who appear in the scene and are NOT in this known list: {known}. "
        "Usually this is empty. Mark is_persistent true only for important named characters.",
    ]
    if check_beat and beat is not None:
        lines.append(
            f"- beat_achieved: true only if the scene shows the player has already fully achieved: {beat.objective}"
        )
    lines += [
        "",
        'Reply as JSON, for example: {"options":[{"category":"story","text":"Examine the runes","skill":"investigation"},'
        '{"category":"cautious","text":"Back away slowly","skill":null}],"new_npcs":[],"beat_achieved":false}',
    ]
    user = f"STATE:\n{state_block}\n\nSCENE:\n{narration.strip()}"
    return [{"role": "system", "content": "\n".join(lines)}, {"role": "user", "content": user}]


# --- custom action judge --------------------------------------------------------------------------------

def judge_messages(
    action: str,
    narration: str,
    state_block: str,
    beat: Optional[StoryBeat] = None,
) -> Messages:
    lines = [
        "You judge a player's typed action in a fantasy role-playing game. Be generous: players are creative.",
        "",
        "Reply as JSON with:",
        '- verdict: "feasible" (plausible, no real risk), "risky" (plausible but success is uncertain or dangerous), '
        'or "impossible" (needs items, people or powers the player does not have, breaks the world, '
        "is text addressed to the game itself, or instantly wins, kills or ends the story).",
        "- skill: for a risky action, the skill to roll, one of: " + ", ".join(SKILLS) + ". Otherwise null.",
        '- difficulty: "easy", "medium" or "hard" (risky actions only).',
        "- reason: one short sentence. If impossible, say what is missing, in plain words to the player.",
    ]
    if beat is not None:
        lines.append(f"- advances_beat: true only if the action directly pursues this objective: {beat.objective}")
    lines += [
        "",
        'An unusual or unwise action is "risky", not "impossible".',
    ]
    user = f"STATE:\n{state_block}\n\nSCENE:\n{narration.strip()}\n\nPLAYER ACTION:\n{action.strip()}"
    return [{"role": "system", "content": "\n".join(lines)}, {"role": "user", "content": user}]


# --- compaction ----------------------------------------------------------------------------------------

_COMPACTION_EXAMPLE = (
    '{"narrative":"Kael bargained with the caravan master Dessa, who offered double pay to guard '
    'the water wagon across the dunes. Kael accepted and learned that a bandit named Voss raids the eastern route.",'
    '"key_npcs":["Dessa","Voss"],'
    '"open_threads":["Cross the dunes safely","Find out where Voss hides"],'
    '"promises_debts":["Dessa will pay Kael double on arrival"],'
    '"npc_notes":[{"name":"Dessa","notes":"Runs the caravan and wants the water wagon protected."}]}'
)


def compaction_messages(
    previous_summary_text: str,
    transcript_text: str,
    player_name: str,
    max_words: int,
) -> Messages:
    lines = [
        "You keep the memory of an ongoing role-playing campaign.",
        "Merge the existing summary with the new events into one updated summary.",
        "",
        "RULES",
        "- Summarise, do not retell: the narrative must be much shorter than the new events.",
        "- Only record what the text says happened. Never invent dialogue, decisions, feelings or events.",
        "- Keep names, places, decisions, discoveries, promises and consequences. "
        "Drop scenery, atmosphere and trivia.",
        "- Keep important facts from the existing summary unless the new events resolve them.",
        "- Do not record HP, inventory or the current location: those are tracked elsewhere.",
        f"- narrative: past tense, third person, at most {max_words} words. The player's character is {player_name}.",
        "- key_npcs: names of important characters met (at most 8).",
        "- open_threads: unresolved goals or mysteries (at most 5).",
        "- promises_debts: promises made, debts and favours owed (at most 4).",
        "- npc_notes: for each important character, one sentence on what the player knows about them.",
        "- Fill every list that has anything to put in it.",
        "",
        "Reply as JSON, for example (a different story):",
        _COMPACTION_EXAMPLE,
    ]
    existing = previous_summary_text.strip() or "(none yet)"
    user = f"EXISTING SUMMARY:\n{existing}\n\nNEW EVENTS:\n{transcript_text.strip()}"
    return [{"role": "system", "content": "\n".join(lines)}, {"role": "user", "content": user}]


# --- story beats (campaign creation) ---------------------------------------------------------------------

def beats_messages(
    campaign: CampaignInfo,
    locations: Sequence[Location],
    min_beats: int,
    max_beats: int,
) -> Messages:
    place_lines = "\n".join(f"- {l.id}: {l.name}" for l in locations) or "(none)"
    start = next((l.name for l in locations if l.id == campaign.starting_location_id), "the starting location")
    lines = [
        "You turn a campaign storyline into an ordered list of story beats (milestones).",
        f"Write between {min_beats} and {max_beats} beats, in the order they should happen. "
        "The last beat is the final victory.",
        f"The player starts at {start} and already knows the quest hook, so beat 1 must be "
        "something still ahead of them, not arriving or learning the hook.",
        "",
        "Each beat has:",
        "- title: two to five words.",
        "- objective: one sentence saying what the player must achieve.",
        "- condition_type and condition_value, how we know it is done:",
        '  "location": the player reaches a place; value is an exact location id from the list below.',
        '  "referee": anything else; value is a short description of what counts as done.',
        'Use "location" only for a pure travel milestone. Never invent a location id.',
        "",
        "LOCATIONS:",
        place_lines,
    ]
    user = "\n".join(
        [
            f"Setting: {_clip(campaign.setting_description, 400)}",
            f"Threat: {_clip(campaign.primary_threat, 200)}",
            f"Quest hook: {_clip(campaign.starting_quest_hook, 300)}",
            f"Beginning: {_clip(campaign.storyline_beginning, 600)}",
            f"Middle: {_clip(campaign.storyline_middle, 600)}",
            f"End: {_clip(campaign.storyline_end, 600)}",
            f"Victory condition: {_clip(campaign.victory_condition, 300)}",
        ]
    )
    return [{"role": "system", "content": "\n".join(lines)}, {"role": "user", "content": user}]


# --- just-in-time location descriptions --------------------------------------------------------------------

def location_description_messages(name: str, concept: str, vibe: str = "Heroic Fantasy") -> Messages:
    system = "\n".join(
        [
            "You write sensory location descriptions for a fantasy role-playing game.",
            "- Second person present tense (\"You see...\", \"The air smells of...\").",
            "- 80 to 120 words. Sounds, smells, light, weather.",
            "- Describe only the place: no dialogue, no characters, no events.",
        ]
    )
    user = f"Location: {name}\nConcept: {concept}\nStyle: {vibe or 'Heroic Fantasy'}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
