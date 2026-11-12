"""Campaign creation as a form that code drives.

Code owns the three fields (setting, threat, quest hook) and asks them in
order, so it always knows which question the player is answering. For each
field one structured call writes three options that fit the earlier answers,
and code renders them as a numbered menu. A reply is read by
`character.interpret_reply`: a number first, long typed text next, and the
small intent classifier only for what is left ("another", "you pick", "okay").

Everything the model returns is cleaned in code (`generate_field_options`,
`build_world`, `normalize_*`), because a 3B model often gives bad ids, broken
connections or the wrong JSON shape. Each step has a fallback, so creation
never dead-ends. Every LLM call happens before anything touches the disk, and
the save is one transaction, so a failure never leaves a half-made campaign.

Flow:

    notes -> setting -> threat -> hook -> name -> storyline -> world map
    -> story beats -> save

Run it on its own:  python -m aether.creation.campaign
"""

from __future__ import annotations

import random
import re
import shutil
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from string import capwords
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field, field_validator

from aether.creation.character import Reply, classify_reply, interpret_reply
from aether.creation.names import pick_patron_name
from aether.db import GameDB, slugify
from aether.llm import calls
from aether.llm.client import LLMClient, LLMError, Messages
from aether.models import CampaignInfo, Location, StoryBeat

CHECKLIST_KEYS = ("setting_description", "primary_threat", "starting_quest_hook")
CHECKLIST_LABELS = {
    "setting_description": "Setting",
    "primary_threat": "Threat",
    "starting_quest_hook": "Hook",
}

MIN_LOCATIONS = 3
MAX_LOCATIONS = 10
MAX_INDEX_GAP = 2   # a connection may skip at most one location in the model's list order
NAME_COUNT = 3
FIELD_MAX_CHARS = 800
MIN_FIELD_WORDS = 3   # "dark" or "decaying ruins" is a fragment, not a setting
OPTION_COUNT = 3
OPTION_MAX_CHARS = 220
FIELD_TEXT_MAX_CHARS = 300

FIELD_OPTIONS_MAX_TOKENS = 350
METADATA_MAX_TOKENS = 300
STORYLINE_MAX_TOKENS = 900
MAP_MAX_TOKENS = 900

FALLBACK_NAMES = ("An Unknown Journey", "Echoes of Adventure", "Tales of the Unknown")
DEFAULT_PATRON_ROLE = "a local guide"
_ARTICLES = ("a", "an", "the", "old", "young")

# What each field is, for the options prompt.
FIELD_DESCRIPTIONS = {
    "setting_description": "the setting: the world's environment, atmosphere and style",
    "primary_threat": "the main threat: a villain, faction or force the player will face",
    "starting_quest_hook": "the quest hook: the opening call to adventure that sets the player in motion",
}

# Used when the model gives too few usable options, so a menu is never short.
FALLBACK_FIELD_OPTIONS: Dict[str, Tuple[str, str, str]] = {
    "setting_description": (
        "A fog-choked marsh kingdom of sunken ruins and drowned bells",
        "A sun-scorched desert of buried cities and wandering caravans",
        "A frozen northern frontier of ice-bound forts and howling storms",
    ),
    "primary_threat": (
        "A secret cult that is slowly corrupting the local villages",
        "An ancient sorcerer returning to reclaim a lost throne",
        "A ruthless warlord whose army marches on the nearest town",
    ),
    "starting_quest_hook": (
        "A frightened stranger begs the player to find someone who vanished",
        "A mysterious letter points the player to a hidden place nearby",
        "Strange events in a nearby village need someone brave to investigate",
    ),
}


# --- schemas ---------------------------------------------------------------------------------

def _joined(value):
    """Small models sometimes return a list where a string is expected."""
    if isinstance(value, list):
        return " ".join(str(v).strip() for v in value if str(v).strip())
    return value


class FieldOptions(BaseModel):
    options: List[str] = Field(default_factory=list)


class DerivedMetadata(BaseModel):
    theme_vibe: str = ""
    patron_role: str = ""       # the role only; the name is chosen in code
    suggested_names: List[str] = Field(default_factory=list)

    @field_validator("theme_vibe", "patron_role", mode="before")
    @classmethod
    def _text(cls, value):
        return _joined(value)


class Storyline(BaseModel):
    beginning: str = ""
    middle: str = ""
    end: str = ""
    victory_condition: str = ""

    @field_validator("*", mode="before")
    @classmethod
    def _text(cls, value):
        return _joined(value)


class LocationDraft(BaseModel):
    id: str = ""
    name: str = ""
    brief_concept: str = ""
    connections: List[str] = Field(default_factory=list)


class WorldMapDraft(BaseModel):
    locations: List[LocationDraft]


class _Cancelled(Exception):
    pass


# --- prompts ---------------------------------------------------------------------------------

def _facts(checklist: Dict[str, str]) -> str:
    return "\n".join(
        [
            f"Setting: {checklist.get('setting_description', '')}",
            f"Threat: {checklist.get('primary_threat', '')}",
            f"Quest hook: {checklist.get('starting_quest_hook', '')}",
        ]
    )


def field_options_messages(key: str, answers: Dict[str, str], notes: str, count: int) -> Messages:
    """Ask for `count` options for one field, given what is already settled."""
    system = "\n".join(
        [
            "You help a player design a fantasy role-playing campaign.",
            f"Write exactly {count} different options for {FIELD_DESCRIPTIONS[key]}.",
            "- Each option is one sentence of 8 to 25 words.",
            "- The options must be clearly different from each other.",
            "- Every option must fit the answers already chosen and the player's ideas, if any.",
            "- Do not use character names.",
            "",
            "Reply as JSON in this shape (replace the <placeholders>):",
            '{"options":["<option 1>","<option 2>","<option 3>"]}',
        ]
    )
    lines = []
    if notes.strip():
        lines.append(f"Player's ideas: {_clip(notes, 300)}")
    for k in CHECKLIST_KEYS:
        if answers.get(k):
            lines.append(f"{CHECKLIST_LABELS[k]} (already chosen): {answers[k]}")
    user = "\n".join(lines) or "(no ideas given yet)"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def metadata_messages(checklist: Dict[str, str], patron_name: str) -> Messages:
    system = "\n".join(
        [
            "You name and style a role-playing campaign.",
            "- theme_vibe: a 2 to 3 word genre tag.",
            f"- patron_role: the role of {patron_name}, a character who helps the player at the start, "
            "2 to 5 words, for example 'a retired ferry-keeper'. Do not write the name.",
            f"- suggested_names: exactly {NAME_COUNT} atmospheric campaign names that fit the setting.",
            "",
            "Reply as JSON in this shape (replace the <placeholders>):",
            '{"theme_vibe":"<genre tag>","patron_role":"<role without a name>",'
            '"suggested_names":["<name 1>","<name 2>","<name 3>"]}',
        ]
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": _facts(checklist)}]


def storyline_messages(checklist: Dict[str, str], patron: str) -> Messages:
    system = "\n".join(
        [
            "You are a master RPG writer. Write a campaign storyline in three parts.",
            "- beginning: the opening situation, 2 to 3 sentences.",
            "- middle: 2 to 3 complications or obstacles on the journey, 2 to 3 sentences.",
            "- end: the climax and confrontation, 2 to 3 sentences.",
            "- victory_condition: one sentence naming the single concrete thing the player must do to win.",
            "- The hero is \"the player\". Never give the hero a name. The starting patron only helps "
            "the hero and is never the hero.",
            "",
            "Reply as JSON in this shape (replace the <placeholders>):",
            '{"beginning":"<text>","middle":"<text>","end":"<text>","victory_condition":"<text>"}',
        ]
    )
    user = f"{_facts(checklist)}\nStarting patron (a helper, not the hero): {patron}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def map_messages(checklist: Dict[str, str], story: Storyline) -> Messages:
    system = "\n".join(
        [
            f"You design the world map for a role-playing campaign: {MIN_LOCATIONS + 2} to {MAX_LOCATIONS - 2} locations.",
            "- The FIRST location is where the story begins.",
            "- The LAST location is where the final confrontation happens.",
            "- Lay the map out mostly as a path from first to last, with one or two short side branches.",
            "- id: lowercase snake_case, unique. name: a readable name. brief_concept: one sentence.",
            "- connections: ids of the locations directly next to this one. Only use ids from your own list.",
            "",
            "Reply as JSON in this shape (replace the <placeholders>):",
            '{"locations":[{"id":"<snake_case_id>","name":"<Name>","brief_concept":"<one sentence>",'
            '"connections":["<other_id>"]}]}',
        ]
    )
    user = "\n".join(
        [
            f"Setting: {checklist.get('setting_description', '')}",
            f"Beginning: {_clip(story.beginning, 300)}",
            f"Middle: {_clip(story.middle, 300)}",
            f"End: {_clip(story.end, 300)}",
        ]
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# --- pure helpers ----------------------------------------------------------------------------

def _clip(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(",;:-. ") + "..."


def pending(values: Dict[str, str]) -> List[str]:
    return [k for k in CHECKLIST_KEYS if not values.get(k)]


def valid_field_text(text: str) -> str:
    """A typed or chosen field value, or '' if it is a fragment."""
    text = _clip(text, FIELD_TEXT_MAX_CHARS)
    return text if len(text.split()) >= MIN_FIELD_WORDS else ""


def generate_field_options(
    client: LLMClient, key: str, answers: Dict[str, str], notes: str, count: int = OPTION_COUNT
) -> List[str]:
    """`count` distinct options for one field. Model output is cleaned, and
    built-in options fill any gap, so the menu is never short."""
    response = client.structured(
        "creation", field_options_messages(key, answers, notes, count),
        FieldOptions, max_tokens=FIELD_OPTIONS_MAX_TOKENS,
    )
    result: List[str] = []
    seen = set()
    for raw in (response.options if response else []):
        text = valid_field_text(_clip(str(raw).strip(" \"'-*"), OPTION_MAX_CHARS))
        if text and text.lower() not in seen:
            seen.add(text.lower())
            result.append(text)
    for fallback in FALLBACK_FIELD_OPTIONS[key]:
        if len(result) >= count:
            break
        if fallback.lower() not in seen:
            seen.add(fallback.lower())
            result.append(fallback)
    return result[:count]


def normalize_metadata(resp: Optional[DerivedMetadata]) -> DerivedMetadata:
    """Fill gaps with defaults and make sure there are exactly NAME_COUNT clean names."""
    vibe = _clip(resp.theme_vibe, 40) if resp else ""
    role = _clip(resp.patron_role, 60) if resp else ""
    names: List[str] = []
    for raw in (resp.suggested_names if resp else []):
        name = " ".join(str(raw).split()).strip(" \"'")[:60]
        if name and name.lower() not in {n.lower() for n in names}:
            names.append(name)
    for fallback in FALLBACK_NAMES:
        if len(names) >= NAME_COUNT:
            break
        if fallback.lower() not in {n.lower() for n in names}:
            names.append(fallback)
    return DerivedMetadata(
        theme_vibe=vibe or "Heroic Fantasy",
        patron_role=role or DEFAULT_PATRON_ROLE,
        suggested_names=names[:NAME_COUNT],
    )


def compose_patron(name: str, role: str) -> str:
    """'Odalric Venn' + 'a retired ferry-keeper' -> 'Odalric Venn, a retired ferry-keeper'.

    If the model wrote a name anyway ('Kaelin, a rebel'), it is dropped, so the
    name always comes from the list.
    """
    role = " ".join((role or "").split())
    role = re.sub(re.escape(name), "", role, flags=re.IGNORECASE)
    if "," in role:
        head, tail = role.split(",", 1)
        words = head.split()
        if words and len(words) <= 3 and all(w[0].isupper() for w in words) and words[0].lower() not in _ARTICLES:
            role = tail
    role = role.strip(" ,;:-.\"'")
    return f"{name}, {role or DEFAULT_PATRON_ROLE}"


def normalize_storyline(resp: Optional[Storyline], checklist: Dict[str, str], patron: str) -> Storyline:
    """Clip each part and replace missing ones, one by one."""
    threat = checklist.get("primary_threat", "the main threat")

    def pick(value: str, default: str) -> str:
        return _clip(value, FIELD_MAX_CHARS) or default

    r = resp or Storyline()
    return Storyline(
        beginning=pick(r.beginning, f"The adventure begins in safety under the guidance of {patron}."),
        middle=pick(r.middle, "The journey presents several dangerous obstacles."),
        end=pick(r.end, "The hero confronts the main threat in a dramatic climax."),
        victory_condition=pick(r.victory_condition, f"Defeat or overcome {threat}"),
    )


@dataclass
class World:
    locations: List[Location]
    connections: List[Tuple[str, str]]
    used_fallback: bool = False


def _reachable(start: str, pairs: Sequence[Tuple[str, str]]) -> set:
    adjacent: Dict[str, List[str]] = {}
    for a, b in pairs:
        adjacent.setdefault(a, []).append(b)
        adjacent.setdefault(b, []).append(a)
    seen, queue = {start}, deque([start])
    while queue:
        for nxt in adjacent.get(queue.popleft(), []):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return seen


def fallback_world() -> World:
    locations = [
        Location("starting_point", "The Starting Point", "Where the adventure begins."),
        Location("the_long_road", "The Long Road", "A dangerous road leading toward the heart of the trouble."),
        Location("final_stronghold", "The Final Stronghold", "The place where the main threat waits."),
    ]
    return World(locations, [("starting_point", "the_long_road"), ("the_long_road", "final_stronghold")], True)


def prettify_name(name: str) -> str:
    """'cult_haven' -> 'Cult Haven', 'oakwood manor - fall of the scholar' -> 'Oakwood Manor'.

    Names that already contain a capital letter are left alone.
    """
    name = " ".join(name.split())
    for sep in (" - ", " \u2013 ", " \u2014 "):
        if sep in name:
            head = name.split(sep, 1)[0].strip()
            if head:
                name = head
            break
    spaced = name.replace("_", " ").strip()
    return capwords(spaced) if spaced.islower() else spaced


def build_world(draft: Optional[WorldMapDraft]) -> World:
    """Turn the model's map into one the database accepts and the player can walk.

    * ids are slugified and duplicates dropped, at most MAX_LOCATIONS are kept
    * names that are just an id ("cult_haven") become titles ("Cult Haven")
    * connections to unknown or identical locations are dropped (a connection
      may name a location by id or by name)
    * shortcuts are dropped: the list is meant to run from the start to the
      final location, so a connection spanning more than MAX_INDEX_GAP places,
      or joining the first and last location, is removed
    * any location the player could not reach is chained onto the one before it
    * fewer than MIN_LOCATIONS usable locations gives the built-in map
    """
    entries: List[Tuple[Location, List[str]]] = []
    alias: Dict[str, str] = {}
    for loc in (draft.locations if draft else [])[:MAX_LOCATIONS]:
        raw = " ".join(loc.name.split())[:60]
        name = prettify_name(raw)
        if not name:
            continue
        lid = slugify(loc.id) if loc.id.strip() else slugify(name)
        if lid in {e[0].id for e in entries}:
            continue
        concept = " ".join(loc.brief_concept.split())[:240] or name
        entries.append((Location(lid, name, concept), loc.connections))
        alias[lid] = lid
        alias.setdefault(slugify(raw), lid)
        alias.setdefault(slugify(name), lid)

    if len(entries) < MIN_LOCATIONS:
        return fallback_world()

    pairs: List[Tuple[str, str]] = []
    seen = set()
    for location, targets in entries:
        for target in targets:
            other = alias.get(slugify(str(target)))
            if not other or other == location.id:
                continue
            key = frozenset((location.id, other))
            if key not in seen:
                seen.add(key)
                pairs.append((location.id, other))

    ids = [location.id for location, _ in entries]
    index = {lid: i for i, lid in enumerate(ids)}
    ends = frozenset((ids[0], ids[-1]))
    pairs = [
        (a, b) for a, b in pairs
        if abs(index[a] - index[b]) <= MAX_INDEX_GAP and frozenset((a, b)) != ends
    ]

    for i in range(1, len(ids)):
        if ids[i] not in _reachable(ids[0], pairs):
            pairs.append((ids[i - 1], ids[i]))   # ids[i-1] is reachable by now

    return World([location for location, _ in entries], pairs)


def choose_name(names: Sequence[str], input_fn: Callable[[str], str], print_fn: Callable[[str], None]) -> str:
    """Let the player pick a suggested name or type their own. Bad input picks the first."""
    print_fn("\nSelect a Campaign Name from the suggested options:")
    for i, name in enumerate(names, start=1):
        print_fn(f"{i}. {name}")
    custom = len(names) + 1
    print_fn(f"{custom}. Input a custom name")
    raw = input_fn(f"\nSelect (1-{custom}): ").strip()
    if raw.isdigit():
        n = int(raw)
        if 1 <= n <= len(names):
            return names[n - 1]
        if n == custom:
            typed = input_fn("Enter custom name: ").strip()
            if typed:
                return typed[:60]
    return names[0]


def save_campaign(
    info: CampaignInfo,
    world: World,
    beats: Sequence[StoryBeat],
    campaigns_dir: Optional[Path] = None,
) -> str:
    """Write a new campaign in one transaction. On any failure the folder is removed."""
    db = GameDB.create(info.slug, campaigns_dir)
    try:
        with db.transaction():
            for loc in world.locations:
                db.add_location(loc.id, loc.name, loc.brief_concept)
            for a, b in world.connections:
                db.add_connection(a, b)
            db.set_campaign(info)       # after the locations: it points at the starting one
            for beat in beats:
                db.add_beat(beat)
    except BaseException:
        folder = db.path.parent
        db.close()
        shutil.rmtree(folder, ignore_errors=True)
        raise
    db.close()
    return info.slug


# --- the interactive creator ---------------------------------------------------------------------

class CampaignCreator:
    def __init__(
        self,
        client: LLMClient,
        *,
        campaigns_dir: Optional[Path] = None,
        input_fn: Callable[[str], str] = input,
        print_fn: Callable[..., None] = print,
        rng: Optional[random.Random] = None,
    ):
        self.client = client
        self.campaigns_dir = campaigns_dir
        self.input = input_fn
        self.print = print_fn
        self.rng = rng or random.Random()
        self.values: Dict[str, str] = {}
        self.notes = ""

    def run(self) -> Optional[str]:
        """Run creation. Returns the new campaign's slug, or None if cancelled or failed."""
        self._intro()
        try:
            self._collect()
            return self._finalize()
        except (_Cancelled, KeyboardInterrupt, EOFError):
            self.print("\nCampaign creation cancelled.")
        except LLMError as e:
            self.print(f"\n[LLM error] {e}")
        return None

    def _intro(self) -> None:
        self.print("\n=======================================================")
        self.print("          AETHER: CAMPAIGN CREATION TERMINAL")
        self.print("=======================================================")
        self.print("You will choose a setting, a threat and a quest hook. Answer with a number,")
        self.print("or type what you want. Type 'cancel' at any time to leave.\n")

    # --- input helpers ---

    def _read(self, prompt: str = "\nYou: ", allow_empty: bool = False) -> str:
        while True:
            text = self.input(prompt).strip()
            if text.lower() == "cancel":
                raise _Cancelled()
            if text or allow_empty:
                return text

    def _show(self, title: str, lines: Sequence[str], custom: Optional[str] = None) -> None:
        self.print(f"\n{title}")
        for i, line in enumerate(lines, start=1):
            self.print(f"  {i}. {line}")
        if custom:
            self.print(f"  {len(lines) + 1}. {custom}")

    def _ask(self, question: str, options: Sequence[str], **kw) -> Reply:
        reply = self._read()
        return interpret_reply(
            reply, options, classify=lambda r: classify_reply(self.client, question, options, r), **kw
        )

    # --- the form ---

    def _collect(self) -> None:
        self.notes = self._read(
            "Any ideas for your campaign? (optional, e.g. 'a dark swamp'; press Enter to skip): ",
            allow_empty=True,
        )
        for key in CHECKLIST_KEYS:
            value = self._ask_field(key)
            self.values[key] = value
            self.print(f"[OK] {CHECKLIST_LABELS[key]}: {value}")

    def _field_options(self, key: str) -> List[str]:
        self.print(f"\n[System] Writing {CHECKLIST_LABELS[key].lower()} ideas...")
        return generate_field_options(self.client, key, self.values, self.notes)

    def _ask_field(self, key: str) -> str:
        label = CHECKLIST_LABELS[key].lower()
        options = self._field_options(key)
        while True:
            self._show(f"Choose the {label}:", options, custom="Type my own")
            self.print("(or type your own, or ask for different ideas, or ask me to pick one)")
            rep = self._ask(f"Choose the campaign's {label}", options, custom=True, open_text=True)
            text = ""
            if rep.kind == "option":
                text = options[rep.index]
            elif rep.kind == "custom":
                text = valid_field_text(self._read(f"Your {label}: "))
            elif rep.kind == "text":
                text = valid_field_text(rep.text)
            elif rep.kind == "surprise":
                text = self.rng.choice(options)
                self.print("[System] Picked one for you.")
            elif rep.kind == "reject":
                options = self._field_options(key)
                continue
            if text:
                return text
            self.print(f"[System] Pick a number, or write at least {MIN_FIELD_WORDS} words of your own.")

    # --- build and save ---

    def _structured(self, messages: Messages, schema, max_tokens: int, temperature: float):
        return self.client.structured("referee", messages, schema, max_tokens=max_tokens, temperature=temperature)

    def _finalize(self) -> Optional[str]:
        self.print("\nBuilding your campaign. This takes a few minutes on a CPU...")

        self.print("\n[1/5] Choosing a style and name suggestions...")
        patron_name = pick_patron_name(self.rng)
        meta = normalize_metadata(
            self._structured(metadata_messages(self.values, patron_name), DerivedMetadata, METADATA_MAX_TOKENS, 0.8)
        )
        patron = compose_patron(patron_name, meta.patron_role)
        name = choose_name(meta.suggested_names, self.input, self.print)
        slug = GameDB.unique_slug(name, self.campaigns_dir)
        self.print(f"\nCampaign name: '{name}' (save slot: {slug})")

        self.print("\n[2/5] Writing the storyline...")
        story = normalize_storyline(
            self._structured(
                storyline_messages(self.values, patron), Storyline, STORYLINE_MAX_TOKENS, 0.7
            ),
            self.values,
            patron,
        )

        self.print("\n[3/5] Designing the world map...")
        world = build_world(self._structured(map_messages(self.values, story), WorldMapDraft, MAP_MAX_TOKENS, 0.5))
        if world.used_fallback:
            self.print("[Note] The model's map was unusable, so a simple default map was used.")

        info = CampaignInfo(
            name=name,
            slug=slug,
            setting_description=self.values["setting_description"],
            primary_threat=self.values["primary_threat"],
            starting_quest_hook=self.values["starting_quest_hook"],
            theme_vibe=meta.theme_vibe,
            starting_patron=patron,
            storyline_beginning=story.beginning,
            storyline_middle=story.middle,
            storyline_end=story.end,
            victory_condition=story.victory_condition,
            starting_location_id=world.locations[0].id,
        )

        self.print("\n[4/5] Planning the story beats...")
        beats = calls.derive_beats(self.client, campaign=info, locations=world.locations)

        self.print("\n[5/5] Saving...")
        save_campaign(info, world, beats, self.campaigns_dir)
        self._report(info, world, beats)
        return slug

    def _report(self, info: CampaignInfo, world: World, beats: Sequence[StoryBeat]) -> None:
        self.print("\n=======================================================")
        self.print("          SUCCESS: CAMPAIGN CREATED")
        self.print("=======================================================")
        self.print(f"Name:      {info.name}")
        self.print(f"Patron:    {info.starting_patron}")
        self.print(f"Locations: {len(world.locations)} (start: {world.locations[0].name})")
        self.print(f"Beats:     {len(beats)}")
        self.print("=======================================================\n")


def main() -> int:
    client = LLMClient()
    slug = CampaignCreator(client).run()
    return 0 if slug else 1


if __name__ == "__main__":
    raise SystemExit(main())
