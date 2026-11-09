"""Campaign creation.

Flow (the same for the player as before):

    brainstorm chat -> checklist complete -> name suggestions -> storyline
    -> world map -> story beats -> save

Every LLM call happens before anything touches the disk, and the save is one
transaction, so a failure never leaves a half-made campaign behind.

Everything the model returns is cleaned in code (`build_world`,
`normalize_*`), because a 3B model often gives bad ids, broken connections or
the wrong JSON shape. Each step has a fallback, so creation never dead-ends.

Run it on its own:  python -m aether.creation.campaign
"""

from __future__ import annotations

import shutil
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple
from string import capwords

from pydantic import BaseModel, Field, field_validator

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

CHECKLIST_MAX_TOKENS = 300
METADATA_MAX_TOKENS = 300
STORYLINE_MAX_TOKENS = 900
MAP_MAX_TOKENS = 900

FALLBACK_NAMES = ("An Unknown Journey", "Echoes of Adventure", "Tales of the Unknown")


# --- schemas ---------------------------------------------------------------------------------

def _joined(value):
    """Small models sometimes return a list where a string is expected."""
    if isinstance(value, list):
        return " ".join(str(v).strip() for v in value if str(v).strip())
    return value


class CampaignChecklist(BaseModel):
    setting_description: Optional[str] = None
    primary_threat: Optional[str] = None
    starting_quest_hook: Optional[str] = None

    @field_validator("*", mode="before")
    @classmethod
    def _text(cls, value):
        return _joined(value)


class DerivedMetadata(BaseModel):
    theme_vibe: str = ""
    starting_patron: str = ""
    suggested_names: List[str] = Field(default_factory=list)

    @field_validator("theme_vibe", "starting_patron", mode="before")
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


# --- prompts ---------------------------------------------------------------------------------

def _facts(checklist: Dict[str, str]) -> str:
    return "\n".join(
        [
            f"Setting: {checklist.get('setting_description', '')}",
            f"Threat: {checklist.get('primary_threat', '')}",
            f"Quest hook: {checklist.get('starting_quest_hook', '')}",
        ]
    )


def guide_messages(checklist: Dict[str, str], history: Sequence[Dict[str, str]]) -> Messages:
    def status(key: str) -> str:
        return "COMPLETED" if checklist.get(key) else "PENDING"

    system = "\n".join(
        [
            "You are the World-Building Dungeon Master. Help the player settle exactly 3 campaign parameters:",
            "1. setting_description (the world's environment and style)",
            "2. primary_threat (the main threat, faction or antagonist)",
            "3. starting_quest_hook (the opening call to adventure)",
            "",
            "Current checklist:",
            f"- setting_description: {status('setting_description')}",
            f"- primary_threat: {status('primary_threat')}",
            f"- starting_quest_hook: {status('starting_quest_hook')}",
            "",
            "INSTRUCTIONS",
            "- Focus on PENDING parameters. Ask about ONE parameter at a time.",
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
        "You are a strict data extraction system. Read the recent conversation and extract the campaign "
        "details into the JSON schema. Only extract values the player has explicitly agreed on or chosen. "
        "Use null for anything not decided. Do not invent details."
    )
    return [{"role": "system", "content": system}, *history[-4:]]


def metadata_messages(checklist: Dict[str, str]) -> Messages:
    system = "\n".join(
        [
            "You name and style a role-playing campaign.",
            "- theme_vibe: a 2 to 3 word genre tag.",
            "- starting_patron: one named character who helps the player at the start, written as "
            "'<Name>, <role>'.",
            f"- suggested_names: exactly {NAME_COUNT} atmospheric campaign names that fit the setting.",
            "",
            "Reply as JSON in this shape (replace the <placeholders>):",
            '{"theme_vibe":"<genre tag>","starting_patron":"<Name>, <role>",'
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


def merge_checklist(current: Dict[str, str], parsed: Optional[CampaignChecklist], source: str) -> Dict[str, str]:
    """Accept at most ONE new value per turn: the first pending item.

    The guide asks about one thing at a time, so a model that fills several
    fields after one answer is guessing. The value must also be a real phrase
    (MIN_FIELD_WORDS or more) and be grounded in `source`, the conversation
    text. A value that is rejected is simply asked about again. Settled
    values are never overwritten.
    """
    merged = dict(current)
    missing = pending(merged)
    if parsed is None or not missing:
        return merged
    key = missing[0]
    value = " ".join((getattr(parsed, key) or "").split())
    if len(value.split()) >= MIN_FIELD_WORDS and calls.is_grounded(value, source):
        merged[key] = value
    return merged


def pending(checklist: Dict[str, str]) -> List[str]:
    return [k for k in CHECKLIST_KEYS if not checklist.get(k)]


def status_line(checklist: Dict[str, str]) -> str:
    parts = [f"{CHECKLIST_LABELS[k]}: {'OK' if checklist.get(k) else 'PENDING'}" for k in CHECKLIST_KEYS]
    return "[Checklist] " + " | ".join(parts)


def normalize_metadata(resp: Optional[DerivedMetadata]) -> DerivedMetadata:
    """Fill gaps with defaults and make sure there are exactly NAME_COUNT clean names."""
    vibe = _clip(resp.theme_vibe, 40) if resp else ""
    patron = _clip(resp.starting_patron, 100) if resp else ""
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
        starting_patron=patron or "Old Tam, a local tavern owner",
        suggested_names=names[:NAME_COUNT],
    )


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
    ):
        self.client = client
        self.campaigns_dir = campaigns_dir
        self.input = input_fn
        self.print = print_fn
        self.history: List[Dict[str, str]] = []
        self.checklist: Dict[str, str] = {}

    def run(self) -> Optional[str]:
        """Run creation. Returns the new campaign's slug, or None if cancelled or failed."""
        self._intro()
        try:
            if not self._brainstorm():
                self.print("\nCampaign creation cancelled.")
                return None
            return self._finalize()
        except (KeyboardInterrupt, EOFError):
            self.print("\nCampaign creation cancelled.")
        except LLMError as e:
            self.print(f"\n[LLM error] {e}")
        return None

    # --- brainstorm ---

    def _intro(self) -> None:
        self.print("\n=======================================================")
        self.print("          AETHER: CAMPAIGN CREATION TERMINAL")
        self.print("=======================================================")
        self.print("Tell me about the campaign you want to build, for example:")
        self.print("'A dark-fantasy swamp where a swamp witch resides.'")
        self.print("Once the setting, threat and quest hook are settled, the campaign builds")
        self.print("automatically. Type 'complete' to build it early, or 'cancel' to leave.\n")

    def _brainstorm(self) -> bool:
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

            self.history.append({"role": "user", "content": text})
            if len(self.history) >= 2:
                self._update_checklist()
                self.print(status_line(self.checklist))
            if not pending(self.checklist):
                self.print("\n[System] All core elements are settled.")
                return True
            self._guide_reply()

    def _update_checklist(self) -> None:
        parsed = self.client.structured(
            "referee",
            checklist_messages(self.history),
            CampaignChecklist,
            max_tokens=CHECKLIST_MAX_TOKENS,
            temperature=0.0,
        )
        source = " ".join(m["content"] for m in self.history)
        self.checklist = merge_checklist(self.checklist, parsed, source)

    def _guide_reply(self) -> None:
        self.print("\nGuide: ", end="", flush=True)
        pieces: List[str] = []
        for piece in self.client.stream("creation", guide_messages(self.checklist, self.history)):
            self.print(piece, end="", flush=True)
            pieces.append(piece)
        self.print()
        reply = "".join(pieces).strip()
        if reply:
            self.history.append({"role": "assistant", "content": reply})
        else:
            self.print("[System] The guide had nothing to say. Keep describing your campaign.")

    # --- build and save ---

    def _structured(self, messages: Messages, schema, max_tokens: int, temperature: float):
        return self.client.structured("referee", messages, schema, max_tokens=max_tokens, temperature=temperature)

    def _finalize(self) -> Optional[str]:
        self.print("\nBuilding your campaign. This takes a few minutes on a CPU...")

        self.print("\n[1/5] Choosing a style and name suggestions...")
        meta = normalize_metadata(
            self._structured(metadata_messages(self.checklist), DerivedMetadata, METADATA_MAX_TOKENS, 0.8)
        )
        name = choose_name(meta.suggested_names, self.input, self.print)
        slug = GameDB.unique_slug(name, self.campaigns_dir)
        self.print(f"\nCampaign name: '{name}' (save slot: {slug})")

        self.print("\n[2/5] Writing the storyline...")
        story = normalize_storyline(
            self._structured(
                storyline_messages(self.checklist, meta.starting_patron), Storyline, STORYLINE_MAX_TOKENS, 0.7
            ),
            self.checklist,
            meta.starting_patron,
        )

        self.print("\n[3/5] Designing the world map...")
        world = build_world(self._structured(map_messages(self.checklist, story), WorldMapDraft, MAP_MAX_TOKENS, 0.5))
        if world.used_fallback:
            self.print("[Note] The model's map was unusable, so a simple default map was used.")

        info = CampaignInfo(
            name=name,
            slug=slug,
            setting_description=self.checklist["setting_description"],
            primary_threat=self.checklist["primary_threat"],
            starting_quest_hook=self.checklist["starting_quest_hook"],
            theme_vibe=meta.theme_vibe,
            starting_patron=meta.starting_patron,
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
