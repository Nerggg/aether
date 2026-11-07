"""The LLM jobs the game needs, one function each.

Each function builds its prompt, makes the call through `LLMClient`, then
cleans and checks the result with the pure logic from Stage 2.

Failure policy (so the game loop stays simple):

* **Bad model output** never raises. Each function falls back to something
  safe: a menu filled from built-in options, a judge that lets the action
  through, no compaction this turn, and so on. The result's `ok` flag says
  whether a fallback was used.
* **Ollama itself being unreachable** raises `LLMError`. That is a problem the
  player has to fix, so it is not hidden.
"""

from __future__ import annotations
import re

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from aether.config import Config
from aether.llm import prompts
from aether.llm.client import LLMClient, Messages
from aether.llm.schemas import (
    BeatsBlueprint,
    CompactionResult,
    JudgeVerdict,
    NewNpc,
    OptionsResponse,
)
from aether.logic.memory import format_transcript, summary_to_text
from aether.logic.options import ActionOption, complete_options, sanitize_options
from aether.logic.stats import normalize_skill
from aether.logic.story import is_valid_condition
from aether.logic.tags import ParseResult, Tag, TagParser
from aether.models import CampaignInfo, Location, Message, StoryBeat, Summary

# Output caps for structured calls (tokens). Small, so a rambling model cannot
# stall a CPU-only laptop.
OPTIONS_MAX_TOKENS = 300
JUDGE_MAX_TOKENS = 150
BEATS_MAX_TOKENS = 700
LOCATION_MAX_TOKENS = 300
COMPACTION_EXTRA_TOKENS = 250   # room for the lists and NPC notes on top of the narrative

MAX_NEW_NPCS = 3
MAX_LIST_ITEMS = {"key_npcs": 8, "open_threads": 5, "promises_debts": 4}

MIN_SUMMARY_WORDS = 60
ALLOWED_BEAT_CONDITIONS = ("location", "referee")


# --- narration -------------------------------------------------------------------------------------

def narrate_stream(client: LLMClient, messages: Messages) -> Iterator[ParseResult]:
    """Stream the DM's reply with state tags separated from the story text.

    Yields `ParseResult`s: show `.text` immediately, apply `.tags` as they arrive.
    """
    parser = TagParser()
    for piece in client.stream("dm", messages):
        result = parser.feed(piece)
        if result.text or result.tags:
            yield result
    tail = parser.flush()
    if tail.text:
        yield tail

# --- tidying narration ---------------------------------------------------------------

_BOUNDARY = re.compile(r'[.!?]["\u201d\u2019)]*\s+')
_LAST_END = re.compile(r'[.!?]["\u201d\u2019)]*(?=\s|$)')
_ASKS_PLAYER = re.compile(r"\byou(?:r|rs|rself)?\b|\bwhat (?:now|next)\b", re.IGNORECASE)


def tidy_narration(text: str) -> str:
    """Clean a finished DM reply before it is saved.

    * An unfinished last sentence (the reply hit the length cap) is cut.
    * A closing question to the player ("What do you do?") is removed, because
      the menu asks that. Questions inside quotes (NPC speech) are kept.
    """
    text = text.strip()
    if not text:
        return text
    if text[-1] not in '.!?"\u201d\u2019)\u2026*':
        ends = list(_LAST_END.finditer(text))
        if ends and ends[-1].end() >= len(text) // 2:
            text = text[: ends[-1].end()]
    for _ in range(2):
        boundaries = list(_BOUNDARY.finditer(text))
        if not boundaries:
            break
        head, tail = text[: boundaries[-1].end()], text[boundaries[-1].end():].strip()
        if tail.endswith("?") and _ASKS_PLAYER.search(tail) and not re.search(r'["\u201c\u201d]', tail):
            text = head.rstrip()
        else:
            break
    return text

def narrate(client: LLMClient, messages: Messages) -> Tuple[str, List[Tag]]:
    """The whole narration at once: (tidied text without tags, tags)."""
    text: List[str] = []
    tags: List[Tag] = []
    for result in narrate_stream(client, messages):
        text.append(result.text)
        tags.extend(result.tags)
    return tidy_narration("".join(text)), tags

def describe_location(client: LLMClient, *, name: str, concept: str, vibe: str = "") -> str:
    """Just-in-time description for a location. Falls back to its concept."""
    reply = client.chat(
        "dm",
        prompts.location_description_messages(name, concept, vibe),
        max_tokens=LOCATION_MAX_TOKENS,
    )
    return reply or concept


# --- options + referee -----------------------------------------------------------------------------------

@dataclass
class OptionsResult:
    options: List[ActionOption]
    new_npcs: List[NewNpc] = field(default_factory=list)
    beat_achieved: bool = False
    ok: bool = True   # False when the menu came entirely from built-in fallbacks


def generate_options(
    client: LLMClient,
    *,
    narration: str,
    categories: Sequence[str],
    state_block: str,
    known_npc_names: Sequence[str] = (),
    beat: Optional[StoryBeat] = None,
    check_beat: bool = False,
) -> OptionsResult:
    """The menu for the next turn, plus any new characters and (optionally) beat progress.

    `categories` come from `options.select_categories`; the returned list always
    has exactly `len(categories)` options. Include the player's own name in
    `known_npc_names` so the model does not report them as a new character.
    """
    limit = len(categories)
    max_words = client.config.gameplay.max_option_words
    messages = prompts.options_messages(
        narration, categories, state_block, max_words, known_npc_names, beat, check_beat
    )
    response = client.structured("options", messages, OptionsResponse, max_tokens=OPTIONS_MAX_TOKENS)

    if response is None:
        return OptionsResult(complete_options([], categories, limit), ok=False)

    raw = []
    for o in response.options:
        cat = (o.category or "").strip().lower()
        raw.append({"text": o.text, "skill": o.skill, "category": cat if cat in categories else None})
    cleaned = sanitize_options(raw, max_words, limit, categories)
    options = complete_options(cleaned, categories, limit)
    order = {c: i for i, c in enumerate(categories)}
    options.sort(key=lambda o: order.get(o.category, len(order)))

    known = {n.strip().lower() for n in known_npc_names}
    npcs: List[NewNpc] = []
    for npc in response.new_npcs:
        name = npc.name.strip()
        if not name or len(name) > 60 or name.lower() in known:
            continue
        known.add(name.lower())
        npcs.append(npc.model_copy(update={"name": name}))
        if len(npcs) >= MAX_NEW_NPCS:
            break

    return OptionsResult(
        options=options,
        new_npcs=npcs,
        beat_achieved=bool(response.beat_achieved and check_beat),
        ok=bool(cleaned),
    )


# --- custom action judge ---------------------------------------------------------------------------------

@dataclass
class JudgeResult:
    verdict: str                      # "feasible", "risky" or "impossible"
    skill: Optional[str] = None       # set only for "risky"
    difficulty: str = "medium"
    reason: str = ""
    advances_beat: bool = False
    ok: bool = True                   # False when the model failed and the action was let through


def judge_action(
    client: LLMClient,
    *,
    action: str,
    narration: str,
    state_block: str,
    beat: Optional[StoryBeat] = None,
) -> JudgeResult:
    """Decide what to do with a typed action. Never blocks the player because the model failed."""
    messages = prompts.judge_messages(action, narration, state_block, beat)
    response = client.structured("judge", messages, JudgeVerdict, max_tokens=JUDGE_MAX_TOKENS)
    if response is None:
        return JudgeResult("feasible", ok=False)

    verdict = response.verdict
    skill = normalize_skill(response.skill)
    if verdict == "risky" and skill is None:
        verdict = "feasible"          # nothing to roll, so just let it happen
    if verdict != "risky":
        skill = None

    reason = " ".join(response.reason.split())[:200]
    if verdict == "impossible" and not reason:
        reason = "That isn't possible right now."

    return JudgeResult(
        verdict=verdict,
        skill=skill,
        difficulty=response.difficulty,
        reason=reason,
        advances_beat=bool(response.advances_beat and beat is not None and verdict != "impossible"),
    )


# --- compaction --------------------------------------------------------------------------------------------

@dataclass
class Compaction:
    summary: Summary              # ready for GameDB.add_summary
    npc_notes: Dict[str, str]     # NPC name -> what the player knows; caller maps names to characters


def _truncate_text(text: str, max_chars: int) -> str:
    """Cut at the last sentence end inside the limit (or hard-cut if there is none)."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "), cut.rfind(".\n"))
    if end >= max_chars // 2:
        return cut[: end + 1]
    return cut.rstrip() + "..."


def _clean_list(items: Sequence[str], limit: int) -> List[str]:
    seen, out = set(), []
    for item in items:
        item = " ".join(str(item).split())
        if item and item.lower() not in seen:
            seen.add(item.lower())
            out.append(item)
        if len(out) >= limit:
            break
    return out


def compact_history(
    client: LLMClient,
    *,
    previous: Optional[Summary],
    messages: Sequence[Message],
    player_name: str = "Player",
) -> Optional[Compaction]:
    """Fold `messages` into the running summary.

    Returns None if the model fails; the caller just tries again on a later
    turn (the raw transcript is untouched either way).
    """
    if not messages:
        return None
    mem = client.config.memory
    previous_text = summary_to_text(previous)
    transcript_text = format_transcript(messages, player_name)
    # The summary may grow by about half the size of the new events, never past the configured cap.
    max_words = min(
        int(mem.summary_max_tokens * 0.75),
        max(MIN_SUMMARY_WORDS, len(previous_text.split()) + len(transcript_text.split()) // 2),
    )
    prompt = prompts.compaction_messages(previous_text, transcript_text, player_name, max_words)
    response = client.structured(
        "compaction",
        prompt,
        CompactionResult,
        max_tokens=mem.summary_max_tokens + COMPACTION_EXTRA_TOKENS,
    )
    if response is None:
        return None

    narrative = " ".join(response.narrative.split())
    limit_chars = mem.summary_max_tokens * 4
    if len(narrative.split()) > max_words * 1.3:      # it retold the story instead of summarising
        limit_chars = min(limit_chars, max_words * 6)
    narrative = _truncate_text(narrative, limit_chars)

    if not narrative:
        return None

    summary = Summary(
        covers_through_message_id=messages[-1].id,
        narrative=narrative,
        key_npcs=_clean_list(response.key_npcs, MAX_LIST_ITEMS["key_npcs"]),
        open_threads=_clean_list(response.open_threads, MAX_LIST_ITEMS["open_threads"]),
        promises_debts=_clean_list(response.promises_debts, MAX_LIST_ITEMS["promises_debts"]),
    )
    notes = {
        n.name.strip(): " ".join(n.notes.split())
        for n in response.npc_notes
        if n.name.strip() and n.notes.strip()
    }
    return Compaction(summary, notes)


# --- story beats (used by campaign creation) -------------------------------------------------------------

def fallback_beats(campaign: CampaignInfo) -> List[StoryBeat]:
    """Three plain beats from the storyline text, used when the model fails."""
    def clip(text: str, default: str) -> str:
        text = " ".join(text.split())
        return (text[:200].rsplit(" ", 1)[0] + "...") if len(text) > 200 else (text or default)

    return [
        StoryBeat(1, "The Call to Adventure", clip(campaign.storyline_beginning, campaign.starting_quest_hook)),
        StoryBeat(2, "Rising Danger", clip(campaign.storyline_middle, "Overcome the obstacles on the way.")),
        StoryBeat(3, "The Final Confrontation", clip(campaign.victory_condition or campaign.storyline_end,
                                                    f"Defeat {campaign.primary_threat}")),
    ]


def derive_beats(
    client: LLMClient,
    *,
    campaign: CampaignInfo,
    locations: Sequence[Location],
) -> List[StoryBeat]:
    """4-6 story beats from the storyline, with every condition validated.

    A condition the code cannot check (an invented location id, an empty
    value) is downgraded to the 'referee' kind. If the model fails, three
    plain fallback beats are returned, so campaign creation never stops here.
    """
    gameplay = client.config.gameplay
    known_ids = {l.id for l in locations}
    messages = prompts.beats_messages(campaign, locations, gameplay.story_beats_min, gameplay.story_beats_max)
    response = client.structured("referee", messages, BeatsBlueprint, max_tokens=BEATS_MAX_TOKENS)

    beats: List[StoryBeat] = []
    for draft in (response.beats if response else [])[: gameplay.story_beats_max]:
        title = " ".join(draft.title.split())
        objective = " ".join(draft.objective.split())
        if not title or not objective:
            continue
        ctype = draft.condition_type.strip().lower()
        cvalue = draft.condition_value.strip()
        if ctype not in ALLOWED_BEAT_CONDITIONS or not is_valid_condition(ctype, cvalue, known_ids):
            ctype, cvalue = "referee", ""
        if not beats and ctype == "location" and cvalue == campaign.starting_location_id:
            ctype, cvalue = "referee", ""
        beats.append(StoryBeat(len(beats) + 1, title, objective, ctype, cvalue))

    if len(beats) < 2:
        return fallback_beats(campaign)
    return beats
