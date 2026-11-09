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

from aether.llm import prompts
from aether.llm.client import LLMClient, Messages
from aether.llm.schemas import (
    BeatsBlueprint,
    MemoryFacts,
    JudgeVerdict,
    NewNpc,
    OptionsResponse,
)
from aether.logic.memory import group_exchanges, summary_to_text
from aether.logic.options import ActionOption, apply_default_skills, complete_options, sanitize_options
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

MAX_NEW_NPCS = 3
MAX_LIST_ITEMS = {"key_npcs": 8, "open_threads": 5, "promises_debts": 4}

ALLOWED_BEAT_CONDITIONS = ("location", "referee")

ACTION_MAX_CHARS = 90          # the player's part of one summary line
EVENT_MAX_CHARS = 150          # the DM-reply part
FACTS_MAX_TOKENS = 250


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

_CAP_WORD = re.compile(r"\b[A-Z][a-z]{2,}\b")


def find_invented_names(narration: str, known_text: str) -> List[str]:
    """Capitalised words in mid-sentence that appear nowhere in `known_text`.

    A cheap check for characters the model made up. It cannot see an invented
    name at the very start of a sentence, and it will occasionally flag a
    capitalised common word, so use it only to trigger a retry, never to block.
    """
    known = {w.lower() for w in re.findall(r"[A-Za-z']+", known_text)}
    found: List[str] = []
    for m in _CAP_WORD.finditer(narration):
        before = narration[: m.start()].rstrip()
        if not before or before[-1] in '.!?"\u201c\n':
            continue
        word = m.group()
        if word.lower() not in known and word not in found:
            found.append(word)
    return found


def narrate_opening(
    client: LLMClient, messages: Messages, *, known_text: str, max_attempts: int = 2
) -> Tuple[str, List[Tag]]:
    """The opening scene, regenerated once if it names someone who does not exist.

    Returns the attempt with the fewest invented names. This is not streamed,
    because the text has to be checked before the player sees it.
    """
    best: Optional[Tuple[str, List[Tag], int]] = None
    for _ in range(max_attempts):
        text, tags = narrate(client, messages)
        invented = find_invented_names(text, known_text)
        if not invented:
            return text, tags
        if best is None or len(invented) < best[2]:
            best = (text, tags, len(invented))
        hint = f" Do not mention anyone named {', '.join(invented)}."
        messages = messages[:-1] + [{**messages[-1], "content": messages[-1]["content"] + hint}]
    return best[0], best[1]

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
    options = apply_default_skills(complete_options(cleaned, categories, limit))
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

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")
_CHECK_RESULT = re.compile(r"-\s*(Success|Failure)\s*$", re.IGNORECASE)
_STOPWORDS = frozenset({
    "that", "this", "with", "from", "into", "about", "have", "will", "what", "when", "where",
    "there", "their", "them", "they", "then", "than", "your", "been", "were", "would", "could",
    "should", "some", "more", "only", "also", "just",
})


def _first_sentence(text: str) -> str:
    text = " ".join(text.split())
    return _SENTENCE_END.split(text, maxsplit=1)[0] if text else ""


def extractive_event(group: Sequence[Message], player_name: str) -> str:
    """One summary line built only from the text itself, so it cannot be invented.

    'Aria chose "Ask about the bell" (success). Mother Hesk says the bell sank long ago.'
    """
    action, result, outcome = "", "", ""
    for m in group:
        if m.role == "player" and not action:
            action = _truncate_text(" ".join(m.content.split()).rstrip(" .!?"), ACTION_MAX_CHARS)
        elif m.role == "dm" and not result:
            result = _truncate_text(_first_sentence(m.content), EVENT_MAX_CHARS)
        elif m.role == "system":
            found = _CHECK_RESULT.search(m.content)
            if found:
                outcome = found.group(1).lower()
    parts = []
    if action:
        parts.append(f'{player_name} chose "{action}"' + (f" ({outcome})" if outcome else "") + ".")
        if result:
            parts.append(result)
    elif result:
        parts.append(f"Opening: {result}")
    return " ".join(parts)


def _norm(word: str) -> str:
    w = word.lower().strip("'-")
    return w[:-2] if w.endswith("'s") else w


def is_grounded(item: str, source: str) -> bool:
    """True if `item` plausibly comes from `source` (a heuristic, not a proof).

    Rejects an item with a capitalised word (other than the first) that is not in
    the source, and one where fewer than half its content words appear there.
    """
    raw = _WORD.findall(item)
    if not raw:
        return False
    source_words = {_norm(w) for w in _WORD.findall(source)}
    for i, w in enumerate(raw):
        if i > 0 and w[0].isupper() and _norm(w) not in source_words:
            return False
    stems = {w[:5] for w in source_words}
    content = [n for n in map(_norm, raw) if len(n) >= 4 and n not in _STOPWORDS]
    if not content:
        return False
    hits = sum(1 for n in content if n[:5] in stems)
    return hits * 2 >= len(content)


def _note_for(name: str, messages: Sequence[Message]) -> str:
    """The latest DM sentence that mentions `name`, or ''."""
    low = name.lower()
    for m in reversed(messages):
        if m.role != "dm":
            continue
        for sentence in _SENTENCE_END.split(" ".join(m.content.split())):
            if low in sentence.lower():
                return _truncate_text(sentence, 160)
    return ""


def compact_history(
    client: LLMClient,
    *,
    previous: Optional[Summary],
    messages: Sequence[Message],
    player_name: str = "Player",
    known_npc_names: Sequence[str] = (),
) -> Optional[Compaction]:
    """Fold `messages` into the running summary, using the model as little as possible.

    The narrative is one extractive line per exchange; when it grows past the
    limit the oldest lines are dropped. Key NPCs and their notes come from
    `known_npc_names` (pass the database's NPC names, not the player's). The
    model only proposes open threads and promises, and anything the transcript
    does not support is discarded. The raw transcript is never touched.
    """
    if not messages:
        return None
    mem = client.config.memory

    events = [e for e in (extractive_event(g, player_name) for g in group_exchanges(messages)) if e]
    if not events:
        return None
    lines = [l.strip() for l in (previous.narrative.splitlines() if previous else []) if l.strip()]
    lines += events
    max_words = int(mem.summary_max_tokens * 0.75)
    while len(lines) > 1 and sum(len(l.split()) for l in lines) > max_words:
        lines.pop(0)
    narrative = "\n".join(lines)

    raw = " ".join(m.content for m in messages if m.role != "system")
    source = " ".join([summary_to_text(previous), raw, player_name, *known_npc_names])

    mentioned = [n.strip() for n in known_npc_names if n.strip() and n.strip().lower() in raw.lower()]
    key_npcs = _clean_list([*(previous.key_npcs if previous else []), *mentioned], MAX_LIST_ITEMS["key_npcs"])
    notes: Dict[str, str] = {}
    for name in mentioned:
        note = _note_for(name, messages)
        if note:
            notes[name] = note

    facts = client.structured(
        "compaction",
        prompts.memory_facts_messages(summary_to_text(previous), "\n".join(events), player_name),
        MemoryFacts,
        max_tokens=FACTS_MAX_TOKENS,
    )
    if facts is None:       # keep what we knew; the lists catch up next time
        threads = previous.open_threads if previous else []
        promises = previous.promises_debts if previous else []
    else:
        threads = [t for t in facts.open_threads if is_grounded(t, source)]
        promises = [p for p in facts.promises_debts if is_grounded(p, source)]

    summary = Summary(
        covers_through_message_id=messages[-1].id,
        narrative=narrative,
        key_npcs=key_npcs,
        open_threads=_clean_list(threads, MAX_LIST_ITEMS["open_threads"]),
        promises_debts=_clean_list(promises, MAX_LIST_ITEMS["promises_debts"]),
    )
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
        if ctype == "location" and cvalue == campaign.starting_location_id:
            ctype, cvalue = "referee", ""
        beats.append(StoryBeat(len(beats) + 1, title, objective, ctype, cvalue))

    # The final beat is the victory. A pure travel milestone there would finish
    # the moment the previous beat does, because the confrontation happens at that place.
    if beats and beats[-1].condition_type == "location":
        beats[-1].condition_type, beats[-1].condition_value = "referee", ""

    if len(beats) < 2:
        return fallback_beats(campaign)
    return beats
