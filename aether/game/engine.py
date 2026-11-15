"""The game loop.

    opener -> menu -> (number | typed action -> judge) -> dice -> narration
    -> tags -> beat checks -> next menu -> ...

Everything is saved as it happens, so quitting at any moment is safe:

* the player's action, dice and movement are saved BEFORE the DM answers, so a
  crash mid-turn is finished on resume (`GameDB.needs_dm_reply`)
* the menu is saved in `pending_options`, so resume shows the same choices

Not in this stage: compaction and RAG (Stage 7b), the main menu (Stage 8).

Run it on its own:  python -m aether.game.engine <campaign_slug> [--timing]
"""

from __future__ import annotations

import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence

from aether.db import GameDB, SchemaVersionError
from aether.game.helpers import (
    all_character_names,
    apply_tags,
    detect_destination,
    difficulty_for,
    register_new_npcs,
)
from aether.llm import calls, prompts
from aether.llm.client import LLMClient, LLMError
from aether.logic.memory import assemble_prompt, trim_to_budget
from aether.logic.options import (
    ActionOption,
    detect_situation,
    format_menu,
    parse_choice,
    select_categories,
)
from aether.logic.stats import ABILITIES, format_roll, roll_check
from aether.logic.story import DbWorldView, build_story_block, drift_level, evaluate_beat_condition
from aether.models import Location

QUIT_WORDS = {"exit", "quit", "/quit", "/exit"}
MAX_ACTION_CHARS = 300


class Quit(Exception):
    """The player asked to leave."""


@dataclass
class Action:
    text: str
    skill: Optional[str] = None
    difficulty: str = "medium"
    advances: bool = False      # the judge said it pursues the current objective


class GameEngine:
    def __init__(
        self,
        client: LLMClient,
        slug: str,
        *,
        campaigns_dir: Optional[Path] = None,
        input_fn: Callable[[str], str] = input,
        print_fn: Callable[..., None] = print,
        rng: Optional[random.Random] = None,
        show_timing: bool = False,
    ):
        self.client = client
        self.cfg = client.config
        self.slug = slug
        self.campaigns_dir = campaigns_dir
        self.input = input_fn
        self.print = print_fn
        self.rng = rng or random.Random()
        self.show_timing = show_timing
        self.db: Optional[GameDB] = None
        self.campaign = None
        self.ended: Optional[str] = None     # None, "dead" or "won"

    # ------------------------------------------------------------------
    # Running
    # ------------------------------------------------------------------

    def run(self) -> bool:
        """Play until the player quits or the game ends. False if it could not start or failed."""
        try:
            self.db = GameDB.open(self.slug, self.campaigns_dir)
        except (FileNotFoundError, SchemaVersionError) as e:
            self.print(f"\n[Error] Cannot open campaign '{self.slug}': {e}")
            return False
        try:
            if not self._prepare():
                return False
            self._play()
        except (Quit, KeyboardInterrupt, EOFError):
            self.print("\nProgress saved. Safe travels!")
        except LLMError as e:
            self.print(f"\n[LLM error] {e}\nYour progress is saved. Continue the campaign to try again.")
            return False
        finally:
            self.db.close()
        return True

    def _prepare(self) -> bool:
        db = self.db
        self.campaign = db.get_campaign()
        player = db.get_player()
        if self.campaign is None or player is None:
            self.print("\n[Error] This campaign has no character yet. Run character creation first.")
            return False
        db.activate_first_beat()
        self.print("\n=======================================================")
        self.print("                 AETHER GAME TERMINAL")
        self.print("=======================================================")
        self.print(f"Campaign: {self.campaign.name}   |   Character: {player.name}")
        self.print("Pick a number. Commands: /stats /inventory /map /help. Type 'exit' to quit.")

        if db.is_campaign_complete():
            self.print("\n[Story] This campaign is already complete.")
            self.ended = "won"
        elif player.hp <= 0:
            self._game_over()
        elif db.count_messages() == 0:
            self._opening()
        else:
            self._resume()
        return True

    def _play(self) -> None:
        while not self.ended:
            options = self._current_options()
            if self.ended:
                break
            self.print("\n" + format_menu(options))
            action = self._choose(options)
            if action is not None:
                self._take_action(action)

    # ------------------------------------------------------------------
    # World access
    # ------------------------------------------------------------------

    def _player(self):
        return self.db.get_player()

    def _location(self) -> Location:
        return self.db.get_location(self._player().location_id)

    def _scene(self):
        """(player, location, connected locations, characters present, inventory)."""
        player = self._player()
        loc = self.db.get_location(player.location_id)
        return (
            player,
            loc,
            self.db.get_connected_locations(loc.id),
            self.db.list_characters_at(loc.id),
            self.db.get_inventory(player.id),
        )

    def _last_narration(self) -> str:
        recent = self.db.get_recent_messages(1, include_system=False)
        return recent[0].content if recent and recent[0].role == "dm" else ""

    def _ensure_description(self, loc: Location) -> Location:
        if loc.description:
            return loc
        self.print(f"\n[System] Describing {loc.name}...")
        text = calls.describe_location(
            self.client, name=loc.name, concept=loc.brief_concept, vibe=self.campaign.theme_vibe
        )
        self.db.set_location_description(loc.id, text)
        return self.db.get_location(loc.id)

    # ------------------------------------------------------------------
    # Starting and resuming
    # ------------------------------------------------------------------

    def _opening(self) -> None:
        db = self.db
        player, loc, connected, npcs, inventory = self._scene()
        loc = self._ensure_description(loc)
        beat = db.get_current_beat()
        messages = prompts.opening_messages(self.campaign, player, inventory, loc, connected, npcs, beat)
        known = prompts.opening_known_text(self.campaign, player, inventory, db.list_locations(), npcs)
        self.print("\n[System] Setting the scene...")
        text, tags = calls.narrate_opening(self.client, messages, known_text=known)
        text = text or "The adventure begins."
        with db.transaction():
            db.add_message("dm", text, turn_number=0)
            apply_tags(db, player.id, tags, self.print)
        self.print(f"\nDungeon Master:\n{text}")
        self._after_narration(text)

    def _resume(self) -> None:
        db = self.db
        self.print("\n=== Continuing your campaign ===")
        name = self._player().name
        for m in db.get_recent_messages(self.cfg.memory.resume_display_messages, include_system=False):
            who = name if m.role == "player" else "Dungeon Master"
            self.print(f"\n{who}: {m.content}")
        if db.needs_dm_reply():
            self.print("\n[System] The last turn was interrupted. Finishing it...")
            self._ensure_description(self._location())
            self._respond()

    def _current_options(self) -> List[ActionOption]:
        raw = self.db.get_state().pending_options
        if raw is None:
            self._make_options(self._last_narration())
            if self.ended:
                return []
            raw = self.db.get_state().pending_options or []
        return [ActionOption.from_dict(d) for d in raw]

    # ------------------------------------------------------------------
    # Reading input
    # ------------------------------------------------------------------

    def _read(self, prompt: str = "\nYou: ", allow_empty: bool = False) -> str:
        while True:
            text = self.input(prompt).strip()
            if text.lower() in QUIT_WORDS:
                raise Quit()
            if text or allow_empty:
                return text

    def _choose(self, options: Sequence[ActionOption]) -> Optional[Action]:
        raw = self._read()
        if raw.startswith("/"):
            self._command(raw)
            return None
        choice = parse_choice(raw, len(options))
        if choice is None:
            self.print(f"[System] Please pick a number from 1 to {len(options) + 1}.")
            return None
        if choice.kind == "option":
            o = options[choice.index]
            return Action(o.text, o.skill, difficulty_for(o.category))
        return self._custom_action()

    def _custom_action(self) -> Optional[Action]:
        """Read a typed action and have the judge look at it. None means back to the menu."""
        rejections = 0
        limit = self.cfg.gameplay.max_consecutive_rejections
        while True:
            text = self._read("\nYour action (Enter for the menu): ", allow_empty=True)
            if not text:
                return None
            if text.startswith("/"):
                self._command(text)
                continue
            text = text[:MAX_ACTION_CHARS]
            player, loc, connected, present, inventory = self._scene()
            self.print("\n[System] Judging your action...")
            verdict = calls.judge_action(
                self.client,
                action=text,
                narration=self._last_narration(),
                state_block=prompts.scene_state_block(player, inventory, loc, connected, present),
                beat=self.db.get_current_beat(),
            )
            if verdict.verdict == "impossible":
                rejections += 1
                self.print(f"[Not possible] {verdict.reason}")
                if rejections >= limit:
                    self.print("[System] Back to the menu.")
                    return None
                continue
            skill = verdict.skill if verdict.verdict == "risky" else None
            return Action(text, skill, verdict.difficulty, verdict.advances_beat)

    def _command(self, raw: str) -> None:
        command = raw.strip().lower()
        player, loc, connected, present, inventory = self._scene()
        if command == "/stats":
            self.print("\n" + prompts.character_sheet_block(player, inventory, include_backstory=True))
        elif command == "/inventory":
            items = ", ".join(f"{n} x{q}" if q > 1 else n for n, q in inventory) or "empty"
            self.print(f"\nInventory: {items}")
        elif command == "/map":
            paths = ", ".join(l.name for l in connected) or "none"
            here = ", ".join(c.name for c in present) or "nobody"
            self.print(f"\nYou are at: {loc.name}\nPaths from here: {paths}\nHere with you: {here}")
        else:
            self.print("\nCommands: /stats /inventory /map /help. Type 'exit' to quit.")

    # ------------------------------------------------------------------
    # One turn
    # ------------------------------------------------------------------

    def _take_action(self, action: Action) -> None:
        db = self.db
        started = time.perf_counter()
        self.client.call_log.clear()

        player, loc, connected, _, _ = self._scene()
        destination = detect_destination(action.text, connected)

        roll_line = None
        if action.skill:
            abilities = {a: getattr(player, a) for a in ABILITIES}
            roll_line = format_roll(roll_check(abilities, action.skill, action.difficulty, self.rng))
            self.print(f"\n[Roll] {roll_line}")

        with db.transaction():
            db.advance_turn()
            db.set_pending_options(None)
            db.add_message("player", action.text)
            if roll_line:
                db.add_message("system", roll_line)
            if destination:
                db.purge_ephemeral_npcs(loc.id)
                db.move_character(player.id, destination.id)
                db.add_message("system", f"Travel: {player.name} arrives at {destination.name}")
            if action.advances:
                db.reset_drift_clock()
        if destination:
            self.print(f"\n[Travel] {destination.name}")
            self._ensure_description(destination)

        self._respond()
        self._print_timing(time.perf_counter() - started)

    def _respond(self) -> None:
        """Narrate the answer to the last player message, then prepare the next menu."""
        text = self._narrate()
        self._after_narration(text)

    def _after_narration(self, narration: str) -> None:
        if self._player().hp <= 0:
            self._game_over()
            return
        self._check_code_beat()
        if self.ended:
            return
        self._make_options(narration)

    def _narrate(self) -> str:
        db = self.db
        mem = self.cfg.memory
        player, loc, connected, present, inventory = self._scene()
        beat = db.get_current_beat()
        state = db.get_state()
        level = drift_level(state.turns_since_beat_progress, self.cfg.gameplay.drift_threshold_turns)
        notes = {c.id: n for c in present if (n := db.get_npc_notes(c.id))}

        # Stable blocks first, the ones that change every turn last.
        blocks = [
            prompts.campaign_block(self.campaign),
            prompts.location_block(loc, connected),
            prompts.npcs_block(present, notes),
            prompts.character_sheet_block(player, inventory),
            build_story_block(beat, level, self.campaign.primary_threat),
        ]
        tail = trim_to_budget(db.get_uncompacted_messages(), mem.history_budget_tokens)
        chat = assemble_prompt(prompts.dm_system_prompt(self.campaign), db.get_latest_summary(), tail, blocks)

        self.print("\nDungeon Master:")
        parts: List[str] = []
        tags = []
        for result in calls.narrate_stream(self.client, chat):
            if result.text:
                self.print(result.text, end="", flush=True)
                parts.append(result.text)
            tags.extend(result.tags)
        self.print()

        text = calls.tidy_narration("".join(parts)) or "The moment passes in silence."
        with db.transaction():
            db.add_message("dm", text)
            apply_tags(db, player.id, tags, self.print)
        return text

    # ------------------------------------------------------------------
    # Story
    # ------------------------------------------------------------------

    def _check_code_beat(self) -> None:
        beat = self.db.get_current_beat()
        if beat and beat.condition_type == "location" and evaluate_beat_condition(beat, DbWorldView(self.db)):
            self._complete_beat()

    def _complete_beat(self) -> None:
        db = self.db
        while True:
            beat = db.get_current_beat()
            if beat is None:
                return
            nxt = db.complete_current_beat()
            self.print(f"\n[Story] Objective complete: {beat.title}")
            if nxt is None:
                self.print("\n=======================================================")
                self.print("   VICTORY: the campaign is complete. Well played!")
                self.print("=======================================================")
                self.ended = "won"
                return
            # the next objective may already be true (the player stands where it wants them)
            if not (nxt.condition_type == "location" and evaluate_beat_condition(nxt, DbWorldView(db))):
                return

    def _game_over(self) -> None:
        self.print("\n=======================================================")
        self.print("   GAME OVER: your character has fallen.")
        self.print("=======================================================")
        self.ended = "dead"

    def _make_options(self, narration: str) -> None:
        """Write the next menu (plus new characters and referee beat progress) and save it."""
        db = self.db
        player, loc, connected, present, inventory = self._scene()
        beat = db.get_current_beat()
        check = beat is not None and beat.condition_type == "referee"
        situation = detect_situation([c.name for c in present], narration)
        categories = select_categories(
            situation, self.cfg.gameplay.num_options, db.get_state().current_turn, has_story_beat=beat is not None
        )
        self.print("\n[System] Thinking of options...")
        result = calls.generate_options(
            self.client,
            narration=narration,
            categories=categories,
            state_block=prompts.scene_state_block(player, inventory, loc, connected, present),
            known_npc_names=all_character_names(db),
            beat=beat,
            check_beat=check,
        )
        with db.transaction():
            created = register_new_npcs(db, result.new_npcs, loc.id)
            db.set_pending_options([o.to_dict() for o in result.options])
        for name in created:
            self.print(f"[Referee] Met: {name}")
        if result.beat_achieved and check:
            self._complete_beat()

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------

    def _print_timing(self, seconds: float) -> None:
        if not self.show_timing:
            return
        self.print(f"\n[Timing] turn took {seconds:.0f}s")
        for s in self.client.call_log:
            self.print(
                f"   {s.role:<10} load {s.load_s:4.1f}s | prompt {s.prompt_tokens:>5} tok in {s.prompt_s:5.1f}s "
                f"({s.prompt_tps:4.0f}/s) | generated {s.gen_tokens:>4} tok at {s.gen_tps:4.1f}/s"
            )


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
    show_timing = "--timing" in args
    args = [a for a in args if a != "--timing"]
    slug = args[0] if args else _pick_campaign()
    if not slug:
        return 1
    client = LLMClient()
    try:
        missing = client.missing_models()
    except LLMError as e:
        print(f"[LLM error] {e}")
        return 1
    if missing:
        print("These models are not pulled yet:")
        for m in missing:
            print(f"  ollama pull {m}")
        return 1
    return 0 if GameEngine(client, slug, show_timing=show_timing).run() else 1


if __name__ == "__main__":
    raise SystemExit(main())
