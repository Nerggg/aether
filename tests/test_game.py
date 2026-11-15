import json
import random

from aether.config import Config
from aether.creation.campaign import World, save_campaign
from aether.creation.character import save_character
from aether.db import GameDB
from aether.game import helpers
from aether.game.engine import GameEngine
from aether.llm.client import LLMClient
from aether.llm.schemas import NewNpc
from aether.logic.tags import Tag
from aether.models import CampaignInfo, Character, Location, StoryBeat
from fakes import FakeBackend


def js(**kw):
    return json.dumps(kw)


def ask(*answers):
    it = iter(answers)
    return lambda prompt="": next(it)


# --- helpers ------------------------------------------------------------------------------------

def make_db(tmp_path):
    db = GameDB.create("g", tmp_path)
    db.add_location("a", "A", "x")
    pid = db.add_character(Character(name="Aria", type="player", location_id="a", hp=10, max_hp=10))
    return db, pid


def test_difficulty_for():
    assert helpers.difficulty_for("bold") == "hard"
    assert helpers.difficulty_for("cautious") == "easy"
    assert helpers.difficulty_for("story") == "medium" and helpers.difficulty_for(None) == "medium"


def test_detect_destination():
    cave = Location("cave", "The Rotting Cave", "x")
    ferry = Location("ferry", "Ferry", "y")
    d = helpers.detect_destination
    assert d("Head into the Rotting Cave", [cave, ferry]) == cave
    assert d("go to the ferry", [cave, ferry]) == ferry
    assert d("Ask about the rotting cave", [cave, ferry]) is None      # no movement verb
    assert d("Walk away", [cave, ferry]) is None                        # no destination
    assert d("Head into the Rotting Cave", [ferry]) is None            # not connected


def test_apply_tags(tmp_path):
    db, pid = make_db(tmp_path)
    said = []
    tags = [Tag("TAKE_DAMAGE", "3"), Tag("TAKE_DAMAGE", "lots"), Tag("HEAL_HP", "50"),
            Tag("ADD_ITEM", "Rope"), Tag("REMOVE_ITEM", "rope"), Tag("REMOVE_ITEM", "Sword")]
    helpers.apply_tags(db, pid, tags, said.append)
    assert db.get_player().hp == 10                      # 10 - 3, then healed (capped at 20) back to max
    assert db.get_inventory(pid) == []
    assert "[Item gained] Rope" in said and "[Item lost] rope" in said
    assert not any("Sword" in s for s in said)
    helpers.apply_tags(db, pid, [Tag("TAKE_DAMAGE", "500")], said.append)
    assert db.get_player().hp == 0                       # 500 is capped at 20, then clamped to 0 HP
    db.close()


def test_register_new_npcs(tmp_path):
    db, _ = make_db(tmp_path)
    db.add_character(Character(name="Hesk", type="ally", location_id="a"))
    created = helpers.register_new_npcs(db, [
        NewNpc(name="Tulle", type="ally", is_persistent=True, brief_backstory="A reed-woman."),
        NewNpc(name="hesk"), NewNpc(name="Aria"), NewNpc(name="Gruk", type="enemy"), NewNpc(name="  "),
    ], "a")
    assert created == ["Tulle", "Gruk"]
    by_name = {c.name: c for c in db.list_characters_at("a")}
    assert (by_name["Tulle"].hp, by_name["Tulle"].backstory) == (12, "A reed-woman.")
    assert (by_name["Gruk"].hp, by_name["Gruk"].is_persistent) == (6, False)
    assert sorted(helpers.all_character_names(db)) == ["Aria", "Gruk", "Hesk", "Tulle"]
    db.close()


# --- the whole loop ---------------------------------------------------------------------------

BEATS = [StoryBeat(1, "Find the Bell", "Discover where the bell rings from."),
         StoryBeat(2, "Win", "Defeat the witch.")]
OPT = js(options=[
    {"category": "story", "text": "Search the pier", "skill": "investigation"},
    {"category": "cautious", "text": "Back away slowly"},
    {"category": "bold", "text": "Wade into the water", "skill": "athletics"},
])


def make_game(tmp_path, beats=BEATS):
    info = CampaignInfo(
        name="Camp", slug="camp", setting_description="A foggy marsh.", primary_threat="A witch.",
        starting_quest_hook="Find the bell.", starting_patron="Mother Hesk, ferry-keeper",
        starting_location_id="ferry",
    )
    world = World(
        [Location("ferry", "Ferry", "A landing."), Location("cave", "Rotting Cave", "A cave."),
         Location("end", "End", "z")],
        [("ferry", "cave"), ("cave", "end")],
    )
    save_campaign(info, world, beats, tmp_path)
    save_character("camp", name="Aria", race="Elf", char_class="Ranger", backstory="A scout.",
                   campaigns_dir=tmp_path)


def make_engine(tmp_path, responses, inputs):
    out = []
    client = LLMClient(Config(), FakeBackend(*responses))
    engine = GameEngine(
        client, "camp", campaigns_dir=tmp_path, input_fn=ask(*inputs),
        print_fn=lambda *a, **k: out.append(" ".join(str(x) for x in a)), rng=random.Random(1),
    )
    return engine, out


def test_new_game_opens_plays_one_turn_and_saves_everything(tmp_path):
    make_game(tmp_path)
    responses = [
        "A rotting landing in the fog.",                          # location description
        ["You stand on the pier. Mother Hesk waits."],            # opener
        OPT,
        ["The fog thickens around you. [TAKE_DAMAGE: 2]"],        # narration for the chosen action
        OPT,
    ]
    engine, out = make_engine(tmp_path, responses, ["1", "exit"])
    assert engine.run() is True
    assert len(engine.client._backend.calls) == 5
    with GameDB.open("camp", tmp_path) as db:
        assert [m.role for m in db.get_recent_messages(10)] == ["dm", "player", "system", "dm"]
        assert db.get_player().hp == 9                            # 11 - 2
        assert db.get_state().current_turn == 1
        assert db.get_state().pending_options is not None
        assert db.get_current_beat().title == "Find the Bell"
    assert any("[Roll]" in line for line in out) and any("Damage" in line for line in out)


def test_resume_shows_the_saved_menu_without_calling_the_model(tmp_path):
    make_game(tmp_path)
    first, _ = make_engine(tmp_path, ["A landing.", ["You stand on the pier."], OPT], ["exit"])
    first.run()
    second, out = make_engine(tmp_path, ["unused"], ["exit"])
    assert second.run() is True
    assert second.client._backend.calls == []
    assert any("Continuing" in line for line in out)
    assert any("1. Search the pier" in line for line in out)


def test_an_interrupted_turn_is_finished_on_resume(tmp_path):
    make_game(tmp_path)
    with GameDB.open("camp", tmp_path) as db:
        db.set_location_description("ferry", "A landing.")
        db.add_message("dm", "You arrive.", turn_number=0)
        db.add_message("player", "Search the pier", turn_number=1)
    engine, out = make_engine(tmp_path, [["The fog parts."], OPT], ["exit"])
    assert engine.run() is True
    with GameDB.open("camp", tmp_path) as db:
        assert db.get_last_message().content == "The fog parts."
        assert db.needs_dm_reply() is False
        assert db.get_state().pending_options is not None


def test_moving_to_a_connected_location_completes_a_location_beat(tmp_path):
    beats = [StoryBeat(1, "Reach the Cave", "Reach the Rotting Cave.", "location", "cave"),
             StoryBeat(2, "Win", "Defeat the witch.")]
    make_game(tmp_path, beats)
    move_opt = js(options=[
        {"category": "story", "text": "Head into the Rotting Cave"},
        {"category": "cautious", "text": "Wait quietly here"},
        {"category": "bold", "text": "Shout across the water"},
    ])
    responses = ["A landing.", ["You stand on the pier."], move_opt,
                 "A damp cave.", ["You duck into the cold dark."], OPT]
    engine, out = make_engine(tmp_path, responses, ["1", "exit"])
    engine.run()
    with GameDB.open("camp", tmp_path) as db:
        assert db.get_player().location_id == "cave"
        assert [b.status for b in db.get_beats()] == ["done", "active"]
        assert db.get_location("cave").description == "A damp cave."
        assert "arrives at Rotting Cave" in db.get_recent_messages(10)[2].content
    assert any("Objective complete" in line for line in out)


def test_rejected_custom_actions_return_to_the_menu_and_save_nothing(tmp_path):
    make_game(tmp_path)
    judge = js(verdict="impossible", reason="You have no fire magic.")
    inputs = ["4", "I cast fireball", "I cast a bigger fireball", "exit"]
    engine, out = make_engine(tmp_path, ["A landing.", ["You stand on the pier."], OPT, judge], inputs)
    assert engine.run() is True
    assert sum("[Not possible]" in line for line in out) == 2
    assert any("Back to the menu" in line for line in out)
    with GameDB.open("camp", tmp_path) as db:
        assert [m.role for m in db.get_recent_messages(10)] == ["dm"]


def test_words_at_the_menu_show_a_hint(tmp_path):
    make_game(tmp_path)
    engine, out = make_engine(tmp_path, ["A landing.", ["You stand on the pier."], OPT], ["hmm", "exit"])
    engine.run()
    assert any("Please pick a number from 1 to 4" in line for line in out)


def test_a_campaign_without_a_character_is_refused(tmp_path):
    with GameDB.create("camp", tmp_path) as db:
        db.add_location("a", "A", "x")
    engine, out = make_engine(tmp_path, ["x"], [])
    assert engine.run() is False
    assert any("no character" in line for line in out)
