import pytest

from models import StoryBeat
from story import (
    DbWorldView,
    DriftLevel,
    build_story_block,
    drift_level,
    evaluate_beat_condition,
    is_valid_condition,
)


def beat(condition_type="referee", value="", title="Find the cave", objective="Reach the witch's cave."):
    return StoryBeat(position=1, title=title, objective=objective,
                     condition_type=condition_type, condition_value=value)


class FakeWorld:
    def __init__(self, location="tavern", items=(), dead=()):
        self.location, self.items, self.dead = location, set(items), set(dead)

    def player_location(self):
        return self.location

    def player_has_item(self, name):
        return name in self.items

    def is_npc_dead(self, name):
        return name in self.dead


# --- drift clock -------------------------------------------------------------------------

@pytest.mark.parametrize("turns,expected", [
    (0, DriftLevel.NONE), (2, DriftLevel.NONE),
    (3, DriftLevel.HINT), (5, DriftLevel.HINT),
    (6, DriftLevel.PRESSURE), (8, DriftLevel.PRESSURE),
    (9, DriftLevel.URGENT), (40, DriftLevel.URGENT),
])
def test_drift_levels_for_threshold_six(turns, expected):
    assert drift_level(turns, 6) == expected


def test_drift_never_decreases_as_turns_pass():
    levels = [drift_level(t, 6) for t in range(30)]
    assert levels == sorted(levels)


def test_drift_threshold_must_be_positive():
    with pytest.raises(ValueError):
        drift_level(3, 0)


def test_threshold_of_one_escalates_quickly():
    assert drift_level(0, 1) == DriftLevel.NONE
    assert drift_level(1, 1) == DriftLevel.PRESSURE


# --- story block ----------------------------------------------------------------------------

def test_no_beat_means_no_block():
    assert build_story_block(None, DriftLevel.URGENT) == ""


def test_block_always_names_the_objective():
    text = build_story_block(beat(), DriftLevel.NONE)
    assert "CURRENT STORY OBJECTIVE:" in text
    assert "Find the cave: Reach the witch's cave." in text
    assert "world moves on" not in text


def test_each_level_adds_its_own_instruction():
    hint = build_story_block(beat(), DriftLevel.HINT)
    pressure = build_story_block(beat(), DriftLevel.PRESSURE, "the Swamp Witch")
    urgent = build_story_block(beat(), DriftLevel.URGENT, "the Swamp Witch")
    assert "more noticeable" in hint
    assert "world moves on" in pressure and "(the Swamp Witch)" in pressure
    assert "decisive" in urgent and "(the Swamp Witch)" in urgent
    assert len({hint, pressure, urgent}) == 3


def test_threat_is_optional():
    assert "()" not in build_story_block(beat(), DriftLevel.PRESSURE)


# --- conditions ---------------------------------------------------------------------------

def test_location_condition():
    b = beat("location", "cave")
    assert evaluate_beat_condition(b, FakeWorld("cave")) is True
    assert evaluate_beat_condition(b, FakeWorld("tavern")) is False


def test_has_item_condition():
    b = beat("has_item", "Silver Key")
    assert evaluate_beat_condition(b, FakeWorld(items=["Silver Key"])) is True
    assert evaluate_beat_condition(b, FakeWorld()) is False


def test_npc_dead_condition():
    b = beat("npc_dead", "Swamp Witch")
    assert evaluate_beat_condition(b, FakeWorld(dead=["Swamp Witch"])) is True
    assert evaluate_beat_condition(b, FakeWorld()) is False


def test_referee_condition_needs_the_llm():
    assert evaluate_beat_condition(beat("referee", "Convince the elder"), FakeWorld()) is None


def test_empty_value_never_counts_as_met():
    assert evaluate_beat_condition(beat("has_item", ""), FakeWorld(items=[""])) is False
    assert evaluate_beat_condition(beat("npc_dead", "  "), FakeWorld(dead=[""])) is False


def test_condition_value_is_trimmed():
    assert evaluate_beat_condition(beat("location", " cave "), FakeWorld("cave")) is True


# --- validating LLM-proposed conditions ---------------------------------------------------------

def test_valid_conditions():
    known = {"tavern", "cave"}
    assert is_valid_condition("referee", "", known)
    assert is_valid_condition("location", "cave", known)
    assert is_valid_condition("has_item", "Silver Key", known)
    assert is_valid_condition("npc_dead", "Witch", known)


def test_invalid_conditions():
    known = {"tavern", "cave"}
    assert not is_valid_condition("location", "atlantis", known)
    assert not is_valid_condition("location", "", known)
    assert not is_valid_condition("has_item", "  ", known)
    assert not is_valid_condition("vibes", "x", known)


# --- adapter over the real database (skipped when db.py isn't importable) ----------------------

def test_db_world_view_against_real_database(tmp_path):
    pytest.importorskip("db")
    from db import GameDB
    from models import CampaignInfo  # noqa: F401  (ensures real models are in use)
    from models import Character

    game = GameDB.create("world_view", campaigns_dir=tmp_path)
    try:
        game.add_location("tavern", "Tavern", "An inn.")
        game.add_location("cave", "Cave", "A cave.")
        game.add_connection("tavern", "cave")
        pid = game.add_character(Character(name="Aria", type="player", location_id="tavern", hp=10, max_hp=10))
        witch = game.add_character(Character(name="Swamp Witch", type="enemy", location_id="cave", hp=9, max_hp=9))
        view = DbWorldView(game)

        assert view.player_location() == "tavern"
        game.move_character(pid, "cave")
        assert view.player_location() == "cave"

        assert view.player_has_item("silver key") is False
        game.add_item(pid, "Silver Key")
        assert view.player_has_item("silver key") is True

        assert view.is_npc_dead("swamp witch") is False
        game.set_character_status(witch, "dead")
        assert view.is_npc_dead("swamp witch") is True
        assert view.is_npc_dead("nobody") is False
    finally:
        game.close()


def test_db_world_view_with_no_player(tmp_path):
    pytest.importorskip("db")
    from db import GameDB

    game = GameDB.create("empty_world", campaigns_dir=tmp_path)
    try:
        view = DbWorldView(game)
        assert view.player_location() is None
        assert not view.player_has_item("anything")
    finally:
        game.close()
