import sqlite3

import pytest

from db import (
    SCHEMA_VERSION,
    CampaignExistsError,
    GameDB,
    SchemaVersionError,
    slugify,
)
from models import CampaignInfo, Character, StoryBeat, Summary


@pytest.fixture
def db(tmp_path):
    game = GameDB.create("test_campaign", campaigns_dir=tmp_path)
    yield game
    game.close()


@pytest.fixture
def world(db):
    """A small map, a campaign row and a player standing in the tavern."""
    db.add_location("tavern", "The Tavern", "A smoky inn.")
    db.add_location("cave", "The Cave", "A damp cave.")
    db.add_location("road", "The Road", "A muddy road.")
    db.add_connection("tavern", "cave")
    db.add_connection("road", "tavern")
    db.set_campaign(
        CampaignInfo(
            name="Test",
            slug="test_campaign",
            setting_description="A swamp.",
            primary_threat="A witch.",
            starting_quest_hook="Find the witch.",
            starting_location_id="tavern",
        )
    )
    player_id = db.add_character(
        Character(
            name="Aria", type="player", location_id="tavern",
            race="Elf", char_class="Ranger", hp=12, max_hp=12, dexterity=16,
        )
    )
    return {"player_id": player_id}


# --- slugs and campaign files ------------------------------------------------

def test_slugify():
    assert slugify("Whispers of the Ancient Dark!") == "whispers_of_the_ancient_dark"
    assert slugify("  A -- B  ") == "a_b"
    assert slugify("???") == "campaign"


def test_invalid_slug_rejected(tmp_path):
    for bad in ("../evil", "Has Space", "UPPER", ""):
        with pytest.raises(ValueError):
            GameDB.create(bad, campaigns_dir=tmp_path)


def test_create_makes_file_and_sets_version(tmp_path):
    game = GameDB.create("one", campaigns_dir=tmp_path)
    game.close()
    path = tmp_path / "one" / "game.db"
    assert path.exists()
    conn = sqlite3.connect(path)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    conn.close()


def test_create_twice_fails(tmp_path):
    GameDB.create("one", campaigns_dir=tmp_path).close()
    with pytest.raises(CampaignExistsError):
        GameDB.create("one", campaigns_dir=tmp_path)


def test_open_missing_campaign(tmp_path):
    with pytest.raises(FileNotFoundError):
        GameDB.open("ghost", campaigns_dir=tmp_path)


def test_open_wrong_schema_version(tmp_path):
    GameDB.create("old", campaigns_dir=tmp_path).close()
    conn = sqlite3.connect(tmp_path / "old" / "game.db")
    conn.execute("PRAGMA user_version = 99")
    conn.commit()
    conn.close()
    with pytest.raises(SchemaVersionError):
        GameDB.open("old", campaigns_dir=tmp_path)


def test_unique_slug(tmp_path):
    GameDB.create("quest", campaigns_dir=tmp_path).close()
    assert GameDB.unique_slug("Quest", tmp_path) == "quest_2"
    GameDB.create("quest_2", campaigns_dir=tmp_path).close()
    assert GameDB.unique_slug("Quest", tmp_path) == "quest_3"
    assert GameDB.unique_slug("Fresh Start", tmp_path) == "fresh_start"


def test_list_campaigns(tmp_path):
    assert GameDB.list_campaigns(tmp_path) == []
    GameDB.create("alpha", campaigns_dir=tmp_path).close()
    GameDB.create("beta", campaigns_dir=tmp_path).close()
    assert sorted(GameDB.list_campaigns(tmp_path)) == ["alpha", "beta"]


def test_reopen_sees_saved_data(tmp_path):
    game = GameDB.create("save", campaigns_dir=tmp_path)
    game.add_message("dm", "Hello adventurer.")
    game.close()
    reopened = GameDB.open("save", campaigns_dir=tmp_path)
    assert reopened.get_last_message().content == "Hello adventurer."
    reopened.close()


# --- transactions ------------------------------------------------------------

def test_transaction_rolls_back_on_error(db):
    with pytest.raises(RuntimeError):
        with db.transaction():
            db.add_message("player", "I attack!")
            raise RuntimeError("crash mid-turn")
    assert db.count_messages() == 0


def test_transaction_commits_on_success(db):
    with db.transaction():
        db.add_message("player", "one")
        db.add_message("dm", "two")
    assert db.count_messages() == 2


def test_nested_transaction_inner_failure_keeps_outer_work(db):
    with db.transaction():
        db.add_message("player", "kept")
        try:
            with db.transaction():
                db.add_message("dm", "discarded")
                raise ValueError("inner failure")
        except ValueError:
            pass
    assert [m.content for m in db.get_recent_messages(10)] == ["kept"]


# --- campaign, locations ----------------------------------------------------------

def test_campaign_roundtrip(db, world):
    campaign = db.get_campaign()
    assert campaign.name == "Test"
    assert campaign.starting_location_id == "tavern"
    assert campaign.embedding_model is None
    db.set_embedding_model("bge-small")
    assert db.get_campaign().embedding_model == "bge-small"


def test_get_campaign_before_creation(db):
    assert db.get_campaign() is None


def test_location_description_starts_empty(db, world):
    assert db.get_location("cave").description is None
    db.set_location_description("cave", "Water drips from the ceiling.")
    assert db.get_location("cave").description == "Water drips from the ceiling."


def test_set_description_unknown_location(db):
    with pytest.raises(LookupError):
        db.set_location_description("nowhere", "x")


def test_connections_work_both_ways(db, world):
    assert {l.id for l in db.get_connected_locations("tavern")} == {"cave", "road"}
    assert {l.id for l in db.get_connected_locations("cave")} == {"tavern"}
    assert db.get_connected_locations("road")[0].id == "tavern"


def test_connection_to_unknown_location_rejected(db, world):
    with pytest.raises(sqlite3.IntegrityError):
        db.add_connection("tavern", "atlantis")


def test_self_connection_rejected(db, world):
    with pytest.raises(ValueError):
        db.add_connection("tavern", "tavern")


# --- characters ------------------------------------------------------------------

def test_player_roundtrip(db, world):
    player = db.get_player()
    assert player.name == "Aria"
    assert (player.race, player.char_class) == ("Elf", "Ranger")
    assert player.location_id == "tavern"
    assert player.is_persistent is True
    assert db.has_player()


def test_no_player_initially(db):
    assert db.get_player() is None
    assert not db.has_player()


def test_only_one_player_allowed(db, world):
    with pytest.raises(sqlite3.IntegrityError):
        db.add_character(Character(name="Bob", type="player", location_id="tavern"))


def test_stat_bounds_enforced(db, world):
    with pytest.raises(sqlite3.IntegrityError):
        db.add_character(Character(name="Hulk", type="enemy", location_id="cave", strength=99))


def test_hp_cannot_exceed_max(db, world):
    with pytest.raises(sqlite3.IntegrityError):
        db.add_character(Character(name="Odd", type="ally", location_id="cave", hp=20, max_hp=10))


def test_character_needs_real_location(db, world):
    with pytest.raises(sqlite3.IntegrityError):
        db.add_character(Character(name="Ghost", type="ally", location_id="atlantis"))


def test_invalid_character_type_rejected(db, world):
    with pytest.raises(sqlite3.IntegrityError):
        db.add_character(Character(name="X", type="wizard", location_id="cave"))


def test_move_player_persists(db, world):
    db.move_character(world["player_id"], "cave")
    assert db.get_player().location_id == "cave"


def test_change_hp_clamps(db, world):
    pid = world["player_id"]
    assert db.change_hp(pid, -5) == 7
    assert db.change_hp(pid, -100) == 0
    assert db.change_hp(pid, 4) == 4
    assert db.change_hp(pid, 100) == 12


def test_change_hp_unknown_character(db):
    with pytest.raises(LookupError):
        db.change_hp(999, -1)


def test_list_characters_at_hides_player_and_dead(db, world):
    alive = db.add_character(Character(name="Gruk", type="enemy", location_id="tavern", hp=5, max_hp=5, is_persistent=False))
    dead = db.add_character(Character(name="Old Sam", type="ally", location_id="tavern"))
    db.set_character_status(dead, "dead")
    names = [c.name for c in db.list_characters_at("tavern")]
    assert names == ["Gruk"]
    all_names = [c.name for c in db.list_characters_at("tavern", include_inactive=True)]
    assert all_names == ["Gruk", "Old Sam"]
    assert alive


def test_purge_ephemeral_npcs_keeps_persistent_and_player(db, world):
    db.add_character(Character(name="Goblin", type="enemy", location_id="tavern", is_persistent=False))
    db.add_character(Character(name="Barkeep", type="merchant", location_id="tavern", is_persistent=True))
    assert db.purge_ephemeral_npcs("tavern") == 1
    assert [c.name for c in db.list_characters_at("tavern")] == ["Barkeep"]
    assert db.has_player()


# --- inventory ------------------------------------------------------------------

def test_inventory_add_and_stack(db, world):
    pid = world["player_id"]
    db.add_item(pid, "Torch")
    db.add_item(pid, "torch", 2)  # same item, different case
    db.add_item(pid, "Rope")
    assert db.get_inventory(pid) == [("Torch", 3), ("Rope", 1)]


def test_inventory_remove(db, world):
    pid = world["player_id"]
    db.add_item(pid, "Potion", 3)
    assert db.remove_item(pid, "potion") is True
    assert db.get_inventory(pid) == [("Potion", 2)]
    assert db.remove_item(pid, "Potion", 2) is True
    assert db.get_inventory(pid) == []
    assert not db.has_item(pid, "Potion")


def test_inventory_remove_more_than_owned_changes_nothing(db, world):
    pid = world["player_id"]
    db.add_item(pid, "Potion", 1)
    assert db.remove_item(pid, "Potion", 5) is False
    assert db.remove_item(pid, "Sword") is False
    assert db.get_inventory(pid) == [("Potion", 1)]


def test_has_item_ignores_case_and_spaces(db, world):
    pid = world["player_id"]
    db.add_item(pid, "  Short Sword ")
    assert db.has_item(pid, "short sword")


def test_empty_item_name_rejected(db, world):
    with pytest.raises(ValueError):
        db.add_item(world["player_id"], "   ")


def test_deleting_character_removes_inventory_and_notes(db, world):
    npc = db.add_character(Character(name="Rat", type="enemy", location_id="tavern", is_persistent=False))
    db.add_item(npc, "Cheese")
    db.set_npc_notes(npc, "Squeaks a lot.")
    db.purge_ephemeral_npcs("tavern")
    assert db.get_inventory(npc) == []
    assert db.get_npc_notes(npc) is None


# --- npc notes ------------------------------------------------------------------

def test_npc_notes_upsert(db, world):
    npc = db.add_character(Character(name="Barnaby", type="merchant", location_id="tavern"))
    assert db.get_npc_notes(npc) is None
    db.set_npc_notes(npc, "Sells rumours for silver.")
    db.set_npc_notes(npc, "Owes the player a favour.")
    assert db.get_npc_notes(npc) == "Owes the player a favour."


# --- transcript -----------------------------------------------------------------

def test_messages_keep_order_and_turns(db, world):
    db.add_message("dm", "Opening scene.", turn_number=0)
    db.advance_turn()
    db.add_message("player", "I look around.")
    db.add_message("dm", "You see a door.")
    messages = db.get_recent_messages(10)
    assert [(m.turn_number, m.role) for m in messages] == [(0, "dm"), (1, "player"), (1, "dm")]


def test_get_recent_messages_limit_and_order(db):
    for i in range(10):
        db.add_message("player" if i % 2 == 0 else "dm", f"m{i}")
    assert [m.content for m in db.get_recent_messages(3)] == ["m7", "m8", "m9"]


def test_get_recent_messages_can_skip_system(db):
    db.add_message("dm", "a")
    db.add_message("system", "Rolled 14.")
    db.add_message("dm", "b")
    assert [m.content for m in db.get_recent_messages(5, include_system=False)] == ["a", "b"]


def test_get_messages_range(db):
    ids = [db.add_message("dm", f"m{i}") for i in range(5)]
    assert [m.content for m in db.get_messages(after_id=ids[1])] == ["m2", "m3", "m4"]
    assert [m.content for m in db.get_messages(after_id=ids[0], up_to_id=ids[2])] == ["m1", "m2"]


def test_invalid_role_rejected(db):
    with pytest.raises(sqlite3.IntegrityError):
        db.add_message("narrator", "hi")


def test_needs_dm_reply(db):
    assert db.needs_dm_reply() is False
    db.add_message("dm", "Opening.")
    assert db.needs_dm_reply() is False
    db.add_message("player", "I open the door.")
    assert db.needs_dm_reply() is True
    db.add_message("system", "Rolled 12.")
    assert db.needs_dm_reply() is True  # dice results don't count as a DM reply
    db.add_message("dm", "It creaks open.")
    assert db.needs_dm_reply() is False


def test_last_played_at_updates(db):
    assert db.get_state().last_played_at is None
    db.add_message("dm", "hi")
    assert db.get_state().last_played_at is not None


# --- summaries --------------------------------------------------------------------

def test_uncompacted_messages_without_summary(db):
    for i in range(3):
        db.add_message("dm", f"m{i}")
    assert len(db.get_uncompacted_messages()) == 3
    assert db.get_latest_summary() is None


def test_summary_marks_where_the_tail_starts(db):
    ids = [db.add_message("dm", f"m{i}") for i in range(6)]
    db.add_summary(
        Summary(
            covers_through_message_id=ids[3],
            narrative="The party reached the cave.",
            key_npcs=["Barnaby"],
            open_threads=["Find the missing child"],
            promises_debts=["Owes Barnaby 5 silver"],
        )
    )
    latest = db.get_latest_summary()
    assert latest.narrative == "The party reached the cave."
    assert latest.key_npcs == ["Barnaby"]
    assert latest.open_threads == ["Find the missing child"]
    assert latest.promises_debts == ["Owes Barnaby 5 silver"]
    assert [m.content for m in db.get_uncompacted_messages()] == ["m4", "m5"]


def test_newest_summary_wins_and_old_ones_are_kept(db):
    ids = [db.add_message("dm", f"m{i}") for i in range(6)]
    db.add_summary(Summary(covers_through_message_id=ids[1], narrative="first"))
    db.add_summary(Summary(covers_through_message_id=ids[4], narrative="second"))
    assert db.get_latest_summary().narrative == "second"
    assert [m.content for m in db.get_uncompacted_messages()] == ["m5"]
    assert db.count_messages() == 6  # raw transcript is never deleted


def test_summary_must_point_at_real_message(db):
    with pytest.raises(sqlite3.IntegrityError):
        db.add_summary(Summary(covers_through_message_id=999, narrative="bad"))


# --- story beats ----------------------------------------------------------------

def make_beats(db, n=3):
    for i in range(1, n + 1):
        db.add_beat(StoryBeat(position=i, title=f"Beat {i}", objective=f"Do thing {i}"))


def test_no_current_beat_until_story_starts(db):
    make_beats(db)
    assert db.get_current_beat() is None
    first = db.activate_first_beat()
    assert first.title == "Beat 1"
    assert db.get_current_beat().status == "active"
    assert db.get_state().current_beat_id == first.id


def test_activate_first_beat_is_idempotent(db):
    make_beats(db)
    a = db.activate_first_beat()
    b = db.activate_first_beat()
    assert a.id == b.id


def test_activate_with_no_beats(db):
    assert db.activate_first_beat() is None


def test_completing_beats_walks_the_story(db):
    make_beats(db, 3)
    db.activate_first_beat()
    db.advance_turn()
    nxt = db.complete_current_beat()
    assert nxt.title == "Beat 2"
    beats = db.get_beats()
    assert [b.status for b in beats] == ["done", "active", "pending"]
    assert beats[0].completed_at_turn == 1
    assert db.get_state().current_beat_id == nxt.id

    db.complete_current_beat()
    assert db.complete_current_beat() is None  # last beat finished
    assert db.get_state().current_beat_id is None
    assert db.is_campaign_complete()


def test_not_complete_while_beats_remain(db):
    assert db.is_campaign_complete() is False  # no beats at all
    make_beats(db, 2)
    db.activate_first_beat()
    assert db.is_campaign_complete() is False


def test_complete_without_active_beat(db):
    make_beats(db)
    with pytest.raises(LookupError):
        db.complete_current_beat()


def test_only_one_active_beat_allowed(db):
    make_beats(db, 2)
    db.activate_first_beat()
    with pytest.raises(sqlite3.IntegrityError):
        db.conn.execute("UPDATE story_beats SET status = 'active' WHERE position = 2")


def test_beat_positions_must_be_unique(db):
    db.add_beat(StoryBeat(position=1, title="A", objective="a"))
    with pytest.raises(sqlite3.IntegrityError):
        db.add_beat(StoryBeat(position=1, title="B", objective="b"))


def test_invalid_condition_type_rejected(db):
    with pytest.raises(sqlite3.IntegrityError):
        db.add_beat(StoryBeat(position=1, title="A", objective="a", condition_type="vibes"))


# --- game state / drift clock -------------------------------------------------------

def test_initial_state(db):
    state = db.get_state()
    assert state.current_turn == 0
    assert state.turns_since_beat_progress == 0
    assert state.pending_options is None


def test_advance_turn_runs_the_drift_clock(db):
    assert db.advance_turn() == 1
    assert db.advance_turn() == 2
    state = db.get_state()
    assert (state.current_turn, state.turns_since_beat_progress) == (2, 2)


def test_beat_completion_resets_drift_clock(db):
    make_beats(db, 2)
    db.activate_first_beat()
    for _ in range(4):
        db.advance_turn()
    assert db.get_state().turns_since_beat_progress == 4
    db.complete_current_beat()
    state = db.get_state()
    assert state.turns_since_beat_progress == 0
    assert state.current_turn == 4  # the turn counter itself keeps going


def test_pending_options_roundtrip(db):
    options = [
        {"text": "Explore the cave", "skill": "perception"},
        {"text": "Shout", "skill": "intimidation"},
    ]
    db.set_pending_options(options)
    assert db.get_state().pending_options == options
    db.set_pending_options(None)
    assert db.get_state().pending_options is None


# --- the whole point: save and resume ---------------------------------------------

def test_full_save_and_resume(tmp_path):
    game = GameDB.create("resume_me", campaigns_dir=tmp_path)
    game.add_location("tavern", "Tavern", "An inn.")
    game.add_location("cave", "Cave", "A cave.")
    game.add_connection("tavern", "cave")
    pid = game.add_character(Character(name="Aria", type="player", location_id="tavern", hp=10, max_hp=10))
    game.add_message("dm", "You stand in the tavern.", turn_number=0)

    # One complete turn saved atomically
    with game.transaction():
        game.advance_turn()
        game.add_message("player", "I go to the cave.")
        game.move_character(pid, "cave")
        game.change_hp(pid, -3)
        game.add_item(pid, "Torch")
        game.add_message("dm", "The cave swallows the light.")
        game.set_pending_options([{"text": "Light the torch"}])
    game.close()

    resumed = GameDB.open("resume_me", campaigns_dir=tmp_path)
    player = resumed.get_player()
    assert (player.location_id, player.hp) == ("cave", 7)
    assert resumed.get_inventory(pid) == [("Torch", 1)]
    assert resumed.get_last_message().content == "The cave swallows the light."
    assert resumed.get_state().pending_options == [{"text": "Light the torch"}]
    assert resumed.get_state().current_turn == 1
    assert resumed.needs_dm_reply() is False
    resumed.close()
