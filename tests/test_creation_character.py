import json
import random

import pytest

from aether.config import Config
from aether.creation import character as ch
from aether.creation import names
from aether.creation.campaign import World, save_campaign
from aether.creation.character import (
    CharacterCreator,
    clean_name,
    extract_name,
    fallback_backstories,
    generate_backstories,
    match_closed,
    name_from_phrase,
    parse_number,
    save_character,
    starting_items,
)
from aether.db import GameDB
from aether.llm.client import LLMClient
from aether.logic.stats import CLASSES, RACES
from aether.models import CampaignInfo, Location
from fakes import FakeBackend


def js(**kw):
    return json.dumps(kw)


def ask(*answers):
    it = iter(answers)
    return lambda prompt="": next(it)


INFO = CampaignInfo(
    name="Camp", slug="camp", setting_description="A foggy marsh.", primary_threat="A witch.",
    starting_quest_hook="Find the bell.", starting_patron="Mother Hesk, ferry-keeper",
    starting_location_id="ferry",
)


def make_campaign(tmp_path, slug="camp"):
    world = World(
        [Location("ferry", "Ferry", "x"), Location("cave", "Cave", "y"), Location("end", "End", "z")],
        [("ferry", "cave"), ("cave", "end")],
    )
    save_campaign(INFO, world, [], tmp_path)


def make_creator(tmp_path, responses, inputs, slug="camp"):
    out = []
    client = LLMClient(Config(), FakeBackend(*responses))
    creator = CharacterCreator(
        client, slug, campaigns_dir=tmp_path, input_fn=ask(*inputs),
        print_fn=lambda *a, **k: out.append(" ".join(str(x) for x in a)),
        rng=random.Random(5),
    )
    return creator, out


STORIES = [
    "Aria lost her squad in the fog and seeks atonement.",
    "Aria fled a burning city and hides her past.",
    "Aria is a wandering scout who owes an old debt.",
]


# --- pure helpers ------------------------------------------------------------------------------

def test_parse_number():
    assert parse_number("2", 3) == 1
    assert parse_number(" (3). ", 3) == 2
    assert parse_number("4", 3) is None
    assert parse_number("0", 3) is None
    assert parse_number("two", 3) is None
    assert parse_number("2 rogues", 3) is None


def test_match_closed():
    assert match_closed("i'd go with paladin", CLASSES) == "Paladin"
    assert match_closed("PALADIN!", CLASSES) == "Paladin"
    assert match_closed("palladin", CLASSES) == "Paladin"
    assert match_closed("an elven scout", RACES, ch.RACE_ALIASES) == "Elf"
    assert match_closed("a stranger", CLASSES) is None          # close to 'ranger' but starts differently
    assert match_closed("myself", RACES) is None
    assert match_closed("a fighter or a wizard", CLASSES) is None    # two options: ambiguous
    assert match_closed("sure", CLASSES) is None
    assert match_closed("", CLASSES) is None


def test_name_helpers():
    assert name_from_phrase("i want a human named baron") == "Baron"
    assert name_from_phrase("a ranger called Thrain Blackwood and a bow") == "Thrain Blackwood"
    assert name_from_phrase("something strong") == ""
    assert name_from_phrase("named elf") == ""                  # a race word is not a name
    assert extract_name("Aria") == "Aria"
    assert extract_name("aria") == "Aria"
    assert extract_name("sure thing, that sounds good to me") == ""
    assert clean_name("  'Aria' ") == "Aria"
    for bad in (None, "", "none", "Elf", "ranger", "a b c d e", "12", "x" * 60):
        assert clean_name(bad) == ""


def test_every_class_has_a_kit_with_valid_items():
    for cls in CLASSES:
        items = starting_items(cls)
        assert len(items) >= 3 and len({n.lower() for n, _ in items}) == len(items)
    assert ("Stale bread", 1) in starting_items("Fighter")


def test_suggested_names_are_distinct_and_from_the_lists():
    options = ch.suggest_names(random.Random(1))
    assert len(set(options)) == 3
    assert all(o.split(" ")[0] in names.FIRST_NAMES for o in options)


# --- backstory options ---------------------------------------------------------------------------

def test_generated_backstories_are_cleaned_and_topped_up():
    client = LLMClient(Config(), FakeBackend(js(options=["too short", STORIES[0], STORIES[0], STORIES[1]])))
    got = generate_backstories(client, INFO, "Aria", "Elf", "Ranger", "", 3)
    assert got[:2] == [STORIES[0], STORIES[1]] and len(got) == 3
    assert got[2] == fallback_backstories("Aria", "Elf", "Ranger")[0]


def test_backstory_model_failure_gives_fallbacks():
    client = LLMClient(Config(), FakeBackend("garbage"))
    assert generate_backstories(client, INFO, "Aria", "Elf", "Ranger", "", 3) == fallback_backstories("Aria", "Elf", "Ranger")


# --- saving ----------------------------------------------------------------------------------------------

def test_save_character_writes_player_stats_location_and_kit(tmp_path):
    make_campaign(tmp_path)
    pid = save_character("camp", name="Aria", race="Elf", char_class="Ranger",
                         backstory="A scout.", campaigns_dir=tmp_path)
    with GameDB.open("camp", tmp_path) as db:
        player = db.get_player()
        assert player.id == pid
        assert (player.name, player.race, player.char_class) == ("Aria", "Elf", "Ranger")
        assert (player.dexterity, player.hp, player.max_hp) == (17, 11, 11)
        assert player.location_id == "ferry" and player.backstory == "A scout."
        assert ("Short bow", 1) in db.get_inventory(pid)


def test_save_character_refuses_a_second_player_and_bad_input(tmp_path):
    make_campaign(tmp_path)
    save_character("camp", name="Aria", race="Elf", char_class="Ranger", backstory="x", campaigns_dir=tmp_path)
    with pytest.raises(ValueError, match="already has a player"):
        save_character("camp", name="Bob", race="Human", char_class="Fighter", backstory="x", campaigns_dir=tmp_path)
    with pytest.raises(ValueError):
        save_character("camp", name="Bob", race="Orc", char_class="Fighter", backstory="x", campaigns_dir=tmp_path)


def test_save_character_falls_back_to_the_first_location(tmp_path):
    with GameDB.create("nostart", tmp_path) as db:
        db.add_location("a", "A", "x")
    save_character("nostart", name="Aria", race="Elf", char_class="Ranger", backstory="x", campaigns_dir=tmp_path)
    with GameDB.open("nostart", tmp_path) as db:
        assert db.get_player().location_id == "a"


# --- the whole flow ----------------------------------------------------------------------------------------

def test_numbers_only_run_makes_just_one_model_call(tmp_path):
    make_campaign(tmp_path)
    creator, out = make_creator(tmp_path, [js(options=STORIES)], ["", "1", "2", "6", "1"])
    assert creator.run() is not None
    assert len(creator.client._backend.calls) == 1              # only the backstory options
    with GameDB.open("camp", tmp_path) as db:
        player = db.get_player()
        assert (player.race, player.char_class, player.backstory) == ("Elf", "Ranger", STORIES[0])
        assert player.name.split(" ")[0] in names.FIRST_NAMES
    assert any("[OK] Class: Ranger" in line for line in out)


NEW_STORIES = [
    "Aria hunts the thief who stole her mentor's lute.",
    "Aria swore an oath at a grave and now keeps it.",
    "Aria sells her blade to whoever pays, but never to cults.",
]


def test_notes_prefill_then_a_typed_class_and_surprise_backstory(tmp_path):
    make_campaign(tmp_path)
    inputs = ["i want to create a human named baron", "i'd go with rogue", "6"]
    creator, out = make_creator(tmp_path, [js(options=STORIES)], inputs)
    assert creator.run() is not None
    assert len(creator.client._backend.calls) == 1               # only the backstory options
    with GameDB.open("camp", tmp_path) as db:
        player = db.get_player()
        assert (player.name, player.race, player.char_class) == ("Baron", "Human", "Rogue")
        assert player.backstory in STORIES


def test_typing_your_own_name_and_backstory(tmp_path):
    make_campaign(tmp_path)
    own = "A ferryman's daughter who learned to read the fog and now hunts what hides inside it"
    inputs = ["", "4", "thrain blackwood", "1", "1", "4", own]
    creator, _ = make_creator(tmp_path, [js(options=STORIES)], inputs)
    assert creator.run() is not None
    with GameDB.open("camp", tmp_path) as db:
        player = db.get_player()
        assert (player.name, player.backstory) == ("Thrain Blackwood", own)


def test_words_instead_of_numbers_show_the_menu_again_and_cancel_leaves(tmp_path):
    make_campaign(tmp_path)
    creator, out = make_creator(tmp_path, ["x"], ["", "hmm", "give me another name", "cancel"])
    assert creator.run() is None
    assert sum("Choose a name:" in line for line in out) == 3
    assert sum("Please pick a number from 1 to 6" in line for line in out) == 2
    assert creator.client._backend.calls == []


def test_new_names_slot_regenerates_without_a_model_call(tmp_path):
    make_campaign(tmp_path)
    creator, out = make_creator(tmp_path, [js(options=STORIES)], ["", "5", "1", "1", "1", "1"])
    assert creator.run() is not None
    assert sum("Choose a name:" in line for line in out) == 2
    assert len(creator.client._backend.calls) == 1


def test_new_backstories_slot_asks_the_model_again(tmp_path):
    make_campaign(tmp_path)
    creator, _ = make_creator(
        tmp_path, [js(options=STORIES), js(options=NEW_STORIES)], ["", "1", "1", "1", "5", "2"]
    )
    assert creator.run() is not None
    with GameDB.open("camp", tmp_path) as db:
        assert db.get_player().backstory == NEW_STORIES[1]
