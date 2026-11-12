import json
import random

import pytest

from aether.config import Config
from aether.creation import character as ch
from aether.creation import names
from aether.creation.campaign import World, save_campaign
from aether.creation.character import (
    CharacterCreator,
    Reply,
    classify_reply,
    clean_name,
    extract_name,
    fallback_backstories,
    generate_backstories,
    interpret_reply,
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


# --- reading replies ---------------------------------------------------------------------------

def no_model(reply):
    raise AssertionError("the classifier should not be called")


def test_numbers_and_closed_matches_never_reach_the_classifier():
    opts = ["Human", "Elf", "Dwarf"]
    assert interpret_reply("2", opts, classify=no_model) == Reply("option", 1)
    assert interpret_reply("i'd pick dwarf", opts, classify=no_model, closed=True) == Reply("option", 2)
    assert interpret_reply("4", opts, classify=no_model, custom=True) == Reply("custom")


def test_long_text_is_taken_for_open_fields_only():
    text = "a sailor who lost his ship to the storm and now wanders the docks"
    assert interpret_reply(text, ["a"], classify=no_model, open_text=True) == Reply("text", text=text)
    got = interpret_reply(text, ["a"], classify=lambda r: ("unclear", None))
    assert got.kind == "unclear"


def test_classifier_results_map_to_replies():
    opts = ["One", "Two"]
    for kind, expected in (("accept", "accept"), ("reject", "reject"), ("surprise_me", "surprise"),
                           ("unclear", "unclear"), ("own_text", "text")):
        assert interpret_reply("okay", opts, classify=lambda r, k=kind: (k, None)).kind == expected
    assert interpret_reply("the second", opts, classify=lambda r: ("choose", 1)) == Reply("option", 1)


def test_classify_reply_validates_the_model_output():
    opts = ["One", "Two"]

    def run(reply):
        return classify_reply(LLMClient(Config(), FakeBackend(reply)), "q", opts, "x")

    assert run(js(intent="accept")) == ("accept", None)
    assert run(js(intent="choose", index=2)) == ("choose", 1)
    assert run(js(intent="choose", index=9)) == ("unclear", None)
    assert run(js(intent="choose")) == ("unclear", None)
    assert run(js(intent="banana")) == ("unclear", None)
    assert run("garbage") == ("unclear", None)


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


def test_notes_prefill_then_typed_class_and_a_random_backstory_with_okay(tmp_path):
    make_campaign(tmp_path)
    responses = [
        js(options=STORIES),                       # the three backstory options
        js(intent="surprise_me"),                  # "just create me some random backstory"
        js(options=["Baron is a loyal rogue who secretly serves the rebellion."]),
        js(intent="accept"),                       # "okay"
    ]
    inputs = ["i want to create a human named baron", "i'd go with rogue",
              "just create me some random backstory", "okay"]
    creator, out = make_creator(tmp_path, responses, inputs)
    assert creator.run() is not None
    assert len(creator.client._backend.calls) == 4
    with GameDB.open("camp", tmp_path) as db:
        player = db.get_player()
        assert (player.name, player.race, player.char_class) == ("Baron", "Human", "Rogue")
        assert player.backstory == "Baron is a loyal rogue who secretly serves the rebellion."


def test_typing_your_own_name_and_backstory(tmp_path):
    make_campaign(tmp_path)
    own = "A ferryman's daughter who learned to read the fog and now hunts what hides inside it"
    creator, _ = make_creator(tmp_path, [js(options=STORIES)], ["", "4", "thrain blackwood", "1", "1", own])
    assert creator.run() is not None
    with GameDB.open("camp", tmp_path) as db:
        player = db.get_player()
        assert (player.name, player.backstory) == ("Thrain Blackwood", own)


def test_unclear_replies_re_ask_and_cancel_leaves(tmp_path):
    make_campaign(tmp_path)
    creator, out = make_creator(tmp_path, [js(intent="unclear")], ["", "hmm", "cancel"])
    assert creator.run() is None
    assert any("did not catch a name" in line for line in out)
    with GameDB.open("camp", tmp_path) as db:
        assert not db.has_player()


def test_missing_campaign_and_existing_player_are_reported(tmp_path):
    creator, out = make_creator(tmp_path, ["x"], [], slug="ghost")
    assert creator.run() is None and any("No usable campaign" in line for line in out)

    make_campaign(tmp_path)
    save_character("camp", name="Aria", race="Elf", char_class="Ranger", backstory="x", campaigns_dir=tmp_path)
    creator, out = make_creator(tmp_path, ["x"], [])
    assert creator.run() is None and any("already has a player" in line for line in out)
