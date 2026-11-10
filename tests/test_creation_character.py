import json

import pytest

from aether.config import Config
from aether.creation import character as ch
from aether.creation.campaign import World, save_campaign
from aether.creation.character import (
    CharacterChecklist,
    CharacterCreator,
    clean_name,
    find_choice,
    guide_messages,
    merge_checklist,
    pending,
    rules_text,
    save_character,
    starting_items,
    apply_choice,
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


def make_campaign(tmp_path, slug="camp"):
    info = CampaignInfo(
        name="Camp", slug=slug, setting_description="A foggy marsh.", primary_threat="A witch.",
        starting_quest_hook="Find the bell.", starting_patron="Mother Hesk, ferry-keeper",
        starting_location_id="ferry",
    )
    world = World(
        [Location("ferry", "Ferry", "x"), Location("cave", "Cave", "y"), Location("end", "End", "z")],
        [("ferry", "cave"), ("cave", "end")],
    )
    save_campaign(info, world, [], tmp_path)
    return info


def make_creator(tmp_path, responses, inputs, slug="camp"):
    out = []
    client = LLMClient(Config(), FakeBackend(*responses))
    creator = CharacterCreator(
        client, slug, campaigns_dir=tmp_path, input_fn=ask(*inputs),
        print_fn=lambda *a, **k: out.append(" ".join(str(x) for x in a)),
    )
    return creator, out


# --- schema and small helpers ------------------------------------------------------------------

def test_lists_are_joined_into_text():
    assert CharacterChecklist.model_validate_json('{"backstory":["a","b"]}').backstory == "a b"
    assert CharacterChecklist.model_validate_json("{}").name is None


def test_find_choice():
    assert find_choice("Elf", RACES) == "Elf"
    assert find_choice("a proud ELVEN scout", RACES, ch.RACE_ALIASES) == "Elf"
    assert find_choice("an Orc", RACES, ch.RACE_ALIASES) is None
    assert find_choice("Ranger", CLASSES) == "Ranger"
    assert find_choice(None, CLASSES) is None


def test_clean_name():
    assert clean_name("  'Aria' ") == "Aria"
    assert clean_name("Kael Stormborn") == "Kael Stormborn"
    for bad in (None, "", "none", "Unknown", "Elf", "ranger", "elven", "a b c d e", "12", "x" * 60):
        assert clean_name(bad) == ""


def test_rules_text_comes_from_the_real_tables():
    text = rules_text()
    assert "- Elf: +2 Dexterity" in text
    assert "- Human: +1 to all abilities" in text
    assert "- Wizard: hit die d6" in text
    assert "armor class" not in text.lower() and " AC" not in text


def test_guide_prompt_has_campaign_checklist_and_format():
    info = CampaignInfo(name="Camp", slug="camp", setting_description="A foggy marsh.", primary_threat="A witch.",
                        starting_quest_hook="Find the bell.", starting_patron="Mother Hesk, ferry-keeper")
    msgs = guide_messages(info, {"name": "Aria"}, [{"role": "user", "content": "hi"}])
    system = msgs[0]["content"]
    assert "A foggy marsh." in system and "Mother Hesk, ferry-keeper" in system
    assert "- name: COMPLETED (Aria)" in system and "- race: PENDING" in system
    assert "D) Something else" in system
    assert msgs[1] == {"role": "user", "content": "hi"}


# --- checklist merging -------------------------------------------------------------------------------

SRC = "My name is Aria, an elf ranger. A scout of the Lantern Guard who lost her squad."
FULL = CharacterChecklist(
    name="Aria", race="elven", character_class="Ranger",
    backstory="A scout of the Lantern Guard who lost her squad.",
)


def test_merge_accepts_valid_grounded_values_and_maps_to_the_real_lists():
    assert merge_checklist({}, FULL, SRC) == {
        "name": "Aria", "race": "Elf", "char_class": "Ranger",
        "backstory": "A scout of the Lantern Guard who lost her squad.",
    }


def test_merge_rejects_unknown_ungrounded_and_placeholder_values():
    bad = CharacterChecklist(name="Zed", race="Orc", character_class="Necromancer")
    assert merge_checklist({}, bad, SRC) == {}
    assert merge_checklist({}, CharacterChecklist(name="Elf"), SRC) == {}
    assert merge_checklist({}, CharacterChecklist(race="Dwarf"), SRC) == {}      # not in the conversation
    assert merge_checklist({}, None, SRC) == {}


def test_backstory_waits_for_the_core_and_must_be_a_real_phrase():
    only_story = CharacterChecklist(backstory="A scout of the Lantern Guard who lost her squad.")
    assert "backstory" not in merge_checklist({}, only_story, SRC)
    core = {"name": "Aria", "race": "Elf", "char_class": "Ranger"}
    assert "backstory" not in merge_checklist(core, CharacterChecklist(backstory="A scout"), SRC)
    invented = CharacterChecklist(backstory="A noble heir of Ravenshire who fled the Dread Court")
    assert "backstory" not in merge_checklist(core, invented, SRC)
    assert "backstory" in merge_checklist(core, only_story, SRC)


def test_settled_values_are_never_replaced():
    current = {"name": "Aria", "race": "Elf", "char_class": "Ranger", "backstory": "Old story here for sure."}
    changed = CharacterChecklist(name="Zed", race="Dwarf", character_class="Fighter", backstory="Something else entirely")
    assert merge_checklist(current, changed, SRC + " Zed dwarf fighter") == current
    assert pending(current) == []
    assert pending({}) == ["name", "race", "char_class", "backstory"]


# --- starting kit and saving ---------------------------------------------------------------------------

def test_every_class_has_a_kit_with_valid_items():
    for cls in CLASSES:
        items = starting_items(cls)
        assert len(items) >= 3
        assert len({n.lower() for n, _ in items}) == len(items)
        assert all(q >= 1 for _, q in items)
    assert ("Stale bread", 1) in starting_items("Fighter")


def test_save_character_writes_player_stats_location_and_kit(tmp_path):
    make_campaign(tmp_path)
    pid = save_character("camp", name="Aria", race="Elf", char_class="Ranger",
                         backstory="A scout.", campaigns_dir=tmp_path)
    with GameDB.open("camp", tmp_path) as db:
        player = db.get_player()
        assert player.id == pid
        assert (player.name, player.race, player.char_class) == ("Aria", "Elf", "Ranger")
        assert (player.dexterity, player.hp, player.max_hp) == (17, 11, 11)
        assert player.location_id == "ferry"
        assert player.backstory == "A scout."
        assert ("Short bow", 1) in db.get_inventory(pid) and ("Arrows", 12) in db.get_inventory(pid)


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

def test_full_run_creates_the_character(tmp_path):
    make_campaign(tmp_path)
    story = "A scout of the Lantern Guard who lost her squad in the fog"
    responses = [
        js(name="Aria", race="Elf", character_class="Ranger"),
        "Guide: pick a backstory. A) Scout B) Thief C) Monk D) Other",
        js(name="Aria", race="Elf", character_class="Ranger", backstory=story),
    ]
    creator, out = make_creator(tmp_path, responses, ["My name is Aria, an elf ranger.", story])
    pid = creator.run()
    assert pid is not None
    with GameDB.open("camp", tmp_path) as db:
        player = db.get_player()
        assert (player.name, player.race, player.char_class) == ("Aria", "Elf", "Ranger")
        assert player.backstory == story
        assert db.get_inventory(player.id)
    assert any("SUCCESS" in line for line in out)
    assert any("Backstory: PENDING" in line for line in out)


def test_complete_with_items_pending_keeps_chatting_and_cancel_leaves(tmp_path):
    make_campaign(tmp_path)
    creator, out = make_creator(tmp_path, ["x"], ["complete", "cancel"])
    assert creator.run() is None
    assert any("Still pending" in line for line in out)
    with GameDB.open("camp", tmp_path) as db:
        assert not db.has_player()


def test_garbage_from_the_model_never_fills_the_checklist(tmp_path):
    make_campaign(tmp_path)
    creator, out = make_creator(tmp_path, ["garbage", "Guide: who are you?"], ["hello there", "cancel"])
    assert creator.run() is None
    assert any("Name: PENDING | Race: PENDING | Class: PENDING | Backstory: PENDING" in line for line in out)


def test_missing_campaign_and_existing_player_are_reported(tmp_path):
    creator, out = make_creator(tmp_path, ["x"], [], slug="ghost")
    assert creator.run() is None and any("No usable campaign" in line for line in out)

    make_campaign(tmp_path)
    save_character("camp", name="Aria", race="Elf", char_class="Ranger", backstory="x", campaigns_dir=tmp_path)
    creator, out = make_creator(tmp_path, ["x"], [])
    assert creator.run() is None and any("already has a player" in line for line in out)


INFO = CampaignInfo(name="Camp", slug="camp", setting_description="A marsh.", primary_threat="A witch.",
                    starting_quest_hook="Find the bell.")


def test_apply_choice_fills_only_the_next_pending_item():
    base = {"name": "Aria"}
    assert apply_choice(base, "Elf: nimble")["race"] == "Elf"
    assert "race" not in apply_choice(base, "Sneaky types")
    core = {"name": "Aria", "race": "Elf", "char_class": "Ranger"}
    assert apply_choice(core, "Aria lost her squad in the fog and seeks atonement.")["backstory"].startswith("Aria lost")
    assert "backstory" not in apply_choice(core, "A scout")
    assert apply_choice({}, "Aria: a name") == {}             # names go through the extractor


def test_guide_prompt_asks_about_one_item_only():
    system = guide_messages(INFO, {"name": "Baron", "race": "Human"}, [])[0]["content"]
    assert "Ask ONLY about the player's class" in system
    done = {"name": "Baron", "race": "Human", "char_class": "Rogue"}
    system = guide_messages(INFO, done, [])[0]["content"]
    assert "Ask ONLY about the player's backstory" in system and "Do NOT write a full backstory" in system


def test_menu_choices_fill_the_checklist_without_extraction_and_finish_at_once(tmp_path):
    make_campaign(tmp_path)
    responses = [
        js(name="Aria"),
        "Guide: pick a race.\nA) Human: versatile\nB) Elf: nimble\nC) Dwarf: hardy\nD) Something else",
        "Guide: pick a class.\nA) Fighter: tough\nB) Rogue: sneaky\nC) Ranger: a hunter\nD) Something else",
        "Guide: pick a backstory.\nA) Aria lost her squad in the fog and seeks atonement.\n"
        "B) Aria fled a burning city.\nC) Aria is a wandering scout.\nD) Something else",
    ]
    creator, out = make_creator(tmp_path, responses, ["My name is Aria", "b", "c", "a"])
    assert creator.run() is not None
    assert len(creator.client._backend.calls) == 4     # no extraction for letters, no guide call after the last pick
    with GameDB.open("camp", tmp_path) as db:
        player = db.get_player()
        assert (player.race, player.char_class) == ("Elf", "Ranger")
        assert player.backstory == "Aria lost her squad in the fog and seeks atonement."


def test_something_else_asks_for_typed_input(tmp_path):
    make_campaign(tmp_path)
    responses = [js(name="Aria"), "Guide: pick a race.\nA) Human: v\nB) Elf: n\nC) Dwarf: h\nD) Something else"]
    creator, out = make_creator(tmp_path, responses, ["My name is Aria", "d", "cancel"])
    assert creator.run() is None
    assert any("type your own answer" in line for line in out)
    assert len(creator.client._backend.calls) == 2
