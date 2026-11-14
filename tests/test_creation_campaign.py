import json

import pytest
import random

from aether.config import Config
from aether.creation import campaign as cc
from aether.creation.campaign import (
    CampaignCreator,
    DerivedMetadata,
    Storyline,
    World,
    WorldMapDraft,
    build_world,
    choose_name,
    normalize_metadata,
    normalize_storyline,
    save_campaign,
    prettify_name,
    storyline_messages,
    compose_patron,
    generate_field_options,
    field_options_messages,
    valid_field_text,
)
from aether.creation import names
from aether.db import GameDB
from aether.llm.client import LLMClient
from aether.models import CampaignInfo, Location, StoryBeat
from fakes import FakeBackend


def js(**kw):
    return json.dumps(kw)


def draft(*locs):
    return WorldMapDraft.model_validate({"locations": list(locs)})


def pairs_of(world):
    return {frozenset(p) for p in world.connections}


# --- schemas ---------------------------------------------------------------------------------

def test_lists_are_joined_into_text():
    assert Storyline.model_validate_json('{"middle":["a","b"]}').middle == "a b"
    assert DerivedMetadata.model_validate_json('{"patron_role":["a","cook"]}').patron_role == "a cook"


def test_everything_is_optional():
    assert Storyline.model_validate_json("{}").beginning == ""


# --- normalising model output ------------------------------------------------------------------

def test_metadata_defaults_and_names():
    meta = normalize_metadata(None)
    assert meta.theme_vibe == "Heroic Fantasy" and meta.patron_role == cc.DEFAULT_PATRON_ROLE
    assert meta.suggested_names == list(cc.FALLBACK_NAMES)

    meta = normalize_metadata(DerivedMetadata(
        theme_vibe="Gothic", patron_role="ferry-keeper",
        suggested_names=['"The Drowned Bell"', "the drowned bell", "", "Marsh Song", "A", "B"],
    ))
    assert meta.suggested_names == ["The Drowned Bell", "Marsh Song", "A"]
    assert meta.patron_role == "ferry-keeper"


def test_storyline_fills_each_missing_part():
    checklist = {"primary_threat": "The Witch"}
    story = normalize_storyline(Storyline(beginning="It starts."), checklist, "Tam")
    assert story.beginning == "It starts."
    assert "dangerous" in story.middle and "climax" in story.end
    assert story.victory_condition == "Defeat or overcome The Witch"
    assert "Tam" in normalize_storyline(None, checklist, "Tam").beginning


def test_storyline_is_clipped():
    story = normalize_storyline(Storyline(end="word " * 400), {}, "Tam")
    assert len(story.end) <= cc.FIELD_MAX_CHARS + 3 and story.end.endswith("...")


# --- the world map ------------------------------------------------------------------------------

def test_build_world_cleans_ids_and_connections():
    w = build_world(draft(
        {"id": "Village Tavern", "name": "Village Tavern", "brief_concept": "An inn.",
         "connections": ["Misty Path", "village tavern", "nowhere"]},
        {"id": "misty_path", "name": "Misty Path", "connections": ["village_tavern"]},
        {"id": "cave", "name": "Cave", "connections": ["misty_path"]},
    ))
    assert [l.id for l in w.locations] == ["village_tavern", "misty_path", "cave"]
    assert w.locations[2].brief_concept == "Cave"                      # empty concept falls back to the name
    assert pairs_of(w) == {frozenset(("village_tavern", "misty_path")), frozenset(("misty_path", "cave"))}
    assert not w.used_fallback


def test_build_world_drops_duplicates_and_blank_names():
    w = build_world(draft(
        {"id": "a", "name": "A", "connections": ["b"]},
        {"id": "a", "name": "A again"},
        {"id": "b", "name": "B", "connections": ["c"]},
        {"id": "x", "name": "   "},
        {"id": "c", "name": "C"},
    ))
    assert [l.id for l in w.locations] == ["a", "b", "c"]


def test_build_world_connects_unreachable_locations():
    w = build_world(draft(
        {"id": "a", "name": "A", "connections": ["b"]},
        {"id": "b", "name": "B"},
        {"id": "c", "name": "C", "connections": ["d"]},
        {"id": "d", "name": "D"},
    ))
    assert frozenset(("b", "c")) in pairs_of(w)
    assert cc._reachable("a", w.connections) == {"a", "b", "c", "d"}


def test_build_world_caps_the_size():
    locs = [{"id": f"l{i}", "name": f"L{i}", "connections": [f"l{i + 1}"]} for i in range(15)]
    assert len(build_world(draft(*locs)).locations) == cc.MAX_LOCATIONS


def test_build_world_falls_back_when_too_small():
    for d in (None, draft({"id": "a", "name": "A"}, {"id": "b", "name": "B"})):
        w = build_world(d)
        assert w.used_fallback and len(w.locations) == 3
        assert cc._reachable(w.locations[0].id, w.connections) == {l.id for l in w.locations}


def test_connections_may_use_names_instead_of_ids():
    w = build_world(draft(
        {"id": "loc1", "name": "Old Mill", "connections": ["Dark Wood"]},
        {"id": "loc2", "name": "Dark Wood", "connections": ["The Keep"]},
        {"id": "loc3", "name": "The Keep"},
    ))
    assert pairs_of(w) == {frozenset(("loc1", "loc2")), frozenset(("loc2", "loc3"))}


# --- choosing a name ------------------------------------------------------------------------------

def ask(*answers):
    it = iter(answers)
    return lambda prompt="": next(it)


def test_choose_name():
    shown, seen = ["One", "Two", "Three"], []
    quiet = lambda *a, **k: None

    def more(text):
        seen.append(text)
        return ["Four", "Five", "Six"]

    def pick(*answers):
        return choose_name(shown, ask(*answers), quiet, more, random.Random(1))

    assert pick("2") == "Two"
    assert pick("4", "My Own") == "My Own"
    assert pick("4", " ", "1") == "One"                  # an empty custom name shows the menu again
    assert pick("banana", "9", "3") == "Three"           # words and bad numbers show it again
    assert pick("5", "salty sea", "2") == "Five" and seen == ["salty sea"]
    assert pick("6", "1") == "Four" and seen[-1] == ""
    assert pick("7") in shown


def test_typing_an_idea_at_the_name_prompt_writes_new_names(tmp_path):
    responses = [OPTS_A, OPTS_THREAT, OPTS_HOOK, BUILD[0],
                 js(suggested_names=["Salt Crown", "Tide Song", "Grey Bell"]),
                 BUILD[1], BUILD[2], BUILD[3]]
    creator, out = make_creator(tmp_path, responses, ["", "1", "2", "3", "5", "something with salt", "2"])
    assert creator.run() == "tide_song"
    prompt = creator.client._backend.calls[4]["messages"][1]["content"]
    assert "something with salt" in prompt
    assert "Do not reuse: The Drowned Bell, Marsh Song, Silent Fog" in prompt


# --- saving -----------------------------------------------------------------------------------------

def make_info(slug="saved"):
    return CampaignInfo(name="Saved", slug=slug, setting_description="s", primary_threat="t",
                        starting_quest_hook="h", starting_location_id="a")


def make_world():
    return World([Location("a", "A", "x"), Location("b", "B", "y")], [("a", "b")])


def test_save_campaign_writes_everything(tmp_path):
    beats = [StoryBeat(1, "One", "Do one."), StoryBeat(2, "Two", "Do two.")]
    assert save_campaign(make_info(), make_world(), beats, tmp_path) == "saved"
    db = GameDB.open("saved", tmp_path)
    assert db.get_campaign().starting_location_id == "a"
    assert [l.id for l in db.get_connected_locations("a")] == ["b"]
    assert [b.title for b in db.get_beats()] == ["One", "Two"]
    assert not db.has_player()
    db.close()


def test_failed_save_leaves_no_folder_behind(tmp_path):
    bad = World([Location("a", "A", "x"), Location("a", "A2", "y")], [])   # duplicate id
    with pytest.raises(Exception):
        save_campaign(make_info("broken"), bad, [], tmp_path)
    assert not (tmp_path / "broken").exists()
    assert GameDB.list_campaigns(tmp_path) == []


# --- the whole flow ---------------------------------------------------------------------------------

def make_creator(tmp_path, responses, inputs):
    out = []
    client = LLMClient(Config(), FakeBackend(*responses))
    creator = CampaignCreator(
        client, campaigns_dir=tmp_path, input_fn=ask(*inputs),
        print_fn=lambda *a, **k: out.append(" ".join(str(x) for x in a)),
        rng=random.Random(7),
    )
    return creator, out


def test_prettify_name():
    assert prettify_name("cult_haven") == "Cult Haven"
    assert prettify_name("chapel") == "Chapel"
    assert prettify_name("everwood's laboratory") == "Everwood's Laboratory"
    assert prettify_name("oakwood manor - fall of the scholar") == "Oakwood Manor"
    assert prettify_name("Old Mill") == "Old Mill"
    assert prettify_name("McGrath") == "McGrath"
    assert prettify_name("  ") == ""


def test_lowercase_names_get_title_case_and_suffixes_are_trimmed():
    w = build_world(draft(
        {"id": "oakwood_manor_fall", "name": "oakwood manor - fall of the scholar", "connections": ["lab"]},
        {"id": "lab", "name": "everwood's laboratory", "connections": ["hall"]},
        {"id": "hall", "name": "darkened hallway"},
    ))
    assert [l.name for l in w.locations] == ["Oakwood Manor", "Everwood's Laboratory", "Darkened Hallway"]


def test_names_that_are_just_ids_become_titles():
    w = build_world(draft(
        {"id": "cult_haven", "name": "cult_haven", "connections": ["ancient_temple"]},
        {"id": "ancient_temple", "name": "ancient_temple", "connections": ["lost_city"]},
        {"id": "lost_city", "name": "Lost City"},
    ))
    assert [l.name for l in w.locations] == ["Cult Haven", "Ancient Temple", "Lost City"]
    assert pairs_of(w) == {frozenset(("cult_haven", "ancient_temple")), frozenset(("ancient_temple", "lost_city"))}


def test_shortcuts_to_the_end_are_removed():
    # the map from a real run: two loops that let the player jump straight to the final location
    w = build_world(draft(
        {"id": "kraelion_island", "name": "kraelion_island", "connections": ["cult_haven"]},
        {"id": "cult_haven", "name": "cult_haven", "connections": ["ancient_temple"]},
        {"id": "ancient_temple", "name": "ancient_temple", "connections": ["lost_city"]},
        {"id": "lost_city", "name": "lost_city", "connections": ["kraelion_island"]},
        {"id": "high_priestess_temple", "name": "high_priestess_temple",
         "connections": ["lost_city", "kraelion_island"]},
    ))
    assert pairs_of(w) == {
        frozenset(("kraelion_island", "cult_haven")),
        frozenset(("cult_haven", "ancient_temple")),
        frozenset(("ancient_temple", "lost_city")),
        frozenset(("lost_city", "high_priestess_temple")),
    }


def test_first_and_last_are_never_joined_even_in_a_small_map():
    w = build_world(draft(
        {"id": "a", "name": "A", "connections": ["b", "c"]},
        {"id": "b", "name": "B", "connections": ["c"]},
        {"id": "c", "name": "C"},
    ))
    assert pairs_of(w) == {frozenset(("a", "b")), frozenset(("b", "c"))}


def test_a_short_side_branch_is_kept():
    w = build_world(draft(
        {"id": "a", "name": "A", "connections": ["b"]},
        {"id": "b", "name": "B", "connections": ["x", "c"]},
        {"id": "x", "name": "X"},
        {"id": "c", "name": "C", "connections": ["d"]},
        {"id": "d", "name": "D"},
    ))
    assert frozenset(("b", "x")) in pairs_of(w)
    assert frozenset(("b", "c")) in pairs_of(w)


def test_storyline_prompt_makes_the_player_the_hero():
    msgs = cc.storyline_messages({"setting_description": "A marsh."}, "Kael, a scholar")
    assert "never the hero" in msgs[0]["content"]
    assert "Kael, a scholar" in msgs[1]["content"] and "not the hero" in msgs[1]["content"]


def test_compose_patron_always_uses_the_given_name():
    assert compose_patron("Odalric Venn", "a retired ferry-keeper") == "Odalric Venn, a retired ferry-keeper"
    assert compose_patron("Odalric Venn", "Odalric Venn, the ferry-keeper") == "Odalric Venn, the ferry-keeper"
    assert compose_patron("Odalric Venn", "Kaelin Darkhaven, Former Rebel Leader") == "Odalric Venn, Former Rebel Leader"
    assert compose_patron("Odalric Venn", "The Oracle, keeper of the bell") == "Odalric Venn, The Oracle, keeper of the bell"
    assert compose_patron("Odalric Venn", "") == f"Odalric Venn, {cc.DEFAULT_PATRON_ROLE}"


def test_metadata_prompt_gives_the_name_and_asks_only_for_the_role():
    system = cc.metadata_messages({"setting_description": "A marsh."}, "Odalric Venn")[0]["content"]
    assert "role of Odalric Venn" in system and "Do not write the name" in system


OPTS_A = js(options=["A fog-choked marsh kingdom of drowned bells", "A sunburnt desert empire of buried gods",
                     "A frozen northern citadel under endless storms"])
OPTS_B = js(options=["A sunken coastal city of salt and ruin", "A haunted pine forest of whispering trees",
                     "A volcanic island of ash and fire"])
OPTS_THREAT = js(options=["A cult of drowned priests", "The Swamp Witch Orsolya of the mire",
                          "A rival warlord with a hungry army"])
OPTS_HOOK = js(options=["Children have gone silent overnight", "A plague sweeps the nearby village",
                        "A stranger offers a map to a hidden vault"])
BUILD = [
    js(theme_vibe="Gothic Dark Fantasy", patron_role="the ferry-keeper",
       suggested_names=["The Drowned Bell", "Marsh Song", "Silent Fog"]),
    js(beginning="It begins.", middle=["Complication one.", "Complication two."], end="It ends.",
       victory_condition="Defeat the witch."),
    js(locations=[
        {"id": "mirewick_ferry", "name": "Mirewick Ferry", "brief_concept": "A landing.", "connections": ["village"]},
        {"id": "village", "name": "Village", "brief_concept": "A silent village.", "connections": ["mirewick_ferry", "causeway"]},
        {"id": "causeway", "name": "Causeway", "brief_concept": "A sunken road.", "connections": ["village", "chapel"]},
        {"id": "chapel", "name": "Chapel", "brief_concept": "A flooded chapel.", "connections": ["causeway"]},
        {"id": "reed_hut", "name": "Reed Hut", "brief_concept": "A hut.", "connections": []},
    ]),
    js(beats=[{"title": f"Beat {i}", "objective": f"Do {i}."} for i in range(1, 5)]),
]


# --- field options ------------------------------------------------------------------------------

def test_field_options_prompt_carries_earlier_answers_and_notes():
    msgs = field_options_messages("primary_threat", {"setting_description": "A foggy marsh kingdom"}, "something spooky", 3)
    assert "exactly 3 different options" in msgs[0]["content"] and "<option 1>" in msgs[0]["content"]
    assert "Setting (already chosen): A foggy marsh kingdom" in msgs[1]["content"]
    assert "Player's ideas: something spooky" in msgs[1]["content"]
    assert "(no ideas given yet)" in field_options_messages("setting_description", {}, "", 3)[1]["content"]


def test_field_options_are_cleaned_and_topped_up():
    client = LLMClient(Config(), FakeBackend(js(options=[
        "  'A fog-choked marsh kingdom'  ", "a fog-choked marsh kingdom", "too short", "A desert of buried gods",
    ])))
    got = generate_field_options(client, "setting_description", {}, "")
    assert got[:2] == ["A fog-choked marsh kingdom", "A desert of buried gods"]
    assert got[2] == cc.FALLBACK_FIELD_OPTIONS["setting_description"][0] and len(got) == 3


def test_field_options_model_failure_gives_fallbacks():
    client = LLMClient(Config(), FakeBackend("garbage"))
    assert generate_field_options(client, "primary_threat", {}, "") == list(cc.FALLBACK_FIELD_OPTIONS["primary_threat"])


def test_valid_field_text():
    assert valid_field_text("  A fog-choked marsh  ") == "A fog-choked marsh"
    assert valid_field_text("dark") == "" and valid_field_text("decaying ruins") == ""


def test_every_field_has_three_usable_fallbacks():
    assert set(cc.FALLBACK_FIELD_OPTIONS) == set(cc.CHECKLIST_KEYS)
    for options in cc.FALLBACK_FIELD_OPTIONS.values():
        assert len(set(options)) == 3 and all(valid_field_text(o) == o for o in options)


# --- the whole flow --------------------------------------------------------------------------------

def test_numbers_only_run_builds_a_campaign_with_no_classifier_calls(tmp_path):
    responses = [OPTS_A, OPTS_THREAT, OPTS_HOOK, *BUILD]
    creator, out = make_creator(tmp_path, responses, ["something spooky", "1", "2", "3", "1"])
    assert creator.run() == "the_drowned_bell"
    assert len(creator.client._backend.calls) == 7        # 3 option menus + metadata, storyline, map, beats

    with GameDB.open("the_drowned_bell", tmp_path) as db:
        info = db.get_campaign()
        assert info.setting_description == "A fog-choked marsh kingdom of drowned bells"
        assert info.primary_threat == "The Swamp Witch Orsolya of the mire"
        assert info.starting_quest_hook == "A stranger offers a map to a hidden vault"
        patron_name, _, patron_role = info.starting_patron.partition(", ")
        first, _, last = patron_name.partition(" ")
        assert first in names.FIRST_NAMES and last in names.LAST_NAMES and patron_role == "the ferry-keeper"
        assert info.starting_location_id == "mirewick_ferry"
        assert info.storyline_middle == "Complication one. Complication two."
        assert len(db.list_locations()) == 5 and len(db.get_beats()) == 4
        assert db.get_connected_locations("reed_hut")
        assert not db.has_player()
    assert any("[OK] Threat: The Swamp Witch Orsolya of the mire" in line for line in out)
    assert any("SUCCESS" in line for line in out)


def test_the_notes_reach_the_options_prompt(tmp_path):
    creator, _ = make_creator(tmp_path, [OPTS_A], ["something spooky", "cancel"])
    assert creator.run() is None
    prompt = creator.client._backend.calls[0]["messages"][1]["content"]
    assert "Player's ideas: something spooky" in prompt


def test_asking_for_different_ideas_regenerates_the_menu(tmp_path):
    creator, _ = make_creator(tmp_path, [OPTS_A, js(intent="reject"), OPTS_B], ["", "show me different ones", "1", "cancel"])
    assert creator.run() is None
    assert creator.values == {"setting_description": "A sunken coastal city of salt and ruin"}


def test_ask_me_to_pick_chooses_one_of_the_options(tmp_path):
    creator, _ = make_creator(tmp_path, [OPTS_A, js(intent="surprise_me"), OPTS_THREAT], ["", "you pick", "cancel"])
    assert creator.run() is None
    assert creator.values["setting_description"] in json.loads(OPTS_A)["options"]


def test_garbage_from_the_model_still_creates_a_playable_campaign(tmp_path):
    creator, _ = make_creator(tmp_path, ["garbage"], ["", "1", "1", "1", "1"])
    slug = creator.run()
    assert slug == "an_unknown_journey"
    with GameDB.open(slug, tmp_path) as db:
        assert db.get_campaign().setting_description == cc.FALLBACK_FIELD_OPTIONS["setting_description"][0]
        assert len(db.list_locations()) == 3 and len(db.get_beats()) == 3
        assert db.get_campaign().starting_location_id == "starting_point"


def test_two_campaigns_with_the_same_name_get_different_slots(tmp_path):
    for _ in range(2):
        creator, _ = make_creator(tmp_path, ["garbage"], ["", "1", "1", "1", "1"])
        creator.run()
    assert sorted(GameDB.list_campaigns(tmp_path)) == ["an_unknown_journey", "an_unknown_journey_2"]


def test_old_ideas_are_cut_before_new_ones():
    notes = "old idea; " + "filler " * 100 + "newest idea"
    content = field_options_messages("primary_threat", {}, notes, 3)[1]["content"]
    assert "newest idea" in content and "old idea" not in content
    assert "around the player's ideas" in field_options_messages("primary_threat", {}, "", 3)[0]["content"]


def test_typed_text_is_an_idea_that_regenerates_the_options(tmp_path):
    creator, _ = make_creator(
        tmp_path, [OPTS_A, js(intent="own_text"), OPTS_B, OPTS_THREAT],
        ["something spooky", "something with an eagle", "2", "cancel"],
    )
    assert creator.run() is None
    assert creator.ideas == ["something spooky", "something with an eagle"]
    assert creator.values == {"setting_description": "A haunted pine forest of whispering trees"}
    regenerated = creator.client._backend.calls[2]["messages"][1]["content"]
    assert "Player's ideas: something spooky; something with an eagle" in regenerated


def test_the_type_my_own_idea_slot_asks_for_an_idea_without_the_classifier(tmp_path):
    creator, _ = make_creator(
        tmp_path, [OPTS_A, OPTS_B, OPTS_THREAT], ["", "4", "a sky eagle", "1", "cancel"],
    )
    assert creator.run() is None
    assert len(creator.client._backend.calls) == 3          # 2 setting menus + the threat menu, no classifier
    assert creator.ideas == ["a sky eagle"]
    assert creator.values == {"setting_description": "A sunken coastal city of salt and ruin"}


def test_words_instead_of_numbers_show_the_menu_again_and_are_not_ideas(tmp_path):
    creator, out = make_creator(
        tmp_path, [OPTS_A, OPTS_THREAT], ["", "okay", "give me another one", "1", "cancel"],
    )
    assert creator.run() is None
    assert sum("Choose the setting:" in line for line in out) == 3
    assert sum("Please pick a number from 1 to 6" in line for line in out) == 2
    assert creator.ideas == [] and len(creator.client._backend.calls) == 2


def test_clean_names():
    assert cc.clean_names(['"One"', "one", "", "Two", "Three", "Four"]) == ["One", "Two", "Three"]
    assert cc.clean_names([]) == list(cc.FALLBACK_NAMES)
    got = cc.clean_names(["New"], exclude=["An Unknown Journey"])
    assert got[0] == "New" and "An Unknown Journey" not in got and len(got) == 3
    assert len(cc.clean_names([], exclude=list(cc.FALLBACK_NAMES))) == 3     # last resort reuses them


def test_names_prompt_carries_ideas_and_names_to_avoid():
    msgs = cc.names_messages({"setting_description": "A marsh."}, ["something salty"], ["Marsh Song"])
    assert "exactly 3 atmospheric" in msgs[0]["content"]
    assert "Player's ideas for the name: something salty" in msgs[1]["content"]
    assert "Do not reuse: Marsh Song" in msgs[1]["content"]


def test_cancel_works_at_the_name_prompt(tmp_path):
    responses = [OPTS_A, OPTS_THREAT, OPTS_HOOK, BUILD[0]]
    creator, out = make_creator(tmp_path, responses, ["", "1", "2", "3", "cancel"])
    assert creator.run() is None
    assert GameDB.list_campaigns(tmp_path) == []


def test_new_options_slot_regenerates_the_menu(tmp_path):
    creator, _ = make_creator(tmp_path, [OPTS_A, OPTS_B], ["", "5", "1", "cancel"])
    assert creator.run() is None
    assert creator.values == {"setting_description": "A sunken coastal city of salt and ruin"}


def test_surprise_me_chooses_one_of_the_options(tmp_path):
    creator, _ = make_creator(tmp_path, [OPTS_A, OPTS_THREAT], ["", "6", "cancel"])
    assert creator.run() is None
    assert creator.values["setting_description"] in json.loads(OPTS_A)["options"]


def test_type_my_own_idea_adds_an_idea_and_regenerates(tmp_path):
    creator, _ = make_creator(
        tmp_path, [OPTS_A, OPTS_B, OPTS_THREAT],
        ["something spooky", "4", "something with an eagle", "2", "cancel"],
    )
    assert creator.run() is None
    assert creator.ideas == ["something spooky", "something with an eagle"]
    assert creator.values == {"setting_description": "A haunted pine forest of whispering trees"}
    regenerated = creator.client._backend.calls[1]["messages"][1]["content"]
    assert "Player's ideas: something spooky; something with an eagle" in regenerated
