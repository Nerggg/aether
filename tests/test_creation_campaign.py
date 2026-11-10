import json

import pytest
import random

from aether.config import Config
from aether.creation import campaign as cc
from aether.creation.campaign import (
    CampaignChecklist,
    CampaignCreator,
    DerivedMetadata,
    Storyline,
    World,
    WorldMapDraft,
    build_world,
    choose_name,
    merge_checklist,
    normalize_metadata,
    normalize_storyline,
    pending,
    save_campaign,
    prettify_name,
    storyline_messages,
    apply_choice,
    compose_patron,
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
    assert CampaignChecklist.model_validate_json('{"primary_threat":["x","y"]}').primary_threat == "x y"
    assert DerivedMetadata.model_validate_json('{"patron_role":["a","cook"]}').patron_role == "a cook"


def test_everything_is_optional():
    assert Storyline.model_validate_json("{}").beginning == ""
    assert CampaignChecklist.model_validate_json("{}").setting_description is None


# --- checklist ---------------------------------------------------------------------------------

SRC = "I want a fog-choked marsh. The Swamp Witch is the threat. Children went silent overnight."
EVERYTHING = CampaignChecklist(
    setting_description="A fog-choked marsh.",
    primary_threat="The Swamp Witch.",
    starting_quest_hook="Children went silent.",
)


def test_merge_checklist_accepts_one_grounded_field_per_turn():
    first = merge_checklist({}, EVERYTHING, SRC)
    assert first == {"setting_description": "A fog-choked marsh."}
    second = merge_checklist(first, EVERYTHING, SRC)
    assert list(second) == ["setting_description", "primary_threat"]
    third = merge_checklist(second, EVERYTHING, SRC)
    assert pending(third) == []
    assert merge_checklist(third, None, SRC) == third
    changed = CampaignChecklist(setting_description="Something else entirely different")
    assert merge_checklist(third, changed, SRC) == third          # settled values are never replaced


def test_merge_checklist_rejects_fragments_and_invented_values():
    # the real failure: one answer filled all three fields with fragments
    src = "dark mystery decaying ruins"
    assert merge_checklist({}, CampaignChecklist(setting_description="dark"), src) == {}
    assert merge_checklist({}, CampaignChecklist(setting_description="decaying ruins"), src) == {}
    assert merge_checklist({}, CampaignChecklist(setting_description="A frozen northern citadel"), "I like marsh") == {}
    assert merge_checklist({}, CampaignChecklist(primary_threat="The Swamp Witch."), SRC) == {}   # not the first pending


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
    names = ["One", "Two", "Three"]
    quiet = lambda *a, **k: None
    assert choose_name(names, ask("2"), quiet) == "Two"
    assert choose_name(names, ask("4", "My Own"), quiet) == "My Own"
    assert choose_name(names, ask("4", " "), quiet) == "One"
    assert choose_name(names, ask("banana"), quiet) == "One"
    assert choose_name(names, ask("9"), quiet) == "One"


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


CHECKLIST = js(setting_description="A fog-choked marsh.", primary_threat="The Swamp Witch.",
               starting_quest_hook="Children went silent.")


# Four chat turns: the checklist now fills one field per turn (setting, threat, hook).
CHAT_INPUTS = [
    "I want a fog-choked marsh",
    "The Swamp Witch is the threat",
    "Children went silent overnight",
    "Yes that hook",
]
CHAT_RESPONSES = [
    "Guide: pick a setting. A) Marsh B) Desert C) Tundra D) Other",
    CHECKLIST,
    "Guide: pick a threat. A) Witch B) Dragon C) Cult D) Other",
    CHECKLIST,
    "Guide: pick a hook. A) Silence B) Plague C) War D) Other",
    CHECKLIST,
]


def test_full_run_creates_a_complete_campaign(tmp_path):
    responses = [
        *CHAT_RESPONSES,
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
    creator, out = make_creator(tmp_path, responses, [*CHAT_INPUTS, "1"])
    assert creator.run() == "the_drowned_bell"

    db = GameDB.open("the_drowned_bell", tmp_path)
    info = db.get_campaign()
    assert info.name == "The Drowned Bell"
    patron_name, _, patron_role = info.starting_patron.partition(", ")
    first, _, last = patron_name.partition(" ")
    assert first in names.FIRST_NAMES and last in names.LAST_NAMES and patron_role == "the ferry-keeper"
    assert info.starting_location_id == "mirewick_ferry"
    assert info.storyline_middle == "Complication one. Complication two."
    assert len(db.list_locations()) == 5
    assert db.get_connected_locations("reed_hut")                      # the isolated hut was connected
    assert len(db.get_beats()) == 4
    assert not db.has_player()
    db.close()
    assert any("SUCCESS" in line for line in out)


def test_complete_with_items_pending_keeps_chatting_and_cancel_leaves(tmp_path):
    creator, out = make_creator(tmp_path, ["x"], ["complete", "cancel"])
    assert creator.run() is None
    assert any("Still pending" in line for line in out)
    assert GameDB.list_campaigns(tmp_path) == []


def test_garbage_from_the_model_still_creates_a_playable_campaign(tmp_path):
    creator, _ = make_creator(tmp_path, [*CHAT_RESPONSES, "garbage"], [*CHAT_INPUTS, "1"])
    slug = creator.run()
    assert slug == "an_unknown_journey"
    db = GameDB.open(slug, tmp_path)
    assert len(db.list_locations()) == 3
    assert len(db.get_beats()) == 3
    assert db.get_campaign().starting_location_id == "starting_point"
    db.close()


def test_two_campaigns_with_the_same_name_get_different_slots(tmp_path):
    for _ in range(2):
        creator, _ = make_creator(tmp_path, [*CHAT_RESPONSES, "garbage"], [*CHAT_INPUTS, "1"])
        creator.run()
    assert sorted(GameDB.list_campaigns(tmp_path)) == ["an_unknown_journey", "an_unknown_journey_2"]


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


def test_apply_choice_fills_the_first_pending_item_and_rejects_fragments():
    assert apply_choice({}, "A fog-choked marsh kingdom") == {"setting_description": "A fog-choked marsh kingdom"}
    assert apply_choice({}, "Dark marsh") == {}
    done = {k: "x y z" for k in cc.CHECKLIST_KEYS}
    assert apply_choice(done, "Something else entirely") == done
    first = apply_choice({}, "A fog-choked marsh kingdom")
    assert list(apply_choice(first, "The Swamp Witch Orsolya")) == ["setting_description", "primary_threat"]


def test_menu_letters_build_a_campaign_without_extraction_calls(tmp_path):
    responses = [
        "Pick a setting.\nA) A fog-choked marsh kingdom\nB) A sunburnt desert empire\nC) A frozen citadel\nD) Something else",
        "Pick a threat.\nA) A cult of drowned priests\nB) The Swamp Witch Orsolya\nC) A rival warlord's army\nD) Something else",
        "Pick a hook.\nA) Children have gone silent\nB) A plague sweeps the village\nC) A stranger offers a map\nD) Something else",
        js(theme_vibe="Gothic", patron_role="a ferry-keeper", suggested_names=["Marsh Song", "Silent Fog", "Drowned Bell"]),
        "garbage",
    ]
    creator, _ = make_creator(tmp_path, responses, ["I want something spooky", "a", "b", "c", "1"])
    assert creator.run() == "marsh_song"
    with GameDB.open("marsh_song", tmp_path) as db:
        info = db.get_campaign()
        assert info.setting_description == "A fog-choked marsh kingdom"
        assert info.primary_threat == "The Swamp Witch Orsolya"
        assert info.starting_quest_hook == "A stranger offers a map"
        assert info.starting_patron.endswith(", a ferry-keeper")


def test_something_else_asks_for_typed_input_in_campaign_creation(tmp_path):
    menu = "Pick a setting.\nA) A fog-choked marsh kingdom\nB) A sunburnt desert empire\nC) A frozen citadel\nD) Something else"
    creator, out = make_creator(tmp_path, [menu], ["I want something spooky", "d", "cancel"])
    assert creator.run() is None
    assert any("type your own answer" in line for line in out)
