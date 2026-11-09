import json

import pytest

from aether.config import Config
from aether.llm import calls
from aether.llm.client import LLMClient, LLMError
from aether.models import CampaignInfo, Location, Message, StoryBeat, Summary
from fakes import FakeBackend


def client_with(*responses, config=None):
    backend = FakeBackend(*responses)
    return LLMClient(config or Config(), backend), backend


def js(**kw):
    return json.dumps(kw)


CATS = ["story", "cautious", "bold"]
BEAT = StoryBeat(1, "Find the Bell", "Discover where the bell rings from.")


# --- narration ----------------------------------------------------------------------------------

def test_narrate_separates_tags_from_text():
    client, backend = client_with(["The rope snaps. [TAKE_", "DAMAGE: 3] You land hard."])
    text, tags = calls.narrate(client, [{"role": "user", "content": "go"}])
    assert text == "The rope snaps.  You land hard."
    assert [(t.name, t.value) for t in tags] == [("TAKE_DAMAGE", "3")]
    assert backend.calls[0]["model"] == "llama3.2"


def test_narrate_stream_shows_text_before_the_stream_ends():
    client, _ = client_with(["Hello ", "there", "[ADD_ITEM: Rope]"])
    results = list(calls.narrate_stream(client, []))
    assert results[0].text == "Hello "
    assert results[-1].tags[0].value == "Rope"


def test_narrate_stream_releases_unfinished_tag_as_text():
    client, _ = client_with(["You see [ADD_ITEM: Rop"])
    text, tags = calls.narrate(client, [])
    assert text == "You see [ADD_ITEM: Rop" and tags == []


def test_narrate_error_propagates():
    client, _ = client_with(ConnectionError("down"))
    with pytest.raises(LLMError):
        calls.narrate(client, [])


def test_describe_location_and_fallback():
    client, _ = client_with("Water drips.")
    assert calls.describe_location(client, name="Cave", concept="A cave.") == "Water drips."
    empty, _ = client_with("   ")
    assert calls.describe_location(empty, name="Cave", concept="A cave.") == "A cave."


# --- options ---------------------------------------------------------------------------------------

def test_generate_options_cleans_and_assigns_categories():
    reply = js(options=[
        {"text": "1. Search for the bell's source", "skill": "Investigation"},
        {"text": "Move quietly down the path", "skill": "Stealth"},
        {"text": "Charge into the fog", "skill": "flying"},
    ])
    client, backend = client_with(reply)
    r = calls.generate_options(client, narration="n", categories=CATS, state_block="s", beat=BEAT)
    assert [o.text for o in r.options] == ["Search for the bell's source", "Move quietly down the path", "Charge into the fog"]
    assert [o.skill for o in r.options] == ["investigation", "stealth", "athletics"]
    assert [o.category for o in r.options] == CATS
    assert r.ok and not r.beat_achieved and r.new_npcs == []
    assert backend.calls[0]["options"]["num_predict"] == calls.OPTIONS_MAX_TOKENS


def test_generate_options_tops_up_a_short_menu():
    client, _ = client_with(js(options=[{"text": "Search the ruins", "skill": None}]))
    r = calls.generate_options(client, narration="n", categories=CATS, state_block="s")
    assert len(r.options) == 3
    assert [o.category for o in r.options] == CATS
    assert r.ok is True


def test_generate_options_trims_a_long_menu_and_duplicates():
    client, _ = client_with(js(options=[
        {"text": "Look around the cave"}, {"text": "Look around the cave carefully"},
        {"text": "Shout into the dark"}, {"text": "Climb the wall"}, {"text": "Hide"},
    ]))
    r = calls.generate_options(client, narration="n", categories=CATS, state_block="s")
    assert len(r.options) == 3
    assert len({o.text for o in r.options}) == 3


def test_generate_options_model_failure_gives_fallback_menu():
    client, _ = client_with("garbage")
    r = calls.generate_options(client, narration="n", categories=CATS, state_block="s")
    assert r.ok is False
    assert [o.category for o in r.options] == CATS
    assert r.new_npcs == [] and r.beat_achieved is False


def test_generate_options_filters_new_npcs():
    client, _ = client_with(js(options=[{"text": "Hide"}], new_npcs=[
        {"name": "Aria", "type": "ally"},           # the player
        {"name": " mother hesk "},                  # already known, other case
        {"name": "Tulle", "type": "ally", "is_persistent": True},
        {"name": "tulle"},                          # duplicate in same reply
        {"name": "  "},
        {"name": "Gruk", "type": "enemy"},
        {"name": "Mog"}, {"name": "Nim"},
    ]))
    r = calls.generate_options(client, narration="n", categories=CATS, state_block="s",
                               known_npc_names=["Aria", "Mother Hesk"])
    assert [n.name for n in r.new_npcs] == ["Tulle", "Gruk", "Mog"]  # capped at 3
    assert r.new_npcs[0].is_persistent is True


def test_beat_achieved_only_counts_when_asked():
    reply = js(options=[{"text": "Hide"}], beat_achieved=True)
    client, _ = client_with(reply)
    assert calls.generate_options(client, narration="n", categories=CATS, state_block="s",
                                  beat=BEAT, check_beat=True).beat_achieved is True
    client, _ = client_with(reply)
    assert calls.generate_options(client, narration="n", categories=CATS, state_block="s",
                                  beat=BEAT, check_beat=False).beat_achieved is False


# --- judge ----------------------------------------------------------------------------------------------

def judge(reply, beat=None):
    client, backend = client_with(reply)
    return calls.judge_action(client, action="a", narration="n", state_block="s", beat=beat), backend


def test_judge_feasible():
    r, backend = judge(js(verdict="feasible", skill="stealth"))
    assert (r.verdict, r.skill, r.ok) == ("feasible", None, True)   # skill ignored unless risky
    assert backend.calls[0]["options"]["num_predict"] == calls.JUDGE_MAX_TOKENS


def test_judge_risky_with_skill_and_difficulty():
    r, _ = judge(js(verdict="risky", skill="Sleight_of_Hand", difficulty="hard", reason="  Tricky.  "))
    assert (r.verdict, r.skill, r.difficulty, r.reason) == ("risky", "sleight of hand", "hard", "Tricky.")


def test_judge_risky_without_usable_skill_becomes_feasible():
    assert judge(js(verdict="risky"))[0].verdict == "feasible"
    assert judge(js(verdict="risky", skill="flying"))[0].verdict == "feasible"


def test_judge_impossible_always_has_a_reason():
    r, _ = judge(js(verdict="impossible"))
    assert r.verdict == "impossible" and r.reason


def test_judge_impossible_keeps_models_reason():
    assert judge(js(verdict="impossible", reason="You have no bow."))[0].reason == "You have no bow."


def test_judge_beat_flag_needs_a_beat_and_a_possible_action():
    reply = js(verdict="feasible", advances_beat=True)
    assert judge(reply, BEAT)[0].advances_beat is True
    assert judge(reply, None)[0].advances_beat is False
    assert judge(js(verdict="impossible", advances_beat=True), BEAT)[0].advances_beat is False


def test_judge_failure_lets_the_action_through():
    r, _ = judge("not json at all")
    assert (r.verdict, r.ok) == ("feasible", False)


# --- compaction ---------------------------------------------------------------------------------------------

MSGS = [Message(10, 1, "player", "Hi."), Message(11, 1, "dm", "Hello, traveller.")]


def test_compaction_of_nothing_makes_no_call():
    client, backend = client_with("x")
    assert calls.compact_history(client, previous=None, messages=[]) is None
    assert backend.calls == []

BELL = [
    Message(1, 0, "dm", "Fog covers the ferry. Mother Hesk waits."),
    Message(2, 1, "player", "Ask Mother Hesk about the bell."),
    Message(3, 1, "system", "Perception check (WIS): d20 15 + 2 = 17 vs DC 13 - Success"),
    Message(4, 1, "dm", "Mother Hesk says the bell belonged to a chapel that sank long ago. She begs you to find it."),
]
NO_FACTS = js(open_threads=[], promises_debts=[])


def test_extractive_event_formats():
    G = calls.extractive_event
    assert G([Message(1, 1, "player", "Run away!")], "Aria") == 'Aria chose "Run away".'
    assert G([Message(1, 0, "dm", "Fog rolls in. More text.")], "Aria") == "Opening: Fog rolls in."
    group = [Message(1, 1, "player", "Sneak past."),
             Message(2, 1, "system", "Stealth check (DEX): d20 3 + 1 = 4 vs DC 13 - Failure"),
             Message(3, 1, "dm", "A guard shouts! You run.")]
    assert G(group, "Aria") == 'Aria chose "Sneak past" (failure). A guard shouts!'
    assert G([], "Aria") == ""


def test_is_grounded():
    src = "Mother Hesk begs Aria to find the drowned bell near the chapel."
    assert calls.is_grounded("Find the drowned bell", src)
    assert not calls.is_grounded("Visit Elwynn in Ravenshire", src)
    assert not calls.is_grounded("Cross the dunes safely", src)
    assert not calls.is_grounded("...", src)


def test_compaction_lines_are_extractive_and_npcs_come_from_the_database():
    client, backend = client_with(NO_FACTS)
    c = calls.compact_history(client, previous=None, messages=BELL, player_name="Aria",
                              known_npc_names=["Mother Hesk", "Gruk"])
    assert c.summary.narrative.splitlines() == [
        "Opening: Fog covers the ferry.",
        'Aria chose "Ask Mother Hesk about the bell" (success). '
        "Mother Hesk says the bell belonged to a chapel that sank long ago.",
    ]
    assert c.summary.covers_through_message_id == 4
    assert c.summary.key_npcs == ["Mother Hesk"]
    assert c.npc_notes == {"Mother Hesk": "Mother Hesk says the bell belonged to a chapel that sank long ago."}
    assert len(backend.calls) == 1 and "format" in backend.calls[0]


def test_compaction_drops_threads_and_promises_the_text_never_mentions():
    client, _ = client_with(js(
        open_threads=["Find the bell", "Find out where Voss hides", "Cross the dunes safely", "..."],
        promises_debts=["Dessa will pay Kael double on arrival", "Mother Hesk wants the bell found"],
    ))
    c = calls.compact_history(client, previous=None, messages=BELL, player_name="Aria",
                              known_npc_names=["Mother Hesk"])
    assert c.summary.open_threads == ["Find the bell"]
    assert c.summary.promises_debts == ["Mother Hesk wants the bell found"]


def test_compaction_keeps_old_lists_when_the_facts_call_fails():
    client, _ = client_with("garbage")
    prev = Summary(covers_through_message_id=0, narrative="Earlier line.", key_npcs=["Hesk"],
                   open_threads=["Find it"], promises_debts=["A favour"])
    c = calls.compact_history(client, previous=prev, messages=BELL, player_name="Aria")
    assert c.summary.narrative.splitlines()[0] == "Earlier line."
    assert (c.summary.key_npcs, c.summary.open_threads, c.summary.promises_debts) == (
        ["Hesk"], ["Find it"], ["A favour"])


def test_compaction_caps_key_npcs():
    names = [f"Npc{i}" for i in range(12)]
    msgs = [Message(1, 1, "player", "Hi."), Message(2, 1, "dm", " ".join(names) + " all wait.")]
    client, _ = client_with(NO_FACTS)
    c = calls.compact_history(client, previous=None, messages=msgs, known_npc_names=names)
    assert c.summary.key_npcs == names[:8]


def test_compaction_drops_the_oldest_lines_when_the_summary_is_too_long():
    cfg = Config()
    cfg.memory.summary_max_tokens = 100                      # 75 words
    prev = Summary(covers_through_message_id=0,
                   narrative="\n".join(f"Old event number {i} " + "filler " * 5 for i in range(10)))
    client, _ = client_with(NO_FACTS, config=cfg)
    narrative = calls.compact_history(client, previous=prev, messages=BELL, player_name="Aria").summary.narrative
    assert "Old event number 0" not in narrative
    assert narrative.splitlines()[-1].startswith("Aria chose")
    assert len(narrative.split()) <= 75

def test_find_invented_names():
    known = "Mother Hesk Aria Mirewick Ferry"
    text = 'The fog rolls in. Ferry-keeper Grimbold waves. "Come," says Mother Hesk.'
    assert calls.find_invented_names(text, known) == ["Grimbold"]
    assert calls.find_invented_names("You see Mother Hesk.", known) == []


def test_opening_is_regenerated_when_a_name_is_invented():
    good, bad = "You step onto the pier. Mother Hesk waits.", "You step onto the pier. Old Grimbold waits."
    client, backend = client_with(bad, good)
    text, _ = calls.narrate_opening(client, [{"role": "user", "content": "go"}], known_text="Mother Hesk")
    assert text == good and len(backend.calls) == 2
    assert "Grimbold" in backend.calls[1]["messages"][-1]["content"]


def test_opening_gives_up_after_the_attempt_limit():
    bad = "You step onto the pier. Old Grimbold waits."
    client, backend = client_with(bad)
    text, _ = calls.narrate_opening(client, [{"role": "user", "content": "go"}], known_text="Mother Hesk")
    assert text == bad and len(backend.calls) == 2

# --- story beats -----------------------------------------------------------------------------------------------

CAMPAIGN = CampaignInfo(
    name="N", slug="n", setting_description="s", primary_threat="The Witch", starting_quest_hook="hook",
    storyline_beginning="Begin.", storyline_middle="Middle.", storyline_end="End.",
    victory_condition="Defeat the witch.", starting_location_id="ferry",
)
LOCS = [Location("ferry", "Ferry", "c"), Location("cave", "Cave", "c")]


def beats_reply(*beats):
    return js(beats=list(beats))


def test_derive_beats_keeps_valid_conditions_and_downgrades_bad_ones():
    reply = beats_reply(
        {"title": "Reach the ferry", "objective": "Be at the ferry.", "condition_type": "location", "condition_value": "ferry"},
        {"title": "Enter the cave", "objective": "Go in.", "condition_type": "location", "condition_value": "cave"},
        {"title": "Find a place", "objective": "Go to Atlantis.", "condition_type": "location", "condition_value": "atlantis"},
        {"title": "Get the key", "objective": "Hold the key.", "condition_type": "has_item", "condition_value": "Silver Key"},
        {"title": "Empty", "objective": "No value.", "condition_type": "npc_dead", "condition_value": ""},
        {"title": "Win", "objective": "Defeat the witch.", "condition_type": "referee", "condition_value": "Witch defeated"},
    )
    client, _ = client_with(reply)
    beats = calls.derive_beats(client, campaign=CAMPAIGN, locations=LOCS)
    assert [b.position for b in beats] == [1, 2, 3, 4, 5, 6]
    assert (beats[0].condition_type, beats[0].condition_value) == ("referee", "")   # would complete instantly
    assert (beats[1].condition_type, beats[1].condition_value) == ("location", "cave")
    assert beats[2].condition_type == "referee"                                     # invented id
    assert (beats[3].condition_type, beats[3].condition_value) == ("referee", "")   # has_item is not allowed any more
    assert (beats[4].condition_type, beats[4].condition_value) == ("referee", "")   # npc_dead neither
    assert beats[5].condition_value == "Witch defeated"
    assert all(b.status == "pending" for b in beats)


def test_derive_beats_respects_the_maximum_and_skips_blank_beats():
    reply = beats_reply(*[{"title": f"B{i}", "objective": f"Do {i}."} for i in range(9)],)
    cfg = Config()
    cfg.gameplay.story_beats_max = 5
    client, _ = client_with(reply, config=cfg)
    assert len(calls.derive_beats(client, campaign=CAMPAIGN, locations=LOCS)) == 5

    reply = beats_reply({"title": "", "objective": "x"}, {"title": "A", "objective": "a"}, {"title": "B", "objective": "b"})
    client, _ = client_with(reply)
    assert [b.title for b in calls.derive_beats(client, campaign=CAMPAIGN, locations=LOCS)] == ["A", "B"]


def test_derive_beats_falls_back_when_the_model_fails_or_gives_too_few():
    for reply in ("garbage", beats_reply({"title": "Only", "objective": "one"})):
        client, _ = client_with(reply)
        beats = calls.derive_beats(client, campaign=CAMPAIGN, locations=LOCS)
        assert [b.position for b in beats] == [1, 2, 3]
        assert beats[-1].objective == "Defeat the witch."
        assert all(b.condition_type == "referee" for b in beats)


def test_fallback_beats_work_with_an_empty_storyline():
    bare = CampaignInfo(name="N", slug="n", setting_description="s", primary_threat="The Witch",
                        starting_quest_hook="Find the child.")
    beats = calls.fallback_beats(bare)
    assert beats[0].objective == "Find the child."
    assert beats[2].objective == "Defeat The Witch"


def test_tidy_narration():
    assert calls.tidy_narration("The fog thickens. What do you do?") == "The fog thickens."
    assert calls.tidy_narration("Fog. What now? What do you do?") == "Fog."
    assert calls.tidy_narration("What do you do?") == "What do you do?"            # nothing else to keep
    quoted = '"Do you hear it?" she whispers.'
    assert calls.tidy_narration(quoted) == quoted
    ends_in_quote = 'Hesk leans close. "Do you hear it?"'
    assert calls.tidy_narration(ends_in_quote) == ends_in_quote
    assert calls.tidy_narration("The bell tolls. Mist rises. The path ben") == "The bell tolls. Mist rises."
    assert calls.tidy_narration("   ") == ""


def test_narrate_tidies_the_saved_text():
    client, _ = client_with(["Mist rises. What do you do?"])
    text, _ = calls.narrate(client, [])
    assert text == "Mist rises."


def test_placeholder_tags_never_reach_the_game():
    client, _ = client_with(["You wait. [ADD_ITEM: None] [TAKE_DAMAGE: none]"])
    text, tags = calls.narrate(client, [])
    assert tags == [] and "None" not in text


def test_generate_options_follows_model_categories_and_sorts_to_slots():
    client, _ = client_with(js(options=[
        {"text": "Charge into the fog", "category": "bold"},
        {"text": "Search for the bell", "category": "story"},
        {"text": "Creep along the edge", "category": "cautious"},
    ]))
    r = calls.generate_options(client, narration="n", categories=CATS, state_block="s")
    assert [o.category for o in r.options] == CATS
    assert r.options[0].text == "Search for the bell"


def test_a_later_beat_aimed_at_the_starting_location_is_downgraded():
    reply = beats_reply(
        {"title": "Enter", "objective": "Go in.", "condition_type": "location", "condition_value": "cave"},
        {"title": "Return", "objective": "Come back.", "condition_type": "location", "condition_value": "ferry"},
        {"title": "Win", "objective": "Win."},
    )
    client, _ = client_with(reply)
    beats = calls.derive_beats(client, campaign=CAMPAIGN, locations=LOCS)
    assert (beats[0].condition_type, beats[0].condition_value) == ("location", "cave")
    assert (beats[1].condition_type, beats[1].condition_value) == ("referee", "")
