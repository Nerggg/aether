from aether.llm import prompts
from aether.models import CampaignInfo, Character, Location, StoryBeat

CAMPAIGN = CampaignInfo(
    name="The Drowned Bell", slug="drowned", theme_vibe="Gothic Dark Fantasy",
    starting_patron="Mother Hesk", setting_description="A foggy marsh.",
    primary_threat="The Swamp Witch.", starting_quest_hook="Children have gone silent.",
    storyline_beginning="B" * 50, storyline_middle="M" * 50, storyline_end="E" * 50,
    victory_condition="Defeat the witch.", starting_location_id="ferry",
)
PLAYER = Character(name="Aria", type="player", location_id="ferry", id=1, race="Elf", char_class="Ranger",
                   hp=9, max_hp=11, dexterity=17, backstory="A scout who lost her squad.")
FERRY = Location("ferry", "Mirewick Ferry", "A rotting landing.")
CAUSEWAY = Location("causeway", "Rotting Causeway", "A sunken road.")
BEAT = StoryBeat(1, "Find the Bell", "Discover where the bell rings from.")


# --- DM prompt -------------------------------------------------------------------------------

def test_dm_prompt_forbids_menus_and_player_control():
    text = prompts.dm_system_prompt()
    assert "never write a question to the player" in text
    assert "Never decide the player's" in text


def test_dm_prompt_lists_only_real_tags_and_not_roll():
    text = prompts.dm_system_prompt()
    for tag in ("ADD_ITEM", "REMOVE_ITEM", "TAKE_DAMAGE", "HEAL_HP"):
        assert tag in text
    assert "[ROLL" not in text


def test_dm_prompt_uses_campaign_tone():
    assert "Gothic Dark Fantasy" in prompts.dm_system_prompt(CAMPAIGN)
    assert "Tone:" not in prompts.dm_system_prompt(None)


# --- blocks ------------------------------------------------------------------------------------

def test_blocks_are_empty_when_nothing_to_say():
    assert prompts.character_sheet_block(None) == ""
    assert prompts.location_block(None) == ""
    assert prompts.npcs_block([]) == ""
    assert prompts.recall_block([]) == ""
    assert prompts.recall_block(["", "  "]) == ""
    assert prompts.campaign_block(None) == ""


def test_character_sheet_shows_live_values_and_real_race_and_class():
    text = prompts.character_sheet_block(PLAYER, [("Torch", 2), ("Rope", 1)])
    assert "Aria (Elf Ranger)" in text
    assert "HP: 9/11" in text
    assert "DEX 17" in text
    assert "Torch x2, Rope" in text
    assert "Backstory" not in text


def test_character_sheet_backstory_optional_and_clipped():
    assert "Backstory: A scout" in prompts.character_sheet_block(PLAYER, [], include_backstory=True)
    long = Character(name="X", type="player", location_id="l", backstory="word " * 400)
    text = prompts.character_sheet_block(long, [], include_backstory=True, backstory_chars=100)
    assert text.endswith("...") and len(text.splitlines()[-1]) < 130


def test_empty_inventory_and_missing_class():
    plain = Character(name="Nobody", type="player", location_id="l")
    text = prompts.character_sheet_block(plain, [])
    assert "Name: Nobody\n" in text
    assert "Inventory: empty" in text


def test_location_block_prefers_description_over_concept():
    assert "A rotting landing." in prompts.location_block(FERRY, [])
    described = Location("ferry", "Mirewick Ferry", "concept", description="Fog hangs low.")
    text = prompts.location_block(described, [CAUSEWAY])
    assert "Fog hangs low." in text and "concept" not in text
    assert "Paths from here: Rotting Causeway" in text


def test_npcs_block_includes_notes_by_character_id():
    hesk = Character(name="Mother Hesk", type="ally", location_id="ferry", id=7)
    gruk = Character(name="Gruk", type="enemy", location_id="ferry", id=8)
    text = prompts.npcs_block([hesk, gruk], {7: "Runs the ferry."})
    assert "- Mother Hesk (ally): Runs the ferry." in text
    assert "- Gruk (enemy)" in text and "Gruk (enemy):" not in text


def test_recall_block_lists_chunks():
    text = prompts.recall_block(["Player: hi\nDM: hello"])
    assert text.startswith("EARLIER EVENTS")
    assert "- Player: hi DM: hello" in text


def test_scene_state_block():
    hesk = Character(name="Mother Hesk", type="ally", location_id="ferry")
    text = prompts.scene_state_block(PLAYER, [("Rope", 1)], FERRY, [CAUSEWAY], [hesk])
    assert "Location: Mirewick Ferry (paths: Rotting Causeway)" in text
    assert "Present: Mother Hesk (ally)" in text
    assert "Inventory: Rope" in text
    assert "Present: nobody else" in prompts.scene_state_block(PLAYER, [], FERRY)


# --- opener ------------------------------------------------------------------------------------------

def test_opening_messages_use_campaign_character_and_forbid_waking_up():
    msgs = prompts.opening_messages(CAMPAIGN, PLAYER, [("Rope", 1)], FERRY, [CAUSEWAY], [], BEAT)
    assert [m["role"] for m in msgs] == ["system", "user"]
    system, user = msgs[0]["content"], msgs[1]["content"]
    assert "Backstory: A scout" in system
    assert "CURRENT LOCATION: Mirewick Ferry" in system
    assert "CURRENT STORY OBJECTIVE:" in system
    assert "Children have gone silent." in user
    assert "Mother Hesk" in user
    assert "Do NOT begin with the character waking up" in user
    assert "open" not in user.lower().replace("opening", "")  # no fake 'I open my eyes' line


def test_opening_works_without_beat_or_patron():
    bare = CampaignInfo(name="N", slug="n", setting_description="s", primary_threat="t", starting_quest_hook="")
    msgs = prompts.opening_messages(bare, PLAYER, [], FERRY)
    assert "CURRENT STORY OBJECTIVE" not in msgs[0]["content"]
    assert "Include" not in msgs[1]["content"]


# --- options -------------------------------------------------------------------------------------------

def test_options_prompt_has_one_slot_per_category_in_order():
    msgs = prompts.options_messages("The cave yawns.", ["story", "cautious", "bold"], "STATE", 8, ["Aria"], BEAT)
    system = msgs[0]["content"]
    assert "exactly 3 actions" in system
    assert "1. [story]" in system and "Find the Bell: Discover where the bell rings from." in system
    assert system.index("[story]") < system.index("[cautious]") < system.index("[bold]")
    assert "at most 8 words" in system
    assert "known list: Aria" in system
    assert "beat_achieved" in system.split("Reply as JSON")[1]  # only in the JSON example
    assert msgs[1]["content"] == "STATE:\nSTATE\n\nSCENE:\nThe cave yawns."


def test_options_prompt_beat_question_only_when_asked():
    plain = prompts.options_messages("s", ["story"], "x", 8, [], BEAT)[0]["content"]
    asked = prompts.options_messages("s", ["story"], "x", 8, [], BEAT, check_beat=True)[0]["content"]
    assert "fully achieved" not in plain
    assert "fully achieved: Discover where the bell rings from." in asked


def test_options_prompt_lists_skills_and_handles_no_known_names():
    system = prompts.options_messages("s", ["bold"], "x", 8)[0]["content"]
    assert "sleight of hand" in system and "persuasion" in system
    assert "known list: none" in system


# --- judge -------------------------------------------------------------------------------------------------

def test_judge_prompt_has_three_verdicts_and_the_action():
    msgs = prompts.judge_messages("I fly away", "Scene text", "STATE", BEAT)
    system, user = msgs[0]["content"], msgs[1]["content"]
    for word in ("feasible", "risky", "impossible"):
        assert word in system
    assert "Discover where the bell rings from." in system
    assert user.endswith("PLAYER ACTION:\nI fly away")


def test_judge_prompt_without_beat_skips_beat_field():
    assert "advances_beat" not in prompts.judge_messages("a", "s", "x")[0]["content"]


# --- compaction, beats, location ---------------------------------------------------------------------------

def test_compaction_prompt():
    msgs = prompts.compaction_messages("STORY SO FAR:\nOld.", "Aria: hi\nDM: hello", "Aria", 300)
    assert "at most 300 words" in msgs[0]["content"]
    assert "Do not record HP, inventory or the current location" in msgs[0]["content"]
    assert "EXISTING SUMMARY:\nSTORY SO FAR:\nOld." in msgs[1]["content"]
    assert "NEW EVENTS:\nAria: hi" in msgs[1]["content"]


def test_compaction_prompt_with_no_previous_summary():
    assert "(none yet)" in prompts.compaction_messages("", "x", "Aria", 300)[1]["content"]


def test_beats_prompt_lists_real_location_ids_and_counts():
    msgs = prompts.beats_messages(CAMPAIGN, [FERRY, CAUSEWAY], 4, 6)
    system = msgs[0]["content"]
    assert "between 4 and 6 beats" in system
    assert "- ferry: Mirewick Ferry" in system and "- causeway: Rotting Causeway" in system
    assert "Victory condition: Defeat the witch." in msgs[1]["content"]


def test_location_description_prompt():
    msgs = prompts.location_description_messages("Chapel", "A flooded chapel.", "Gothic")
    assert "Location: Chapel" in msgs[1]["content"] and "Style: Gothic" in msgs[1]["content"]
    assert "Heroic Fantasy" in prompts.location_description_messages("A", "b", "")[1]["content"]


def test_options_prompt_example_carries_category_labels():
    system = prompts.options_messages("s", ["story", "bold"], "x", 8)[0]["content"]
    assert '"category":"story"' in system and "label in square brackets" in system


def test_opening_limits_the_cast_and_length():
    user = prompts.opening_messages(CAMPAIGN, PLAYER, [], FERRY)[1]["content"]
    assert "only named characters in the scene are Mother Hesk" in user
    assert "2 short paragraphs" in user
    bare = CampaignInfo(name="N", slug="n", setting_description="s", primary_threat="t", starting_quest_hook="h")
    assert "Do not name any character" in prompts.opening_messages(bare, PLAYER, [], FERRY)[1]["content"]


def test_compaction_prompt_has_example_and_forbids_retelling():
    system = prompts.compaction_messages("", "x", "Aria", 100)[0]["content"]
    assert "do not retell" in system and '"key_npcs"' in system and "Dessa" in system


def test_beats_prompt_only_offers_location_and_referee():
    system = prompts.beats_messages(CAMPAIGN, [FERRY, CAUSEWAY], 4, 6)[0]["content"]
    assert "has_item" not in system and "npc_dead" not in system
    assert "beat 1 must be something still ahead" in system and "Mirewick Ferry" in system
