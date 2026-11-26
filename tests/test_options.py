import pytest

from aether.logic.options import (
    CATEGORY_POOLS,
    CONVERSATION,
    EXPLORATION,
    FALLBACK_OPTIONS,
    STORY,
    ActionOption,
    Choice,
    clean_option_text,
    complete_options,
    detect_situation,
    format_menu,
    parse_choice,
    sanitize_options,
    select_categories,
    apply_default_skills,
    ROLLING_CATEGORIES,
    avoid_repeating,
    filter_unusable_options,
    is_spell_option,
    restrict_rolls,
    rolled_skill,
)


# --- situation and categories ------------------------------------------------------

def test_detect_situation():
    assert detect_situation(["Barnaby"], "Barnaby waves you over.") == CONVERSATION
    assert detect_situation(["Barnaby"], "barnaby waves you over.") == CONVERSATION
    assert detect_situation(["Barnaby"], "The cave is quiet.") == EXPLORATION
    assert detect_situation([], "Anyone there?") == EXPLORATION
    assert detect_situation([" "], "text") == EXPLORATION


def test_story_slot_comes_first():
    cats = select_categories(EXPLORATION, 3, turn_number=0)
    assert cats[0] == STORY
    assert len(cats) == 3


def test_no_story_slot_without_a_beat():
    cats = select_categories(EXPLORATION, 3, has_story_beat=False)
    assert STORY not in cats
    assert len(cats) == 3


def test_same_inputs_give_same_categories():
    assert select_categories(CONVERSATION, 3, 5) == select_categories(CONVERSATION, 3, 5)


def test_categories_rotate_with_turns():
    seen = {tuple(select_categories(EXPLORATION, 3, t)) for t in range(4)}
    assert len(seen) == 4


def test_categories_are_distinct_and_from_pool():
    for situation in (EXPLORATION, CONVERSATION):
        for n in range(1, 6):
            for turn in range(8):
                cats = select_categories(situation, n, turn)
                assert len(cats) == len(set(cats)) == n
                assert set(cats) - {STORY} <= set(CATEGORY_POOLS[situation])


def test_five_options_possible_in_both_situations():
    assert len(select_categories(EXPLORATION, 5)) == 5
    assert len(select_categories(CONVERSATION, 5)) == 5


def test_too_many_options_is_capped_by_pool():
    assert len(select_categories(EXPLORATION, 9)) == 5


def test_bad_inputs():
    with pytest.raises(ValueError):
        select_categories("combat", 3)
    with pytest.raises(ValueError):
        select_categories(EXPLORATION, 0)


def test_every_category_has_a_fallback():
    every = {c for pool in CATEGORY_POOLS.values() for c in pool} | {STORY}
    assert every <= set(FALLBACK_OPTIONS)


# --- cleaning ---------------------------------------------------------------------------

def test_clean_strips_numbering_and_quotes():
    assert clean_option_text("1. Explore further", 8) == "Explore further"
    assert clean_option_text("2) 'Shout loudly'", 8) == "Shout loudly"
    assert clean_option_text("- Prepare gear", 8) == "Prepare gear"
    assert clean_option_text("(3) Turn back", 8) == "Turn back"
    assert clean_option_text('"Run!"', 8) == "Run!"


def test_clean_truncates_and_collapses_space():
    assert clean_option_text("Slowly   creep  toward the glowing door in the wall", 5) == "Slowly creep toward the glowing"
    assert clean_option_text("one two three, four", 3) == "one two three"


def test_clean_handles_junk():
    assert clean_option_text(None, 8) == ""
    assert clean_option_text("   ", 8) == ""
    assert clean_option_text(42, 8) == "42"


def test_sanitize_normalises_skills_and_fills_categories():
    raw = [
        {"text": "1. Sneak past the guard", "skill": "Stealth"},
        {"text": "Charge the guard", "skill": "flying"},
        "Ask the guard a question",
    ]
    result = sanitize_options(raw, 8, 3, categories=["story", "bold", "probing"])
    assert [o.text for o in result] == ["Sneak past the guard", "Charge the guard", "Ask the guard a question"]
    assert [o.skill for o in result] == ["stealth", None, None]
    assert [o.category for o in result] == ["story", "bold", "probing"]


def test_sanitize_drops_empty_and_duplicates():
    raw = ["Explore further", "", "Explore further!", "explore  further", "Light a torch"]
    result = sanitize_options(raw, 8, 5)
    assert [o.text for o in result] == ["Explore further", "Light a torch"]


def test_sanitize_drops_near_duplicates():
    raw = ["Look around the cave", "Look around the cave carefully", "Shout into the dark"]
    assert [o.text for o in sanitize_options(raw, 8, 5)] == ["Look around the cave", "Shout into the dark"]


def test_sanitize_respects_limit():
    raw = ["Alpha move", "Bravo plan", "Charlie trick", "Delta dash"]
    assert len(sanitize_options(raw, 8, 2)) == 2


def test_sanitize_keeps_explicit_category():
    result = sanitize_options([{"text": "Hide", "category": "cautious"}], 8, 3, categories=["story"])
    assert result[0].category == "cautious"


# --- completing a short menu -----------------------------------------------------------

def test_complete_fills_missing_categories():
    opts = [ActionOption("Sneak ahead", "stealth", "story")]
    full = complete_options(opts, ["story", "cautious", "bold"], 3)
    assert [o.category for o in full] == ["story", "cautious", "bold"]
    assert full[1].text == FALLBACK_OPTIONS["cautious"].text


def test_complete_does_nothing_when_full():
    opts = [ActionOption("a", None, "story"), ActionOption("b", None, "bold"), ActionOption("c", None, "cautious")]
    assert complete_options(opts, ["story", "bold", "cautious"], 3) == opts


def test_complete_from_empty_gives_a_full_menu():
    full = complete_options([], ["story", "cautious", "bold"], 3)
    assert len(full) == 3


def test_complete_does_not_mutate_fallbacks():
    full = complete_options([], ["bold"], 1)
    full[0].text = "changed"
    assert FALLBACK_OPTIONS["bold"].text != "changed"


def test_fallback_skills_are_valid():
    from aether.logic.stats import normalize_skill
    for opt in FALLBACK_OPTIONS.values():
        assert opt.skill is None or normalize_skill(opt.skill) == opt.skill
        assert len(opt.text.split()) <= 8


# --- saving and showing ------------------------------------------------------------------

def test_option_dict_roundtrip():
    o = ActionOption("Sneak", "stealth", "cautious")
    assert ActionOption.from_dict(o.to_dict()) == o


def test_from_dict_is_forgiving():
    o = ActionOption.from_dict({"text": "Hide", "skill": "Stealth!"})
    assert o == ActionOption("Hide", None, None)


def test_format_menu_adds_custom_slot():
    menu = format_menu([ActionOption("Explore", None), ActionOption("Shout", None)])
    assert menu == "1. Explore\n2. Shout\n3. Write your own action"


# --- reading the choice ------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("1", Choice("option", 0)),
    (" 3 ", Choice("option", 2)),
    ("2.", Choice("option", 1)),
    ("2)", Choice("option", 1)),
    ("4", Choice("custom", None)),
])
def test_parse_choice_valid(raw, expected):
    assert parse_choice(raw, 3) == expected


@pytest.mark.parametrize("raw", ["0", "5", "-1", "two", "I attack the guard", "", "1 2", "3rd"])
def test_parse_choice_invalid(raw):
    assert parse_choice(raw, 3) is None

def test_clean_does_not_leave_a_dangling_end():
    assert clean_option_text("Tell Mother Hesk to get us off the ferry quickly", 8) == "Tell Mother Hesk"
    assert clean_option_text("Back away slowly", 8) == "Back away slowly"


def test_default_skills_fill_gaps_only():
    opts = [ActionOption("Sneak", None, "cautious"), ActionOption("Hide", "stealth", "cautious"),
            ActionOption("Ask", None, "probing"), ActionOption("Go", None, "story")]
    # only bold has a default skill now; a skill the model gave is left for restrict_rolls to judge
    assert [o.skill for o in apply_default_skills(opts)] == [None, "stealth", None, None]
    bold = [ActionOption("Charge", None, "bold")]
    assert apply_default_skills(bold)[0].skill == "athletics"

def test_firm_replaces_confrontational_and_never_rolls_by_default():
    assert "firm" in CATEGORY_POOLS[CONVERSATION]
    assert "confrontational" not in CATEGORY_POOLS[CONVERSATION]
    assert apply_default_skills([ActionOption("Demand answers", None, "firm")])[0].skill is None


def test_only_bold_keeps_a_skill():
    opts = [ActionOption("Search", "investigation", "investigative"), ActionOption("Sneak", "stealth", "cautious"),
            ActionOption("Make a snack", "survival", "resourceful"), ActionOption("Charge", "athletics", "bold"),
            ActionOption("Go on", "perception", "story"), ActionOption("Chat", "persuasion", "friendly")]
    assert [o.skill for o in restrict_rolls(opts)] == [None, None, None, "athletics", None, None]
    assert ROLLING_CATEGORIES == {"bold"}


def test_rolled_skill_ignores_skills_on_menus_saved_by_older_versions():
    assert rolled_skill(ActionOption("Sneak", "stealth", "cautious")) is None
    assert rolled_skill(ActionOption("Charge", "athletics", "bold")) == "athletics"
    assert rolled_skill(ActionOption("Charge", None, "bold")) is None


def test_fallbacks_roll_only_when_bold():
    for category, opt in FALLBACK_OPTIONS.items():
        assert (opt.skill is not None) == (category in ROLLING_CATEGORIES)


# --- spells for classes that cannot cast -----------------------------------------------

@pytest.mark.parametrize("text", ["Cast a detection spell", "Conjure a flame", "Chant the incantation",
                                  "Hurl a fireball", "Enchant the door"])
def test_spell_words_are_detected(text):
    assert is_spell_option(text)


@pytest.mark.parametrize("text", ["Search the castle wall", "Pick the lock",
                                  "Examine the runes", "Sneak past the guard"])
def test_ordinary_actions_are_not_spells(text):
    assert not is_spell_option(text)


def test_rogue_and_fighter_lose_spell_options_but_casters_keep_them():
    opts = [ActionOption("Examine the runes", None, "investigative"),
            ActionOption("Cast a detection spell", None, "resourceful")]
    for cls in ("Rogue", "fighter", " Fighter "):
        out = filter_unusable_options(opts, cls)
        assert out[0].text == "Examine the runes"
        assert out[1] == ActionOption("Try something clever", None, "resourceful")
    for cls in ("Wizard", "Cleric", "Bard", "Paladin", "Ranger", None, "", "Warlock"):
        assert filter_unusable_options(opts, cls)[1].text == "Cast a detection spell"


def test_filter_does_not_touch_the_original_options_or_fallback_table():
    opts = [ActionOption("Cast a spell", None, "bold")]
    out = filter_unusable_options(opts, "Rogue")
    out[0].text = "changed"
    assert opts[0].text == "Cast a spell" and FALLBACK_OPTIONS["bold"].text == "Act boldly"


def test_unknown_category_falls_back_to_the_story_option():
    out = filter_unusable_options([ActionOption("Cast a spell", None, None)], "Rogue")
    assert out[0].text == FALLBACK_OPTIONS[STORY].text


# --- not repeating the player's last action -------------------------------------------

def test_options_that_repeat_the_last_action_are_replaced():
    opts = [ActionOption("Ration the stale bread", "survival", "resourceful"),
            ActionOption("Examine the runes", None, "investigative")]
    out = avoid_repeating(opts, "Ration the stale bread for later")
    assert out[0] == ActionOption("Try something clever", None, "resourceful")
    assert out[1].text == "Examine the runes"


def test_exact_menu_text_counts_as_a_repeat():
    out = avoid_repeating([ActionOption("Act boldly", "athletics", "bold")], "Act boldly")
    assert out[0].text == "Act boldly"        # the fallback repeats too, so the original is kept


def test_no_last_action_changes_nothing():
    opts = [ActionOption("Examine the runes", None, "investigative")]
    assert avoid_repeating(opts, "") == opts and avoid_repeating(opts, "   ") == opts
