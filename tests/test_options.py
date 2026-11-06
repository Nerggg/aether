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
