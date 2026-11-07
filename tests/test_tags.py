import pytest

from aether.logic.tags import MAX_TAG_LENGTH, Tag, TagParser, parse_amount, parse_tags

def test_placeholder_values_are_swallowed():
    text, tags = parse_tags("Fine. [ADD_ITEM: None][TAKE_DAMAGE: N/A][HEAL_HP: null] Done.")
    assert tags == []
    assert text == "Fine.  Done."

def stream(text, size):
    """Feed text in chunks of `size` and return (shown text, tags)."""
    parser = TagParser()
    shown, tags = [], []
    for i in range(0, len(text), size):
        r = parser.feed(text[i:i + size])
        shown.append(r.text)
        tags.extend(r.tags)
    last = parser.flush()
    shown.append(last.text)
    return "".join(shown), tags


def test_plain_text_passes_through():
    assert parse_tags("The door creaks open.") == ("The door creaks open.", [])


def test_single_tag_is_removed():
    text, tags = parse_tags("You are hit. [TAKE_DAMAGE: 3]")
    assert text == "You are hit."
    assert tags == [Tag("TAKE_DAMAGE", "3")]


def test_multiple_tags_in_order():
    text, tags = parse_tags("You find a rope. [ADD_ITEM: Rope] A trap! [TAKE_DAMAGE: 2][REMOVE_ITEM: Torch]")
    assert text == "You find a rope.  A trap!"
    assert tags == [Tag("ADD_ITEM", "Rope"), Tag("TAKE_DAMAGE", "2"), Tag("REMOVE_ITEM", "Torch")]


def test_tag_names_are_case_insensitive_and_spacing_tolerant():
    _, tags = parse_tags("[ heal_hp :  4 ]")
    assert tags == [Tag("HEAL_HP", "4")]


def test_value_keeps_inner_spaces():
    _, tags = parse_tags("[ADD_ITEM: Silver Key of the Vault]")
    assert tags == [Tag("ADD_ITEM", "Silver Key of the Vault")]


def test_stray_bracket_does_not_swallow_text():
    # The old interceptor would have eaten everything up to the "]".
    text, tags = parse_tags("He muttered [something odd and then walked away, but the road [went on] forever.")
    assert text == "He muttered [something odd and then walked away, but the road [went on] forever."
    assert tags == []


def test_bracketed_non_tag_text_survives():
    text, tags = parse_tags("A sign reads [KEEP OUT] in red paint.")
    assert text == "A sign reads [KEEP OUT] in red paint."
    assert tags == []


def test_unknown_tag_name_stays_as_text():
    text, tags = parse_tags("[ROLL: stealth] You sneak.")
    assert text == "[ROLL: stealth] You sneak."
    assert tags == []


def test_empty_value_is_not_a_tag():
    text, tags = parse_tags("Hmm [ADD_ITEM: ] ok")
    assert tags == []
    assert text == "Hmm [ADD_ITEM: ] ok"


def test_bracket_restarts_inside_unfinished_tag():
    text, tags = parse_tags("[ADD_ITEM: broken [ADD_ITEM: Rope]")
    assert tags == [Tag("ADD_ITEM", "Rope")]
    assert text == "[ADD_ITEM: broken"


def test_unclosed_tag_is_released_on_flush():
    text, tags = parse_tags("You see [ADD_ITEM: Rop")
    assert text == "You see [ADD_ITEM: Rop"
    assert tags == []


def test_lone_bracket_at_end_is_released():
    assert parse_tags("Maybe [") == ("Maybe [", [])


def test_overlong_candidate_is_released_as_text():
    long_value = "x" * (MAX_TAG_LENGTH + 20)
    text, tags = parse_tags(f"[ADD_ITEM: {long_value}]")
    assert tags == []
    assert text == f"[ADD_ITEM: {long_value}]"


def test_newline_ends_a_candidate_tag():
    text, tags = parse_tags("[ADD_ITEM: Rope\nand more]")
    assert tags == []
    assert text == "[ADD_ITEM: Rope\nand more]"


def test_custom_tag_names():
    text, tags = parse_tags("[ROLL: stealth] go", names=("ROLL",))
    assert tags == [Tag("ROLL", "stealth")]
    assert text == " go"


SAMPLES = [
    "The rope snaps. [TAKE_DAMAGE: 4] You land hard.",
    "Look [here] and [ADD_ITEM: Lantern] then [there",
    "[HEAL_HP: 2][HEAL_HP: 3]",
    "No tags at all, just a [bracket.",
    "Odd [A very long note that is not a tag but keeps going and going for a while yet] end.",
    "[ADD_ITEM: Key] at the start and [REMOVE_ITEM: Key]",
]


@pytest.mark.parametrize("sample", SAMPLES)
@pytest.mark.parametrize("size", [1, 2, 3, 5, 7, 11, 1000])
def test_streaming_matches_whole_text_for_any_chunking(sample, size):
    expected_text, expected_tags = parse_tags(sample)
    # parse_tags strips trailing whitespace; compare on that basis
    shown, tags = stream(sample, size)
    assert shown.rstrip() == expected_text
    assert tags == expected_tags


def test_text_is_not_held_back_unnecessarily():
    parser = TagParser()
    assert parser.feed("Hello, ").text == "Hello, "
    assert parser.feed("[Z").text == "[Z"      # cannot be a tag, released at once
    assert parser.feed("[A").text == ""        # could still become ADD_ITEM


def test_parse_amount():
    assert parse_amount("3") == 3
    assert parse_amount("3 damage") == 3
    assert parse_amount("-2") == -2
    assert parse_amount("a lot") is None
    assert parse_amount("") is None
