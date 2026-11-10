import random

from aether.creation import names
from aether.creation.menus import CUSTOM_CHOICE, last_assistant_reply, parse_menu, resolve_choice

MENU = "Pick one.\nA) Human: versatile\nB) Elf: nimble\nC) Dwarf: hardy\nD) Something else / Player's custom input"


def test_parse_menu_and_resolve_choice():
    assert parse_menu(MENU) == {"A": "Human: versatile", "B": "Elf: nimble", "C": "Dwarf: hardy",
                                "D": "Something else / Player's custom input"}
    assert resolve_choice("b", MENU) == "Elf: nimble"
    assert resolve_choice(" C. ", MENU) == "Dwarf: hardy"
    assert resolve_choice("2", MENU) == "Elf: nimble"
    assert resolve_choice("d", MENU) == CUSTOM_CHOICE
    assert resolve_choice("e", MENU) is None
    assert resolve_choice("0", MENU) is None
    assert resolve_choice("a rogue named Bo", MENU) is None
    assert resolve_choice("a", "no menu here") is None


def test_a_menu_without_a_custom_option_treats_every_letter_as_real():
    menu = "A) One thing\nB) Two things\nC) Three things\nD) Four things"
    assert resolve_choice("d", menu) == "Four things"


def test_last_assistant_reply():
    history = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "x"},
               {"role": "user", "content": "b"}, {"role": "assistant", "content": "y"}]
    assert last_assistant_reply(history) == "y"
    assert last_assistant_reply([]) == ""


def test_name_lists_are_unique_single_words_and_varied():
    for pool in (names.FIRST_NAMES, names.LAST_NAMES):
        assert len(pool) >= 40
        assert len(set(pool)) == len(pool)
        assert all(" " not in n for n in pool)
        assert not any("kael" in n.lower() for n in pool)


def test_pick_patron_name_is_seedable_and_combines_both_lists():
    a = names.pick_patron_name(random.Random(3))
    assert a == names.pick_patron_name(random.Random(3))
    first, last = a.split(" ")
    assert first in names.FIRST_NAMES and last in names.LAST_NAMES
    assert len({names.pick_patron_name(random.Random(i)) for i in range(30)}) > 20


def test_pick_patron_name_respects_exclusions(monkeypatch):
    monkeypatch.setattr(names, "FIRST_NAMES", ("Ann", "Bo"))
    monkeypatch.setattr(names, "LAST_NAMES", ("Cole",))
    assert names.pick_patron_name(random.Random(1), exclude=["ann cole"]) == "Bo Cole"
    assert names.pick_patron_name(random.Random(1), exclude=["Ann Cole", "Bo Cole"]) in ("Ann Cole", "Bo Cole")
