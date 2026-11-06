import pytest

from stats import (
    ABILITIES,
    CLASSES,
    DIFFICULTY_DC,
    RACES,
    SKILL_ABILITIES,
    STAT_PRIORITIES,
    ability_for_skill,
    ability_modifier,
    calculate_character_stats,
    format_roll,
    normalize_skill,
    roll_check,
    skill_modifier,
)


class FakeRng:
    def __init__(self, value):
        self.value = value

    def randint(self, a, b):
        assert (a, b) == (1, 20)
        return self.value


def test_modifier_formula():
    assert [ability_modifier(s) for s in (1, 8, 9, 10, 11, 12, 15, 20, 30)] == [-5, -1, -1, 0, 0, 1, 2, 5, 10]


def test_every_priority_list_covers_all_abilities():
    assert set(STAT_PRIORITIES) == set(CLASSES)
    for order in STAT_PRIORITIES.values():
        assert sorted(order) == sorted(ABILITIES)


def test_every_race_and_class_combination_is_valid():
    for race in RACES:
        for cls in CLASSES:
            s = calculate_character_stats(race, cls)
            assert set(s.abilities) == set(ABILITIES)
            assert all(1 <= v <= 30 for v in s.abilities.values())
            assert s.hp >= 1


def test_human_wizard():
    s = calculate_character_stats("Human", "Wizard")
    assert s.abilities == {
        "intelligence": 16, "dexterity": 15, "constitution": 14,
        "wisdom": 13, "charisma": 11, "strength": 9,
    }
    assert s.hp == 6 + 2


def test_elf_ranger_gets_dexterity_bonus_and_no_longer_hardcoded():
    s = calculate_character_stats("Elf", "Ranger")
    assert s.abilities["dexterity"] == 17
    assert s.abilities["wisdom"] == 14
    assert s.hp == 10 + 1


def test_dragonborn_fighter():
    s = calculate_character_stats("Dragonborn", "Fighter")
    assert s.abilities["strength"] == 17
    assert s.abilities["charisma"] == 11
    assert s.hp == 10 + 2


def test_tiefling_bard():
    s = calculate_character_stats("Tiefling", "Bard")
    assert s.abilities["charisma"] == 17
    assert s.abilities["intelligence"] == 11


def test_race_and_class_are_case_insensitive():
    assert calculate_character_stats("elf", "RANGER") == calculate_character_stats("Elf", "Ranger")


def test_unknown_race_or_class_fails_loudly():
    with pytest.raises(ValueError, match="race"):
        calculate_character_stats("Orc", "Fighter")
    with pytest.raises(ValueError, match="class"):
        calculate_character_stats("Elf", "Necromancer")


def test_normalize_skill_variants():
    assert normalize_skill("Sleight_of_Hand") == "sleight of hand"
    assert normalize_skill("sleight-of-hand") == "sleight of hand"
    assert normalize_skill("  STEALTH ") == "stealth"
    assert normalize_skill("DEX") == "dexterity"
    assert normalize_skill("wisdom") == "wisdom"
    assert normalize_skill("flying") is None
    assert normalize_skill("") is None
    assert normalize_skill(None) is None


def test_ability_for_skill():
    assert ability_for_skill("athletics") == "strength"
    assert ability_for_skill("Animal_Handling") == "wisdom"
    assert ability_for_skill("charisma") == "charisma"
    with pytest.raises(ValueError):
        ability_for_skill("flying")


def test_all_skills_map_to_real_abilities():
    assert set(SKILL_ABILITIES.values()) <= set(ABILITIES)
    assert len(SKILL_ABILITIES) == 18


def test_skill_modifier():
    abilities = {a: 10 for a in ABILITIES}
    abilities["dexterity"] = 16
    assert skill_modifier(abilities, "stealth") == 3
    assert skill_modifier(abilities, "athletics") == 0


def test_roll_success_and_failure():
    abilities = {a: 10 for a in ABILITIES}
    abilities["dexterity"] = 16  # +3
    hit = roll_check(abilities, "stealth", "medium", FakeRng(10))
    assert (hit.die, hit.modifier, hit.total, hit.dc, hit.success) == (10, 3, 13, 13, True)
    miss = roll_check(abilities, "stealth", "medium", FakeRng(9))
    assert miss.success is False


def test_difficulty_changes_dc():
    abilities = {a: 10 for a in ABILITIES}
    assert roll_check(abilities, "athletics", "easy", FakeRng(10)).success is True
    assert roll_check(abilities, "athletics", "hard", FakeRng(10)).success is False
    assert roll_check(abilities, "athletics", "hard", FakeRng(10)).dc == DIFFICULTY_DC["hard"]


def test_roll_rejects_bad_input():
    abilities = {a: 10 for a in ABILITIES}
    with pytest.raises(ValueError):
        roll_check(abilities, "flying")
    with pytest.raises(ValueError):
        roll_check(abilities, "stealth", "impossible")


def test_roll_without_rng_stays_in_range():
    abilities = {a: 10 for a in ABILITIES}
    for _ in range(50):
        assert 1 <= roll_check(abilities, "stealth").die <= 20


def test_format_roll():
    abilities = {a: 10 for a in ABILITIES}
    abilities["dexterity"] = 8  # -1
    text = format_roll(roll_check(abilities, "sleight_of_hand", "medium", FakeRng(14)))
    assert text == "Sleight Of Hand check (DEX): d20 14 - 1 = 13 vs DC 13 - Success"
