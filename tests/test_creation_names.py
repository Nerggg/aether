import random

from aether.creation import names


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
