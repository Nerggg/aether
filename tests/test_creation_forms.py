from aether.creation.forms import CUSTOM, SURPRISE, Reply, ask_menu


def run(answers, options=("A", "B", "C"), extras=(), **kw):
    it, out = iter(answers), []
    reply = ask_menu(lambda prompt="": next(it), lambda *a, **k: out.append(" ".join(map(str, a))),
                     "Pick:", options, extras, **kw)
    return reply, out


EXTRAS = [(CUSTOM, "Own"), (SURPRISE, "Surprise me")]


def test_numbers_pick_options_and_extras():
    assert run(["2"], extras=EXTRAS)[0] == Reply("option", 1)
    assert run([" (4). "], extras=EXTRAS)[0] == Reply(CUSTOM)
    assert run(["5"], extras=EXTRAS)[0] == Reply(SURPRISE)


def test_words_and_bad_numbers_show_the_list_again():
    reply, out = run(["another one", "0", "9", "1"], extras=EXTRAS)
    assert reply == Reply("option", 0)
    assert sum(line == "\nPick:" for line in out) == 4
    assert sum("Please pick a number from 1 to 5" in line for line in out) == 3


def test_closed_menus_accept_a_name_and_show_lines():
    reply, out = run(["i'd go with paladin"], options=("Fighter", "Paladin"),
                     lines=["Fighter: tough", "Paladin: holy"], closed=True)
    assert reply == Reply("option", 1)
    assert "  2. Paladin: holy" in out


def test_shortcut_runs_before_the_hint():
    reply, _ = run(["named Baron"], shortcut=lambda r: Reply("text", text=r) if "named" in r else None)
    assert reply == Reply("text", text="named Baron")
