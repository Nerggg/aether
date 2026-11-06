from config import MemoryConfig
from memory import (
    CONTINUES_MARKER,
    MESSAGE_OVERHEAD_TOKENS,
    assemble_prompt,
    build_history_messages,
    estimate_tokens,
    format_transcript,
    message_tokens,
    should_compact,
    split_for_compaction,
    summary_to_text,
    total_tokens,
    trim_to_budget,
)
from models import Message, Summary


def msgs(roles, content="x", start_id=1):
    """Messages with the given roles; `content` can be a string or a list."""
    out = []
    for i, role in enumerate(roles):
        text = content[i] if isinstance(content, list) else content
        out.append(Message(id=start_id + i, turn_number=i // 2, role=role, content=text))
    return out


def alternating(n, content="x"):
    """dm, player, dm, player ... (a game that began with an opener)."""
    return msgs(["dm" if i % 2 == 0 else "player" for i in range(n)], content)


# --- tokens -----------------------------------------------------------------------------

def test_estimate_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2
    assert estimate_tokens("a" * 400) == 100


def test_message_and_total_tokens():
    m = msgs(["dm"], "a" * 40)[0]
    assert message_tokens(m) == 10 + MESSAGE_OVERHEAD_TOKENS
    assert total_tokens([m, m]) == 2 * (10 + MESSAGE_OVERHEAD_TOKENS)
    assert total_tokens([]) == 0


# --- summary text -----------------------------------------------------------------------

def test_no_summary_gives_empty_text():
    assert summary_to_text(None) == ""
    assert summary_to_text(Summary(covers_through_message_id=1, narrative="   ")) == ""


def test_summary_text_includes_labelled_fields():
    s = Summary(
        covers_through_message_id=5,
        narrative="The party reached the cave.",
        key_npcs=["Barnaby", " "],
        open_threads=["Find the child"],
        promises_debts=[],
    )
    text = summary_to_text(s)
    assert text.startswith("STORY SO FAR:\nThe party reached the cave.")
    assert "Key characters met: Barnaby" in text
    assert "Unresolved threads: Find the child" in text
    assert "Promises" not in text  # empty lists are omitted


# --- splitting for compaction ---------------------------------------------------------

def test_split_keeps_at_least_keep_recent_and_starts_on_a_player_message():
    messages = alternating(12)  # dm,player,... ids 1..12; player messages have even ids
    split = split_for_compaction(messages, 6)
    assert len(split.keep) >= 6
    assert split.keep[0].role == "player"
    assert split.to_compact + split.keep == messages
    assert split.covers_through_id == split.to_compact[-1].id


def test_split_moves_boundary_back_to_a_player_message():
    # index 12 - 6 = 6 is a 'dm' message, so the boundary slides back one to a player message
    messages = alternating(12)
    split = split_for_compaction(messages, 6)
    assert split.keep[0].role == "player"
    assert len(split.keep) == 7


def test_split_with_too_few_messages_compacts_nothing():
    messages = alternating(6)
    split = split_for_compaction(messages, 6)
    assert split.to_compact == []
    assert split.keep == messages
    assert split.covers_through_id is None


def test_split_with_no_player_message_to_anchor_compacts_nothing():
    messages = msgs(["dm"] * 10)
    assert split_for_compaction(messages, 4).to_compact == []


def test_split_keeps_dice_notes_with_their_player_message():
    roles = ["dm", "player", "system", "dm", "player", "system", "dm", "player", "dm", "player"]
    messages = msgs(roles)
    split = split_for_compaction(messages, 3)
    assert split.keep[0].role == "player"
    assert split.to_compact[-1].id == split.keep[0].id - 1


def test_split_never_drops_an_unanswered_player_action():
    messages = alternating(11)  # last message is a player action with no DM reply yet
    split = split_for_compaction(messages, 6)
    assert split.keep[-1] == messages[-1]


# --- when to compact ----------------------------------------------------------------------

def cfg(**kw):
    base = dict(history_budget_tokens=3000, compaction_trigger_ratio=0.7,
                max_uncompacted_messages=20, keep_recent_messages=6)
    base.update(kw)
    return MemoryConfig(**base)


def test_short_conversation_does_not_compact():
    assert should_compact(alternating(8, "short"), cfg()) is False


def test_big_messages_trigger_compaction():
    # 12 messages x ~254 tokens = ~3050 tokens > 0.7 * 3000
    assert should_compact(alternating(12, "a" * 1000), cfg()) is True


def test_message_cap_triggers_even_when_tiny():
    assert should_compact(alternating(20, "hi"), cfg()) is True
    assert should_compact(alternating(19, "hi"), cfg()) is False


def test_existing_summary_lowers_the_threshold():
    messages = alternating(12, "a" * 600)   # ~ 12 * 154 = 1848 tokens
    assert should_compact(messages, cfg()) is False   # 1848 < 0.7 * 3000 = 2100
    big_summary = Summary(covers_through_message_id=0, narrative="s" * 4000)  # ~1000 tokens
    assert should_compact(messages, cfg(), big_summary) is True  # 1848 > 0.7 * ~2000


def test_does_not_compact_a_tiny_slice():
    # Huge tail but only 2 messages fall outside the kept window
    messages = alternating(8, "a" * 3000)
    assert len(split_for_compaction(messages, 6).to_compact) < 4
    assert should_compact(messages, cfg()) is False


def test_nothing_to_compact_means_no_compaction():
    assert should_compact(msgs(["dm"] * 25), cfg()) is False


# --- transcript text ----------------------------------------------------------------------

def test_format_transcript():
    messages = msgs(["dm", "player", "system"], ["You enter.", "I look.", "Perception check: success"])
    assert format_transcript(messages, "Aria") == "DM: You enter.\nAria: I look.\n[Perception check: success]"


# --- building the chat -----------------------------------------------------------------

def test_history_maps_roles():
    chat = build_history_messages(alternating(4, "t"))
    assert [m["role"] for m in chat] == ["assistant", "user", "assistant", "user"]


def test_dice_note_is_folded_into_the_player_message():
    messages = msgs(["dm", "player", "system", "dm"], ["Opening.", "I sneak.", "Stealth check: success", "You slip by."])
    chat = build_history_messages(messages)
    assert [m["role"] for m in chat] == ["assistant", "user", "assistant"]
    assert chat[1]["content"] == "I sneak.\n[Stealth check: success]"


def test_note_with_no_player_message_before_it_becomes_a_user_message():
    chat = build_history_messages(msgs(["dm", "system"], ["Opening.", "Event"]))
    assert chat[-1] == {"role": "user", "content": "[Event]"}


def test_consecutive_same_role_messages_are_merged():
    chat = build_history_messages(msgs(["player", "player", "dm"], ["a", "b", "c"]))
    assert chat[0] == {"role": "user", "content": "a\n\nb"}
    assert len(chat) == 2


def test_history_of_nothing_is_empty():
    assert build_history_messages([]) == []


def test_trim_to_budget_drops_oldest_first():
    messages = alternating(6, "a" * 400)  # ~104 tokens each
    kept = trim_to_budget(messages, 300)   # two messages fit (208), three do not (312)
    assert kept == messages[-2:]


def test_trim_never_goes_below_min_keep():
    messages = alternating(6, "a" * 4000)
    assert len(trim_to_budget(messages, 10)) == 2
    assert len(trim_to_budget(messages, 10, min_keep=3)) == 3


def test_trim_leaves_a_fitting_tail_alone():
    messages = alternating(4, "hi")
    assert trim_to_budget(messages, 1000) == messages


def test_assemble_without_summary():
    chat = assemble_prompt("You are the DM.", None, alternating(2, "t"), ["SHEET", "", "LOCATION"])
    assert chat[0]["role"] == "system"
    assert chat[0]["content"] == "You are the DM.\n\nSHEET\n\nLOCATION"
    assert len(chat) == 3


def test_assemble_with_summary_and_tail_adds_marker():
    s = Summary(covers_through_message_id=1, narrative="Long ago.")
    chat = assemble_prompt("SYS", s, alternating(2, "t"))
    system = chat[0]["content"]
    assert "STORY SO FAR:\nLong ago." in system
    assert system.endswith(CONTINUES_MARKER)


def test_assemble_with_summary_but_no_tail_has_no_marker():
    s = Summary(covers_through_message_id=1, narrative="Long ago.")
    chat = assemble_prompt("SYS", s, [])
    assert CONTINUES_MARKER not in chat[0]["content"]
    assert len(chat) == 1
