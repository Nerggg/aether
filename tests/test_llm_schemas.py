import pytest
from pydantic import ValidationError

from aether.llm.schemas import BeatsBlueprint, MemoryFacts, JudgeVerdict, OptionsResponse


def test_options_response_only_needs_options():
    r = OptionsResponse.model_validate_json('{"options":[{"text":"Hide"}]}')
    assert r.options[0].skill is None
    assert r.new_npcs == [] and r.beat_achieved is False


def test_new_npc_defaults():
    r = OptionsResponse.model_validate_json('{"options":[],"new_npcs":[{"name":"Tulle"}]}')
    assert r.new_npcs[0].type == "ally"
    assert r.new_npcs[0].is_persistent is False


def test_options_response_rejects_missing_options():
    with pytest.raises(ValidationError):
        OptionsResponse.model_validate_json("{}")


def test_judge_verdict_defaults_and_validation():
    v = JudgeVerdict.model_validate_json('{"verdict":"impossible","reason":"No bow."}')
    assert v.difficulty == "medium" and v.advances_beat is False
    with pytest.raises(ValidationError):
        JudgeVerdict.model_validate_json('{"verdict":"maybe"}')
    with pytest.raises(ValidationError):
        JudgeVerdict.model_validate_json('{"verdict":"risky","difficulty":"extreme"}')


def test_memory_facts_all_optional():
    r = MemoryFacts.model_validate_json("{}")
    assert r.open_threads == [] and r.promises_debts == []


def test_beats_condition_defaults_to_referee_and_accepts_any_string():
    r = BeatsBlueprint.model_validate_json('{"beats":[{"title":"A","objective":"b"}]}')
    assert r.beats[0].condition_type == "referee"
    r = BeatsBlueprint.model_validate_json('{"beats":[{"title":"A","objective":"b","condition_type":"has_item"}]}')
    assert r.beats[0].condition_type == "has_item"   # calls.py downgrades it


def test_option_category_is_optional():
    r = OptionsResponse.model_validate_json('{"options":[{"text":"Hide","category":"cautious"},{"text":"Run"}]}')
    assert [o.category for o in r.options] == ["cautious", None]


def test_schemas_export_json_schema():
    for model in (OptionsResponse, JudgeVerdict, MemoryFacts, BeatsBlueprint):
        assert "properties" in model.model_json_schema()
