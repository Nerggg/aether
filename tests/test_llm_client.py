from types import SimpleNamespace

import pytest

from aether.config import Config
from aether.llm.client import LLMClient, LLMError, model_installed, strip_json_fence
from aether.llm.schemas import JudgeVerdict
from fakes import FakeBackend


MSGS = [{"role": "user", "content": "hi"}]
GOOD = '{"verdict": "feasible"}'


def make(*responses, config=None, **kw):
    backend = FakeBackend(*responses, **kw)
    return LLMClient(config or Config(), backend), backend


# --- request building ----------------------------------------------------------------------

def test_every_call_sends_context_window_and_keep_alive():
    client, backend = make("ok")
    client.chat("dm", MSGS)
    call = backend.calls[0]
    assert call["options"]["num_ctx"] == 6144
    assert call["keep_alive"] == "30m"
    assert call["model"] == "llama3.2"
    assert call["stream"] is False
    assert "format" not in call


def test_dm_is_capped_by_narration_limit_and_uses_its_temperature():
    client, backend = make("ok")
    client.chat("dm", MSGS)
    assert backend.calls[0]["options"]["num_predict"] == 320
    assert backend.calls[0]["options"]["temperature"] == 0.7


def test_other_roles_have_no_default_cap():
    client, backend = make("ok")
    client.chat("judge", MSGS)
    assert "num_predict" not in backend.calls[0]["options"]
    assert backend.calls[0]["options"]["temperature"] == 0.0


def test_unlimited_narration_sends_no_cap():
    cfg = Config()
    cfg.ollama.narration_max_tokens = 0
    client, backend = make("ok", config=cfg)
    client.chat("dm", MSGS)
    assert "num_predict" not in backend.calls[0]["options"]


def test_explicit_limits_override_defaults():
    client, backend = make("ok")
    client.chat("dm", MSGS, max_tokens=50, temperature=1.2)
    assert backend.calls[0]["options"]["num_predict"] == 50
    assert backend.calls[0]["options"]["temperature"] == 1.2


def test_zero_temperature_is_respected_not_treated_as_missing():
    client, backend = make("ok")
    client.chat("dm", MSGS, temperature=0.0)
    assert backend.calls[0]["options"]["temperature"] == 0.0


def test_role_can_use_its_own_model():
    cfg = Config()
    cfg.models.judge = "qwen3:4b"
    client, backend = make("ok", config=cfg)
    client.chat("judge", MSGS)
    client.chat("dm", MSGS)
    assert [c["model"] for c in backend.calls] == ["qwen3:4b", "llama3.2"]


def test_unknown_role_rejected():
    client, _ = make("ok")
    with pytest.raises(ValueError):
        client.chat("wizard", MSGS)


def test_chat_strips_whitespace():
    client, _ = make("  hello \n")
    assert client.chat("dm", MSGS) == "hello"


def test_object_style_responses_are_understood():
    from aether.llm.client import _content
    assert _content(SimpleNamespace(message=SimpleNamespace(content="hi"))) == "hi"
    assert _content(SimpleNamespace(message=SimpleNamespace(content=None))) == ""
    assert _content({}) == ""


# --- streaming -------------------------------------------------------------------------------

def test_stream_yields_pieces_and_skips_empty_ones():
    client, backend = make(["Hel", "", "lo"])
    assert list(client.stream("dm", MSGS)) == ["Hel", "lo"]
    assert backend.calls[0]["stream"] is True


def test_stream_failure_becomes_llm_error():
    client, _ = make(ConnectionError("refused"))
    with pytest.raises(LLMError, match="dm"):
        list(client.stream("dm", MSGS))


def test_chat_failure_names_host_and_model():
    client, _ = make(ConnectionError("refused"))
    with pytest.raises(LLMError, match="localhost:11434.*llama3.2"):
        client.chat("dm", MSGS)


# --- structured output -----------------------------------------------------------------------

def test_structured_sends_schema_and_returns_model():
    client, backend = make(GOOD)
    result = client.structured("judge", MSGS, JudgeVerdict)
    assert result.verdict == "feasible"
    assert backend.calls[0]["format"] == JudgeVerdict.model_json_schema()


def test_structured_accepts_fenced_json():
    client, _ = make('```json\n{"verdict": "risky", "skill": "stealth"}\n```')
    assert client.structured("judge", MSGS, JudgeVerdict).skill == "stealth"


def test_strip_json_fence():
    assert strip_json_fence('```\n{"a":1}\n```') == '{"a":1}'
    assert strip_json_fence('  {"a":1} ') == '{"a":1}'


def test_structured_retries_once_with_warmer_temperature():
    client, backend = make("not json", GOOD)
    assert client.structured("judge", MSGS, JudgeVerdict).verdict == "feasible"
    assert len(backend.calls) == 2
    assert backend.calls[0]["options"]["temperature"] == 0.0
    assert backend.calls[1]["options"]["temperature"] == 0.3


def test_structured_gives_up_and_explains():
    client, backend = make('{"verdict": "maybe"}')
    assert client.structured("judge", MSGS, JudgeVerdict) is None
    assert len(backend.calls) == 2
    assert "JudgeVerdict" in client.last_structured_error


def test_structured_error_is_cleared_on_success():
    client, _ = make("bad", GOOD, GOOD)
    client.structured("judge", MSGS, JudgeVerdict, retries=0)
    assert client.last_structured_error
    client.structured("judge", MSGS, JudgeVerdict)
    assert client.last_structured_error is None


def test_structured_retries_zero_means_one_attempt():
    client, backend = make("bad")
    assert client.structured("judge", MSGS, JudgeVerdict, retries=0) is None
    assert len(backend.calls) == 1


def test_structured_connection_error_raises_without_retrying():
    client, backend = make(ConnectionError("down"))
    with pytest.raises(LLMError):
        client.structured("judge", MSGS, JudgeVerdict)
    assert len(backend.calls) == 1


# --- setup checks -------------------------------------------------------------------------------

def test_model_installed_matching():
    assert model_installed("llama3.2", ["llama3.2:latest"])
    assert model_installed("LLAMA3.2", ["llama3.2:latest"])
    assert model_installed("qwen3:4b", ["qwen3:4b"])
    assert not model_installed("qwen3:4b", ["qwen3:8b"])
    assert not model_installed("qwen3", ["qwen3:4b"])
    assert not model_installed("llama3.2", [])


def test_missing_models_lists_only_what_is_needed_and_absent():
    cfg = Config()
    cfg.models.judge = "qwen3:4b"
    client, _ = make("x", config=cfg, models=["llama3.2:latest"])
    assert client.missing_models() == ["qwen3:4b"]


def test_missing_models_none_missing():
    client, _ = make("x", models=["llama3.2:latest"])
    assert client.missing_models() == []


def test_missing_models_when_ollama_is_down():
    client, _ = make("x", models=ConnectionError("down"))
    with pytest.raises(LLMError, match="list"):
        client.missing_models()

class StatBackend:
    STATS = {"load_duration": 2_000_000_000, "prompt_eval_count": 100, "prompt_eval_duration": 1_000_000_000,
             "eval_count": 50, "eval_duration": 4_000_000_000}

    def chat(self, **kw):
        if kw["stream"]:
            return iter([{"message": {"content": "a"}}, {"message": {"content": "b"}, "done": True, **self.STATS}])
        return {"message": {"content": "hi"}, **self.STATS}


def test_timing_stats_are_recorded_for_chat_and_stream():
    client = LLMClient(Config(), StatBackend())
    client.chat("dm", MSGS)
    list(client.stream("dm", MSGS))
    assert len(client.call_log) == 2
    s = client.call_log[0]
    assert (s.role, s.load_s, s.prompt_tokens, s.gen_tokens) == ("dm", 2.0, 100, 50)
    assert s.gen_tps == 12.5 and s.prompt_tps == 100.0


def test_missing_timing_fields_record_nothing():
    client, _ = make("ok")
    client.chat("dm", MSGS)
    assert len(client.call_log) == 0
