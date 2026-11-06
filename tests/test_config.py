from pathlib import Path

import pytest

from aether.config import (
    ROLES,
    Config,
    ConfigError,
    load_config,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def write(tmp_path, text: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults_are_valid():
    assert Config().validate() == []


def test_missing_default_file_uses_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("AETHER_CONFIG", raising=False)
    monkeypatch.setattr("aether.config.DEFAULT_CONFIG_PATH", tmp_path / "does_not_exist.toml")
    assert load_config() == Config()


def test_explicit_missing_file_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml")


def test_env_var_selects_file(tmp_path, monkeypatch):
    path = write(tmp_path, '[models]\ndefault = "qwen3:4b"\n')
    monkeypatch.setenv("AETHER_CONFIG", str(path))
    assert load_config().models.default == "qwen3:4b"


def test_example_file_matches_defaults():
    """config.example.toml must never drift from the real defaults."""
    assert load_config(PROJECT_ROOT / "config.example.toml") == Config()


def test_role_falls_back_to_default_model(tmp_path):
    cfg = load_config(write(tmp_path, '[models]\ndefault = "a"\njudge = "b"\n'))
    assert cfg.models.for_role("dm") == "a"
    assert cfg.models.for_role("judge") == "b"


def test_every_role_resolves():
    cfg = Config()
    for role in ROLES:
        assert cfg.models.for_role(role)
        assert isinstance(cfg.temperature.for_role(role), float)


def test_unknown_role_rejected():
    with pytest.raises(ValueError):
        Config().models.for_role("wizard")


def test_partial_override_keeps_other_defaults(tmp_path):
    cfg = load_config(write(tmp_path, "[memory]\nkeep_recent_messages = 8\n"))
    assert cfg.memory.keep_recent_messages == 8
    assert cfg.memory.summary_max_tokens == Config().memory.summary_max_tokens


def test_integer_accepted_for_float_setting(tmp_path):
    cfg = load_config(write(tmp_path, "[temperature]\ndm = 1\n"))
    assert cfg.temperature.dm == 1.0
    assert isinstance(cfg.temperature.dm, float)


def test_unknown_key_rejected(tmp_path):
    with pytest.raises(ConfigError, match="unknown key.*num_ctxx"):
        load_config(write(tmp_path, "[ollama]\nnum_ctxx = 4096\n"))


def test_unknown_section_rejected(tmp_path):
    with pytest.raises(ConfigError, match="unknown section"):
        load_config(write(tmp_path, "[oops]\nx = 1\n"))


def test_wrong_type_rejected(tmp_path):
    with pytest.raises(ConfigError, match="num_ctx must be int"):
        load_config(write(tmp_path, '[ollama]\nnum_ctx = "big"\n'))


def test_bool_rejected_for_number(tmp_path):
    with pytest.raises(ConfigError, match="num_ctx must be int"):
        load_config(write(tmp_path, "[ollama]\nnum_ctx = true\n"))


def test_invalid_toml_syntax(tmp_path):
    with pytest.raises(ConfigError, match="not valid TOML"):
        load_config(write(tmp_path, "[ollama\nnum_ctx = 1"))


def test_all_problems_reported_together(tmp_path):
    text = "[rag]\ntop_k = 0\n[gameplay]\nnum_options = 9\n"
    with pytest.raises(ConfigError) as exc:
        load_config(write(tmp_path, text))
    message = str(exc.value)
    assert "top_k" in message and "num_options" in message


def test_keep_recent_must_be_below_message_cap():
    cfg = Config()
    cfg.memory.keep_recent_messages = 30
    assert any("keep_recent_messages" in p for p in cfg.validate())


def test_history_must_fit_in_context():
    cfg = Config()
    cfg.ollama.num_ctx = 4096  # too small for the default 3000-token history budget
    assert any("num_ctx" in p for p in cfg.validate())


def test_unlimited_narration_skips_context_check():
    cfg = Config()
    cfg.ollama.num_ctx = 4096
    cfg.ollama.narration_max_tokens = 0
    assert cfg.validate() == []


def test_path_helpers():
    cfg = Config()
    assert cfg.paths.campaigns_dir == Path("data") / "campaigns"
    assert cfg.paths.vector_db_dir == Path("data") / "vector_db"
