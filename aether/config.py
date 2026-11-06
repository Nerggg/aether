"""Central configuration for Aether.

Every tunable value lives here. Settings are resolved in this order:

    built-in defaults  <  config.toml  (optional)

Copy ``config.example.toml`` to ``config.toml`` and edit only the values you
want to change. Set the AETHER_CONFIG environment variable to load a config
file from somewhere else.

Usage:
    from aether.config import get_config
    cfg = get_config()
    model = cfg.models.for_role("dm")
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, get_type_hints

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None

CONFIG_ENV_VAR = "AETHER_CONFIG"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.toml"

# Every LLM job in the game. Each can use its own model and temperature.
ROLES = ("dm", "options", "judge", "referee", "compaction", "creation")


class ConfigError(Exception):
    """Raised when the config file is missing, malformed, or has invalid values."""


@dataclass
class ModelsConfig:
    """Ollama model names. Empty role entries fall back to `default`."""

    default: str = "llama3.2"
    dm: str = ""          # narration and opening scene
    options: str = ""     # action menu generation
    judge: str = ""       # custom-action feasibility check
    referee: str = ""     # structured extraction (new NPCs, beat progress, metadata)
    compaction: str = ""  # summarising old history
    creation: str = ""    # conversational guide in campaign/character creation

    def for_role(self, role: str) -> str:
        if role not in ROLES:
            raise ValueError(f"Unknown role '{role}'. Expected one of {ROLES}")
        return getattr(self, role) or self.default


@dataclass
class TemperatureConfig:
    """Sampling temperature per role. Structured roles should stay near 0."""

    dm: float = 0.7
    options: float = 0.5
    judge: float = 0.0
    referee: float = 0.0
    compaction: float = 0.3
    creation: float = 0.7

    def for_role(self, role: str) -> float:
        if role not in ROLES:
            raise ValueError(f"Unknown role '{role}'. Expected one of {ROLES}")
        return getattr(self, role)


@dataclass
class OllamaConfig:
    host: str = "http://localhost:11434"
    num_ctx: int = 6144               # context window passed on every call
    keep_alive: str = "30m"           # how long Ollama keeps the model loaded
    narration_max_tokens: int = 500   # cap on one DM reply; 0 = unlimited


@dataclass
class EmbeddingConfig:
    model_name: str = "BAAI/bge-small-en-v1.5"
    # Prefix added to search queries (not documents). It is model-specific:
    # bge models want this one, other models want a different one or none.
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    document_prefix: str = ""
    device: str = "cpu"               # "cpu", "cuda", "mps"; keeps VRAM free for Ollama


@dataclass
class RagConfig:
    enabled: bool = True
    top_k: int = 2                    # transcript exchanges injected per turn
    max_distance: float = 0.45        # discard matches farther than this (tuned in the RAG stage)


@dataclass
class MemoryConfig:
    history_budget_tokens: int = 3000       # room for summary + raw recent messages
    compaction_trigger_ratio: float = 0.7   # compact when raw messages exceed this share of the budget
    max_uncompacted_messages: int = 20      # safety net regardless of token count
    keep_recent_messages: int = 6           # kept verbatim after a compaction
    summary_max_tokens: int = 400
    resume_display_messages: int = 6        # messages shown when continuing a campaign


@dataclass
class GameplayConfig:
    num_options: int = 3                    # generated options (a "custom action" slot is always added)
    max_option_words: int = 8
    max_consecutive_rejections: int = 2     # custom actions rejected before returning to the menu
    drift_threshold_turns: int = 6          # turns without beat progress before pressure events
    story_beats_min: int = 4
    story_beats_max: int = 6


@dataclass
class PathsConfig:
    data_dir: str = "data"

    @property
    def campaigns_dir(self) -> Path:
        return Path(self.data_dir) / "campaigns"

    @property
    def vector_db_dir(self) -> Path:
        return Path(self.data_dir) / "vector_db"


SECTIONS = {
    "models": ModelsConfig,
    "temperature": TemperatureConfig,
    "ollama": OllamaConfig,
    "embedding": EmbeddingConfig,
    "rag": RagConfig,
    "memory": MemoryConfig,
    "gameplay": GameplayConfig,
    "paths": PathsConfig,
}


@dataclass
class Config:
    models: ModelsConfig = field(default_factory=ModelsConfig)
    temperature: TemperatureConfig = field(default_factory=TemperatureConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    rag: RagConfig = field(default_factory=RagConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    gameplay: GameplayConfig = field(default_factory=GameplayConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    def validate(self) -> List[str]:
        """Return a list of human-readable problems (empty if the config is valid)."""
        problems: List[str] = []

        def need(condition: bool, message: str) -> None:
            if not condition:
                problems.append(message)

        need(bool(self.models.default.strip()), "[models] default must not be empty")

        for role in ROLES:
            t = self.temperature.for_role(role)
            need(0.0 <= t <= 2.0, f"[temperature] {role} must be between 0 and 2 (got {t})")

        o = self.ollama
        need(bool(o.host.strip()), "[ollama] host must not be empty")
        need(o.num_ctx >= 1024, f"[ollama] num_ctx must be at least 1024 (got {o.num_ctx})")
        need(o.narration_max_tokens >= 0, "[ollama] narration_max_tokens must be 0 or more")

        e = self.embedding
        need(bool(e.model_name.strip()), "[embedding] model_name must not be empty")
        need(bool(e.device.strip()), "[embedding] device must not be empty")

        r = self.rag
        need(r.top_k >= 1, "[rag] top_k must be at least 1")
        need(r.max_distance > 0, "[rag] max_distance must be greater than 0")

        m = self.memory
        need(m.history_budget_tokens >= 500, "[memory] history_budget_tokens must be at least 500")
        need(
            0.1 <= m.compaction_trigger_ratio <= 1.0,
            f"[memory] compaction_trigger_ratio must be between 0.1 and 1.0 (got {m.compaction_trigger_ratio})",
        )
        need(m.keep_recent_messages >= 2, "[memory] keep_recent_messages must be at least 2")
        need(
            m.max_uncompacted_messages > m.keep_recent_messages,
            "[memory] max_uncompacted_messages must be greater than keep_recent_messages",
        )
        need(m.summary_max_tokens >= 100, "[memory] summary_max_tokens must be at least 100")
        need(m.resume_display_messages >= 1, "[memory] resume_display_messages must be at least 1")

        g = self.gameplay
        need(2 <= g.num_options <= 5, f"[gameplay] num_options must be between 2 and 5 (got {g.num_options})")
        need(g.max_option_words >= 3, "[gameplay] max_option_words must be at least 3")
        need(g.max_consecutive_rejections >= 1, "[gameplay] max_consecutive_rejections must be at least 1")
        need(g.drift_threshold_turns >= 1, "[gameplay] drift_threshold_turns must be at least 1")
        need(g.story_beats_min >= 2, "[gameplay] story_beats_min must be at least 2")
        need(g.story_beats_max >= g.story_beats_min, "[gameplay] story_beats_max must be >= story_beats_min")

        need(bool(self.paths.data_dir.strip()), "[paths] data_dir must not be empty")

        # Cross-section check: the history must fit in the context window with
        # room left over for the system prompt, sheet and RAG chunks.
        # (narration_max_tokens = 0 means unlimited, so it is skipped.)
        if o.narration_max_tokens > 0:
            reserved = m.history_budget_tokens + m.summary_max_tokens + o.narration_max_tokens
            need(
                reserved <= o.num_ctx - 1000,
                f"history_budget_tokens + summary_max_tokens + narration_max_tokens = {reserved}, "
                f"which leaves under 1000 tokens of num_ctx ({o.num_ctx}) for the system prompt. "
                "Raise num_ctx or lower one of those values.",
            )

        return problems


def _build_section(cls, data: Dict[str, Any], section: str):
    """Build one section dataclass from a dict, collecting every error found."""
    errors: List[str] = []
    hints = get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}

    unknown = sorted(set(data) - known)
    if unknown:
        errors.append(f"[{section}] unknown key(s): {', '.join(unknown)}")

    kwargs: Dict[str, Any] = {}
    for name, value in data.items():
        if name not in known:
            continue
        expected = hints[name]
        # TOML writes 1 for a float setting; accept it. Reject bools for numbers.
        if expected is float and isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        is_bool = isinstance(value, bool)
        type_ok = isinstance(value, expected) and not (is_bool and expected is not bool)
        if not type_ok:
            errors.append(
                f"[{section}] {name} must be {expected.__name__}, got {type(value).__name__} ({value!r})"
            )
            continue
        kwargs[name] = value

    return cls(**kwargs), errors


def _read_toml(path: Path) -> Dict[str, Any]:
    if tomllib is None:
        raise ConfigError("Reading config.toml needs Python 3.11 or newer.")
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path} is not valid TOML: {e}") from e
    except OSError as e:
        raise ConfigError(f"Could not read {path}: {e}") from e


def load_config(path: Optional[os.PathLike] = None) -> Config:
    """Load and validate the config.

    If `path` (or the AETHER_CONFIG env var) is given, the file must exist.
    Otherwise config.toml in the project root is used if present, and plain
    defaults if not.
    """
    explicit = path or os.environ.get(CONFIG_ENV_VAR)
    config_path = Path(explicit) if explicit else DEFAULT_CONFIG_PATH

    raw: Dict[str, Any] = {}
    if config_path.exists():
        raw = _read_toml(config_path)
    elif explicit:
        raise ConfigError(f"Config file not found: {config_path}")

    errors: List[str] = []

    unknown_sections = sorted(set(raw) - set(SECTIONS))
    if unknown_sections:
        errors.append(f"unknown section(s): {', '.join(unknown_sections)}")

    built: Dict[str, Any] = {}
    for name, cls in SECTIONS.items():
        section_data = raw.get(name, {})
        if not isinstance(section_data, dict):
            errors.append(f"[{name}] must be a table")
            section_data = {}
        built[name], section_errors = _build_section(cls, section_data, name)
        errors.extend(section_errors)

    if errors:
        raise ConfigError(_format_errors(config_path, errors))

    cfg = Config(**built)
    problems = cfg.validate()
    if problems:
        raise ConfigError(_format_errors(config_path, problems))
    return cfg


def _format_errors(path: Path, errors: List[str]) -> str:
    lines = "\n".join(f"  - {e}" for e in errors)
    return f"Invalid configuration ({path}):\n{lines}"


_config: Optional[Config] = None


def get_config() -> Config:
    """Return the shared Config, loading it on first use."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reload_config(path: Optional[os.PathLike] = None) -> Config:
    """Discard the cached config and load it again."""
    global _config
    _config = load_config(path)
    return _config


if __name__ == "__main__":
    cfg = get_config()
    print("Config loaded OK.\n")
    for role in ROLES:
        print(f"{role:<11} model={cfg.models.for_role(role):<14} temperature={cfg.temperature.for_role(role)}")
    print(f"\nnum_ctx={cfg.ollama.num_ctx}  embedding={cfg.embedding.model_name} on {cfg.embedding.device}")
