"""The one place that talks to Ollama.

Every call goes through `LLMClient`, which:

* picks the model and temperature for the role (dm, options, judge, ...),
* always sends `num_ctx`, so Ollama can never silently use a smaller window
  and drop the start of the prompt,
* caps reply length (`narration_max_tokens` for the DM, an explicit limit for
  structured calls so a runaway answer can't hang a CPU-only laptop),
* turns every connection or server problem into a single `LLMError`,
* validates structured output against a pydantic schema, retrying once.

The Ollama connection is injectable (`backend=`) so tests never need a server.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import re
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Type, TypeVar

import ollama
from pydantic import BaseModel, ValidationError

from aether.config import ROLES, Config, get_config

T = TypeVar("T", bound=BaseModel)
Messages = List[Dict[str, str]]

RETRY_MIN_TEMPERATURE = 0.3  # a retry at temperature 0 would just repeat the same bad answer

_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


class LLMError(Exception):
    """Ollama could not be reached, or the request failed."""


def strip_json_fence(text: str) -> str:
    """Remove a ```json fence if the model added one."""
    match = _FENCE.match(text)
    return match.group(1) if match else text.strip()


def _get(obj: Any, key: str) -> Any:
    """Read `key` from a dict or from an object attribute (the Ollama client returns objects)."""
    try:
        return obj[key]
    except (KeyError, TypeError, IndexError):
        return getattr(obj, key, None)


def _content(response: Any) -> str:
    message = _get(response, "message")
    return (_get(message, "content") if message is not None else None) or ""


def model_installed(name: str, installed: Iterable[str]) -> bool:
    """True if `name` is among the installed models ('llama3.2' matches 'llama3.2:latest')."""
    wanted = name.strip().lower()
    have = {n.strip().lower() for n in installed}
    if wanted in have:
        return True
    return ":" not in wanted and f"{wanted}:latest" in have


@dataclass(frozen=True)
class CallStats:
    """Ollama's own timing for one call (durations arrive in nanoseconds)."""
    role: str
    load_s: float = 0.0        # time spent loading the model into memory
    prompt_tokens: int = 0     # prompt tokens actually evaluated (low = Ollama reused a cached prefix)
    prompt_s: float = 0.0
    gen_tokens: int = 0
    gen_s: float = 0.0

    @property
    def prompt_tps(self) -> float:
        return self.prompt_tokens / self.prompt_s if self.prompt_s > 0 else 0.0

    @property
    def gen_tps(self) -> float:
        return self.gen_tokens / self.gen_s if self.gen_s > 0 else 0.0


def _stats_from(role: str, response: Any) -> Optional[CallStats]:
    """Read Ollama's timing fields, or None when the response has none."""
    def seconds(key: str) -> float:
        return float(_get(response, key) or 0) / 1e9

    def count(key: str) -> int:
        return int(_get(response, key) or 0)

    prompt_tokens, gen_tokens = count("prompt_eval_count"), count("eval_count")
    if not (prompt_tokens or gen_tokens):
        return None
    return CallStats(
        role, seconds("load_duration"), prompt_tokens, seconds("prompt_eval_duration"),
        gen_tokens, seconds("eval_duration"),
    )

class LLMClient:
    def __init__(self, config: Optional[Config] = None, backend: Any = None):
        self.config = config or get_config()
        self._backend = backend if backend is not None else ollama.Client(host=self.config.ollama.host)
        self.last_structured_error: Optional[str] = None  # why the last structured call failed
        self.call_log: "deque[CallStats]" = deque(maxlen=50)  # timing of recent calls

    # ------------------------------------------------------------------
    # Request building
    # ------------------------------------------------------------------

    def model_for(self, role: str) -> str:
        return self.config.models.for_role(role)

    def _request(
        self,
        role: str,
        messages: Messages,
        *,
        stream: bool,
        max_tokens: Optional[int],
        temperature: Optional[float],
        schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        cfg = self.config
        options: Dict[str, Any] = {
            "temperature": cfg.temperature.for_role(role) if temperature is None else temperature,
            "num_ctx": cfg.ollama.num_ctx,
        }
        limit = max_tokens if max_tokens is not None else (cfg.ollama.narration_max_tokens if role == "dm" else 0)
        if limit and limit > 0:
            options["num_predict"] = limit
        request: Dict[str, Any] = {
            "model": self.model_for(role),
            "messages": messages,
            "options": options,
            "stream": stream,
            "keep_alive": cfg.ollama.keep_alive,
        }
        if schema is not None:
            request["format"] = schema
        return request

    def _wrap(self, role: str, error: Exception) -> LLMError:
        return LLMError(
            f"Ollama request for '{role}' failed ({type(error).__name__}: {error}). "
            f"Is Ollama running at {self.config.ollama.host} and is the model "
            f"'{self.model_for(role)}' pulled?"
        )

    def _record(self, role: str, response: Any) -> None:
        stats = _stats_from(role, response)
        if stats:
            self.call_log.append(stats)

    # ------------------------------------------------------------------
    # Calls
    # ------------------------------------------------------------------

    def chat(
        self,
        role: str,
        messages: Messages,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """One complete reply as text."""
        request = self._request(role, messages, stream=False, max_tokens=max_tokens, temperature=temperature)
        try:
            response = self._backend.chat(**request)
            self._record(role, response)
        except Exception as e:
            raise self._wrap(role, e) from e
        return _content(response).strip()

    def stream(
        self,
        role: str,
        messages: Messages,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Iterator[str]:
        """A reply as it is generated, piece by piece."""
        request = self._request(role, messages, stream=True, max_tokens=max_tokens, temperature=temperature)
        try:
            for chunk in self._backend.chat(**request):
                self._record(role, chunk)
                piece = _content(chunk)
                if piece:
                    yield piece
        except Exception as e:
            raise self._wrap(role, e) from e

    def structured(
        self,
        role: str,
        messages: Messages,
        schema: Type[T],
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        retries: int = 1,
    ) -> Optional[T]:
        """A reply that must match `schema`.

        Returns None when the model keeps producing invalid output (the reason
        is kept in `last_structured_error`). Connection problems raise
        `LLMError` instead, since retrying will not help.
        """
        self.last_structured_error = None
        schema_json = schema.model_json_schema()
        base_temp = self.config.temperature.for_role(role) if temperature is None else temperature

        for attempt in range(retries + 1):
            temp = base_temp if attempt == 0 else max(base_temp, RETRY_MIN_TEMPERATURE)
            request = self._request(
                role, messages, stream=False, max_tokens=max_tokens, temperature=temp, schema=schema_json
            )
            try:
                response = self._backend.chat(**request)
            except Exception as e:
                raise self._wrap(role, e) from e
            self._record(role, response)
            raw = strip_json_fence(_content(response))
            try:
                return schema.model_validate_json(raw)
            except ValidationError as e:
                self.last_structured_error = f"{schema.__name__}: {e.errors()[:2]} | raw={raw[:200]!r}"
        return None

    # ------------------------------------------------------------------
    # Setup checks
    # ------------------------------------------------------------------

    def installed_models(self) -> List[str]:
        try:
            response = self._backend.list()
        except Exception as e:
            raise LLMError(
                f"Could not list Ollama models ({type(e).__name__}: {e}). "
                f"Is Ollama running at {self.config.ollama.host}?"
            ) from e
        names = []
        for entry in _get(response, "models") or []:
            name = _get(entry, "model") or _get(entry, "name")
            if name:
                names.append(name)
        return names

    def missing_models(self, roles: Sequence[str] = ROLES) -> List[str]:
        """Models the config needs that are not pulled yet (sorted, no duplicates)."""
        installed = self.installed_models()
        needed = {self.model_for(r) for r in roles}
        return sorted(m for m in needed if not model_installed(m, installed))
