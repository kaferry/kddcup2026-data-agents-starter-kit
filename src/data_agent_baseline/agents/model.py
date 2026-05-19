from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError

try:
    from openai import InternalServerError
except ImportError:
    InternalServerError = None  # type: ignore[assignment]


TRANSIENT_HTTP_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
TRANSIENT_ERROR_PATTERNS = (
    "429", "bad gateway", "connection", "gateway timeout",
    "internal server error", "rate limit", "server disconnected",
    "service unavailable", "timeout", "too many requests",
)


@dataclass(frozen=True, slots=True)
class ModelMessage:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class ModelAction:
    action: str
    action_input: dict[str, Any]

@dataclass(frozen=True, slots=True)
class ModelStep:
    thought: str
    actions: list[ModelAction]
    raw_response: str


class ModelAdapter(Protocol):
    def complete(self, messages: list[ModelMessage]) -> str:
        raise NotImplementedError


def _is_transient_error(exc: BaseException) -> bool:
    transient_types: tuple[type[BaseException], ...]
    if InternalServerError is None:
        transient_types = (APIConnectionError, APITimeoutError, RateLimitError)
    else:
        transient_types = (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)
    if isinstance(exc, transient_types):
        return True
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and status_code in TRANSIENT_HTTP_STATUS_CODES:
        return True
    message = str(exc).lower()
    return any(p in message for p in TRANSIENT_ERROR_PATTERNS)


class OpenAIModelAdapter:
    def __init__(
        self,
        *,
        model: str,
        api_base: str,
        api_key: str,
        temperature: float,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
    ) -> None:
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay

    def complete(self, messages: list[ModelMessage]) -> str:
        if not self.api_key:
            raise RuntimeError("Missing model API key in config.agent.api_key.")

        client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            timeout=httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=5.0),
            max_retries=0,
        )

        request_messages = [{"role": m.role, "content": m.content} for m in messages]
        for attempt in range(self.max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=request_messages,
                    temperature=self.temperature,
                )
                break
            except Exception as exc:
                if attempt >= self.max_retries or not _is_transient_error(exc):
                    raise RuntimeError(f"Model request failed: {exc}") from exc
                delay = min(self.retry_base_delay * (2 ** attempt), 8.0)
                time.sleep(delay)
        else:
            raise RuntimeError("Model request failed without returning a response.")

        choices = response.choices or []
        if not choices:
            raise RuntimeError("Model response missing choices.")
        content = choices[0].message.content
        if not isinstance(content, str):
            raise RuntimeError("Model response missing text content.")
        return content


class ScriptedModelAdapter:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def complete(self, messages: list[ModelMessage]) -> str:
        del messages
        if not self._responses:
            raise RuntimeError("No scripted model responses remaining.")
        return self._responses.pop(0)
