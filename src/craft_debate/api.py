"""OpenAI API client (async) plus a deterministic mock provider for dry runs."""

from __future__ import annotations

import asyncio
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

def load_api_key(
    project_root: Path,
    key_name: str = "openai_api_key",
    env_var: str = "OPENAI_API_KEY",
) -> Optional[str]:
    """Key precedence: env var, then ``.secret/<key_name>`` file."""
    env_key = os.getenv(env_var, "").strip()
    if env_key:
        return env_key
    key_file = project_root / ".secret" / key_name
    if key_file.is_file():
        for line in key_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            return line
    return None


class LLMClient:
    """Thin async wrapper over OpenAI Chat Completions with retry/backoff."""

    def __init__(
        self,
        *,
        model: str,
        temperature: float,
        max_completion_tokens: int,
        api_key: str,
        timeout_seconds: float = 120.0,
        max_retries: int = 6,
        backoff_seconds: float = 2.0,
        base_url: Optional[str] = None,
        provider: str = "openai",
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.provider = provider
        self.extra_body = dict(extra_body or {})
        self.backoff_seconds = backoff_seconds
        self.max_retries = max_retries
        self._client = None
        self._client_kwargs = {
            "api_key": api_key,
            "timeout": timeout_seconds,
            "max_retries": 0,
        }
        if base_url:
            self._client_kwargs["base_url"] = base_url

    async def _get_client(self):
        if self._client is None:
            # Created lazily inside the event loop so the connection pool is loop-local.
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(**self._client_kwargs)
        return self._client

    async def complete(
        self, system: str, user: str, meta: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                started = time.monotonic()
                client = await self._get_client()
                request: Dict[str, Any] = {
                    "model": self.model,
                    "temperature": self.temperature,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                }
                if self.provider in ("deepseek", "ollama"):
                    # DeepSeek and Ollama OpenAI-compatible endpoints accept
                    # `max_tokens` (their docs list max_tokens, not
                    # max_completion_tokens).
                    request["max_tokens"] = self.max_completion_tokens
                else:
                    request["max_completion_tokens"] = self.max_completion_tokens
                if self.extra_body:
                    request["extra_body"] = self.extra_body
                response = await client.chat.completions.create(**request)
                content = response.choices[0].message.content or ""
                # Thinking-mode providers return the chain-of-thought separately;
                # keep it in the record without breaking downstream tag parsing.
                reasoning = getattr(response.choices[0].message, "reasoning_content", None)
                usage = response.usage
                return {
                    "content": content,
                    "reasoning_content": reasoning or "",
                    "model": self.model,
                    "latency_seconds": round(time.monotonic() - started, 3),
                    "usage": {
                        "prompt_tokens": getattr(usage, "prompt_tokens", None),
                        "completion_tokens": getattr(usage, "completion_tokens", None),
                        "total_tokens": getattr(usage, "total_tokens", None),
                    },
                    "attempts": attempt + 1,
                }
            except Exception as exc:  # noqa: BLE001 - retried below, then re-raised
                last_error = exc
                retriable = any(
                    marker in type(exc).__name__.lower()
                    for marker in ("ratelimit", "apiconnection", "timeout", "server")
                ) or getattr(exc, "status_code", None) in (429, 500, 502, 503, 504)
                if not retriable or attempt >= self.max_retries:
                    break
                delay = self.backoff_seconds * (2**attempt) + random.uniform(0, 0.5)
                await asyncio.sleep(delay)
        raise RuntimeError(
            f"LLM call failed after {self.max_retries + 1} attempts: {last_error}"
        ) from last_error


class MockLLM:
    """Deterministic provider used by `--mock` to verify the pipeline offline."""

    def __init__(self, model: str) -> None:
        self.model = model

    async def complete(
        self, system: str, user: str, meta: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        meta = meta or {}
        kind = meta.get("kind", "agent")
        if kind == "director":
            content = (
                "<analysis>Mock Director checks the private wall for a missing block.</analysis>\n"
                "<message>Place the next missing block on my bottom layer.</message>"
            )
        elif kind == "judge":
            oracle_moves = meta.get("oracle_moves") or []
            if not oracle_moves:
                content = "<move>CLARIFY:no candidate move (mock)</move>"
            else:
                move = oracle_moves[0]
                if move["action"] == "place":
                    line = f"PLACE:{move['block']}:{move['position']}:{move['layer']}"
                else:
                    line = f"REMOVE:{move['position']}:{move['layer']}"
                if move.get("span_to"):
                    line += f":{move['span_to']}"
                content = f"<move>{line}:CONFIRM:mock paper-protocol move</move>"
        else:
            content = "mock"
        return {
            "content": content,
            "model": self.model,
            "latency_seconds": 0.0,
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "attempts": 1,
        }
