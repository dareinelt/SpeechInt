"""Client for the llama.cpp server that hosts the dictation model.

llama.cpp exposes an OpenAI-compatible API (``/v1/chat/completions``) plus a
``/health`` probe. Two payload details matter for dictation:

* ``chat_template_kwargs.enable_thinking = false`` switches the hybrid-reasoning
  chat template of Qwen3.5 off;
* ``reasoning_budget = 0`` is the belt-and-braces variant, because the template
  switch alone is not honoured on every llama.cpp build – the model would then
  fill its whole token budget with a thinking block and return an empty answer.

The same payload is used for the OpenAI-compatible passthrough route, so a
client that talks to ``/v1/chat/completions`` gets identical behaviour.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Mapping, Optional, Sequence

import httpx

from .config import LlmSettings
from .errors import not_configured, upstream_timeout, upstream_unavailable

log = logging.getLogger("speechint.llm")


def _headers(token: str, *, json_body: bool = False) -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if json_body:
        headers["Content-Type"] = "application/json"
    if token:
        headers["X-Auth-Token"] = token
    return headers


async def health(settings: LlmSettings, *, timeout: float = 10.0) -> dict:
    """Live health check of the llama.cpp server (``GET /health``).

    llama.cpp answers 503 with ``{"error":{"message":"Loading model"}}`` while it
    is reading the GGUF, and 200 with ``{"status":"ok"}`` once the model is
    ready; both cases are reported here.
    """

    if not settings.configured:
        return {
            "ok": False,
            "loading": False,
            "configured": False,
            "message": "Keine Modell-URL konfiguriert.",
            "http": 0,
            "url": "",
            "model": settings.model,
        }

    result = {"configured": True, "url": settings.url, "model": settings.model}
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(settings.url + "/health", headers=_headers(settings.token))
    except httpx.TimeoutException:
        return {
            **result,
            "ok": False,
            "loading": False,
            "http": 0,
            "message": "Modellserver antwortet nicht (Timeout).",
        }
    except httpx.HTTPError as exc:
        return {
            **result,
            "ok": False,
            "loading": False,
            "http": 0,
            "message": f"Modellserver nicht erreichbar: {exc}",
        }

    decoded: Any = None
    try:
        decoded = response.json()
    except ValueError:
        decoded = None

    if response.status_code == 503:
        return {
            **result,
            "ok": False,
            "loading": True,
            "http": 503,
            "message": _llama_message(decoded) or "Modellserver lädt das Modell noch (HTTP 503).",
        }

    if response.status_code != 200:
        return {
            **result,
            "ok": False,
            "loading": False,
            "http": response.status_code,
            "message": f"Modellserver meldet HTTP {response.status_code}.",
        }

    status = "ok"
    if isinstance(decoded, dict):
        status = str(decoded.get("status") or "ok")

    if status != "ok":
        return {
            **result,
            "ok": False,
            "loading": True,
            "http": 200,
            "message": f"Modellserver lädt noch ({status}).",
        }

    return {
        **result,
        "ok": True,
        "loading": False,
        "http": 200,
        "message": f"Modellserver erreichbar ({status}).",
    }


def _llama_message(decoded: Any) -> str:
    """Pull the human-readable text out of a llama.cpp error body."""

    if not isinstance(decoded, dict):
        return ""
    error = decoded.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or "")
    if error:
        return str(error)
    return ""


def build_payload(
    settings: LlmSettings,
    messages: Sequence[Mapping[str, Any]],
    *,
    model: str = "",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> dict:
    """Assemble the chat-completion payload with thinking switched off."""

    payload: dict[str, Any] = {
        "model": model or settings.model,
        "stream": False,
        "messages": [dict(message) for message in messages],
        "temperature": 0.1 if temperature is None else float(temperature),
        "chat_template_kwargs": {"enable_thinking": False},
        "reasoning_budget": 0,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max(32, min(4096, int(max_tokens)))
    return payload


async def chat_completion(
    settings: LlmSettings, payload: Mapping[str, Any], *, timeout: Optional[int] = None
) -> dict:
    """Run one non-streaming chat completion.

    Returns ``{"ok", "text", "model", "latency_ms", "usage"}`` on success and
    raises :class:`~app.errors.SpeechIntError` on a transport or upstream error.
    """

    if not settings.configured:
        raise not_configured("Es ist kein Modellserver für das Diktat konfiguriert.")

    body = dict(payload)
    body.setdefault("stream", False)

    started_at = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=timeout or settings.timeout) as client:
            response = await client.post(
                settings.url + "/v1/chat/completions",
                json=body,
                headers=_headers(settings.token, json_body=True),
            )
    except httpx.TimeoutException:
        raise upstream_timeout(
            "llm_timeout",
            f"Der Modellserver hat nach {timeout or settings.timeout} s nicht geantwortet.",
        )
    except httpx.HTTPError as exc:
        raise upstream_unavailable("llm_unavailable", f"Modellserver nicht erreichbar: {exc}")

    latency_ms = int(round((time.monotonic() - started_at) * 1000))

    try:
        data = response.json()
    except ValueError:
        data = None

    if response.status_code != 200 or not isinstance(data, dict):
        message = ""
        if isinstance(data, dict):
            error = data.get("error")
            if isinstance(error, dict):
                message = str(error.get("message") or "")
            elif error:
                message = str(error)
        raise upstream_unavailable(
            "llm_error", message or f"Modell-Fehler (HTTP {response.status_code})"
        )

    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    if isinstance(content, list):
        # Some builds answer with typed content parts.
        content = "".join(
            str(part.get("text") or "")
            for part in content
            if isinstance(part, Mapping) and part.get("type") == "text"
        )
    if not isinstance(content, str):
        content = ""

    usage = data.get("usage") or {}
    return {
        "ok": True,
        "text": content,
        "model": str(body.get("model") or settings.model),
        "latency_ms": latency_ms,
        "usage": {
            "prompt": int(usage.get("prompt_tokens") or 0),
            "completion": int(usage.get("completion_tokens") or 0),
            "total": int(usage.get("total_tokens") or 0),
        },
        "raw": data,
    }
