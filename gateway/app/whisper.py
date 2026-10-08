"""Client for the whisper.cpp HTTP server.

whisper.cpp is not a streaming recogniser, so a dictation is cut into short
segments by the browser and each segment is transcribed on its own. The server
exposes two routes that matter here: ``GET /health`` and ``POST /inference``
(multipart upload, ``response_format=json``).
"""

from __future__ import annotations

import logging
import time

import httpx

from .config import WhisperSettings
from .errors import not_configured, upstream_timeout, upstream_unavailable

log = logging.getLogger("speechint.whisper")


def _headers(token: str) -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if token:
        headers["X-Auth-Token"] = token
    return headers


async def health(settings: WhisperSettings, *, timeout: float = 10.0) -> dict:
    """Live health check of the whisper server (``GET /health``).

    The result carries ``ok`` (the model is usable) and ``loading`` (the server
    is up but says it is still loading a model), which is what turns a bare
    "unreachable" into the friendlier "wird noch geladen".
    """

    if not settings.configured:
        return {
            "ok": False,
            "loading": False,
            "configured": False,
            "message": "Keine Whisper-URL konfiguriert.",
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
            "message": "Whisper antwortet nicht (Timeout).",
        }
    except httpx.HTTPError as exc:
        return {
            **result,
            "ok": False,
            "loading": False,
            "http": 0,
            "message": f"Whisper nicht erreichbar: {exc}",
        }

    if response.status_code == 200:
        return {
            **result,
            "ok": True,
            "loading": False,
            "http": 200,
            "message": "Whisper erreichbar (HTTP 200).",
        }

    # Older builds do not expose /health; a 404 still proves the server is up.
    if response.status_code == 404:
        return {
            **result,
            "ok": True,
            "loading": False,
            "http": 404,
            "message": "Whisper erreichbar, aber ohne /health-Endpunkt (ältere Version).",
        }

    # whisper.cpp answers 503 while it is still reading the model.
    if response.status_code == 503:
        return {
            **result,
            "ok": False,
            "loading": True,
            "http": 503,
            "message": "Whisper lädt das Modell noch (HTTP 503).",
        }

    return {
        **result,
        "ok": False,
        "loading": False,
        "http": response.status_code,
        "message": f"Whisper meldet HTTP {response.status_code}.",
    }


async def transcribe(
    settings: WhisperSettings,
    content: bytes,
    filename: str,
    mime: str = "",
    *,
    language: str = "",
    model: str = "",
) -> dict:
    """Send one audio segment to the whisper server and return the transcript.

    Returns ``{"ok", "text", "error", "http", "duration_ms", "model"}``. Failures
    are reported in the result rather than raised, because the caller decides
    whether a failed segment is fatal (it never is for an empty one).
    """

    if not settings.configured:
        raise not_configured("Es ist keine Spracherkennung (Whisper) konfiguriert.")

    data = {
        "response_format": "json",
        "temperature": "0.0",
    }
    if language:
        data["language"] = language
    if model:
        data["model"] = model

    files = {"file": (filename, content, mime or "application/octet-stream")}

    started_at = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=settings.timeout) as client:
            response = await client.post(
                settings.url + "/inference",
                data=data,
                files=files,
                headers=_headers(settings.token),
            )
    except httpx.TimeoutException:
        raise upstream_timeout(
            "transcription_timeout",
            f"Die Spracherkennung hat nach {settings.timeout} s nicht geantwortet.",
        )
    except httpx.HTTPError as exc:
        raise upstream_unavailable(
            "transcription_failed", f"Whisper nicht erreichbar: {exc}"
        )

    duration_ms = int(round((time.monotonic() - started_at) * 1000))
    body = response.text

    if response.status_code != 200:
        message = ""
        try:
            decoded = response.json()
            if isinstance(decoded, dict):
                message = str(decoded.get("error") or decoded.get("message") or "")
        except ValueError:
            message = ""
        if message == "":
            message = f"Whisper-Fehler (HTTP {response.status_code})"
        raise upstream_unavailable("transcription_failed", message)

    text = body
    try:
        decoded = response.json()
    except ValueError:
        decoded = None
    if isinstance(decoded, dict) and "text" in decoded:
        text = str(decoded.get("text") or "")
    elif isinstance(decoded, str):
        text = decoded

    return {
        "ok": True,
        "text": text.strip(),
        "error": "",
        "http": response.status_code,
        "duration_ms": duration_ms,
        "model": model or settings.model,
    }
