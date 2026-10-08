"""Shared fixtures.

The model probes are replaced globally so no test ever touches the network: by
default both servers report "ready", and a test that wants to exercise the
loading window overwrites ``models["llm"]`` / ``models["whisper"]``.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config, health  # noqa: E402  (needs the path above)

READY_WHISPER = {
    "ok": True,
    "loading": False,
    "configured": True,
    "http": 200,
    "message": "Whisper erreichbar (HTTP 200).",
    "url": "http://whisper:8080",
    "model": "small",
}

READY_LLM = {
    "ok": True,
    "loading": False,
    "configured": True,
    "http": 200,
    "message": "Modellserver erreichbar (ok).",
    "url": "http://llm:8080",
    "model": "Qwen3.5-2B Q4",
}

LOADING_WHISPER = {
    **READY_WHISPER,
    "ok": False,
    "loading": True,
    "http": 503,
    "message": "Whisper lädt das Modell noch (HTTP 503).",
}

LOADING_LLM = {
    **READY_LLM,
    "ok": False,
    "loading": True,
    "http": 503,
    "message": "Loading model",
}

UNREACHABLE_WHISPER = {
    **READY_WHISPER,
    "ok": False,
    "loading": False,
    "http": 0,
    "message": "Whisper nicht erreichbar: Connection refused",
}

UNREACHABLE_LLM = {
    **READY_LLM,
    "ok": False,
    "loading": False,
    "http": 0,
    "message": "Modellserver nicht erreichbar: Connection refused",
}

UNCONFIGURED_LLM = {
    **READY_LLM,
    "ok": False,
    "loading": False,
    "configured": False,
    "http": 0,
    "url": "",
    "message": "Keine Modell-URL konfiguriert.",
}


@pytest.fixture(autouse=True)
def models(monkeypatch):
    """Fake both model probes; yields the mutable state they read from."""

    state = {
        "whisper": dict(READY_WHISPER),
        "llm": dict(READY_LLM),
        "probes": 0,
    }

    async def probe_whisper(settings, *, timeout=10.0):
        state["probes"] += 1
        return dict(state["whisper"])

    async def probe_llm(settings, *, timeout=10.0):
        state["probes"] += 1
        return dict(state["llm"])

    monkeypatch.setattr(health, "whisper_health", probe_whisper)
    monkeypatch.setattr(health, "llm_health", probe_llm)
    health.reset()
    yield state
    health.reset()


@pytest.fixture(autouse=True)
def clean_settings(monkeypatch):
    """Start every test from the built-in defaults, not the developer's shell."""

    for name in (
        "SPEECHINT_TOKEN",
        "SPEECHINT_MAX_AUDIO_MB",
        "SPEECHINT_MAX_CONTEXT_CHARS",
        "SPEECHINT_DEFAULT_LANGUAGE",
        "SPEECHINT_TEMPERATURE",
        "SPEECHINT_LOG_LEVEL",
        "SPEECHINT_HEALTH_CACHE_SECONDS",
        "SPEECHINT_LOADING_RETRY_AFTER",
        "SPEECHINT_STARTING_GRACE_SECONDS",
        "WHISPER_URL",
        "WHISPER_TOKEN",
        "WHISPER_TIMEOUT",
        "WHISPER_MODEL",
        "WHISPER_LANGUAGE",
        "LLM_URL",
        "LLM_TOKEN",
        "LLM_TIMEOUT",
        "LLM_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    config.reset_settings()
    yield
    config.reset_settings()
