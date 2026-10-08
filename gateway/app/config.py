"""Environment-driven configuration of the SpeechInt gateway.

Every value has a working default so the container starts without an .env file.
See ``.env.example`` in the repository root for the documented variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

DEFAULT_WHISPER_URL = "http://whisper:8080"
DEFAULT_LLM_URL = "http://llm:8080"
DEFAULT_LLM_MODEL = "Qwen3.5-2B Q4"
LOG_LEVELS = ("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG")


def _text(name: str, default: str = "") -> str:
    value = (os.getenv(name) or "").strip()
    return value or default


def _number(name: str, default: int, low: int, high: int) -> int:
    raw = _text(name)
    if raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(low, min(high, value))


@dataclass(frozen=True)
class WhisperSettings:
    """Connection to the whisper.cpp server."""

    url: str = ""
    token: str = ""
    timeout: int = 120
    model: str = "small"
    language: str = "de"

    @property
    def configured(self) -> bool:
        return self.url != ""


@dataclass(frozen=True)
class LlmSettings:
    """Connection to the llama.cpp server that hosts the dictation model."""

    url: str = ""
    token: str = ""
    timeout: int = 60
    model: str = DEFAULT_LLM_MODEL

    @property
    def configured(self) -> bool:
        return self.url != ""


@dataclass(frozen=True)
class Settings:
    """Complete gateway configuration."""

    token: str = ""
    max_audio_bytes: int = 25 * 1024 * 1024
    max_context_chars: int = 400
    default_language: str = "de"
    default_temperature: float = 0.1
    log_level: str = "INFO"
    health_cache_seconds: int = 5
    loading_retry_after: int = 15
    starting_grace_seconds: int = 900
    whisper: WhisperSettings = field(default_factory=WhisperSettings)
    llm: LlmSettings = field(default_factory=LlmSettings)

    @property
    def auth_required(self) -> bool:
        return self.token != ""


def load_settings() -> Settings:
    """Build a :class:`Settings` instance from the current environment."""

    try:
        temperature = float(_text("SPEECHINT_TEMPERATURE", "0.1"))
    except ValueError:
        temperature = 0.1

    log_level = _text("SPEECHINT_LOG_LEVEL", "INFO").upper()
    if log_level not in LOG_LEVELS:
        log_level = "INFO"

    return Settings(
        token=_text("SPEECHINT_TOKEN"),
        max_audio_bytes=_number("SPEECHINT_MAX_AUDIO_MB", 25, 1, 500) * 1024 * 1024,
        max_context_chars=_number("SPEECHINT_MAX_CONTEXT_CHARS", 400, 0, 20000),
        default_language=_text("SPEECHINT_DEFAULT_LANGUAGE", "de"),
        default_temperature=max(0.0, min(1.0, temperature)),
        log_level=log_level,
        health_cache_seconds=_number("SPEECHINT_HEALTH_CACHE_SECONDS", 5, 0, 300),
        loading_retry_after=_number("SPEECHINT_LOADING_RETRY_AFTER", 15, 1, 3600),
        starting_grace_seconds=_number("SPEECHINT_STARTING_GRACE_SECONDS", 900, 0, 86400),
        whisper=WhisperSettings(
            url=_text("WHISPER_URL", DEFAULT_WHISPER_URL).rstrip("/"),
            token=_text("WHISPER_TOKEN"),
            timeout=_number("WHISPER_TIMEOUT", 120, 5, 1800),
            model=_text("WHISPER_MODEL", "small"),
            language=_text("WHISPER_LANGUAGE", "de"),
        ),
        llm=LlmSettings(
            url=_text("LLM_URL", DEFAULT_LLM_URL).rstrip("/"),
            token=_text("LLM_TOKEN"),
            timeout=_number("LLM_TIMEOUT", 60, 5, 900),
            model=_text("LLM_MODEL", DEFAULT_LLM_MODEL),
        ),
    )


_cache: Optional[Settings] = None


def get_settings() -> Settings:
    """Return the process-wide settings, reading the environment once."""

    global _cache
    if _cache is None:
        _cache = load_settings()
    return _cache


def reset_settings() -> None:
    """Drop the cached settings; used by tests and after environment changes."""

    global _cache
    _cache = None
