"""Readiness of the two model servers.

The first start of this stack is slow and mostly unattended: whisper.cpp
downloads a GGML model and llama.cpp pulls a ~1.1 GB GGUF from Hugging Face
before either of them answers a request. A client that only sees "connection
refused" cannot tell that apart from a broken deployment, so this module turns
the startup phase into an explicit, reportable state.

Component states
----------------
``ready``         the server answered and the model is loaded
``loading``       the server answered, but explicitly reports a model loading
``starting``      not reachable yet and has never been reachable – most likely
                  still downloading or loading
``unreachable``   was reachable before (or the grace window expired) and is not
``error``         reachable, but answering with an unexpected status
``unconfigured``  no URL configured for this component

The overall status aggregates them into ``ready``, ``loading``, ``degraded`` or
``unavailable``. While a component is in ``loading``/``starting``, requests that
need it are answered with HTTP 503 ``service_loading`` and a ``Retry-After``
header, so a client can say "die Modelle werden noch geladen" instead of showing
an error.

Probes are cached for a few seconds (``SPEECHINT_HEALTH_CACHE_SECONDS``) because
clients poll this while waiting.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

from .config import Settings
from .errors import service_loading
from .llm import health as llm_health
from .whisper import health as whisper_health

log = logging.getLogger("speechint.health")

READY = "ready"
LOADING = "loading"
STARTING = "starting"
UNREACHABLE = "unreachable"
ERROR = "error"
UNCONFIGURED = "unconfigured"

#: States that mean "not yet, but keep trying".
TRANSIENT = (LOADING, STARTING)

#: States that mean "this will not work right now".
BROKEN = (UNREACHABLE, ERROR, UNCONFIGURED)

COMPONENT_LABELS = {
    "whisper": "Spracherkennung (Whisper)",
    "llm": "Diktat-Modell (Qwen)",
}

TRANSIENT_MESSAGES = {
    LOADING: "{label} lädt das Modell noch.",
    STARTING: "{label} startet noch oder lädt das Modell herunter.",
}

_STATE_LABELS = {
    READY: "bereit",
    LOADING: "lädt Modell",
    STARTING: "startet",
    UNREACHABLE: "nicht erreichbar",
    ERROR: "Fehler",
    UNCONFIGURED: "nicht konfiguriert",
}

_cache: dict[str, Any] = {}
_ever_ready: dict[str, bool] = {}
_first_seen: dict[str, float] = {}


def reset() -> None:
    """Forget cached probes and readiness history (used by tests)."""

    _cache.clear()
    _ever_ready.clear()
    _first_seen.clear()


def _derive_state(name: str, result: dict, settings: Settings) -> str:
    if not result.get("configured", True):
        return UNCONFIGURED

    if result.get("ok"):
        _ever_ready[name] = True
        _first_seen.pop(name, None)
        return READY

    if result.get("loading"):
        return LOADING

    if int(result.get("http") or 0) == 0:
        # A closed port is the normal look of a container that is still
        # downloading a model, so treat it as "starting" – but only until the
        # grace window expires, after which it really is unreachable.
        if _ever_ready.get(name):
            return UNREACHABLE
        first = _first_seen.setdefault(name, time.monotonic())
        if time.monotonic() - first < settings.starting_grace_seconds:
            return STARTING
        return UNREACHABLE

    return ERROR


def _describe(name: str, state: str, result: dict) -> dict:
    return {
        "state": state,
        "state_label": _STATE_LABELS[state],
        "ok": state == READY,
        "http": int(result.get("http") or 0),
        "message": str(result.get("message") or ""),
        "url": str(result.get("url") or ""),
        "model": str(result.get("model") or ""),
    }


def _overall(components: dict[str, dict]) -> str:
    states = [component["state"] for component in components.values()]
    if all(state == READY for state in states):
        return READY
    if any(state in TRANSIENT for state in states) and not any(
        state in BROKEN for state in states
    ):
        return LOADING
    if any(state == READY for state in states):
        return "degraded"
    return "unavailable"


def _summary(status: str, components: dict[str, dict], retry_after: int) -> str:
    parts = [
        f"{COMPONENT_LABELS[name]}: {component['state_label']}"
        for name, component in components.items()
    ]
    joined = "; ".join(parts)

    if status == READY:
        return f"Bereit. {joined}."
    if status == LOADING:
        return (
            f"Noch nicht bereit – die Modelle werden geladen oder heruntergeladen. "
            f"{joined}. Bitte in {retry_after} s erneut versuchen."
        )
    return f"Nicht einsatzbereit. {joined}."


def _build_payload(settings: Settings, states: dict[str, str], results: dict[str, dict]) -> dict:
    components = {
        name: _describe(name, states[name], results[name]) for name in ("whisper", "llm")
    }
    status = _overall(components)
    retry_after = settings.loading_retry_after

    return {
        "ok": status == READY,
        "service": "speechint",
        "status": status,
        "ready": status == READY,
        "message": _summary(status, components, retry_after),
        "retry_after": retry_after if status in (LOADING, "degraded") else 0,
        "components": components,
    }


async def probe(settings: Settings, *, force: bool = False) -> dict:
    """Aggregated health of both model servers (cached for a few seconds)."""

    ttl = settings.health_cache_seconds
    now = time.monotonic()
    if not force and ttl > 0:
        cached = _cache.get("payload")
        if cached is not None and now - cached[0] < ttl:
            return cached[1]

    whisper_result, llm_result = await asyncio.gather(
        whisper_health(settings.whisper),
        llm_health(settings.llm),
    )
    results = {"whisper": whisper_result, "llm": llm_result}
    states = {
        name: _derive_state(name, results[name], settings) for name in ("whisper", "llm")
    }
    payload = _build_payload(settings, states, results)

    if ttl > 0:
        _cache["payload"] = (time.monotonic(), payload)
    return payload


async def require_ready(settings: Settings, *components: str) -> None:
    """Raise a ``service_loading`` error while a needed component is starting.

    Only the transient states are reported this way. A component that is
    reachable-but-broken or genuinely unreachable is left to the caller, so the
    dictation fallback and the normal upstream errors keep working.
    """

    payload = await probe(settings)
    for name in components:
        component = payload["components"][name]
        if component["state"] not in TRANSIENT:
            continue

        message = TRANSIENT_MESSAGES[component["state"]].format(
            label=COMPONENT_LABELS[name]
        )
        log.info("Anfrage abgelehnt: %s ist %s.", name, component["state"])
        raise service_loading(name, message, payload["retry_after"] or settings.loading_retry_after)


def component_of(settings: Settings, name: str) -> Optional[str]:
    """Synchronously read the cached state of one component, if there is one."""

    cached = _cache.get("payload")
    if cached is None:
        return None
    return cached[1]["components"][name]["state"]
