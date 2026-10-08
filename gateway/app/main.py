"""SpeechInt – HTTP API for whisper.cpp transcription and llama.cpp dictation.

The service owns the two compute-heavy parts of the LLMInt dictation feature and
exposes them as a small REST API so they can run on a separate Docker host.

Routes
------
``GET  /health``                     liveness probe (no auth)
``GET  /v1/ready``                   readiness probe: 200 ready / 503 loading (no auth)
``GET  /v1/health``                  component health of whisper and the model
``GET  /v1/config``                  capabilities, defaults and limits
``GET  /v1/models``                  available models
``POST /v1/audio/transcriptions``    audio segment -> raw transcript
``POST /v1/chat/completions``        OpenAI-compatible passthrough to llama.cpp
``POST /v1/dictate/process``         transcript fragment -> finished text
``POST /v1/dictate``                 audio -> finished text (one-shot)
"""

from __future__ import annotations

import hmac
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import Depends, FastAPI, File, Form, Header, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

from . import __version__, dictation, health, host, whisper
from .commands import DEFAULT_COMMANDS, normalize_commands
from .config import get_settings
from .errors import SpeechIntError, bad_request, payload_too_large, unauthorized
from .llm import chat_completion
from .pipeline import process_fragment

settings = get_settings()

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("speechint")

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Report how this endpoint compares to the reference machine at startup.

    SpeechInt is dimensioned for six CPU cores with AVX2 and 16 GB RAM; a host
    below that still runs, so this only warns.
    """

    host.log_sizing()
    yield


app = FastAPI(
    title="SpeechInt",
    version=__version__,
    description="Spracherkennung (whisper.cpp) und Diktat-Nachbearbeitung (llama.cpp) als eigener Dienst.",
    lifespan=lifespan,
)


# ── Authentication ────────────────────────────────────────────────────────────


async def require_token(
    x_auth_token: Optional[str] = Header(default=None),
    authorization: Optional[str] = Header(default=None),
) -> None:
    """Check the shared secret of the API.

    Both ``X-Auth-Token`` (the header the whisper.cpp and llama.cpp servers use)
    and the more common ``Authorization: Bearer …`` are accepted, so a client can
    pick whichever fits its HTTP helper.
    """

    expected = get_settings().token
    if expected == "":
        return

    supplied = (x_auth_token or "").strip()
    if supplied == "" and authorization:
        scheme, _, value = authorization.partition(" ")
        if scheme.lower() == "bearer":
            supplied = value.strip()

    if supplied == "" or not hmac.compare_digest(supplied, expected):
        raise unauthorized()


# ── Error handling ────────────────────────────────────────────────────────────


@app.exception_handler(SpeechIntError)
async def speechint_error_handler(_request: Request, exc: SpeechIntError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=exc.detail, headers=exc.headers)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    first = exc.errors()[0] if exc.errors() else {}
    field = ".".join(str(part) for part in first.get("loc", ()) if part != "body")
    message = str(first.get("msg") or "Ungültige Anfrage.")
    return JSONResponse(
        status_code=400,
        content={
            "ok": False,
            "error": "bad_request",
            "message": f"Ungültige Anfrage ({field}): {message}" if field else message,
        },
    )


# ── Request models ────────────────────────────────────────────────────────────


class ProcessRequest(BaseModel):
    """Body of ``POST /v1/dictate/process``."""

    fragment: str = ""
    context: str = ""
    prompt: Optional[str] = None
    commands: Optional[Any] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


# ── Helpers ───────────────────────────────────────────────────────────────────


def _commands_of(raw: Any) -> list[dict[str, str]]:
    return normalize_commands(raw)


def _prompt_of(raw: Optional[str], commands: list[dict[str, str]]) -> str:
    prompt = (raw or "").strip()
    return prompt or dictation.default_prompt(commands)


async def _read_audio(file: UploadFile) -> bytes:
    limit = get_settings().max_audio_bytes
    content = await file.read()
    if len(content) > limit:
        raise payload_too_large(
            f"Die Aufnahme ist zu groß (max. {round(limit / 1048576, 1)} MB)."
        )
    return content


def _safe_filename(name: Optional[str]) -> str:
    candidate = (name or "").rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    cleaned = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in candidate)
    return cleaned or "segment.wav"


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/")
async def root() -> dict:
    return {
        "ok": True,
        "service": "speechint",
        "version": __version__,
        "docs": "/docs",
        "endpoints": [
            "/health",
            "/v1/health",
            "/v1/config",
            "/v1/models",
            "/v1/audio/transcriptions",
            "/v1/chat/completions",
            "/v1/dictate/process",
            "/v1/dictate",
        ],
    }


@app.get("/health")
async def health_liveness() -> dict:
    """Liveness probe; deliberately unauthenticated for container health checks.

    This only says that the gateway process is up – it does not look at the
    model servers, so a container is never restarted while they are loading.
    Use ``/v1/ready`` to find out whether the models are available.
    """

    return {"ok": True, "service": "speechint", "version": __version__, "status": "ok"}


@app.get("/v1/ready")
async def ready() -> JSONResponse:
    """Readiness probe: HTTP 200 once both models are usable, otherwise 503.

    Unauthenticated so an orchestrator can use it. While a model is still being
    downloaded or loaded the body carries ``status: "loading"``, a human-readable
    ``message`` and ``retry_after``, and the response has a ``Retry-After``
    header – everything a client needs to show progress instead of an error.
    """

    payload = await health.probe(get_settings())
    if payload["ready"]:
        return JSONResponse(status_code=200, content=payload)

    retry_after = payload["retry_after"] or get_settings().loading_retry_after
    return JSONResponse(
        status_code=503,
        content=payload,
        headers={"Retry-After": str(max(1, retry_after))},
    )


@app.get("/v1/health", dependencies=[Depends(require_token)])
async def component_health() -> JSONResponse:
    """Reachability and loading state of both backing services.

    Answers 200 while everything is ready and 503 while a model is still being
    downloaded or loaded, so a client can poll it and report progress. The body
    is the same in both cases.
    """

    payload = await health.probe(get_settings())
    if payload["ready"]:
        return JSONResponse(status_code=200, content=payload)

    retry_after = payload["retry_after"] or get_settings().loading_retry_after
    return JSONResponse(
        status_code=503,
        content=payload,
        headers={"Retry-After": str(max(1, retry_after))},
    )


@app.get("/v1/config", dependencies=[Depends(require_token)])
async def config() -> dict:
    """Capabilities, defaults, limits and host sizing of this instance.

    A client uses this to prefill its own settings and to offer the service's
    default command table and prompt. ``host`` reports how the endpoint compares
    to the reference machine (six cores with AVX2, 16 GB RAM).
    """

    current = get_settings()
    return {
        "ok": True,
        "service": "speechint",
        "version": __version__,
        "capabilities": ["transcribe", "process", "dictate", "chat"],
        "transcription": {
            "configured": current.whisper.configured,
            "model": current.whisper.model,
            "language": current.whisper.language,
            "timeout_seconds": current.whisper.timeout,
        },
        "dictation": {
            "configured": current.llm.configured,
            "model": current.llm.model,
            "timeout_seconds": current.llm.timeout,
            "max_context_chars": current.max_context_chars,
            "temperature": current.default_temperature,
            "commands": [dict(command) for command in DEFAULT_COMMANDS],
            "default_prompt": dictation.default_prompt(DEFAULT_COMMANDS),
        },
        "limits": {
            "max_audio_bytes": current.max_audio_bytes,
            "max_audio_mb": round(current.max_audio_bytes / 1048576, 1),
            "loading_retry_after": current.loading_retry_after,
        },
        # Reference machine of this deployment: six cores with AVX2, 16 GB RAM.
        # An administrator can see here whether the endpoint is sized for it.
        "host": host.inspect(),
    }


@app.get("/v1/models", dependencies=[Depends(require_token)])
async def models() -> dict:
    """Models this instance serves, in the OpenAI ``list`` envelope."""

    current = get_settings()
    return {
        "ok": True,
        "object": "list",
        "data": [
            {
                "id": current.whisper.model,
                "kind": "transcription",
                "provider": "whisper.cpp",
            },
            {
                "id": current.llm.model,
                "kind": "chat",
                "provider": "llama.cpp",
            },
        ],
    }


@app.post("/v1/audio/transcriptions", dependencies=[Depends(require_token)])
async def transcriptions(
    file: UploadFile = File(...),
    language: str = Form(""),
    model: str = Form(""),
    response_format: str = Form("json"),
) -> Any:
    """Transcribe one recorded audio segment.

    The recognised text is returned unchanged and is never stored.
    """

    current = get_settings()
    content = await _read_audio(file)

    if not content:
        # An empty segment is not an error: it simply carries no speech.
        return {"ok": True, "text": "", "empty": True, "duration_ms": 0, "bytes": 0}

    # Report the model download/load window as "still loading" rather than as a
    # failed transcription.
    await health.require_ready(current, "whisper")

    started_at = time.monotonic()
    result = await whisper.transcribe(
        current.whisper,
        content,
        _safe_filename(file.filename),
        file.content_type or "",
        language=(language or current.whisper.language).strip(),
        model=model.strip(),
    )

    text = result["text"]
    log.info(
        "Segment transkribiert (%s, %d Bytes, %d ms).",
        dictation.log_summary(text),
        len(content),
        result["duration_ms"],
    )

    if response_format == "text":
        return PlainTextResponse(content=text)

    return {
        "ok": True,
        "text": text,
        "empty": text == "",
        "model": result["model"],
        "language": (language or current.whisper.language).strip(),
        "duration_ms": result["duration_ms"],
        "bytes": len(content),
        "total_ms": int(round((time.monotonic() - started_at) * 1000)),
    }


@app.post("/v1/chat/completions", dependencies=[Depends(require_token)])
async def chat_completions(request: Request) -> dict:
    """OpenAI-compatible passthrough to the llama.cpp server.

    The request body is forwarded as received; only the two dictation defaults
    (thinking off, non-streaming) are filled in when the caller omitted them.
    """

    current = get_settings()
    try:
        body = await request.json()
    except ValueError:
        raise bad_request("Der Anfragekörper ist kein gültiges JSON.")
    if not isinstance(body, dict):
        raise bad_request("Der Anfragekörper muss ein JSON-Objekt sein.")
    if not body.get("messages"):
        raise bad_request('Das Feld "messages" fehlt oder ist leer.')

    body.setdefault("chat_template_kwargs", {"enable_thinking": False})
    body.setdefault("reasoning_budget", 0)
    body["stream"] = False

    await health.require_ready(current, "llm")

    result = await chat_completion(current.llm, body)
    return result["raw"]


@app.post("/v1/dictate/process", dependencies=[Depends(require_token)])
async def dictate_process(payload: ProcessRequest) -> dict:
    """Turn one transcript fragment into finished text.

    Called once per recognised fragment – the client keeps the accumulated text,
    this service stays stateless.
    """

    current = get_settings()
    commands = _commands_of(payload.commands)
    prompt = _prompt_of(payload.prompt, commands)

    await health.require_ready(current, "llm")

    result = await process_fragment(
        payload.fragment,
        payload.context,
        settings=current,
        commands=commands,
        prompt=prompt,
        model=(payload.model or "").strip(),
        temperature=payload.temperature,
        max_tokens=payload.max_tokens,
    )

    if result["fallback"]:
        log.warning("Diktat-Modell nicht verfügbar – Regel-Fallback verwendet: %s", result["warning"])
    else:
        log.info("Fragment verarbeitet (%s, %d ms).", dictation.log_summary(result["text"]), result["duration_ms"])

    return {"ok": True, **result}


@app.post("/v1/dictate", dependencies=[Depends(require_token)])
async def dictate(
    file: UploadFile = File(...),
    language: str = Form(""),
    context: str = Form(""),
    prompt: str = Form(""),
    commands: str = Form(""),
    model: str = Form(""),
) -> dict:
    """Transcribe an audio segment and process it in one call.

    Convenience route for clients that do not need the two steps separately.
    """

    current = get_settings()
    content = await _read_audio(file)

    command_table = _commands_of(commands)
    effective_prompt = _prompt_of(prompt, command_table)

    await health.require_ready(current, "whisper", "llm")

    started_at = time.monotonic()
    if not content:
        transcript = ""
        transcribe_ms = 0
    else:
        transcription = await whisper.transcribe(
            current.whisper,
            content,
            _safe_filename(file.filename),
            file.content_type or "",
            language=(language or current.whisper.language).strip(),
            model="",
        )
        transcript = transcription["text"]
        transcribe_ms = transcription["duration_ms"]

    result = await process_fragment(
        transcript,
        context,
        settings=current,
        commands=command_table,
        prompt=effective_prompt,
        model=model.strip(),
    )

    return {
        "ok": True,
        "text": result["text"],
        "transcript": transcript,
        "fallback": result["fallback"],
        "model": result["model"],
        "warning": result["warning"],
        "whisper_model": current.whisper.model,
        "transcribe_ms": transcribe_ms,
        "process_ms": result["duration_ms"],
        "total_ms": int(round((time.monotonic() - started_at) * 1000)),
    }
