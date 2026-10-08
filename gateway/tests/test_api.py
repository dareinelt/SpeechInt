"""Route-level tests with both model servers mocked.

Nothing here talks to a real whisper.cpp or llama.cpp; the two upstream clients
are replaced with fakes so the tests exercise the HTTP contract, the auth check
and the pipeline wiring.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import config, errors, whisper
from app import pipeline
from app.main import app as gateway_app
from conftest import (
    LOADING_LLM,
    LOADING_WHISPER,
    READY_WHISPER,
    UNCONFIGURED_LLM,
    UNREACHABLE_LLM,
    UNREACHABLE_WHISPER,
)

AUDIO = ("segment.wav", b"RIFF....WAVEfmt ", "audio/wav")


@pytest.fixture
def client():
    with TestClient(gateway_app) as test_client:
        yield test_client


@pytest.fixture
def fake_whisper(monkeypatch):
    """Replace the whisper client; returns the list of captured calls."""

    calls: list[dict] = []

    async def transcribe(settings, content, filename, mime="", *, language="", model=""):
        calls.append(
            {
                "content": content,
                "filename": filename,
                "mime": mime,
                "language": language,
                "model": model,
            }
        )
        return {
            "ok": True,
            "text": "hallo welt",
            "error": "",
            "http": 200,
            "duration_ms": 7,
            "model": model or settings.model,
        }

    monkeypatch.setattr(whisper, "transcribe", transcribe)
    return calls


@pytest.fixture
def fake_model(monkeypatch):
    """Replace the dictation model client; returns the list of captured payloads."""

    payloads: list[dict] = []
    replies: list[str] = ["Hallo welt"]

    async def chat_completion(settings, payload, *, timeout=None):
        payloads.append(dict(payload))
        text = replies[min(len(payloads) - 1, len(replies) - 1)]
        return {
            "ok": True,
            "text": text,
            "model": payload.get("model", ""),
            "latency_ms": 3,
            "usage": {"prompt": 1, "completion": 1, "total": 2},
            "raw": {"choices": [{"message": {"content": text}}]},
        }

    monkeypatch.setattr(pipeline, "chat_completion", chat_completion)
    return {"payloads": payloads, "replies": replies}


# ── liveness and auth ─────────────────────────────────────────────────────────


def test_health_needs_no_token(client, monkeypatch):
    monkeypatch.setenv("SPEECHINT_TOKEN", "geheim")
    config.reset_settings()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert response.json()["service"] == "speechint"


def test_protected_route_rejects_missing_token(client, monkeypatch):
    monkeypatch.setenv("SPEECHINT_TOKEN", "geheim")
    config.reset_settings()

    response = client.get("/v1/config")

    assert response.status_code == 401
    assert response.json() == {
        "ok": False,
        "error": "unauthorized",
        "message": "Ungültiger oder fehlender Token.",
    }


def test_protected_route_rejects_wrong_token(client, monkeypatch):
    monkeypatch.setenv("SPEECHINT_TOKEN", "geheim")
    config.reset_settings()

    response = client.get("/v1/config", headers={"X-Auth-Token": "falsch"})

    assert response.status_code == 401


def test_protected_route_accepts_x_auth_token(client, monkeypatch):
    monkeypatch.setenv("SPEECHINT_TOKEN", "geheim")
    config.reset_settings()

    response = client.get("/v1/config", headers={"X-Auth-Token": "geheim"})

    assert response.status_code == 200


def test_protected_route_accepts_bearer_token(client, monkeypatch):
    monkeypatch.setenv("SPEECHINT_TOKEN", "geheim")
    config.reset_settings()

    response = client.get("/v1/config", headers={"Authorization": "Bearer geheim"})

    assert response.status_code == 200


def test_authentication_is_optional(client):
    assert client.get("/v1/config").status_code == 200


# ── discovery ─────────────────────────────────────────────────────────────────


def test_config_reports_capabilities_and_defaults(client):
    body = client.get("/v1/config").json()

    assert body["ok"] is True
    assert set(body["capabilities"]) == {"transcribe", "process", "dictate", "chat"}
    assert body["transcription"]["model"] == "small"
    assert body["transcription"]["language"] == "de"
    assert body["dictation"]["model"] == "Qwen3.5-2B Q4"
    assert body["limits"]["max_audio_mb"] == 25.0
    assert any(row["phrase"] == "neue zeile" for row in body["dictation"]["commands"])
    assert "Diktatprozessor" in body["dictation"]["default_prompt"]


def test_models_lists_both_services(client):
    body = client.get("/v1/models").json()

    kinds = {row["kind"] for row in body["data"]}
    assert kinds == {"transcription", "chat"}


def test_health_reports_components(client, models):
    models["llm"] = UNREACHABLE_LLM

    body = client.get("/v1/health").json()

    assert body["components"]["whisper"]["ok"] is True
    assert body["components"]["whisper"]["state"] == "ready"
    assert body["components"]["llm"]["ok"] is False
    assert body["components"]["llm"]["state"] == "starting"
    assert body["components"]["llm"]["state_label"] == "startet"
    assert body["ok"] is False
    assert body["ready"] is False
    assert body["status"] == "loading"


def test_health_is_ok_once_both_models_are_ready(client):
    response = client.get("/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["status"] == "ready"
    assert body["ready"] is True
    assert body["retry_after"] == 0
    assert body["message"].startswith("Bereit.")


def test_health_reports_unconfigured_components(client, models):
    models["llm"] = UNCONFIGURED_LLM

    body = client.get("/v1/health").json()

    assert body["components"]["llm"]["state"] == "unconfigured"
    assert body["components"]["llm"]["state_label"] == "nicht konfiguriert"
    assert body["status"] == "degraded"


def test_a_component_that_was_ready_is_unreachable_not_starting(client, models, monkeypatch):
    monkeypatch.setenv("SPEECHINT_HEALTH_CACHE_SECONDS", "0")
    config.reset_settings()

    assert client.get("/v1/ready").status_code == 200

    models["llm"] = UNREACHABLE_LLM

    body = client.get("/v1/health").json()

    assert body["components"]["llm"]["state"] == "unreachable"
    assert body["status"] == "degraded"


# ── loading feedback ──────────────────────────────────────────────────────────


def test_ready_reports_the_download_in_progress(client, models):
    models["whisper"] = LOADING_WHISPER
    models["llm"] = LOADING_LLM

    response = client.get("/v1/ready")

    assert response.status_code == 503
    assert response.headers["Retry-After"] == "15"

    body = response.json()
    assert body["ok"] is False
    assert body["status"] == "loading"
    assert body["ready"] is False
    assert body["retry_after"] == 15
    assert "Modelle werden geladen" in body["message"]
    assert body["components"]["whisper"]["state"] == "loading"
    assert body["components"]["whisper"]["state_label"] == "lädt Modell"
    assert body["components"]["llm"]["state"] == "loading"


def test_ready_names_the_component_that_is_still_starting(client, models):
    models["llm"] = UNREACHABLE_LLM

    response = client.get("/v1/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["components"]["whisper"]["state"] == "ready"
    assert body["components"]["llm"]["state"] == "starting"
    assert "Diktat-Modell (Qwen): startet" in body["message"]


def test_ready_needs_no_token(client, monkeypatch):
    monkeypatch.setenv("SPEECHINT_TOKEN", "geheim")
    config.reset_settings()

    assert client.get("/v1/ready").status_code == 200


def test_ready_does_not_probe_twice_within_the_cache_window(client, models):
    client.get("/v1/ready")
    client.get("/v1/ready")

    assert models["probes"] == 2  # one whisper + one llm probe


def test_transcription_is_rejected_while_whisper_loads(client, fake_whisper, models):
    models["whisper"] = LOADING_WHISPER

    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": AUDIO},
        data={"language": "de"},
    )

    assert response.status_code == 503
    assert response.headers["Retry-After"] == "15"
    assert response.json() == {
        "ok": False,
        "error": "service_loading",
        "component": "whisper",
        "message": "Spracherkennung (Whisper) lädt das Modell noch.",
        "retry_after": 15,
    }
    assert fake_whisper == []


def test_process_is_rejected_while_the_model_loads(client, fake_model, models):
    models["llm"] = LOADING_LLM

    response = client.post("/v1/dictate/process", json={"text": "hallo welt"})

    assert response.status_code == 503
    assert response.headers["Retry-After"] == "15"
    body = response.json()
    assert body["error"] == "service_loading"
    assert body["component"] == "llm"
    assert body["retry_after"] == 15
    assert fake_model["payloads"] == []


def test_chat_completions_is_rejected_while_the_model_loads(client, models):
    models["llm"] = LOADING_LLM

    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hallo"}]},
    )

    assert response.status_code == 503
    assert response.json()["component"] == "llm"


def test_dictate_reports_whisper_first(client, models):
    models["whisper"] = LOADING_WHISPER
    models["llm"] = LOADING_LLM

    response = client.post("/v1/dictate", files={"file": AUDIO})

    assert response.status_code == 503
    assert response.json()["component"] == "whisper"


def test_dictate_reports_the_model_without_transcribing(client, fake_whisper, models):
    models["llm"] = LOADING_LLM

    response = client.post("/v1/dictate", files={"file": AUDIO})

    assert response.status_code == 503
    assert response.json()["component"] == "llm"
    assert fake_whisper == []


def test_empty_audio_is_answered_without_waiting_for_whisper(client, models):
    models["whisper"] = LOADING_WHISPER

    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("leer.wav", b"", "audio/wav")},
    )

    assert response.status_code == 200
    assert response.json()["text"] == ""


def test_dictation_falls_back_while_the_model_is_unreachable(client, models, monkeypatch):
    monkeypatch.setenv("SPEECHINT_HEALTH_CACHE_SECONDS", "0")
    config.reset_settings()

    # Seen as ready once, so the later failure is reported as unreachable and
    # not as a still-starting model: the fallback must keep working.
    assert client.get("/v1/ready").status_code == 200
    models["llm"] = UNREACHABLE_LLM

    async def failing(settings, payload, *, timeout=None):
        raise errors.upstream_unavailable("llm", "Connection refused")

    monkeypatch.setattr(pipeline, "chat_completion", failing)

    response = client.post("/v1/dictate/process", json={"fragment": "hallo welt punkt"})

    assert response.status_code == 200
    assert response.json()["fallback"] is True


# ── transcription ─────────────────────────────────────────────────────────────


def test_transcribe_returns_text(client, fake_whisper):
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": AUDIO},
        data={"language": "de"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["text"] == "hallo welt"
    assert body["empty"] is False
    assert body["bytes"] == len(AUDIO[1])
    assert body["language"] == "de"
    assert fake_whisper[0]["filename"] == "segment.wav"
    assert fake_whisper[0]["language"] == "de"


def test_transcribe_defaults_to_configured_language(client, fake_whisper):
    client.post("/v1/audio/transcriptions", files={"file": AUDIO})

    assert fake_whisper[0]["language"] == "de"


def test_transcribe_short_circuits_empty_audio(client, fake_whisper):
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("leer.wav", b"", "audio/wav")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "text": "",
        "empty": True,
        "duration_ms": 0,
        "bytes": 0,
    }
    assert fake_whisper == []


def test_transcribe_supports_plain_text_response(client, fake_whisper):
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": AUDIO},
        data={"response_format": "text"},
    )

    assert response.status_code == 200
    assert response.text == "hallo welt"
    assert response.headers["content-type"].startswith("text/plain")


def test_transcribe_rejects_oversized_audio(client, monkeypatch):
    monkeypatch.setenv("SPEECHINT_MAX_AUDIO_MB", "1")
    config.reset_settings()

    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("gross.wav", b"x" * (1024 * 1024 + 1), "audio/wav")},
    )

    assert response.status_code == 413
    assert response.json()["error"] == "payload_too_large"


def test_transcribe_reports_upstream_failure(client, monkeypatch):
    async def transcribe(*args, **kwargs):
        raise errors.upstream_unavailable("transcription_failed", "Whisper nicht erreichbar: kaputt")

    monkeypatch.setattr(whisper, "transcribe", transcribe)

    response = client.post("/v1/audio/transcriptions", files={"file": AUDIO})

    assert response.status_code == 502
    assert response.json() == {
        "ok": False,
        "error": "transcription_failed",
        "message": "Whisper nicht erreichbar: kaputt",
    }


def test_transcribe_sanitises_the_filename(client, fake_whisper):
    client.post(
        "/v1/audio/transcriptions",
        files={"file": ("../../etc/pass wd.wav", b"audio", "audio/wav")},
    )

    assert fake_whisper[0]["filename"] == "pass_wd.wav"


# ── dictation processing ──────────────────────────────────────────────────────


def test_process_returns_model_text(client, fake_model):
    response = client.post("/v1/dictate/process", json={"fragment": "hallo welt"})

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["text"] == "Hallo welt"
    assert body["fallback"] is False
    assert body["model"] == "Qwen3.5-2B Q4"


def test_process_disables_thinking(client, fake_model):
    client.post("/v1/dictate/process", json={"fragment": "hallo welt"})

    payload = fake_model["payloads"][0]
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["reasoning_budget"] == 0
    assert payload["stream"] is False
    assert payload["temperature"] == 0.1
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][1]["role"] == "user"


def test_process_returns_empty_text_for_empty_fragment(client, fake_model):
    response = client.post("/v1/dictate/process", json={"fragment": "   "})

    assert response.json()["text"] == ""
    assert fake_model["payloads"] == []


def test_process_passes_client_prompt_and_model_settings(client, fake_model):
    client.post(
        "/v1/dictate/process",
        json={
            "fragment": "hallo stern",
            "prompt": "EIGENER PROMPT",
            "commands": [{"phrase": "stern", "type": "insert", "value": "*"}],
            "model": "eigenes-modell",
            "temperature": 0.5,
        },
    )

    payload = fake_model["payloads"][0]
    assert payload["model"] == "eigenes-modell"
    assert payload["temperature"] == 0.5
    # A client-supplied prompt is used verbatim: it already contains the
    # client's own command list, so nothing is appended to it.
    assert payload["messages"][0]["content"] == "EIGENER PROMPT"


def test_process_uses_the_client_command_table(client, fake_model):
    fake_model["replies"][:] = ["Hallo welt", "Und weiter"]

    response = client.post(
        "/v1/dictate/process",
        json={
            "fragment": "hallo welt absatz und weiter",
            "commands": [
                {"phrase": "absatz", "type": "paragraph", "value": ""},
                {"phrase": "stern", "type": "insert", "value": "*"},
            ],
        },
    )

    assert response.json()["text"] == "Hallo welt\n\nund weiter"
    # The generated default prompt describes the client's table, not ours.
    system = fake_model["payloads"][0]["messages"][0]["content"]
    assert '"stern"→"*"' in system
    assert '"punkt"' not in system


def test_process_splits_at_break_commands_and_fixes_casing(client, fake_model):
    fake_model["replies"][:] = ["Hallo welt", "Ich wollte fragen"]

    response = client.post(
        "/v1/dictate/process",
        json={"fragment": "hallo welt neue zeile ich wollte fragen"},
    )

    assert response.json()["text"] == "Hallo welt\nich wollte fragen"
    assert len(fake_model["payloads"]) == 2
    # The second call continues the same dictation and sees the first part.
    assert "Hallo welt" in fake_model["payloads"][1]["messages"][0]["content"]


def test_process_uses_the_previous_context(client, fake_model):
    client.post(
        "/v1/dictate/process",
        json={"fragment": "hallo welt", "context": "früher gesagter text"},
    )

    system = fake_model["payloads"][0]["messages"][0]["content"]
    assert "früher gesagter text" in system
    assert "nicht wiederholen" in system


def test_process_drops_an_invented_sentence_end(client, fake_model):
    fake_model["replies"][:] = ["Hallo welt."]

    response = client.post("/v1/dictate/process", json={"fragment": "hallo welt"})

    assert response.json()["text"] == "Hallo welt"


def test_process_falls_back_when_the_model_is_unavailable(client, monkeypatch):
    async def chat_completion(*args, **kwargs):
        raise errors.upstream_unavailable("llm_unavailable", "Modellserver nicht erreichbar")

    monkeypatch.setattr(pipeline, "chat_completion", chat_completion)

    response = client.post("/v1/dictate/process", json={"fragment": "hallo welt punkt"})

    body = response.json()
    assert response.status_code == 200
    assert body["fallback"] is True
    assert body["text"] == "hallo welt."
    assert "nicht verfügbar" in body["warning"]


def test_process_falls_back_on_an_empty_model_answer(client, fake_model):
    fake_model["replies"][:] = [""]

    body = client.post("/v1/dictate/process", json={"fragment": "hallo welt"}).json()

    assert body["fallback"] is True
    assert body["text"] == "hallo welt"


def test_process_ignores_an_empty_fragment_command_table(client, fake_model):
    body = client.post(
        "/v1/dictate/process",
        json={"fragment": "hallo welt", "commands": []},
    ).json()

    assert body["text"] == "Hallo welt"
    # An unusable table falls back to the service default, not to "no commands".
    assert '"punkt"→"."' in fake_model["payloads"][0]["messages"][0]["content"]


# ── chat completions passthrough ──────────────────────────────────────────────


def test_chat_completions_forwards_and_returns_the_raw_answer(client, monkeypatch):
    seen: list[dict] = []

    async def chat_completion(settings, payload, *, timeout=None):
        seen.append(dict(payload))
        return {"ok": True, "raw": {"choices": [{"message": {"content": "Hallo"}}]}}

    import app.main as main_module

    monkeypatch.setattr(main_module, "chat_completion", chat_completion)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "Qwen3.5-2B Q4", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "Hallo"
    assert seen[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert seen[0]["reasoning_budget"] == 0
    assert seen[0]["stream"] is False


def test_chat_completions_keeps_client_thinking_settings(client, monkeypatch):
    seen: list[dict] = []

    async def chat_completion(settings, payload, *, timeout=None):
        seen.append(dict(payload))
        return {"ok": True, "raw": {}}

    import app.main as main_module

    monkeypatch.setattr(main_module, "chat_completion", chat_completion)

    client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "chat_template_kwargs": {"enable_thinking": True},
        },
    )

    assert seen[0]["chat_template_kwargs"] == {"enable_thinking": True}


def test_chat_completions_rejects_a_body_without_messages(client):
    response = client.post("/v1/chat/completions", json={"model": "x"})

    assert response.status_code == 400
    assert response.json()["error"] == "bad_request"


def test_chat_completions_rejects_invalid_json(client):
    response = client.post(
        "/v1/chat/completions",
        content=b"{kein json",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 400
    assert response.json()["error"] == "bad_request"


def test_validation_error_uses_the_standard_envelope(client):
    response = client.post("/v1/dictate/process", json={"fragment": 5})

    assert response.status_code == 400
    body = response.json()
    assert body["ok"] is False
    assert body["error"] == "bad_request"


# ── one-shot dictation ────────────────────────────────────────────────────────


def test_dictate_combines_transcription_and_processing(client, fake_whisper, fake_model):
    response = client.post(
        "/v1/dictate",
        files={"file": AUDIO},
        data={"context": "vorher"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["transcript"] == "hallo welt"
    assert body["text"] == "Hallo welt"
    assert body["fallback"] is False
    assert body["whisper_model"] == "small"
    assert body["transcribe_ms"] == 7


def test_dictate_skips_whisper_for_empty_audio(client, fake_whisper, fake_model):
    response = client.post("/v1/dictate", files={"file": ("leer.wav", b"", "audio/wav")})

    body = response.json()
    assert body["transcript"] == ""
    assert body["text"] == ""
    assert body["transcribe_ms"] == 0
    assert fake_whisper == []
    assert fake_model["payloads"] == []


def test_dictate_accepts_a_json_command_table(client, fake_whisper, fake_model):
    import json

    response = client.post(
        "/v1/dictate",
        files={"file": AUDIO},
        data={"commands": json.dumps([{"phrase": "stern", "type": "insert", "value": "*"}])},
    )

    assert response.status_code == 200
    assert '"stern"→"*"' in fake_model["payloads"][0]["messages"][0]["content"]


def test_root_lists_the_endpoints(client):
    body = client.get("/").json()

    assert body["ok"] is True
    assert "/v1/dictate" in body["endpoints"]
