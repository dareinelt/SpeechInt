"""Uniform error model of the SpeechInt API.

Every failing request answers with the same JSON envelope::

    {"ok": false, "error": "<code>", "message": "<German text>"}

The codes are stable identifiers the client may branch on; the message is meant
for the administrator and is shown verbatim in the LLMInt admin area.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from fastapi import HTTPException


class SpeechIntError(HTTPException):
    """HTTPException carrying a machine-readable ``error`` code."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        headers: Optional[Mapping[str, str]] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        detail: dict[str, Any] = {"ok": False, "error": code, "message": message}
        if extra:
            detail.update(extra)
        super().__init__(status_code=status_code, detail=detail, headers=dict(headers or {}))
        self.code = code
        self.message = message


def unauthorized(message: str = "Ungültiger oder fehlender Token.") -> SpeechIntError:
    return SpeechIntError(401, "unauthorized", message)


def bad_request(message: str) -> SpeechIntError:
    return SpeechIntError(400, "bad_request", message)


def payload_too_large(message: str) -> SpeechIntError:
    return SpeechIntError(413, "payload_too_large", message)


def not_configured(message: str) -> SpeechIntError:
    return SpeechIntError(503, "not_configured", message)


def upstream_unavailable(code: str, message: str) -> SpeechIntError:
    return SpeechIntError(502, code, message)


def upstream_timeout(code: str, message: str) -> SpeechIntError:
    return SpeechIntError(504, code, message)


def service_loading(component: str, message: str, retry_after: int) -> SpeechIntError:
    """The request is valid but a model server is still downloading or loading.

    This is a *transient* state, not a failure: the client should keep its input
    and try again after ``Retry-After`` seconds instead of showing an error.
    """

    seconds = max(1, int(retry_after))
    return SpeechIntError(
        503,
        "service_loading",
        message,
        headers={"Retry-After": str(seconds)},
        extra={"component": component, "retry_after": seconds},
    )
