"""The dictation pipeline: transcript in, finished text out.

One fragment is processed in three steps:

1. line-break commands are cut out of the fragment and remembered;
2. every remaining text part is sent to the dictation model on its own, with
   the text produced so far as read-only context;
3. the breaks are re-inserted and the result is cleaned up.

The pipeline never fails hard for non-empty input. When the model is
unreachable the deterministic command processor takes over and the answer is
marked ``fallback: true``, so a recognised sentence is never lost.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Mapping, Optional, Sequence

from . import dictation
from .config import Settings
from .errors import SpeechIntError
from .llm import build_payload, chat_completion

log = logging.getLogger("speechint.pipeline")


def _max_tokens_for(fragment: str) -> int:
    return max(128, len(fragment) * 2 + 128)


def _fallback(fragment: str, commands: Sequence[Mapping[str, str]], model: str, reason: str) -> dict:
    return {
        "text": dictation.apply_commands(fragment, commands),
        "fallback": True,
        "model": model,
        "warning": f"Diktat-Modell nicht verfügbar – Rohtext übernommen ({reason})",
        "duration_ms": 0,
    }


async def process_fragment(
    fragment: str,
    context: str = "",
    *,
    settings: Settings,
    commands: Sequence[Mapping[str, str]],
    prompt: str,
    model: str = "",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> dict[str, Any]:
    """Turn one dictated fragment into finished text.

    Returns ``{"text", "fallback", "model", "warning", "duration_ms"}``.
    """

    fragment = fragment.strip()
    if fragment == "":
        return {"text": "", "fallback": False, "model": "", "warning": "", "duration_ms": 0}

    context = dictation.limit_context(context, settings.max_context_chars)
    effective_model = model or settings.llm.model
    started_at = time.monotonic()

    pieces: list[str] = []
    after_break = False

    for part in dictation.split_at_breaks(fragment, commands):
        if part["break"] is not None:
            pieces.append(str(part["break"]))
            after_break = True
            continue

        text = str(part["text"] or "").strip()
        if text == "":
            continue

        payload = build_payload(
            settings.llm,
            [
                {"role": "system", "content": dictation.build_system_message(prompt, context)},
                {"role": "user", "content": dictation.build_user_message(text)},
            ],
            model=effective_model,
            temperature=settings.default_temperature if temperature is None else temperature,
            max_tokens=max_tokens if max_tokens is not None else _max_tokens_for(text),
        )

        try:
            result = await chat_completion(settings.llm, payload)
            produced = dictation.clean_model_output(str(result["text"]))
        except SpeechIntError as exc:
            if exc.code == "service_loading":
                # A model that is still downloading is not a reason to degrade to
                # the rule-based output: the client should wait and retry.
                raise
            return _fallback(fragment, commands, effective_model, exc.message)

        if produced == "":
            return _fallback(
                fragment,
                commands,
                effective_model,
                "Das Diktat-Modell hat keinen Text geliefert.",
            )

        if after_break:
            produced = dictation.match_leading_case(text, produced)
        pieces.append(produced)

        # Each following segment continues the same dictation, so it sees what
        # has been produced so far – that keeps mid-sentence breaks lowercase.
        context = dictation.limit_context(f"{context} {produced}", settings.max_context_chars)

    text = dictation.clean_model_output("".join(pieces))
    text = dictation.drop_invented_sentence_end(fragment, text, commands)

    if text == "":
        return _fallback(
            fragment,
            commands,
            effective_model,
            "Das Diktat-Modell hat keinen Text geliefert.",
        )

    return {
        "text": text,
        "fallback": False,
        "model": effective_model,
        "warning": "",
        "duration_ms": int(round((time.monotonic() - started_at) * 1000)),
    }
