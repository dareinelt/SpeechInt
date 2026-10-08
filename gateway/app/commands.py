"""The dictation command table.

A *command* is a spoken phrase that turns into punctuation, formatting or an
edit instead of appearing in the text ("punkt" → ".", "neue zeile" → line
break). The table is data, not code: clients (LLMInt) send their configured
table with every dictation request so an administrator can extend it without
touching this service. The table below is the default used when a request
carries none.

Command types
-------------
``insert``           append the literal ``value``
``newline``          insert a single line break
``paragraph``        insert a blank line (two breaks)
``delete_word``      remove the word before the cursor
``delete_sentence``  remove the sentence before the cursor
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping

COMMAND_TYPES = (
    "insert",
    "newline",
    "paragraph",
    "delete_word",
    "delete_sentence",
)

BREAK_TYPES = ("newline", "paragraph")

DEFAULT_COMMANDS: tuple[dict[str, str], ...] = (
    {"phrase": "punkt", "type": "insert", "value": "."},
    {"phrase": "komma", "type": "insert", "value": ","},
    {"phrase": "fragezeichen", "type": "insert", "value": "?"},
    {"phrase": "ausrufezeichen", "type": "insert", "value": "!"},
    {"phrase": "doppelpunkt", "type": "insert", "value": ":"},
    {"phrase": "semikolon", "type": "insert", "value": ";"},
    {"phrase": "gedankenstrich", "type": "insert", "value": " – "},
    {"phrase": "bindestrich", "type": "insert", "value": "-"},
    {"phrase": "prozent", "type": "insert", "value": "%"},
    {"phrase": "klammer auf", "type": "insert", "value": " ("},
    {"phrase": "klammer zu", "type": "insert", "value": ")"},
    {"phrase": "anführungszeichen auf", "type": "insert", "value": " „"},
    {"phrase": "anführungszeichen zu", "type": "insert", "value": '"'},
    {"phrase": "neue zeile", "type": "newline", "value": ""},
    {"phrase": "neuer absatz", "type": "paragraph", "value": ""},
    {"phrase": "lösche letztes wort", "type": "delete_word", "value": ""},
    {"phrase": "lösche letzten satz", "type": "delete_sentence", "value": ""},
)


def normalize_commands(raw: Any) -> list[dict[str, str]]:
    """Validate and normalise a client-supplied command table.

    Unknown entries are dropped rather than rejected: a stale client should
    never lose a dictation because it sent one unusable row. An empty or
    entirely unusable table falls back to :data:`DEFAULT_COMMANDS`.
    """

    if raw is None:
        return [dict(command) for command in DEFAULT_COMMANDS]

    if isinstance(raw, Mapping):
        # Accept {"commands": [...]}-style wrappers defensively.
        raw = raw.get("commands")

    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            return [dict(command) for command in DEFAULT_COMMANDS]

    if not isinstance(raw, Iterable) or isinstance(raw, (bytes, bytearray)):
        return [dict(command) for command in DEFAULT_COMMANDS]

    commands: list[dict[str, str]] = []
    for entry in raw:
        if not isinstance(entry, Mapping):
            continue
        phrase = str(entry.get("phrase") or "").strip()
        if phrase == "":
            continue
        kind = str(entry.get("type") or "insert")
        if kind not in COMMAND_TYPES:
            kind = "insert"
        commands.append(
            {
                "phrase": phrase,
                "type": kind,
                "value": str(entry.get("value") or ""),
            }
        )

    return commands or [dict(command) for command in DEFAULT_COMMANDS]


def command_preview(command: Mapping[str, str]) -> str:
    """Short human-readable description of what a command inserts or does."""

    kind = str(command.get("type") or "insert")
    if kind == "newline":
        return "Zeilenumbruch"
    if kind == "paragraph":
        return "Absatzumbruch (Leerzeile)"
    if kind == "delete_word":
        return "lösche das zuletzt gesprochene Wort"
    if kind == "delete_sentence":
        return "lösche den zuletzt gesprochenen Satz"
    value = str(command.get("value") or "")
    return f'"{value}"' if value else ""


def break_phrases(commands: Iterable[Mapping[str, str]]) -> dict[str, str]:
    """Map every break phrase (lower case) to the break it produces."""

    breaks: dict[str, str] = {}
    for command in commands:
        kind = str(command.get("type") or "")
        phrase = str(command.get("phrase") or "").strip().lower()
        if phrase == "" or kind not in BREAK_TYPES:
            continue
        breaks[phrase] = "\n\n" if kind == "paragraph" else "\n"
    return breaks


def punctuation_phrases(commands: Iterable[Mapping[str, str]]) -> list[str]:
    """Lower-case phrases of commands that insert a pure sentence mark.

    Used to decide whether a trailing "." in the model output was dictated or
    invented – see :func:`app.dictation.drop_invented_sentence_end`.
    """

    marks = {".", "!", "?", "…"}
    phrases: list[str] = []
    for command in commands:
        phrase = str(command.get("phrase") or "").strip().lower()
        value = str(command.get("value") or "").strip()
        if phrase != "" and value in marks:
            phrases.append(phrase)
    return phrases
