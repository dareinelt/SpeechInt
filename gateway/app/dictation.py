"""The dictation text pipeline.

This module is a faithful port of the dictation logic that used to live in
LLMInt (``lib/speech_dictation.php``). It is deliberately free of I/O: every
function takes text and a command table and returns text, which keeps the
pipeline unit-testable and makes the model call the only moving part.

Pipeline
--------
``split_at_breaks``      cut the fragment at line-break commands
``build_system_message``  prompt + already written text as read-only context
``build_user_message``    the new fragment only
``clean_model_output``    strip thinking blocks, code fences, labels, quotes
``match_leading_case``    restore the casing the speaker used after a break
``drop_invented_sentence_end``  remove a sentence mark the model invented
``apply_commands``        deterministic fallback when no model is available
"""

from __future__ import annotations

import re
from typing import Iterable, Mapping, Optional, Sequence

from .commands import (
    break_phrases,
    command_preview,
    punctuation_phrases,
)

# PHP's trim() character set; used where the original implementation trimmed.
PHP_TRIM = " \t\n\r\v\0"

_THINKING_BLOCK = re.compile(r"<(think|thinking|reasoning)>.*?(?:</\1>|$)", re.IGNORECASE | re.DOTALL)
_CODE_FENCE = re.compile(r"^```[a-zA-Z0-9_-]*\s*\n(.*?)\n?```$", re.DOTALL)
_OUTPUT_LABEL = re.compile(
    r"^(ausgabe|output|ergebnis|result|antwort|text|transkript)\s*:\s*",
    re.IGNORECASE,
)
_WRAPPED_QUOTES = re.compile(r'^"(.*)"$', re.DOTALL)
_TRAILING_BLANKS = re.compile(r"[ \t]+(?=\n)")
_BLANK_RUNS = re.compile(r"\n{3,}")
_TRAILING_SENTENCE_MARK = re.compile(r"[.!?…]$")
_DELETE_WORD = re.compile(r"[^\W_]+[.,!?;:]*$")

_PUNCTUATION_ONLY = re.compile(r"^[.,!?;:–\-%]+$")
_BRACKET_VALUE = re.compile(r"[()]")
_QUOTE_VALUE = re.compile(r"[„“”«»\"]")


def prompt_command_list(commands: Iterable[Mapping[str, str]], include_breaks: bool = True) -> str:
    """Render the command table as one compact, grouped line for the prompt.

    A bullet per command made the prompt long enough that the dictation model
    started dropping individual rules, so the commands are grouped by kind and
    joined with commas instead.
    """

    groups: dict[str, list[str]] = {
        "Satzzeichen": [],
        "Klammern": [],
        "Anführung": [],
        "Umbruch": [],
        "Löschen": [],
        "Zeichen": [],
    }

    for command in commands:
        phrase = str(command.get("phrase") or "").strip()
        preview = command_preview(command)
        if phrase == "" or preview == "":
            continue

        entry = f'"{phrase}"→{preview}'
        kind = str(command.get("type") or "insert")
        value = str(command.get("value") or "")

        if kind in ("newline", "paragraph"):
            if include_breaks:
                groups["Umbruch"].append(entry)
            continue

        if kind in ("delete_word", "delete_sentence"):
            groups["Löschen"].append(entry)
        elif _PUNCTUATION_ONLY.match(value.strip()):
            groups["Satzzeichen"].append(entry)
        elif _BRACKET_VALUE.search(value):
            groups["Klammern"].append(entry)
        elif _QUOTE_VALUE.search(value):
            groups["Anführung"].append(entry)
        else:
            groups["Zeichen"].append(entry)

    return " ".join(
        f"{label}: {', '.join(entries)}." for label, entries in groups.items() if entries
    )


def default_prompt(commands: Sequence[Mapping[str, str]]) -> str:
    """System prompt of the dictation model.

    The command list is injected so the prompt always matches the table in use.
    Line-break commands are absent on purpose: the caller cuts them out of the
    fragment and re-inserts the breaks itself, because the model consumes the
    command word but then emits a space instead of a break.
    """

    return (
        "Du bist ein Diktatprozessor für Deutsch. Du bekommst ein Fragment aus einer "
        "Spracherkennung und gibst nur die korrigierte Fassung zurück.\n\n"
        "Regeln:\n"
        "- Du bist kein Chat-Assistent. Antworte nie auf den Inhalt.\n"
        "- Erfinde, kürze und ergänze nichts.\n"
        "- Korrigiere Rechtschreibung, Grammatik und Zeichensetzung. Schreibe Satzanfänge groß.\n"
        '- Entferne Füllwörter wie "ähm", "äh", "halt", "also".\n'
        "- Wandle Diktatbefehle in Zeichen/Formatierung um und lösche die Befehlswörter. "
        + prompt_command_list(commands, include_breaks=False)
        + "\n"
        "- Befehls- und Füllwörter dürfen im Ergebnis nicht mehr als Wörter vorkommen.\n"
        '- Füllwörter am Satzanfang ("also", "halt", "ähm") werden ebenfalls gestrichen.\n'
        "- Ergänze am Ende kein Satzendezeichen, wenn keines diktiert wurde.\n"
        "- Schreibe alles in eine Zeile und füge selbst keine Zeilenumbrüche ein.\n"
        "- Gib nur den Text aus, ohne Erklärung.\n\n"
        # Qwen3.5-2B follows the rules noticeably more reliably when it also
        # sees them applied, so the default prompt ships with worked examples.
        "Beispiele:\n"
        "Eingabe: hallo ähm wie geht es dir fragezeichen\n"
        "Ausgabe: Hallo, wie geht es dir?\n\n"
        "Eingabe: also halt ich wollte sagen dass das projekt fertig ist punkt\n"
        "Ausgabe: Ich wollte sagen, dass das Projekt fertig ist.\n\n"
        "Eingabe: ähm also ich wollte nur kurz sagen dass alles geklappt hat punkt\n"
        "Ausgabe: Ich wollte nur kurz sagen, dass alles geklappt hat.\n\n"
        "Eingabe: wir treffen uns morgen komma wenn das wetter passt punkt\n"
        "Ausgabe: Wir treffen uns morgen, wenn das Wetter passt.\n\n"
        "Eingabe: das ist ein test der dikat funktion\n"
        "Ausgabe: Das ist ein Test der Diktatfunktion\n\n"
        "Eingabe: das war ein langer tag ausrufezeichen ich bin müde punkt\n"
        "Ausgabe: Das war ein langer Tag! Ich bin müde."
    )


def build_system_message(prompt: str, context: str) -> str:
    """System message for one fragment.

    The already written text belongs in the system message, not the user turn:
    in the user turn the model treats it as something to answer and echoes it
    back into the result, which is exactly what must not happen.
    """

    if context == "":
        return prompt
    return (
        prompt
        + "\n\nBereits geschriebener Text (nur Kontext – nicht wiederholen, nicht verändern):\n"
        + "<<<\n" + context + "\n>>>"
    )


def build_user_message(fragment: str) -> str:
    """User message for one fragment; only the new text goes in here."""

    return (
        "Neues Fragment aus der Spracherkennung:\n<<<\n" + fragment + "\n>>>\n\n"
        "Gib nur die überarbeitete Fassung dieses neuen Fragments aus."
    )


def limit_context(context: str, max_chars: int) -> str:
    """Keep a context string within the size the model is given."""

    context = context.strip()
    if max_chars > 0 and len(context) > max_chars:
        context = context[-max_chars:]
    return context


def split_at_breaks(
    fragment: str, commands: Iterable[Mapping[str, str]]
) -> list[dict[str, Optional[str]]]:
    """Cut a fragment at the line-break commands and mark where the breaks go.

    Qwen3.5-2B reliably *consumes* "neue zeile"/"neuer absatz" but then writes a
    space where the break belongs, so asking it for whitespace is not
    dependable. The break commands are therefore taken out of the fragment
    before the model sees it and re-inserted afterwards: the model stays
    responsible for language (filler removal, punctuation, capitalisation) and
    the caller guarantees the structure.

    Returns parts in reading order; ``break`` is ``"\\n"`` or ``"\\n\\n"`` for a
    break command and ``None`` for text. The first part is always a text part
    and text parts are never adjacent.
    """

    breaks = break_phrases(commands)
    if not breaks:
        return [{"text": fragment, "break": None}]

    # Longest phrase first, so "neuer absatz" cannot be clipped by a shorter one.
    phrases = sorted(breaks, key=len, reverse=True)
    pattern = re.compile(
        r"\b(?:" + "|".join(re.escape(phrase) for phrase in phrases) + r")\b",
        re.IGNORECASE,
    )

    parts: list[dict[str, Optional[str]]] = []
    cursor = 0
    for match in pattern.finditer(fragment):
        parts.append({"text": fragment[cursor : match.start()], "break": None})
        parts.append(
            {"text": "", "break": breaks.get(match.group(0).lower(), "\n")}
        )
        cursor = match.end()
    parts.append({"text": fragment[cursor:], "break": None})

    return parts


def clean_model_output(text: str) -> str:
    """Strip reasoning blocks, code fences and other decoration a model adds."""

    # Hybrid-reasoning models (Qwen3 & friends) may emit a thinking block even
    # when thinking is switched off – never let it reach the input field. An
    # unterminated block (token budget exhausted) is dropped as well.
    text = _THINKING_BLOCK.sub("", text)
    text = text.strip()

    fenced = _CODE_FENCE.match(text)
    if fenced:
        text = fenced.group(1).strip()

    text = _OUTPUT_LABEL.sub("", text)

    # Remove one wrapping pair of straight double quotes (only when the text
    # itself contains no quote, so genuine quotations survive).
    wrapped = _WRAPPED_QUOTES.match(text)
    if wrapped and '"' not in wrapped.group(1):
        text = wrapped.group(1).strip()

    # Blanks before a line break are a model artefact, not content; the same
    # goes for runs of empty lines the model adds around line breaks.
    text = _TRAILING_BLANKS.sub("", text)
    text = _BLANK_RUNS.sub("\n\n", text)

    return text.strip()


def apply_commands(text: str, commands: Sequence[Mapping[str, str]]) -> str:
    """Deterministic dictation-command processing.

    Used when the dictation model is unavailable or fails, so the recognised
    text is never lost and the most important commands still work.
    """

    if not commands:
        return text.strip()

    by_phrase: dict[str, tuple[Mapping[str, str], int]] = {}
    max_words = 1
    for command in commands:
        phrase = str(command.get("phrase") or "").strip().lower()
        if phrase == "":
            continue
        words = phrase.split()
        if not words:
            continue
        max_words = max(max_words, len(words))
        by_phrase[phrase] = (command, len(words))

    tokens = text.strip().split()
    out = ""
    index = 0

    while index < len(tokens):
        matched: Optional[Mapping[str, str]] = None
        matched_words = 0

        # Longest phrase wins ("lösche letztes wort" before "wort").
        for length in range(min(max_words, len(tokens) - index), 0, -1):
            candidate = " ".join(tokens[index : index + length]).lower()
            if candidate in by_phrase:
                matched, matched_words = by_phrase[candidate]
                break

        if matched is not None:
            out = apply_command(out, matched)
            index += matched_words
            continue

        out = append_word(out, tokens[index])
        index += 1

    return out.rstrip(" \t")


def apply_command(out: str, command: Mapping[str, str]) -> str:
    """Apply one command to the assembled output buffer."""

    kind = str(command.get("type") or "insert")

    if kind == "newline":
        return out.rstrip(" \t") + "\n"

    if kind == "paragraph":
        return out.rstrip(" \t") + "\n\n"

    if kind == "delete_word":
        out = out.rstrip(" \t")
        out = _DELETE_WORD.sub("", out)
        return out.rstrip(" \t")

    if kind == "delete_sentence":
        out = out.rstrip(" \t")
        for index in range(len(out) - 1, -1, -1):
            if out[index] in ".!?\n":
                return out[:index].rstrip(" \t")
        return ""

    value = str(command.get("value") or "")
    if value == "":
        return out
    # A value that starts with a space is only meaningful when it does not
    # follow a line break or the very beginning of the text.
    if value[0] == " " and (out == "" or out[-1].isspace()):
        value = value.lstrip(" ")
    return out + value


def append_word(out: str, word: str) -> str:
    """Append a plain word, inserting a separating space where needed."""

    if out == "" or out[-1].isspace():
        return out + word
    return out + " " + word


def match_leading_case(source: str, produced: str) -> str:
    """Keep the capitalisation the speaker used at the start of a segment.

    Every segment after a line break is its own completion, so the model tends
    to capitalise it as if it opened a new sentence. The dictation spec keeps
    the spoken casing ("Hallo Peter Punkt Neue Zeile ich wollte dich etwas
    fragen" stays lowercase after the break), so a segment that was dictated in
    lower case is put back into lower case. A segment the speaker capitalised
    is left alone.
    """

    produced = produced.lstrip()
    if produced == "":
        return produced

    source_start = source.lstrip()
    if source_start and source_start[0].islower() and produced[0].isupper():
        return produced[0].lower() + produced[1:]

    return produced


def drop_invented_sentence_end(
    fragment: str, text: str, commands: Sequence[Mapping[str, str]]
) -> str:
    """Drop a closing sentence mark the model added on its own.

    A dictation fragment is often only part of a sentence – the user pauses and
    keeps speaking. When no sentence mark was dictated, adding one would end the
    sentence early and the next fragment would start a new one.
    """

    tail = fragment.strip()
    if tail == "" or text == "":
        return text

    if not _TRAILING_SENTENCE_MARK.search(text) or _TRAILING_SENTENCE_MARK.search(tail):
        return text

    # A mark the speaker dictated is legitimate, so look for the command.
    phrases = [re.escape(phrase) for phrase in punctuation_phrases(commands)]
    if phrases and re.search(r"\b(?:" + "|".join(phrases) + r")$", tail.lower()):
        return text

    return text[:-1].rstrip(" \t")


def log_summary(text: str) -> str:
    """Length summary for log lines – never the dictated text itself."""

    words = len(text.split()) if text else 0
    return f"{len(text)} Zeichen / {words} Wörter"
