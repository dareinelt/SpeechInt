"""Unit tests for the pure dictation text pipeline."""

from __future__ import annotations

import pytest

from app.commands import DEFAULT_COMMANDS, normalize_commands
from app.dictation import (
    apply_commands,
    build_system_message,
    build_user_message,
    clean_model_output,
    default_prompt,
    drop_invented_sentence_end,
    limit_context,
    match_leading_case,
    prompt_command_list,
    split_at_breaks,
)


def commands(*rows: dict[str, str]) -> list[dict[str, str]]:
    return [dict(row) for row in rows]


# ── prompt ────────────────────────────────────────────────────────────────────


def test_default_prompt_contains_examples_and_no_break_commands():
    prompt = default_prompt(DEFAULT_COMMANDS)
    assert "Diktatprozessor" in prompt
    assert "Ausgabe: Hallo, wie geht es dir?" in prompt
    # Break commands are handled by the caller, never by the model.
    assert "neue zeile" not in prompt
    assert "neuer absatz" not in prompt


def test_prompt_command_list_groups_by_kind():
    rendered = prompt_command_list(DEFAULT_COMMANDS, include_breaks=False)
    assert 'Satzzeichen: "punkt"→"."' in rendered
    assert "Umbruch" not in rendered

    with_breaks = prompt_command_list(DEFAULT_COMMANDS, include_breaks=True)
    assert 'Umbruch: "neue zeile"→Zeilenumbruch' in with_breaks
    assert '"neuer absatz"→Absatzumbruch (Leerzeile)' in with_breaks


def test_system_message_marks_context_as_read_only():
    assert build_system_message("PROMPT", "") == "PROMPT"
    with_context = build_system_message("PROMPT", "Hallo welt")
    assert with_context.startswith("PROMPT")
    assert "nicht wiederholen" in with_context
    assert "Hallo welt" in with_context


def test_user_message_wraps_only_the_fragment():
    message = build_user_message("hallo welt")
    assert "hallo welt" in message
    assert "Neues Fragment" in message


# ── context ───────────────────────────────────────────────────────────────────


def test_limit_context_keeps_the_tail():
    assert limit_context("  hallo  ", 100) == "hallo"
    assert limit_context("abcdefghij", 4) == "ghij"
    assert limit_context("abcdefghij", 0) == "abcdefghij"


# ── break splitting ───────────────────────────────────────────────────────────


def test_split_at_breaks_without_commands_returns_single_part():
    assert split_at_breaks("hallo welt", []) == [{"text": "hallo welt", "break": None}]


def test_split_at_breaks_splits_and_marks_breaks():
    parts = split_at_breaks("hallo welt neue zeile wie gehts", DEFAULT_COMMANDS)
    assert [part["break"] for part in parts] == [None, "\n", None]
    assert parts[0]["text"] == "hallo welt "
    assert parts[2]["text"] == " wie gehts"


def test_split_at_breaks_prefers_the_longer_phrase():
    parts = split_at_breaks("ende neuer absatz weiter", DEFAULT_COMMANDS)
    assert parts[1]["break"] == "\n\n"
    # "neuer absatz" must not be clipped into "neue zeile"-style leftovers.
    assert parts[0]["text"] == "ende "
    assert parts[2]["text"] == " weiter"


def test_split_at_breaks_is_case_insensitive():
    parts = split_at_breaks("hallo Neue Zeile weiter", DEFAULT_COMMANDS)
    assert parts[1]["break"] == "\n"


def test_split_at_breaks_accepts_client_specific_phrases():
    table = commands({"phrase": "absatz", "type": "paragraph", "value": ""})
    parts = split_at_breaks("hallo absatz welt", table)
    assert parts[1]["break"] == "\n\n"


# ── output cleaning ───────────────────────────────────────────────────────────


def test_clean_model_output_strips_thinking_blocks():
    # Built by concatenation so no literal tag appears in this file.
    raw = "<" + "think>interne überlegung<" + "/think>Hallo welt"
    assert clean_model_output(raw) == "Hallo welt"


def test_clean_model_output_strips_unterminated_thinking_block():
    assert clean_model_output("Hallo welt<thinking>abgeschnitten") == "Hallo welt"


def test_clean_model_output_strips_code_fence_and_label():
    assert clean_model_output("```\nHallo welt.\n```") == "Hallo welt."
    assert clean_model_output("Ausgabe: Hallo welt.") == "Hallo welt."


def test_clean_model_output_keeps_inner_quotes():
    assert clean_model_output('Er sagte "hallo" laut.') == 'Er sagte "hallo" laut.'
    assert clean_model_output('"Hallo welt"') == "Hallo welt"


def test_clean_model_output_collapses_blank_line_runs():
    assert clean_model_output("a  \n\n\n\nb") == "a\n\nb"


# ── commands (deterministic fallback) ─────────────────────────────────────────


def test_apply_commands_inserts_punctuation():
    assert apply_commands("hallo welt punkt", DEFAULT_COMMANDS) == "hallo welt."


def test_apply_commands_uses_the_longest_phrase():
    assert apply_commands("hallo welt lösche letztes wort", DEFAULT_COMMANDS) == "hallo"


def test_apply_commands_deletes_the_last_sentence():
    text = "hallo welt punkt ich bin da lösche letzten satz"
    assert apply_commands(text, DEFAULT_COMMANDS) == "hallo welt"


def test_apply_commands_inserts_line_break():
    assert apply_commands("hallo neue zeile welt", DEFAULT_COMMANDS) == "hallo\nwelt"


def test_apply_commands_inserts_blank_line():
    assert apply_commands("hallo neuer absatz welt", DEFAULT_COMMANDS) == "hallo\n\nwelt"


def test_apply_commands_keeps_space_before_opening_bracket():
    # "klammer auf" carries a leading space in its value, so the bracket is
    # separated from the preceding word; the following word gets its own space.
    assert apply_commands("hallo klammer auf welt klammer zu", DEFAULT_COMMANDS) == "hallo ( welt)"


def test_apply_commands_without_table_only_trims():
    assert apply_commands("  hallo welt  ", []) == "hallo welt"


def test_apply_commands_ignores_unknown_words():
    assert apply_commands("hallo welt", DEFAULT_COMMANDS) == "hallo welt"


# ── casing after a break ──────────────────────────────────────────────────────


def test_match_leading_case_restores_spoken_lower_case():
    assert match_leading_case("ich wollte fragen", "Ich wollte fragen") == "ich wollte fragen"


def test_match_leading_case_keeps_spoken_capitalisation():
    assert match_leading_case("Ich wollte fragen", "Ich wollte fragen") == "Ich wollte fragen"


def test_match_leading_case_handles_empty_output():
    assert match_leading_case("irgendwas", "   ") == ""


# ── invented sentence end ─────────────────────────────────────────────────────


def test_drop_invented_sentence_end_removes_model_mark():
    assert drop_invented_sentence_end("hallo welt", "Hallo welt.", DEFAULT_COMMANDS) == "Hallo welt"


def test_drop_invented_sentence_end_keeps_dictated_mark():
    text = "Hallo welt."
    assert drop_invented_sentence_end("hallo welt punkt", text, DEFAULT_COMMANDS) == text


def test_drop_invented_sentence_end_keeps_literal_mark():
    text = "Hallo welt!"
    assert drop_invented_sentence_end("hallo welt!", text, DEFAULT_COMMANDS) == text


def test_drop_invented_sentence_end_ignores_text_without_mark():
    assert drop_invented_sentence_end("hallo welt", "Hallo welt", DEFAULT_COMMANDS) == "Hallo welt"


# ── command normalisation ─────────────────────────────────────────────────────


def test_normalize_commands_defaults_when_missing():
    assert normalize_commands(None) == [dict(row) for row in DEFAULT_COMMANDS]
    assert normalize_commands([]) == [dict(row) for row in DEFAULT_COMMANDS]
    assert normalize_commands("kein json") == [dict(row) for row in DEFAULT_COMMANDS]


def test_normalize_commands_parses_json_string():
    raw = '[{"phrase": "stern", "type": "insert", "value": "*"}]'
    assert normalize_commands(raw) == [{"phrase": "stern", "type": "insert", "value": "*"}]


def test_normalize_commands_drops_unusable_rows():
    raw = [
        {"phrase": "  ", "type": "insert", "value": "."},
        "kaputt",
        {"type": "insert", "value": "."},
        {"phrase": "stern", "type": "erfunden", "value": "*"},
    ]
    assert normalize_commands(raw) == [{"phrase": "stern", "type": "insert", "value": "*"}]


def test_normalize_commands_accepts_wrapper_object():
    raw = {"commands": [{"phrase": "stern", "type": "insert", "value": "*"}]}
    assert normalize_commands(raw) == [{"phrase": "stern", "type": "insert", "value": "*"}]


@pytest.mark.parametrize("kind", ["newline", "paragraph", "delete_word", "delete_sentence"])
def test_normalize_commands_keeps_known_types(kind):
    raw = [{"phrase": "befehl", "type": kind, "value": ""}]
    assert normalize_commands(raw) == [{"phrase": "befehl", "type": kind, "value": ""}]
