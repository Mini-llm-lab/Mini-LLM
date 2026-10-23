from mini_llm.prompt import DEFAULT_SYSTEM_PROMPT, build_system_prompt, format_history, format_prompt


def test_format_prompt_trims_question():
    assert format_prompt(" hi ") == "User: hi\nAssistant:"


def test_format_prompt_with_system():
    assert format_prompt(" hi ", system=" S ") == "System: S\nUser: hi\nAssistant:"
    assert format_prompt(" hi ", system=None) == "User: hi\nAssistant:"


def test_build_system_prompt():
    assert build_system_prompt(DEFAULT_SYSTEM_PROMPT, date="Oct 23, 2026").startswith("Today is Oct 23, 2026.")
    assert build_system_prompt(" S ", date=None) == "S"


def test_format_history():
    h = format_history("S", [("q1", "a1"), ("q2", None)])
    assert h == "System: S\nUser: q1\nAssistant: a1\nUser: q2\nAssistant:"
    assert format_history(None, [("q1", "a1"), ("q2", None)]) == "User: q1\nAssistant: a1\nUser: q2\nAssistant:"


def test_format_history_rejects_bad_shapes():
    import pytest

    with pytest.raises(ValueError):
        format_history(None, [])
    with pytest.raises(ValueError):
        format_history(None, [("q1", "a1"), ("q2", "a2")])
    with pytest.raises(ValueError):
        format_history(None, [("q1", None), ("q2", None)])
    assert format_history(None, [("q1", None)]) == "User: q1\nAssistant:"
