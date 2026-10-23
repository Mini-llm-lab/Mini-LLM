import re

from mini_llm import math_data

PLAIN = re.compile(r"^([0-9]+) ([+\-x/]) ([0-9]+) = ([0-9]+)\.$")
CARRIED = re.compile(r"^([0-9]+) ([+\-]) ([0-9]+) = [^,]+, then [0-9]+ [+\-] [0-9]+ = ([0-9]+)\.$")


def check_answer(question, answer):
    got = None
    for pattern in (PLAIN, CARRIED):
        m = pattern.match(answer)
        if not m:
            continue
        groups = m.groups()
        a, sign, b, result = int(groups[0]), groups[1], int(groups[2]), int(groups[3])
        expected = a + b if sign == "+" else a - b if sign == "-" else a * b if sign in ("x", "*") else a // b
        assert result == expected, (question, answer)
        got = result
    assert got is not None, (question, answer)


def test_table_pairs_are_exhaustive_and_correct():
    pairs = math_data.table_pairs(0)
    assert len(pairs) == 144 + 132  # tables 1-12 x 1-12, plus inverse divisions
    for question, answer in pairs:
        check_answer(question, answer)


def test_synthetic_pairs_are_correct_and_deterministic():
    pairs = math_data.synthetic_pairs(300, seed=5)
    assert len(pairs) == 300
    assert pairs == math_data.synthetic_pairs(300, seed=5)
    for question, answer in pairs:
        check_answer(question, answer)


def test_skill_pairs_shape():
    pairs = math_data.skill_pairs(50, seed=2)
    assert len(pairs) == 50
    assert pairs == math_data.skill_pairs(50, seed=2)
    for question, answer in pairs:
        assert question and answer

    count = [a for q, a in pairs if a.endswith(" words.")]
    assert count, "expected word-count examples"

    for q, a in pairs:
        if a.endswith(" words."):
            words = re.search(r'"([^"]+)"', q).group(1).split()
            assert int(a.split()[0]) == len(words)


def test_digit_pre_tokenizer_isolates_digits():
    from mini_llm.prep import digit_pre_tokenizer

    pieces = [t for t, _ in digit_pre_tokenizer().pre_tokenize_str("abc 1234 GPT4 3,472")]
    assert "1234" not in pieces and "472" not in pieces
    assert all(len(p) <= 1 or not p[-1].isdigit() or not p[0].isdigit() for p in pieces)
