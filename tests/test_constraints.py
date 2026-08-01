import time

import pytest

from app.constraints import compatible, extract

# One case per dimension: text_a/text_b share a base and differ only in
# the constraint under test, mirroring eval/build_dataset.py's P3 families.
DIMENSION_CASES = [
    ("count", "Answer in exactly 3 bullet points.", "Answer in exactly 10 bullet points."),
    ("negation", "Include examples from Python.", "Do not include anything related to Python."),
    ("language", "Answer in French.", "Answer in German."),
    ("entities", "Focus on Python.", "Focus on Go."),
    ("format", "Answer as a markdown table.", "Answer as a single prose paragraph."),
]


@pytest.mark.parametrize("dimension,text_a,text_b", DIMENSION_CASES)
def test_dimension_extracted_and_makes_pair_incompatible(dimension, text_a, text_b):
    a, b = extract(text_a), extract(text_b)
    # At least one side must extract something on this dimension, or the
    # case isn't exercising it at all. Both sides non-None isn't required
    # here — negation's pair is legitimately asymmetric (one side has no
    # negation cue at all), and that asymmetry is exactly what compatible()
    # is supposed to catch via None != <scope>.
    assert a[dimension] is not None or b[dimension] is not None, (
        f"{dimension} not extracted from either {text_a!r} or {text_b!r}"
    )
    assert a[dimension] != b[dimension]

    ok, diff = compatible(a, b)
    assert ok is False
    assert dimension in diff


def test_count_extracts_digits_and_spelled_out_numbers():
    assert extract("Answer in exactly 3 bullet points.")["count"] == 3
    assert extract("Answer in exactly three bullet points.")["count"] == 3
    assert extract("Give me 5 steps.")["count"] == 5


def test_negation_captures_cue_and_scope():
    result = extract("Do not include anything related to Python.")["negation"]
    assert result is not None
    assert "python" in result

    assert extract("Include examples from Python.")["negation"] is None


def test_language_lexicon():
    assert extract("Answer in French.")["language"] == "french"
    assert extract("Please respond in Spanish now.")["language"] == "spanish"
    assert extract("Tell me a story.")["language"] is None


def test_entities_set_equality_and_excludes_sentence_initial_token():
    # "Focus" is sentence-initial (grammar, not an entity); "Python" is.
    # Without excluding sentence-initial tokens, negation's own pair
    # ("Include..." vs "Do not include...") would register a spurious
    # entities diff alongside the real negation diff.
    assert extract("Focus on Python.")["entities"] == {"Python"}
    assert extract("Include examples from Python.")["entities"] == {"Python"}
    assert extract("Do not include anything related to Python.")["entities"] == {"Python"}


def test_entities_includes_numerals():
    assert "2019" in extract("Answer as of 2019.")["entities"]


def test_format_lexicon():
    assert extract("Answer as a markdown table.")["format"] == "table"
    assert extract("Give the answer as a bulleted list.")["format"] == "bullets"
    assert extract("Respond as a JSON object.")["format"] == "json"
    assert extract("Write it as a single prose paragraph.")["format"] == "prose"
    assert extract("Show me a code snippet.")["format"] == "code"


def test_compatible_both_none_on_every_dimension():
    a = extract("What is a tardigrade?")
    b = extract("Tell me about tardigrades please")
    ok, diff = compatible(a, b)
    assert ok is True
    assert diff == {}


def test_compatible_diff_reports_both_values_not_just_a_bool():
    a = extract("Answer in exactly 3 bullet points.")
    b = extract("Answer in exactly 10 bullet points.")
    ok, diff = compatible(a, b)
    assert ok is False
    assert diff["count"] == [3, 10]


def test_p3_control_reworded_pairs_stay_compatible():
    # Same constraint, trivially reworded — must stay compatible, this is
    # exactly what P3_control (safe_to_reuse=True) depends on.
    a = extract("Answer in exactly 3 bullet points.")
    b = extract("Respond using exactly 3 bullet points.")
    ok, diff = compatible(a, b)
    assert ok is True
    assert diff == {}


def test_extract_runs_under_1ms_average():
    text = (
        "Please answer in exactly 3 bullet points, in French, focused on "
        "Python and Go, without including anything about Rust."
    )
    n = 1000
    t0 = time.perf_counter()
    for _ in range(n):
        extract(text)
    elapsed_ms = (time.perf_counter() - t0) / n * 1000
    assert elapsed_ms < 1.0, f"extract() averaged {elapsed_ms:.4f}ms over {n} calls, wanted <1ms"
