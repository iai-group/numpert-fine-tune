import pytest
from src.data_preprocessing.perturbations import approximation


def test_approximate_number_prefers_human_steps():
    assert approximation.approximate_number(1234) == pytest.approx(1200)
    assert approximation.approximate_number(5000) == pytest.approx(5000)
    assert approximation.approximate_number(7.2) == pytest.approx(7)


def test_extract_numeric_spans_includes_currency_and_exaggeration(monkeypatch):
    monkeypatch.setattr(approximation.random, "choice", lambda seq: seq[-1])
    text = "Revenue was $1,234 last year."
    tokens = ["Revenue", "was", "$", "1,234", "last", "year", "."]
    ner_tags = ["O", "O", "MONEY", "MONEY", "O", "O", "O"]

    spans = approximation.extract_numeric_spans(text, tokens, ner_tags)

    assert len(spans) == 1
    span = spans[0]
    assert span.char_start == text.index("$")
    assert span.original_value == 1234.0
    assert span.approx_true == pytest.approx(1200.0)
    assert span.approx_false == pytest.approx(2500.0)


def test_apply_numeric_spans_adds_about_and_formats():
    text = "Revenue hit 1,000 units last year."
    char_start = text.index("1,000")
    span = approximation.NumericSpan(
        token_start=2,
        token_end=3,
        char_start=char_start,
        char_end=char_start + len("1,000"),
        number_token_idx=2,
        original_number="1,000",
        original_value=1000.0,
        approx_true=1200.0,
        approx_false=2500.0,
    )

    updated = approximation.apply_numeric_spans(text, [span], True)

    assert updated == "Revenue hit about 1,200 units last year."


def test_apply_numeric_spans_respects_existing_about():
    text = "It was about 1,000 units."
    char_start = text.index("1,000")
    span = approximation.NumericSpan(
        token_start=3,
        token_end=4,
        char_start=char_start,
        char_end=char_start + len("1,000"),
        number_token_idx=3,
        original_number="1,000",
        original_value=1000.0,
        approx_true=1200.0,
        approx_false=2500.0,
    )

    updated = approximation.apply_numeric_spans(text, [span], True)

    assert updated == "It was about 1,200 units."


def test_perturb_claim_generates_true_and_false_variants(monkeypatch):
    monkeypatch.setattr(approximation.random, "choice", lambda seq: seq[-1])
    claim = "100 people attended the rally."
    tokens = ["100", "people", "attended", "the", "rally", "."]
    ner_tags = ["CARDINAL", "O", "O", "O", "O", "O"]

    true_claim, false_claim = approximation.perturb_claim(
        claim, tokens, ner_tags
    )

    assert true_claim == "About 100 people attended the rally."
    assert false_claim == "About 200 people attended the rally."
