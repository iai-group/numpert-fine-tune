"""Tests for label_next (feedback) mode and domain-anchored keyword rules."""

from src.data_preprocessing.raw_dataset_preprocessing.verdict_strip.stripper import (  # noqa: E501
    strip,
)


# --- label_next: feedback DETAILS <rating>: <explanation> ---

def test_label_next_removes_label_and_following_sentence():
    doc = (
        "Header context. DETAILS Misleading: The presence of a short "
        "sequence is not proof. Later evidence sentence stays."
    )
    res = strip(doc, "healthfeedback.org")
    assert "DETAILS Misleading:" not in res.cleaned
    assert "is not proof" not in res.cleaned          # explanation removed
    assert "Header context." in res.cleaned
    assert "Later evidence sentence stays." in res.cleaned
    assert res.removals[0].mode == "label_next"


def test_label_next_bare_label_without_details_prefix():
    doc = "Intro. Flawed reasoning: The study was misread. Real finding here."
    res = strip(doc, "sciencefeedback.co")
    assert "Flawed reasoning:" not in res.cleaned
    assert "study was misread" not in res.cleaned
    assert "Real finding here." in res.cleaned


# --- keyword rules: domain-anchored debunked / similar false claim ---

def test_debunked_sentence_stripped_on_anchored_domain():
    doc = "We have debunked this exact claim before. The photo is from 2017."
    res = strip(doc, "factcheck.afp.com")
    assert "debunked" not in res.cleaned
    assert "The photo is from 2017." in res.cleaned


def test_debunked_untouched_on_non_fact_checker_domain():
    doc = "We have debunked this exact claim before. The photo is from 2017."
    res = strip(doc, "randomnewssite.com")  # not in KEYWORD_DOMAINS
    assert res.cleaned == doc
    assert res.removals == []


def test_similar_false_claim_sentence_stripped():
    doc = "This is a similar false claim to one we checked. Evidence follows."
    res = strip(doc, "boomlive.in")
    assert "similar false claim" not in res.cleaned
    assert "Evidence follows." in res.cleaned


def test_keyword_does_not_touch_plain_evidence():
    doc = "The bridge was built in 1932 and spans two kilometres."
    res = strip(doc, "politifact.com")
    assert res.cleaned == doc
    assert res.removals == []
