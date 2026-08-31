"""Sanity tests for the rule registry (data integrity, not behavior)."""

import re

from src.data_preprocessing.raw_dataset_preprocessing.verdict_strip.rules import (  # noqa: E501
    RULES,
    MODES,
    domain_of,
)


def test_domain_of_strips_www_and_scheme():
    assert (
        domain_of("https://www.politifact.com/factchecks/x")
        == "politifact.com"
    )
    assert domain_of("http://africacheck.org/y") == "africacheck.org"
    assert domain_of("") == ""


def test_every_rule_has_valid_mode_and_compiled_anchor():
    for domain, rules in RULES.items():
        assert rules, f"{domain} has no rules"
        for rule in rules:
            assert rule.mode in MODES, (
                f"{domain}/{rule.name}: bad mode {rule.mode}"
            )
            assert isinstance(rule.anchor, re.Pattern)


def test_expected_domains_present():
    for d in [
        "politifact.com", "snopes.com", "africacheck.org",
        "factcheck.afp.com", "factly.in", "logically.ai",
        "indiatoday.in", "checkyourfact.com", "washingtonpost.com",
        "usatoday.com", "healthfeedback.org", "sciencefeedback.co",
        "climatefeedback.org", "aap.com.au",
    ]:
        assert d in RULES, f"missing rules for {d}"
