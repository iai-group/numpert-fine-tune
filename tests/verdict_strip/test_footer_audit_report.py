"""Footer highlighting in the audit report."""

from src.data_preprocessing.raw_dataset_preprocessing.verdict_strip.hybrid_audit_report import (  # noqa: E501
    build_hybrid_audit_records,
    highlight_html,
)


def test_highlight_html_wraps_footer_span():
    doc = "Real evidence here. Donate to Alt News! Please contribute."
    out = highlight_html(
        doc, removed=[], rejected=(), rule_removed=(),
        footer_removed=["Donate to Alt News! Please contribute."],
    )
    assert '<span class="footer">' in out
    assert "Real evidence here." in out


def test_build_hybrid_audit_records_splits_footer_from_rule():
    pristine = [{"datasplit": "train", "id": 1, "claim": "c",
                 "doc": "Evidence. Donate to Alt News! The verdict is false."}]
    hybrid = [{"datasplit": "train", "id": 1, "doc": "Evidence."}]
    hybrid_difflog = [{"datasplit": "train", "id": 1, "kept": ["x"]}]
    rule_difflog = [{
        "datasplit": "train", "id": 1,
        "removals": [
            {"removed_text": "Donate to Alt News!", "category": "footer"},
            {"removed_text": "The verdict is false.", "category": "verdict"},
        ],
    }]
    out = build_hybrid_audit_records(
        pristine, hybrid, hybrid_difflog, rule_difflog
    )
    assert out[0]["footer_removed"] == ["Donate to Alt News!"]
    assert out[0]["rule_removed"] == ["The verdict is false."]
