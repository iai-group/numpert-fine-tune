"""Tests for compare_classifications.py."""

import json
import sys
from pathlib import Path
import pytest
import os

# Add the current directory to the path for imports
current_dir = os.path.dirname(
    os.path.abspath("code/error_analysis/compare_classifications.py")
)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


from compare_classifications import (
    CustomIdParser,
    extract_label_from_content,
    extract_label_from_openai_response,
    load_jsonl_predictions,
    evaluate_predictions,
    compare_classifications,
    print_summary,
    main as compare_main,
)


def test_custom_id_parser_baseline_and_categories():
    """Test parsing of custom_id strings."""
    category, gt = CustomIdParser.parse("id-True-math-arithmetic-baseline")
    assert category == "math-arithmetic"
    assert gt is True

    category, gt = CustomIdParser.parse("id-false-sci-physics-baseline")
    assert category == "sci-physics"
    assert gt is False


def test_custom_id_parser_perturbations_and_errors():
    """Test parsing of custom_id strings with perturbations."""
    # t2f -> False
    category, gt = CustomIdParser.parse("foo-true-topic-t2f")
    assert category == "topic"
    assert gt is False

    # f2t -> True
    category, gt = CustomIdParser.parse("bar-false-topic-f2t")
    assert category == "topic"
    assert gt is True

    # unknown perturbation
    with pytest.raises(ValueError):
        CustomIdParser.parse("bad-true-topic-unknown")

    # too few parts
    with pytest.raises(ValueError):
        CustomIdParser.parse("bad")


@pytest.mark.parametrize(
    "content,expected",
    [
        ('{"label": true}', True),
        ("{ 'label' : false }", False),
        ("label: true", True),
        ("no label here", None),
    ],
)
def test_extract_label_from_content(content, expected):
    """Test extraction of boolean label from content strings."""
    assert extract_label_from_content(content) == expected


def test_extract_label_from_openai_response_variants():
    """Test extraction of label from OpenAI-like response dicts."""
    data_true = {
        "response": {
            "body": {
                "choices": [
                    {"message": {"content": '{"label": true}'}},
                ]
            }
        }
    }
    assert extract_label_from_openai_response(data_true) is True

    data_none = {"response": {"body": {"choices": []}}}
    assert extract_label_from_openai_response(data_none) is None

    data_unparsable = {
        "response": {
            "body": {"choices": [{"message": {"content": "no label"}}]}
        }
    }
    assert extract_label_from_openai_response(data_unparsable) is None


def test_load_jsonl_predictions_handles_formats_and_skips(tmp_path, capsys):
    """Test loading predictions from JSONL with various formats and errors."""
    p = tmp_path / "preds.jsonl"

    # 1) Valid OpenAI response with custom_id
    line1 = {
        "custom_id": "a-true-math-baseline",
        "response": {
            "body": {"choices": [{"message": {"content": '{"label": true}'}}]}
        },
    }
    # 2) Valid "content" field with label and 'id' instead of custom_id
    line2 = {
        "id": "b-false-logic-baseline",
        "content": "some text {'label': false} more text",
    }
    # 3) Invalid JSON
    bad_json_line = '{"this is": "not closed"'
    # 4) Missing custom_id/id
    line4 = {"content": '{"label": true}'}
    # 5) Has id but no label anywhere -> skipped
    line5 = {"custom_id": "c-true-chem-baseline", "content": "no label here"}

    with p.open("w", encoding="utf-8") as f:
        f.write(json.dumps(line1) + "\n")
        f.write(json.dumps(line2) + "\n")
        f.write(bad_json_line + "\n")
        f.write(json.dumps(line4) + "\n")
        f.write(json.dumps(line5) + "\n")

    preds = load_jsonl_predictions(p, verbose=True)
    # Should have captured print about loaded count
    captured = capsys.readouterr()
    assert "Loaded 2 predictions from" in captured.out

    assert preds == {
        "a-true-math-baseline": True,
        "b-false-logic-baseline": False,
    }


def test_evaluate_predictions_and_skip_invalid(capsys):
    """Test evaluation of predictions against ground truth, skipping invalid
    IDs."""
    preds = {
        "x1-true-cat-baseline": True,  # gt True => correct
        "x2-false-cat-ff2f": False,  # gt False => correct
        "bad_custom_id": True,  # invalid -> skipped
    }
    evaluated = evaluate_predictions(preds)
    # bad_custom_id should be skipped and not present
    assert set(evaluated.keys()) == {
        "x1-true-cat-baseline",
        "x2-false-cat-ff2f",
    }
    assert evaluated["x1-true-cat-baseline"] == (True, True)
    assert evaluated["x2-false-cat-ff2f"] == (False, True)

    out = capsys.readouterr().out
    assert "Skipping bad_custom_id:" in out


def test_compare_classifications_details_and_ids():
    """Test comparison of classification results with details and IDs."""
    # Ground truths:
    # x1-true-...-baseline -> gt True
    # x2-false-...-ff2f -> gt False
    # x3-true-...-t2t -> gt True
    eval1 = {
        "x1-true-stat-baseline": (True, True),
        "x2-false-bio-ff2f": (False, True),
        "x3-true-chem-t2t": (True, True),
    }
    eval2 = {
        "x1-true-stat-baseline": (False, False),
        "x2-false-bio-ff2f": (True, False),
        "x3-true-chem-t2t": (True, True),
    }

    improved_ids = compare_classifications(eval1, eval2, details=False)
    assert set(improved_ids) == {"x1-true-stat-baseline", "x2-false-bio-ff2f"}

    improved_details = compare_classifications(eval1, eval2, details=True)
    # Sort for deterministic assertion
    improved_details_sorted = sorted(improved_details, key=lambda t: t[0])
    assert improved_details_sorted == [
        ("x1-true-stat-baseline", True, False, True),
        ("x2-false-bio-ff2f", False, True, False),
    ]


def test_print_summary_output(capsys):
    """Test printing of summary output."""
    eval1 = {
        "x1-true-stat-baseline": (True, True),
        "x2-false-bio-ff2f": (False, True),
        "x3-true-chem-t2t": (True, True),
    }
    eval2 = {
        "x1-true-stat-baseline": (False, False),
        "x2-false-bio-ff2f": (True, False),
        "x3-true-chem-t2t": (True, True),
    }
    improved = ["x1-true-stat-baseline", "x2-false-bio-ff2f"]

    print_summary(eval1, eval2, improved)
    out = capsys.readouterr().out
    assert "=== COMPARISON SUMMARY ===" in out
    assert "Common IDs: 3" in out
    assert "File1 accuracy: 3/3 = 100.00%" in out
    assert "File2 accuracy: 1/3 = 33.33%" in out
    assert "Improved (file1 correct, file2 wrong): 2" in out


def _write_jsonl(path: Path, lines: list[dict | str]):
    """Helper to write a list of dicts or strings as JSONL to a file."""
    with path.open("w", encoding="utf-8") as f:
        for ln in lines:
            if isinstance(ln, str):
                f.write(ln)
            else:
                f.write(json.dumps(ln))
            f.write("\n")


def test_main_integration(tmp_path, monkeypatch, capsys):
    """Integration test for main() function with two JSONL files."""
    # Build two JSONL files for main()
    f1 = tmp_path / "file1.jsonl"
    f2 = tmp_path / "file2.jsonl"
    out_path = tmp_path / "improved.txt"

    # IDs and ground truth:
    # x1-true-stat-baseline -> gt True
    # x2-false-bio-ff2f -> gt False
    # x3-true-chem-t2t -> gt True
    _write_jsonl(
        f1,
        [
            {
                "custom_id": "x1-true-stat-baseline",
                "response": {
                    "body": {
                        "choices": [{"message": {"content": '{"label": true}'}}]
                    }
                },
            },
            {
                "custom_id": "x2-false-bio-ff2f",
                "content": "prefix 'label': false suffix",
            },
            {
                "id": "x3-true-chem-t2t",
                "content": '{"label": true}',
            },
        ],
    )
    _write_jsonl(
        f2,
        [
            {
                "custom_id": "x1-true-stat-baseline",
                "content": '{"label": false}',
            },
            {
                "custom_id": "x2-false-bio-ff2f",
                "response": {
                    "body": {
                        "choices": [{"message": {"content": '{"label": true}'}}]
                    }
                },
            },
            {
                "id": "x3-true-chem-t2t",
                "response": {
                    "body": {
                        "choices": [{"message": {"content": '{"label": true}'}}]
                    }
                },
            },
        ],
    )

    monkeypatch.setenv("PYTHONIOENCODING", "utf-8")
    monkeypatch.setenv("LC_ALL", "C")

    argv = [
        "compare_classifications",
        str(f1),
        str(f2),
        "--details",
        "--output",
        str(out_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    rc = compare_main()
    assert rc == 0

    out = capsys.readouterr().out
    # Summary appears
    assert "=== COMPARISON SUMMARY ===" in out
    # Improved section appears and includes both improved IDs with details
    assert "=== IMPROVED CUSTOM IDs (file1 correct, file2 incorrect) ===" in out
    assert "x1-true-stat-baseline | file1=True | file2=False | gt=True" in out
    assert "x2-false-bio-ff2f | file1=False | file2=True | gt=False" in out
    # Output file contains only the IDs (one per line)
    saved = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert set(saved) == {"x1-true-stat-baseline", "x2-false-bio-ff2f"}
