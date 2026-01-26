import json
import sys
import tempfile
from pathlib import Path
import os

# Get the directory containing this test file
current_dir = Path(__file__).parent.absolute()
# Go up two levels: tests/test_error_analysis -> tests -> aa (root)
root_dir = current_dir.parent.parent
# Then go to code/error_analysis
module_dir = root_dir / "code" / "error_analysis"

if str(module_dir) not in sys.path:
    sys.path.insert(0, str(module_dir))

# Import with the actual filename (with typo)
import main_compare_thoughts as ct


def test_extract_fields_openai_batch():
    """It should correctly extract both content and thoughts from an OpenAI
    batch record."""
    record = {
        "response": {
            "body": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"label": true}',
                            "thoughts": "Reasoning text here.",
                        }
                    }
                ]
            }
        }
    }
    fields = ct.extract_fields(record)
    assert fields["content"] == '{"label": true}'
    assert fields["thoughts"] == "Reasoning text here."


def test_extract_fields_fallback():
    """It should fallback to top-level content/thoughts if response.body.choices
    is missing."""
    record = {"content": '{"label": false}', "thoughts": "Fallback reasoning."}
    fields = ct.extract_fields(record)
    assert fields["content"] == '{"label": false}'
    assert fields["thoughts"] == "Fallback reasoning."


def test_build_comparison_creates_expected_json(tmp_path: Path):
    """It should build a nested JSON file with ground_truth, preds, content, and
    thoughts."""
    # Minimal fake data
    cid = "123-True-baseline"
    fake_pred_true = {cid: True}
    fake_eval_true = {cid: (True, True)}  # pred == gt
    fake_eval_false = {cid: (False, False)}  # pred != gt

    # Store original functions to restore later
    original_load_jsonl = ct.cmp.load_jsonl_predictions
    original_evaluate = ct.cmp.evaluate_predictions
    original_compare = ct.cmp.compare_classifications
    original_load_raw = ct.load_jsonl_raw

    try:
        # Mock the functions with proper parameter handling
        ct.cmp.load_jsonl_predictions = lambda f: fake_pred_true

        def mock_evaluate(preds):
            # This will be called twice, once for each file
            # We can differentiate based on call order or other logic
            return fake_eval_true

        ct.cmp.evaluate_predictions = mock_evaluate
        ct.cmp.compare_classifications = lambda e1, e2, details=True: [
            (cid, True, False, True)
        ]

        # Fake raw JSONL lookups
        raw_entry = {
            "response": {
                "body": {
                    "choices": [
                        {
                            "message": {
                                "content": '{"label": true}',
                                "thoughts": "Some reasoning",
                            }
                        }
                    ]
                }
            }
        }
        ct.load_jsonl_raw = lambda f: {cid: raw_entry}

        f1 = tmp_path / "f1.jsonl"
        f2 = tmp_path / "f2.jsonl"
        output = tmp_path / "out.json"

        ct.build_comparison(f1, f2, output)

        # Validate file structure
        data = json.loads(output.read_text())
        assert cid in data
        assert data[cid]["ground_truth"] is True
        assert "content" in data[cid]["f1"]
        assert "thoughts" in data[cid]["f1"]
        assert "content" in data[cid]["f2"]
        assert "thoughts" in data[cid]["f2"]

    finally:
        # Restore original functions
        ct.cmp.load_jsonl_predictions = original_load_jsonl
        ct.cmp.evaluate_predictions = original_evaluate
        ct.cmp.compare_classifications = original_compare
        ct.load_jsonl_raw = original_load_raw
