"""Test suite for the metrics calculation module."""

import pytest

from code.metrics.calculate_metrics_jsonl import (
    ResponseValidator,
    CustomIdParser,
    JsonlProcessor,
    MetricsCalculator,
)


class TestResponseValidator:
    @pytest.fixture
    def validator(self):
        """Fixture to create a ResponseValidator instance."""
        return ResponseValidator(debug_mode=True)

    def test_extract_label_from_json_with_boolean(self, validator):
        """Test extracting boolean labels from JSON strings."""
        assert validator.extract_label_from_json('{"label": true}') is True
        assert validator.extract_label_from_json('{"label": false}') is False

    def test_extract_label_from_json_with_string(self, validator):
        """Test extracting string labels from JSON strings."""
        assert validator.extract_label_from_json('{"label": "True"}') is True
        assert validator.extract_label_from_json('{"label": "false"}') is False

    def test_extract_label_from_json_invalid(self, validator):
        """Test extracting labels from invalid JSON strings."""
        assert validator.extract_label_from_json("not a json") is None
        assert validator.extract_label_from_json('{"no_label": true}') is None

    def test_extract_label_from_regex(self, validator):
        """Test extracting labels using regex."""
        content = 'Some text {"label": true} and more'
        matches = validator.extract_label_from_regex(content)
        assert matches == ["true"]
        content = 'Some text {"label": false} and more'
        matches = validator.extract_label_from_regex(content)
        assert matches == ["false"]
        content = 'Some text {"label": false} and {"label": true}'
        matches = validator.extract_label_from_regex(content)
        assert matches == ["false", "true"]

    def test_validate_and_extract_empty(self, validator):
        """Test validation and extraction with empty content."""
        pred, err = validator.validate_and_extract("")
        assert pred is None and err == "empty_content"

    def test_validate_and_extract_json_success(self, validator):
        """Test validation and extraction with valid JSON content."""
        p, e = validator.validate_and_extract('{"label":false}')
        assert p is False and e == ""

    def test_validate_and_extract_regex_success(self, validator):
        """Test validation and extraction with valid regex content."""
        p, e = validator.validate_and_extract("Answer is label:true")
        assert p is True and e == ""

    def test_validate_and_extract_no_label(self, validator):
        """Test validation and extraction when no label is found."""
        p, e = validator.validate_and_extract("no labels here")
        assert p is None and e == "no_label_found"

    def test_validate_and_extract_conflict(self, validator):
        """Test validation and extraction with conflicting labels."""
        content = 'Some text {"label": false} and {"label": true}'
        p, e = validator.validate_and_extract(content)
        assert p is None and e == "conflicting_labels"


class TestCustomIdParser:
    """Test suite for parsing custom IDs in metrics calculations."""

    def test_parse_true_t2t(self):
        """Test parsing a custom ID for True category with t2t."""
        cid = "173-True-approximation-t2t"
        category, gt = CustomIdParser.parse(cid)
        assert category == "True-approximation-t2t"
        assert gt is True

    def test_parse_false_baseline(self):
        """Test parsing a custom ID for False category with baseline."""
        cid = "42-False-normalized_claim-baseline"
        category, gt = CustomIdParser.parse(cid)
        assert category == "False-normalized_claim-baseline"
        assert gt is True

    def test_parse_false_ft2f(self):
        """Test parsing a custom ID for False category with ft2f."""
        cid = "123-False-numeration-ff2f"
        cat, gt = CustomIdParser.parse(cid)
        assert cat == "False-numeration-ff2f"
        assert gt is False

    def test_parse_invalid_format(self):
        """Test parsing an invalid custom ID format."""
        with pytest.raises(ValueError):
            CustomIdParser.parse("too-short-id")

    def test_parse_unknown_label_in_baseline(self):
        """Test parsing a custom ID with an unknown label in baseline."""
        with pytest.raises(ValueError):
            CustomIdParser.parse("1-Unknown-cat-baseline")

    def test_parse_unknown_perturbation(self):
        """Test parsing a custom ID with an unknown perturbation."""
        with pytest.raises(ValueError):
            CustomIdParser.parse("1-True-cat-xyz")


class TestJsonlProcessor:
    """Test suite for processing JSONL records in metrics calculations."""

    @pytest.fixture
    def processor(self):
        """Fixture to create a JsonlProcessor instance."""
        return JsonlProcessor(debug_mode=False)

    def test_extract_response_api_error(self, processor):
        """Test extracting response content when an API error occurs."""
        rec = {"error": "some error"}
        content, err = processor.extract_response_content(rec)
        assert content is None and err == "api_error"

    def test_extract_response_no_response(self, processor):
        """Test extracting response content when no response is present."""
        rec = {}
        content, err = processor.extract_response_content(rec)
        assert content is None and err == "no_response"

    def test_extract_response_no_message(self, processor):
        """Test extracting response content when no message is present."""
        rec = {"response": {"body": {"choices": [{}]}}}
        c, e = processor.extract_response_content(rec)
        assert c is None and e == "no_message"

    def test_extract_response_empty_content(self, processor):
        """Test extracting response content when the content is empty."""
        rec = {
            "response": {"body": {"choices": [{"message": {"content": ""}}]}}
        }
        c, e = processor.extract_response_content(rec)
        assert c is None and e == "empty_content"

    def test_extract_response_success(self, processor):
        """Test extracting response content when successful."""
        rec = {
            "response": {
                "body": {
                    "choices": [{"message": {"content": '{"label": true}'}}]
                }
            }
        }
        c, e = processor.extract_response_content(rec)
        assert c.strip() == '{"label": true}' and e == ""

    def test_process_single_record_success(self, processor):
        """Test processing a single record successfully."""
        rec = {
            "custom_id": "1-True-cat-t2f",
            "response": {
                "body": {"choices": [{"message": {"content": "label:false"}}]}
            },
        }
        result, err = processor.process_single_record(rec, line_num=1)
        assert err == ""
        assert result["custom_id"] == rec["custom_id"]
        assert (
            result["ground_truth"] is False
        )  # t2f -> gt False? actually t2f => False
        assert result["prediction"] is False

    def test_process_single_record_missing_id(self, processor):
        """Test processing a single record with missing custom ID."""
        rec = {"response": {}}
        res, err = processor.process_single_record(rec, 1)
        assert res is None and err == "missing_custom_id"

    def test_process_single_record_parse_error(self, processor):
        """Test processing a single record with a parsing error."""
        rec = {"custom_id": "bad-format", "response": {}}
        res, err = processor.process_single_record(rec, 1)
        assert res is None
        assert err.startswith("custom_id_parse_error")

    def test_process_single_record_validation_error(self, processor):
        """Test processing a single record with validation error."""
        rec = {
            "custom_id": "2-True-cat-t2t",
            "response": {
                "body": {"choices": [{"message": {"content": "no label"}}]}
            },
        }
        res, err = processor.process_single_record(rec, 1)
        assert res is None and err == "no_label_found"


class TestMetricsCalculator:
    """Test suite for the MetricsCalculator class."""

    @pytest.fixture
    def calc(self):
        """Fixture to create a MetricsCalculator instance."""
        return MetricsCalculator()

    def test_calculate_binary_metrics_zero(self, calc):
        """Test calculating binary metrics when all counts are zero."""
        m = calc.calculate_binary_metrics(0, 0, 0, 0)
        assert all(v == 0.0 for v in m.values())

    def test_calculate_binary_metrics_nonzero(self, calc):
        """Test calculating binary metrics with non-zero counts."""
        m = calc.calculate_binary_metrics(tp=5, tn=3, fp=2, fn=4)
        # manual
        total = 5 + 3 + 2 + 4
        assert pytest.approx(m["accuracy"]) == (5 + 3) / total
        assert pytest.approx(m["precision"]) == 5 / (5 + 2)
        assert pytest.approx(m["recall"]) == 5 / (5 + 4)
        # f1
        prec = 5 / 7
        rec = 5 / 9
        assert pytest.approx(m["f1"]) == 2 * (prec * rec) / (prec + rec)

    def test_calculate_confusion_matrix(self, calc):
        """Test calculating confusion matrix from data."""
        data = [
            {"ground_truth": True, "prediction": True},
            {"ground_truth": True, "prediction": False},
            {"ground_truth": False, "prediction": True},
            {"ground_truth": False, "prediction": False},
            {"ground_truth": False, "prediction": False},
        ]
        tp, tn, fp, fn = calc.calculate_confusion_matrix(data)
        assert (tp, tn, fp, fn) == (1, 2, 1, 1)

    def test_calculate_all_metrics(self, calc):
        """Test calculating all metrics with a sample dataset."""
        data = [
            {"category": "A-cat", "ground_truth": True, "prediction": True},
            {"category": "A-cat", "ground_truth": True, "prediction": False},
            {"category": "B-cat", "ground_truth": False, "prediction": False},
        ]
        errors = {"no_label": 1}
        failed = {"no_label": ["id1"]}
        results = calc.calculate_all_metrics(data, errors, failed)
        # summary
        summary = results["summary"]
        assert summary["total_samples"] == 3
        assert (
            summary["total_invalid"]
            if "total_invalid" in summary
            else summary["invalid_responses"] == 1
        )
        # overall
        overall = results["overall"]
        assert "metrics" in overall and "confusion_matrix" in overall
        # categories
        cats = results["categories"]
        assert set(cats.keys()) == {"A-cat", "B-cat"}
        # category A-cat confusion
        cm_a = cats["A-cat"]["confusion_matrix"]
        assert cm_a["tp"] == 1 and cm_a["fn"] == 1
        # category B-cat confusion
        cm_b = cats["B-cat"]["confusion_matrix"]
        assert cm_b["tn"] == 1
