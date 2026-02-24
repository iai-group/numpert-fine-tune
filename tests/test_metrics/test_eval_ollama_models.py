"""Tests for eval_ollama_models.py."""

import json
import pytest
from unittest.mock import patch, mock_open, MagicMock

from src.metrics.eval_ollama_models import (
    flip_label,
    convert_bool_label_to_str,
    check_file_exists_and_readable,
    get_configuration,
    get_result_file_path,
    process_result_file,
    evaluate_label_combination,
    main,
)


def test_flip_label():
    """Test that flip_label correctly flips boolean labels."""
    assert flip_label("True") == "False"
    assert flip_label("False") == "True"


def test_convert_bool_label_to_str():
    """Test that bool to str conversion works correctly."""
    assert convert_bool_label_to_str(True) == "True"
    assert convert_bool_label_to_str(False) == "False"


@pytest.mark.parametrize(
    "exists, is_file, size, json_valid, expected",
    [
        (False, True, 100, True, (False, "File does not exist: test.json")),
        (True, False, 100, True, (False, "Path is not a file: test.json")),
        (True, True, 0, True, (False, "File is empty: test.json")),
        (True, True, 100, False, (False, "Invalid JSON in file test.json: ")),
        (True, True, 100, True, (True, "")),
    ],
)
def test_check_file_exists_and_readable(
    exists, is_file, size, json_valid, expected
):
    """Test that check_file_exists_and_readable correctly validates files."""
    with (
        patch("os.path.exists", return_value=exists),
        patch("os.path.isfile", return_value=is_file),
        patch("os.path.getsize", return_value=size),
    ):

        if exists and is_file and size > 0:
            json_content = '{"valid": true}' if json_valid else "invalid json"
            m = mock_open(read_data=json_content)

            if json_valid:
                with patch("builtins.open", m):
                    result = check_file_exists_and_readable("test.json")
                    assert result[0] == expected[0]
                    assert result[1].startswith(expected[1])
            else:
                with (
                    patch("builtins.open", m),
                    patch(
                        "json.loads",
                        side_effect=json.JSONDecodeError("", "", 0),
                    ),
                ):
                    result = check_file_exists_and_readable("test.json")
                    assert result[0] == expected[0]
                    assert result[1].startswith(expected[1])
        else:
            result = check_file_exists_and_readable("test.json")
            assert result == expected


def test_get_configuration():
    """Test that get_configuration returns the expected configuration."""
    perturbations, models, short_pert_names = get_configuration()

    assert "normalized_claim" in perturbations
    assert "approximation" in perturbations
    assert len(perturbations) == 7

    assert "deepseek-r1:32b" in models
    assert "qwen3:32b" in models
    assert len(models) == 7

    assert short_pert_names["normalized_claim"] == "original"
    assert short_pert_names["approximation"] == "approx"
    assert len(short_pert_names) == 7


@pytest.mark.parametrize(
    "shot, perturbation, original_label, model, is_normalized, expected",
    [
        (
            "2-shot",
            "approximation",
            "true",
            "llama3.3:70b",
            False,
            "results/ollama_runs/2-shot/approximation_true_llama3.3:70b_results.jsonl",  # noqa E501
        ),
        (
            "0-shot",
            "masking",
            "false",
            "mistral",
            False,
            "results/ollama_runs/0-shot/masking_false_mistral_results.jsonl",  # noqa E501
        ),
        (
            "2-shot",
            "normalized_claim",
            "true",
            "qwen3:32b",
            True,
            "results/ollama_runs/2-shot/numeration_true_qwen3:32b_results.jsonl",  # noqa E501
        ),
    ],
)
def test_get_result_file_path(
    shot, perturbation, original_label, model, is_normalized, expected
):
    """Test that get_result_file_path returns the correct file path."""
    result = get_result_file_path(
        shot, perturbation, original_label, model, is_normalized
    )
    assert result == expected


def test_process_result_file_file_not_found():
    """Test that process_result_file handles non-existent files correctly."""
    with patch(
        "src.metrics.eval_ollama_models.check_file_exists_and_readable",
        return_value=(False, "File does not exist"),
    ):

        result = process_result_file(
            "nonexistent.jsonl", "approx", "true", "false"
        )
        assert result == ([], [], 0, 0, 0)


def test_process_result_file_valid_file():
    """Test that process_result_file processes valid files correctly."""
    json_lines = [
        '{"claim_id": 1, "label": "True", "approximation_false_pred_label": true}',  # noqa E501
        '{"claim_id": 2, "label": "False", "approximation_false_pred_label": false}',  # noqa E501
    ]

    with (
        patch(
            "src.metrics.eval_ollama_models.check_file_exists_and_readable",
            return_value=(True, ""),
        ),
        patch("builtins.open", mock_open(read_data="\n".join(json_lines))),
    ):

        true_labels, pred_labels, line_count, valid_entries, invalid_entries = (
            process_result_file("valid.jsonl", "approximation", "true", "false")
        )

        assert set(true_labels) == set(["True", "False"])
        assert set(pred_labels) == set(["True", "False"])
        assert line_count == 2
        assert valid_entries == 2
        assert invalid_entries == 0


def test_process_result_file_normalized_claim():
    """Test that process_result_file handles normalized_claim correctly."""
    json_lines = [
        '{"claim_id": 1, "label": "True", "normalized_claim_pred_label": true}'
    ]

    with (
        patch(
            "src.metrics.eval_ollama_models.check_file_exists_and_readable",
            return_value=(True, ""),
        ),
        patch("builtins.open", mock_open(read_data="\n".join(json_lines))),
    ):

        true_labels, pred_labels, line_count, valid_entries, invalid_entries = (
            process_result_file(
                "valid.jsonl", "normalized_claim", "true", "true"
            )
        )

        assert true_labels == ["True"]
        assert pred_labels == ["True"]
        assert line_count == 1
        assert valid_entries == 1
        assert invalid_entries == 0


def test_process_result_file_with_invalid_entries():
    """Test that process_result_file handles invalid entries correctly."""
    json_lines = [
        '{"claim_id": 1, "label": "True", "approximation_false_pred_label": null}',  # noqa E501
        '{"claim_id": 2, "label": "False", "approximation_false_pred_label": false}',  # noqa E501
        "invalid json",
    ]

    with (
        patch(
            "src.metrics.eval_ollama_models.check_file_exists_and_readable",
            return_value=(True, ""),
        ),
        patch("builtins.open", mock_open(read_data="\n".join(json_lines))),
    ):

        true_labels, pred_labels, line_count, valid_entries, invalid_entries = (
            process_result_file("valid.jsonl", "approximation", "true", "false")
        )

        assert true_labels == ["True"]
        assert pred_labels == ["False"]
        assert line_count == 3
        assert valid_entries == 1
        assert invalid_entries == 1


@patch("src.metrics.eval_ollama_models.process_result_file")
@patch("src.metrics.eval_ollama_models.get_result_file_path")
@patch("builtins.open", new_callable=mock_open)
@patch("src.metrics.eval_ollama_models.DeltaEvalTable")
def test_evaluate_label_combination(
    mock_delta_eval_table,
    mock_open_file,
    mock_get_result_file_path,
    mock_process_result_file,
):
    """Test that evaluate_label_combination correctly processes results."""
    # Setup mocks
    mock_table = MagicMock()
    mock_delta_eval_table.return_value = mock_table
    mock_process_result_file.return_value = (["True"], ["True"], 1, 1, 0)
    mock_get_result_file_path.return_value = "test/path.jsonl"

    # Test data
    original_label = "true"
    perturbed_label = "false"
    shot = "2-shot"
    perturbations = ["normalized_claim", "approximation"]
    models = ["model1", "model2"]
    short_pert_names = {
        "normalized_claim": "original",
        "approximation": "approx",
    }
    error_file = mock_open_file.return_value

    # Call function
    evaluate_label_combination(
        original_label,
        perturbed_label,
        shot,
        perturbations,
        models,
        short_pert_names,
        error_file,
    )

    # Assertions
    assert mock_delta_eval_table.call_count == 1
    assert mock_table.add_result.call_count == 4  # 2 models * 2 perturbations
    assert mock_table.to_latex.call_count == 1
    assert mock_open_file.call_count == 1  # Error file and LaTeX file


@patch("src.metrics.eval_ollama_models.evaluate_label_combination")
@patch("src.metrics.eval_ollama_models.get_configuration")
@patch("os.path.exists")
@patch("builtins.open", new_callable=mock_open)
def test_main(
    mock_open_file,
    mock_path_exists,
    mock_get_configuration,
    mock_evaluate_label_combination,
):
    """Test the main function."""
    # Setup mocks
    mock_path_exists.return_value = True
    mock_get_configuration.return_value = (
        ["normalized_claim", "approximation"],
        ["model1", "model2"],
        {"normalized_claim": "original", "approximation": "approx"},
    )

    # Call main
    main()

    # Assertions
    assert mock_get_configuration.call_count == 1
    assert mock_open_file.call_count == 1
    assert (
        mock_evaluate_label_combination.call_count == 4
    )  # 2 orig labels * 2 pert labels


@patch("os.path.exists")
def test_main_base_dir_not_exists(mock_path_exists):
    """Test that main handles missing base directory correctly."""
    mock_path_exists.return_value = False

    # Call main
    main()

    # No exceptions should be raised
    assert mock_path_exists.call_count >= 1
