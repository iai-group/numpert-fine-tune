"""Test module for main_perturb functionality."""

from unittest.mock import Mock
from src.data_preprocessing.perturbations import main_perturb as mp


def test_remove_unperturbed_claims_filters_and_flattens(capsys):
    """Test remove_unperturbed_claims function.

    Verifies that:
    - Claims where perturbation equals normalized_claim are removed
    - Nested lists are flattened
    - Non-dictionary items trigger a warning
    - Removal count is printed
    """
    data = [
        [
            {"normalized_claim": "A", "approximation_true": "A"},
            {"normalized_claim": "B", "approximation_false": "X"},
        ],
        {"normalized_claim": "C"},
        "not a dict",
    ]
    keys = ["approximation_true", "approximation_false"]

    result = mp.remove_unperturbed_claims(data, keys)
    out = capsys.readouterr().out

    assert isinstance(result, list)
    assert len(result) == 2
    assert {"normalized_claim": "B", "approximation_false": "X"} in result
    assert {"normalized_claim": "C"} in result

    assert "Warning: Skipping non-dictionary item" in out
    assert "Removed 1 unperturbed claims from dataset" in out


def test_remove_unperturbed_claims_no_removal_no_message(capsys):
    """Test that no removal message is printed when no claims are removed.

    Verifies that:
    - Claims with different perturbation values are kept
    - Claims missing perturbation keys are kept
    - No removal message is printed when nothing is removed
    """
    data = [
        {"normalized_claim": "A", "approximation_true": "DIFF"},
        {"normalized_claim": "B", "other_key": "B"},
        {"normalized_claim": "C"},
    ]
    keys = ["approximation_true", "approximation_false"]

    result = mp.remove_unperturbed_claims(data, keys)
    out = capsys.readouterr().out

    assert result == data
    assert "Removed" not in out


def test_main_runs_pipeline_and_saves_filtered_outputs(monkeypatch):
    """Test the complete main pipeline with all perturbation modules.

    This test verifies the full workflow:
    - Loading true/false datasets
    - Running all perturbation modules
    - Filtering unperturbed claims
    - Saving filtered results to appropriate output files
    - Handling the special obfuscate module that returns tuples of three lists
    """
    load_mock = Mock(
        side_effect=[
            [{"id": 1}],
            [{"id": 2}],
        ]
    )
    save_mock = Mock()
    monkeypatch.setattr(mp, "load_data", load_mock)
    monkeypatch.setattr(mp, "save_data", save_mock)

    monkeypatch.setattr(
        mp.approximation,
        "process_json",
        Mock(
            side_effect=[
                [
                    {"normalized_claim": "n1", "approximation_true": "n1"},
                    {"normalized_claim": "n2", "approximation_false": "DIFF"},
                ],
                [
                    {"normalized_claim": "m1", "approximation_false": "m1"},
                    {"normalized_claim": "m2", "approximation_true": "other"},
                ],
            ]
        ),
    )

    monkeypatch.setattr(
        mp.negative,
        "process_json",
        Mock(
            side_effect=[
                [
                    {"normalized_claim": "a", "negative_true": "a"},
                    {"normalized_claim": "b", "negative_false": "x"},
                ],
                [
                    {"normalized_claim": "c", "negative_true": "y"},
                    {"normalized_claim": "d", "negative_false": "d"},
                ],
            ]
        ),
    )

    monkeypatch.setattr(
        mp.numeration,
        "process_json",
        Mock(
            side_effect=[
                [
                    {"normalized_claim": "e", "numeration_true": "z"},
                    {"normalized_claim": "f", "numeration_false": "f"},
                ],
                [
                    {"normalized_claim": "g"},
                ],
            ]
        ),
    )

    monkeypatch.setattr(
        mp.obfuscate,
        "process_json",
        Mock(
            side_effect=[
                (
                    [
                        {
                            "normalized_claim": "h",
                            "random_replacement_false": "h",
                        }
                    ],
                    [{"normalized_claim": "i", "masking_false": "keep"}],
                    [{"normalized_claim": "j", "value_removal_false": "j"}],
                ),
                (
                    [
                        {
                            "normalized_claim": "k",
                            "random_replacement_false": "keep",
                        }
                    ],
                    [{"normalized_claim": "l", "masking_false": "l"}],
                    [],
                ),
            ]
        ),
    )

    monkeypatch.setattr(
        mp.range,
        "process_json",
        Mock(
            side_effect=[
                [
                    {"normalized_claim": "m", "range_true": "m"},
                    {"normalized_claim": "n", "range_false": "q"},
                ],
                [
                    {"normalized_claim": "o", "range_false": "o"},
                    {"normalized_claim": "p"},
                ],
            ]
        ),
    )

    mp.main()

    assert load_mock.call_count == 2
    assert save_mock.call_count == 10

    saved = [
        (call.args[1], len(call.args[0])) for call in save_mock.call_args_list
    ]

    expected = [
        ("data/processed/perturbations_test/approximation_true.json", 1),
        ("data/processed/perturbations_test/approximation_false.json", 1),
        ("data/processed/perturbations_test/negative_true.json", 1),
        ("data/processed/perturbations_test/negative_false.json", 1),
        ("data/processed/perturbations_test/numeration_true.json", 1),
        ("data/processed/perturbations_test/numeration_false.json", 1),
        ("data/processed/perturbations_test/obfuscation_true.json", 1),
        ("data/processed/perturbations_test/obfuscation_false.json", 1),
        ("data/processed/perturbations_test/range_true.json", 1),
        ("data/processed/perturbations_test/range_false.json", 1),
    ]

    assert saved == expected
