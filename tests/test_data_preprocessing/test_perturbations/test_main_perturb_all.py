"""Tests for the main perturbation script."""

from unittest.mock import MagicMock, call, patch

import pytest

from src.data_preprocessing.perturbations.main_perturb_all import (
    INPUT_FALSE,
    INPUT_TRUE,
    OUTPUT_DIR,
    get_paths,
    main,
    remove_unperturbed_claims,
    run_perturbations,
    save_splits,
    split_by_label,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_claims() -> list[dict]:
    """Provides a list of sample claim dictionaries for testing."""
    return [
        {
            "normalized_claim": "The population is 1000.",
            "perturb_type1": "The population is one thousand.",
            "perturb_type2": "The population is ~1000.",
            "label": "True",
        },
        {
            "normalized_claim": "The rate is 5%.",
            "perturb_type1": "The rate is five percent.",
            "perturb_type2": "The rate is ~5%.",
            "label": "False",
        },
        {
            "normalized_claim": "The count is 42.",
            "perturb_type1": "The count is 42.",  # Unperturbed.
            "perturb_type2": "The count is ~42.",
            "label": "True",
        },
    ]


@pytest.fixture
def flat_claims() -> list[dict]:
    """Provides a flat list of claims with mixed labels."""
    return [
        {"label": "True", "claim": "Claim A."},
        {"label": "False", "claim": "Claim B."},
        {"label": "True", "claim": "Claim C."},
        {"label": "False", "claim": "Claim D."},
        {"label": "True", "claim": "Claim E."},
    ]


# ── get_paths ─────────────────────────────────────────────────────────────────


class TestGetPaths:
    """Tests for the get_paths function."""

    @pytest.mark.parametrize("split", ["train", "val", "test"])
    def test_returns_tuple_of_three_strings(self, split: str) -> None:
        """Verifies that get_paths returns a 3-tuple of strings."""
        result = get_paths(split)
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert all(isinstance(p, str) for p in result)

    @pytest.mark.parametrize("split", ["train", "val", "test"])
    def test_split_name_present_in_all_paths(self, split: str) -> None:
        """Verifies that the split name appears in all returned paths."""
        input_true, input_false, output_dir = get_paths(split)
        assert split in input_true
        assert split in input_false
        assert split in output_dir

    def test_paths_match_module_level_templates(self) -> None:
        """Verifies that get_paths uses the module-level path templates."""
        split = "test"
        input_true, input_false, output_dir = get_paths(split)
        assert input_true == INPUT_TRUE.format(split=split)
        assert input_false == INPUT_FALSE.format(split=split)
        assert output_dir == OUTPUT_DIR.format(split=split)

    def test_different_splits_produce_different_paths(self) -> None:
        """Verifies that different splits produce distinct paths."""
        paths_train = get_paths("train")
        paths_val = get_paths("val")
        paths_test = get_paths("test")
        assert paths_train != paths_val
        assert paths_val != paths_test
        assert paths_train != paths_test

    def test_true_and_false_paths_differ(self) -> None:
        """Verifies that the true and false input paths are distinct."""
        input_true, input_false, _ = get_paths("test")
        assert input_true != input_false

    def test_true_path_contains_true(self) -> None:
        """Verifies that the true input path contains 'true'."""
        input_true, _, _ = get_paths("test")
        assert "true" in input_true.lower()

    def test_false_path_contains_false(self) -> None:
        """Verifies that the false input path contains 'false'."""
        _, input_false, _ = get_paths("test")
        assert "false" in input_false.lower()


# ── remove_unperturbed_claims ─────────────────────────────────────


class TestRemoveUnperturbedClaims:
    """Tests for the remove_unperturbed_claims function."""

    def test_removes_claim_matching_any_perturbation_key(
        self, sample_claims: list[dict]
    ) -> None:
        """Verifies that claims with an unperturbed key are removed."""
        result = remove_unperturbed_claims(
            sample_claims,
            ["perturb_type1", "perturb_type2"],
        )
        normalized = [c["normalized_claim"] for c in result]
        assert "The count is 42." not in normalized

    def test_keeps_fully_perturbed_claims(
        self, sample_claims: list[dict]
    ) -> None:
        """Verifies that claims with all keys perturbed are retained."""
        result = remove_unperturbed_claims(
            sample_claims,
            ["perturb_type1", "perturb_type2"],
        )
        normalized = [c["normalized_claim"] for c in result]
        assert "The population is 1000." in normalized
        assert "The rate is 5%." in normalized

    def test_returns_correct_count_after_filtering(
        self, sample_claims: list[dict]
    ) -> None:
        """Verifies that the returned list has the expected length."""
        result = remove_unperturbed_claims(
            sample_claims,
            ["perturb_type1", "perturb_type2"],
        )
        assert len(result) == 2

    def test_empty_input_returns_empty_list(self) -> None:
        """Verifies that an empty input list returns an empty list."""
        result = remove_unperturbed_claims([], ["perturb_type1"])
        assert result == []

    def test_no_matching_keys_keeps_all_claims(
        self, sample_claims: list[dict]
    ) -> None:
        """Verifies that non-existent keys cause no claims to be removed."""
        result = remove_unperturbed_claims(
            sample_claims,
            ["nonexistent_key"],
        )
        assert len(result) == len(sample_claims)

    def test_skips_non_dict_items(self, capsys: pytest.CaptureFixture) -> None:
        """Verifies that non-dict items are skipped with a warning."""
        data: list = [
            "not a dict",
            {"normalized_claim": "X.", "perturb_type1": "Y.", "label": "True"},
        ]
        result = remove_unperturbed_claims(data, ["perturb_type1"])
        assert len(result) == 1
        captured = capsys.readouterr()
        assert "Warning" in captured.out

    def test_all_unperturbed_returns_empty_list(self) -> None:
        """Verifies that all unperturbed claims result in an empty list."""
        data = [
            {
                "normalized_claim": "Same text.",
                "perturb_type1": "Same text.",
                "label": "True",
            }
        ]
        result = remove_unperturbed_claims(data, ["perturb_type1"])
        assert result == []

    def test_only_checks_specified_keys(self) -> None:
        """Verifies that only specified keys are considered for filtering."""
        data = [
            {
                "normalized_claim": "Text.",
                "perturb_type1": "Text.",  # Matches.
                "perturb_type2": "Different text.",
                "label": "True",
            }
        ]
        # Only checking perturb_type2, which differs — should keep.
        result = remove_unperturbed_claims(data, ["perturb_type2"])
        assert len(result) == 1

    def test_prints_removed_count_when_claims_removed(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        """Verifies that the removed count is printed when items are removed."""
        data = [
            {
                "normalized_claim": "Same.",
                "perturb_type1": "Same.",
                "label": "True",
            }
        ]
        remove_unperturbed_claims(data, ["perturb_type1"])
        captured = capsys.readouterr()
        assert "Removed 1" in captured.out

    def test_no_print_when_nothing_removed(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        """Verifies that nothing is printed when no claims are removed."""
        data = [
            {
                "normalized_claim": "Original.",
                "perturb_type1": "Different.",
                "label": "True",
            }
        ]
        remove_unperturbed_claims(data, ["perturb_type1"])
        captured = capsys.readouterr()
        assert "Removed" not in captured.out


# ── split_by_label ─────────────────────────────────────────────────


class TestSplitByLabel:
    """Tests for the split_by_label function."""

    def test_groups_entries_by_label(self, flat_claims: list[dict]) -> None:
        """Verifies that entries are grouped correctly by their label."""
        result = split_by_label(flat_claims)
        assert "True" in result
        assert "False" in result

    def test_correct_counts_per_label(self, flat_claims: list[dict]) -> None:
        """Verifies that each label group has the expected number of entries."""
        result = split_by_label(flat_claims)
        assert len(result["True"]) == 3
        assert len(result["False"]) == 2

    def test_empty_input_returns_empty_dict(self) -> None:
        """Verifies that an empty input list returns an empty dict."""
        result = split_by_label([])
        assert result == {}

    def test_unknown_label_used_as_fallback(self) -> None:
        """Verifies that missing labels are grouped under 'UNKNOWN'."""
        data = [{"claim": "No label here."}]
        result = split_by_label(data)
        assert "UNKNOWN" in result
        assert len(result["UNKNOWN"]) == 1

    def test_preserves_entry_content(self, flat_claims: list[dict]) -> None:
        """Verifies that entry contents are preserved after splitting."""
        result = split_by_label(flat_claims)
        true_claims = [e["claim"] for e in result["True"]]
        assert "Claim A." in true_claims
        assert "Claim C." in true_claims
        assert "Claim E." in true_claims

    def test_single_label_dataset(self) -> None:
        """Verifies correct behaviour when all entries share one label."""
        data = [{"label": "True", "claim": f"Claim {i}."} for i in range(5)]
        result = split_by_label(data)
        assert list(result.keys()) == ["True"]
        assert len(result["True"]) == 5


# ── save_splits ────────────────────────────────────────────────────


class TestSaveSplits:
    """Tests for the save_splits function."""

    @patch("src.data_preprocessing.perturbations.main_perturb_all.save_data")
    @patch(
        "src.data_preprocessing.perturbations.main_perturb_all."
        "remove_unperturbed_claims"
    )
    def test_calls_save_data_for_each_label(
        self,
        mock_remove: MagicMock,
        mock_save: MagicMock,
    ) -> None:
        """Verifies that save_data is called once per non-empty label."""
        mock_remove.side_effect = lambda data, keys: data
        results = {
            "True": [{"claim": "A."}],
            "False": [{"claim": "B."}],
        }
        save_splits(results, "numeration", ["perturb_type1"], "/out")
        assert mock_save.call_count == 2

    @patch("src.data_preprocessing.perturbations.main_perturb_all.save_data")
    @patch(
        "src.data_preprocessing.perturbations.main_perturb_all."
        "remove_unperturbed_claims"
    )
    def test_output_path_uses_name_and_label(
        self,
        mock_remove: MagicMock,
        mock_save: MagicMock,
    ) -> None:
        """Verifies that the output path contains the name and label."""
        mock_remove.side_effect = lambda data, keys: data
        results = {"True": [{"claim": "A."}]}
        save_splits(results, "numeration", ["perturb_type1"], "/out")
        saved_path = mock_save.call_args[0][1]
        assert "numeration" in saved_path
        assert "True" in saved_path

    @patch("src.data_preprocessing.perturbations.main_perturb_all.save_data")
    @patch(
        "src.data_preprocessing.perturbations.main_perturb_all."
        "remove_unperturbed_claims"
    )
    def test_skips_empty_splits_after_filtering(
        self,
        mock_remove: MagicMock,
        mock_save: MagicMock,
    ) -> None:
        """Verifies that save_data is not called for empty filtered splits."""
        mock_remove.return_value = []
        results = {"True": [{"claim": "A."}]}
        save_splits(results, "numeration", ["perturb_type1"], "/out")
        mock_save.assert_not_called()

    @patch("src.data_preprocessing.perturbations.main_perturb_all.save_data")
    @patch(
        "src.data_preprocessing.perturbations.main_perturb_all."
        "remove_unperturbed_claims"
    )
    def test_output_dir_used_in_path(
        self,
        mock_remove: MagicMock,
        mock_save: MagicMock,
    ) -> None:
        """Verifies that the output directory appears in the saved path."""
        mock_remove.side_effect = lambda data, keys: data
        results = {"True": [{"claim": "A."}]}
        save_splits(results, "negative", ["perturb_type2"], "/custom/dir")
        saved_path = mock_save.call_args[0][1]
        assert saved_path.startswith("/custom/dir")


# ── run_perturbations ──────────────────────────────────────────────


class TestRunPerturbations:
    """Tests for the run_perturbations function."""

    @patch("src.data_preprocessing.perturbations.main_perturb_all.obfuscate")
    @patch("src.data_preprocessing.perturbations.main_perturb_all.range_pert")
    @patch("src.data_preprocessing.perturbations.main_perturb_all.numeration")
    @patch("src.data_preprocessing.perturbations.main_perturb_all.negative")
    @patch(
        "src.data_preprocessing.perturbations.main_perturb_all.approximation"
    )
    @patch("src.data_preprocessing.perturbations.main_perturb_all.save_splits")
    @patch(
        "src.data_preprocessing.perturbations.main_perturb_all.split_by_label"
    )
    @patch("src.data_preprocessing.perturbations.main_perturb_all.load_data")
    def test_loads_both_input_files(
        self,
        mock_load: MagicMock,
        mock_split: MagicMock,
        mock_save: MagicMock,
        mock_approx: MagicMock,
        mock_neg: MagicMock,
        mock_num: MagicMock,
        mock_range: MagicMock,
        mock_obf: MagicMock,
    ) -> None:
        """Verifies that load_data is called for both input file paths."""
        mock_load.return_value = []
        mock_split.return_value = {}
        mock_neg.process_json.return_value = []
        mock_num.process_json.return_value = []
        mock_range.process_json.return_value = []
        mock_obf.process_json.return_value = {}

        run_perturbations("path/true.json", "path/false.json", "/out")

        mock_load.assert_any_call("path/true.json")
        mock_load.assert_any_call("path/false.json")

    @patch("src.data_preprocessing.perturbations.main_perturb_all.obfuscate")
    @patch("src.data_preprocessing.perturbations.main_perturb_all.range_pert")
    @patch("src.data_preprocessing.perturbations.main_perturb_all.numeration")
    @patch("src.data_preprocessing.perturbations.main_perturb_all.negative")
    @patch(
        "src.data_preprocessing.perturbations.main_perturb_all.approximation"
    )
    @patch("src.data_preprocessing.perturbations.main_perturb_all.save_splits")
    @patch(
        "src.data_preprocessing.perturbations.main_perturb_all.split_by_label"
    )
    @patch("src.data_preprocessing.perturbations.main_perturb_all.load_data")
    def test_calls_approximation_with_both_inputs(
        self,
        mock_load: MagicMock,
        mock_split: MagicMock,
        mock_save: MagicMock,
        mock_approx: MagicMock,
        mock_neg: MagicMock,
        mock_num: MagicMock,
        mock_range: MagicMock,
        mock_obf: MagicMock,
    ) -> None:
        """Verifies that approximation is called for both input files."""
        mock_load.return_value = []
        mock_split.return_value = {}
        mock_neg.process_json.return_value = []
        mock_num.process_json.return_value = []
        mock_range.process_json.return_value = []
        mock_obf.process_json.return_value = {}

        run_perturbations("path/true.json", "path/false.json", "/out")

        expected_calls = [
            call("path/true.json", "/out/approximation_True.json"),
            call("path/false.json", "/out/approximation_False.json"),
        ]
        mock_approx.process_claims.assert_has_calls(expected_calls)

    @patch("src.data_preprocessing.perturbations.main_perturb_all.obfuscate")
    @patch("src.data_preprocessing.perturbations.main_perturb_all.range_pert")
    @patch("src.data_preprocessing.perturbations.main_perturb_all.numeration")
    @patch("src.data_preprocessing.perturbations.main_perturb_all.negative")
    @patch(
        "src.data_preprocessing.perturbations.main_perturb_all.approximation"
    )
    @patch("src.data_preprocessing.perturbations.main_perturb_all.save_splits")
    @patch(
        "src.data_preprocessing.perturbations.main_perturb_all.split_by_label"
    )
    @patch("src.data_preprocessing.perturbations.main_perturb_all.load_data")
    def test_calls_all_perturbation_modules(
        self,
        mock_load: MagicMock,
        mock_split: MagicMock,
        mock_save: MagicMock,
        mock_approx: MagicMock,
        mock_neg: MagicMock,
        mock_num: MagicMock,
        mock_range: MagicMock,
        mock_obf: MagicMock,
    ) -> None:
        """Verifies that all four perturbation modules are invoked."""
        mock_load.return_value = []
        mock_split.return_value = {}
        mock_neg.process_json.return_value = []
        mock_num.process_json.return_value = []
        mock_range.process_json.return_value = []
        mock_obf.process_json.return_value = {}

        run_perturbations("path/true.json", "path/false.json", "/out")

        mock_neg.process_json.assert_called_once()
        mock_num.process_json.assert_called_once()
        mock_range.process_json.assert_called_once()
        mock_obf.process_json.assert_called_once()


# ── main (CLI) ─────────────────────────────────────────────────────


class TestMain:
    """Tests for the main CLI entry point."""

    @patch(
        "src.data_preprocessing.perturbations.main_perturb_all."
        "run_perturbations"
    )
    def test_default_split_is_used(self, mock_run: MagicMock) -> None:
        """Verifies that the default split is used when no arg is passed."""
        with patch("sys.argv", ["main.py"]):
            main()
        mock_run.assert_called_once()

    @patch(
        "src.data_preprocessing.perturbations.main_perturb_all."
        "run_perturbations"
    )
    def test_single_split_calls_run_once(self, mock_run: MagicMock) -> None:
        """Verifies that a single split triggers exactly one run."""
        with patch("sys.argv", ["main.py", "--split", "val"]):
            main()
        mock_run.assert_called_once()

    @patch(
        "src.data_preprocessing.perturbations.main_perturb_all."
        "run_perturbations"
    )
    def test_all_split_calls_run_three_times(self, mock_run: MagicMock) -> None:
        """Verifies that '--split all' triggers three separate runs."""
        with patch("sys.argv", ["main.py", "--split", "all"]):
            main()
        assert mock_run.call_count == 3

    @patch(
        "src.data_preprocessing.perturbations.main_perturb_all."
        "run_perturbations"
    )
    def test_all_split_covers_train_val_test(self, mock_run: MagicMock) -> None:
        """Verifies that all three split names are passed when using 'all'."""
        with patch("sys.argv", ["main.py", "--split", "all"]):
            main()
        called_with = [c[0] for c in mock_run.call_args_list]
        input_trues = [args[0] for args in called_with]
        for split in ["train", "val", "test"]:
            assert any(split in p for p in input_trues)

    @patch(
        "src.data_preprocessing.perturbations.main_perturb_all."
        "run_perturbations"
    )
    def test_correct_paths_passed_for_test_split(
        self, mock_run: MagicMock
    ) -> None:
        """Verifies that the correct paths are passed for the test split."""
        with patch("sys.argv", ["main.py", "--split", "test"]):
            main()
        input_true, input_false, output_dir = mock_run.call_args[0]
        assert "test" in input_true
        assert "test" in input_false
        assert "test" in output_dir

    def test_invalid_split_raises_system_exit(self) -> None:
        """Verifies that an invalid split argument causes a SystemExit."""
        with patch("sys.argv", ["main.py", "--split", "invalid"]):
            with pytest.raises(SystemExit):
                main()
