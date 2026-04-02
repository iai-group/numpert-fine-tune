"""Tests for src/llm_finetune/utils/data_loading/loader.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset

from src.llm_finetune.utils.data_loading.loader import (
    _build_allowed_types_by_file,
    _build_doc_lookup,
    _enrich_samples,
    _resolve_file_paths,
    load_split,
)
from src.llm_finetune.utils.data_loading.perturbation_types import (
    FileLoadConfig,
    PerturbationType,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LOADER = "src.llm_finetune.utils.data_loading.loader"


def _make_raw_record(claim_id: str, doc: str = "evidence text") -> dict:
    return {
        "claim_id": claim_id,
        "doc": doc,
        "claim": "some claim",
        "label": "True",
    }


def _make_balanced_sample(
    source_claim_id: str,
    is_perturbed: bool = False,
    type_: str = "T→T",
    perturbation_method: str | None = None,
    source_file: str = "file.json",
    label: str = "True",
) -> dict:
    return {
        "source_claim_id": source_claim_id,
        "text": f"claim text for {source_claim_id}",
        "label": label,
        "is_perturbed": is_perturbed,
        "type": type_,
        "perturbation_method": perturbation_method,
        "source_file": source_file,
    }


# ---------------------------------------------------------------------------
# _resolve_file_paths
# ---------------------------------------------------------------------------


class TestResolveFilePaths:
    def test_uses_file_configs_when_provided(self):
        configs = [FileLoadConfig(path="a.json"), FileLoadConfig(path="b.json")]
        result = _resolve_file_paths(configs, json_paths=["c.json"])
        assert result == ["a.json", "b.json"]

    def test_falls_back_to_json_paths(self):
        result = _resolve_file_paths(
            file_configs=None, json_paths=["x.json", "y.json"]
        )
        assert result == ["x.json", "y.json"]

    def test_empty_file_configs_falls_back_to_json_paths(self):
        result = _resolve_file_paths(file_configs=[], json_paths=["z.json"])
        assert result == ["z.json"]

    def test_both_none_returns_empty(self):
        assert _resolve_file_paths(None, None) == []

    def test_both_empty_returns_empty(self):
        assert _resolve_file_paths([], []) == []


# ---------------------------------------------------------------------------
# _build_doc_lookup
# ---------------------------------------------------------------------------


class TestBuildDocLookup:
    def test_maps_claim_id_to_doc(self):
        raw = [
            (
                "file.json",
                [
                    _make_raw_record("c1", "doc one"),
                    _make_raw_record("c2", "doc two"),
                ],
            )
        ]
        with patch(f"{_LOADER}._load_raw_json_files", return_value=raw):
            result = _build_doc_lookup(["file.json"])
        assert result == {"c1": "doc one", "c2": "doc two"}

    def test_first_occurrence_wins_across_files(self):
        raw = [
            ("file1.json", [_make_raw_record("c1", "first")]),
            ("file2.json", [_make_raw_record("c1", "second")]),
        ]
        with patch(f"{_LOADER}._load_raw_json_files", return_value=raw):
            result = _build_doc_lookup(["file1.json", "file2.json"])
        assert result["c1"] == "first"

    def test_records_missing_claim_id_are_skipped(self):
        raw = [("file.json", [{"doc": "no id here"}])]
        with patch(f"{_LOADER}._load_raw_json_files", return_value=raw):
            result = _build_doc_lookup(["file.json"])
        assert result == {}

    def test_records_missing_doc_default_to_empty_string(self):
        raw = [("file.json", [{"claim_id": "c1"}])]
        with patch(f"{_LOADER}._load_raw_json_files", return_value=raw):
            result = _build_doc_lookup(["file.json"])
        assert result == {"c1": ""}


# ---------------------------------------------------------------------------
# _enrich_samples
# ---------------------------------------------------------------------------


class TestEnrichSamples:
    def test_non_perturbed_gets_original_normalized_type(self):
        sample = _make_balanced_sample("c1", is_perturbed=False, type_="T→T")
        doc_lookup = {"c1": "evidence"}
        result = _enrich_samples([sample], doc_lookup)
        assert (
            result[0]["perturbation_type"]
            == PerturbationType.ORIGINAL_NORMALIZED.value
        )

    def test_perturbed_arrow_replaced_with_2(self):
        sample = _make_balanced_sample("c1", is_perturbed=True, type_="T→F")
        result = _enrich_samples([sample], {"c1": "doc"})
        assert result[0]["perturbation_type"] == "T2F"

    def test_doc_filled_from_lookup(self):
        sample = _make_balanced_sample("c1")
        result = _enrich_samples([sample], {"c1": "the evidence"})
        assert result[0]["doc"] == "the evidence"

    def test_missing_claim_id_in_lookup_gives_empty_doc(self):
        sample = _make_balanced_sample("c99")
        result = _enrich_samples([sample], {})
        assert result[0]["doc"] == ""

    def test_none_perturbation_method_becomes_empty_string(self):
        sample = _make_balanced_sample("c1", perturbation_method=None)
        result = _enrich_samples([sample], {"c1": "doc"})
        assert result[0]["perturbation_method"] == ""

    def test_output_keys(self):
        sample = _make_balanced_sample(
            "c1", is_perturbed=True, type_="T→F", perturbation_method="approx"
        )
        result = _enrich_samples([sample], {"c1": "doc"})
        expected_keys = {
            "claim_id",
            "claim",
            "doc",
            "label",
            "perturbation_type",
            "source_file",
            "perturbation_method",
        }
        assert set(result[0].keys()) == expected_keys

    def test_multiple_samples_all_enriched(self):
        samples = [
            _make_balanced_sample("c1", is_perturbed=False),
            _make_balanced_sample("c2", is_perturbed=True, type_="T→F"),
        ]
        result = _enrich_samples(samples, {"c1": "doc1", "c2": "doc2"})
        assert len(result) == 2
        assert (
            result[0]["perturbation_type"]
            == PerturbationType.ORIGINAL_NORMALIZED.value
        )
        assert result[1]["perturbation_type"] == "T2F"


# ---------------------------------------------------------------------------
# load_split — unbalanced path
# ---------------------------------------------------------------------------


class TestLoadSplitUnbalanced:
    def test_delegates_to_load_fact_checking_data(self):
        mock_ds = MagicMock(spec=Dataset)
        with patch(
            f"{_LOADER}.load_fact_checking_data", return_value=mock_ds
        ) as mock_lfc:
            result = load_split(json_paths=["t.json"], split="train")
        mock_lfc.assert_called_once_with(
            json_paths=["t.json"],
            file_configs=None,
            split="train",
            expand_perturbations_flag=True,
            include_doc=True,
        )
        assert result is mock_ds

    def test_expand_perturbations_forwarded(self):
        with patch(
            f"{_LOADER}.load_fact_checking_data", return_value=MagicMock()
        ) as mock_lfc:
            load_split(json_paths=["t.json"], expand_perturbations=False)
        assert mock_lfc.call_args[1]["expand_perturbations_flag"] is False

    def test_include_doc_forwarded(self):
        with patch(
            f"{_LOADER}.load_fact_checking_data", return_value=MagicMock()
        ) as mock_lfc:
            load_split(json_paths=["t.json"], include_doc=False)
        assert mock_lfc.call_args[1]["include_doc"] is False

    def test_file_configs_forwarded(self):
        configs = [FileLoadConfig(path="a.json")]
        with patch(
            f"{_LOADER}.load_fact_checking_data", return_value=MagicMock()
        ) as mock_lfc:
            load_split(file_configs=configs)
        assert mock_lfc.call_args[1]["file_configs"] == configs

    def test_balancer_not_called_on_unbalanced_path(self):
        with patch(
            f"{_LOADER}.load_fact_checking_data", return_value=MagicMock()
        ):
            with patch(f"{_LOADER}.build_balanced_dataset") as mock_bal:
                load_split(json_paths=["t.json"], use_balancer=False)
        mock_bal.assert_not_called()


# ---------------------------------------------------------------------------
# load_split — balanced path
# ---------------------------------------------------------------------------


def _patch_balanced_path(balanced_samples, doc_lookup_raw, conversation_data):
    """Context-manager stack for the balanced path."""
    import contextlib

    @contextlib.contextmanager
    def ctx():
        with patch(
            f"{_LOADER}._load_raw_json_files", return_value=doc_lookup_raw
        ) as p1, patch(
            f"{_LOADER}.build_balanced_dataset", return_value=balanced_samples
        ) as p2, patch(
            f"{_LOADER}.convert_to_conversations",
            return_value=conversation_data,
        ) as p3:
            yield p1, p2, p3

    return ctx()


class TestLoadSplitBalanced:
    def _balanced_sample(self, cid="c1"):
        return _make_balanced_sample(cid, is_perturbed=False)

    def _raw_file_data(self, cid="c1", doc="evidence"):
        return [("f.json", [_make_raw_record(cid, doc)])]

    def _conversation_row(self, cid="c1"):
        return {
            "claim_id": cid,
            "conversations": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ],
            "label": '{"label": true}',
            "perturbation_type": PerturbationType.ORIGINAL_NORMALIZED.value,
        }

    def test_raises_when_no_paths(self):
        with pytest.raises(ValueError, match="at least one file path"):
            load_split(use_balancer=True)

    def test_returns_dataset(self):
        samples = [self._balanced_sample()]
        rows = [self._conversation_row()]
        with _patch_balanced_path(samples, self._raw_file_data(), rows):
            result = load_split(json_paths=["f.json"], use_balancer=True)
        assert isinstance(result, Dataset)
        assert len(result) == 1

    def test_build_balanced_dataset_kwargs_passed(self):
        samples = [self._balanced_sample()]
        rows = [self._conversation_row()]
        with _patch_balanced_path(samples, self._raw_file_data(), rows) as (
            _,
            mock_bal,
            _c,
        ):
            load_split(
                json_paths=["f.json"],
                use_balancer=True,
                balance_labels=True,
                perturb_ratio=0.5,
                false_perturb_t_to_f_ratio=0.3,
                equal_perturbation_methods=True,
                seed=42,
            )
        mock_bal.assert_called_once_with(
            file_paths=["f.json"],
            balance_labels=True,
            perturb_ratio=0.5,
            false_perturb_t_to_f_ratio=0.3,
            equal_method_size=True,
            seed=42,
            allowed_types_by_file=None,
        )

    def test_convert_to_conversations_receives_split_and_include_doc(self):
        samples = [self._balanced_sample()]
        rows = [self._conversation_row()]
        with _patch_balanced_path(samples, self._raw_file_data(), rows) as (
            _r,
            _b,
            mock_conv,
        ):
            load_split(
                json_paths=["f.json"],
                use_balancer=True,
                split="validation",
                include_doc=False,
            )
        mock_conv.assert_called_once()
        _, kwargs = mock_conv.call_args
        assert kwargs["split"] == "validation"
        assert kwargs["include_doc"] is False

    def test_doc_enriched_from_raw_files(self):
        samples = [self._balanced_sample("c1")]
        raw = self._raw_file_data("c1", "expected doc")
        rows = [self._conversation_row("c1")]
        captured: list[list[dict]] = []

        def capture_conv(data, **kw):
            captured.append(data)
            return rows

        with patch(f"{_LOADER}._load_raw_json_files", return_value=raw), patch(
            f"{_LOADER}.build_balanced_dataset", return_value=samples
        ), patch(
            f"{_LOADER}.convert_to_conversations", side_effect=capture_conv
        ):
            load_split(json_paths=["f.json"], use_balancer=True)

        assert captured[0][0]["doc"] == "expected doc"

    def test_file_configs_resolved_to_paths(self):
        configs = [FileLoadConfig(path="cfg.json")]
        samples = [self._balanced_sample()]
        rows = [self._conversation_row()]
        with _patch_balanced_path(samples, self._raw_file_data(), rows) as (
            _r,
            mock_bal,
            _c,
        ):
            load_split(file_configs=configs, use_balancer=True)
        assert mock_bal.call_args[1]["file_paths"] == ["cfg.json"]
        assert mock_bal.call_args[1]["allowed_types_by_file"] is None

    def test_load_fact_checking_data_not_called_on_balanced_path(self):
        samples = [self._balanced_sample()]
        rows = [self._conversation_row()]
        with _patch_balanced_path(samples, self._raw_file_data(), rows):
            with patch(f"{_LOADER}.load_fact_checking_data") as mock_lfc:
                load_split(json_paths=["f.json"], use_balancer=True)
        mock_lfc.assert_not_called()

    def test_perturb_types_forwarded_to_balancer(self):
        """FileLoadConfig.perturb_types should reach build_balanced_dataset."""
        configs = [
            FileLoadConfig(
                path="/data/approx.json",
                perturb_types=[PerturbationType.PERTURB_TYPE1],
            ),
        ]
        samples = [self._balanced_sample()]
        rows = [self._conversation_row()]
        with _patch_balanced_path(samples, self._raw_file_data(), rows) as (
            _r,
            mock_bal,
            _c,
        ):
            load_split(file_configs=configs, use_balancer=True)
        expected_mapping = {"approx.json": {"type1"}}
        assert (
            mock_bal.call_args[1]["allowed_types_by_file"] == expected_mapping
        )

    def test_mixed_configs_with_and_without_perturb_types(self):
        """Only configs with explicit perturb_types appear in the mapping."""
        configs = [
            FileLoadConfig(
                path="/data/approx.json",
                perturb_types=[
                    PerturbationType.PERTURB_TYPE1,
                    PerturbationType.PERTURB_TYPE2,
                ],
            ),
            FileLoadConfig(path="/data/all_types.json"),  # None = all types
        ]
        samples = [self._balanced_sample()]
        rows = [self._conversation_row()]
        with _patch_balanced_path(samples, self._raw_file_data(), rows) as (
            _r,
            mock_bal,
            _c,
        ):
            load_split(file_configs=configs, use_balancer=True)
        mapping = mock_bal.call_args[1]["allowed_types_by_file"]
        assert mapping == {"approx.json": {"type1", "type2"}}
        # all_types.json not in mapping → no restriction
        assert "all_types.json" not in mapping


# ---------------------------------------------------------------------------
# _build_allowed_types_by_file
# ---------------------------------------------------------------------------


class TestBuildAllowedTypesByFile:
    def test_returns_none_when_no_configs(self):
        assert _build_allowed_types_by_file(None) is None
        assert _build_allowed_types_by_file([]) is None

    def test_returns_none_when_no_config_has_types(self):
        configs = [
            FileLoadConfig(path="a.json"),
            FileLoadConfig(path="b.json"),
        ]
        assert _build_allowed_types_by_file(configs) is None

    def test_builds_mapping_for_typed_configs(self):
        configs = [
            FileLoadConfig(
                path="/dir/a.json",
                perturb_types=[PerturbationType.PERTURB_TYPE1],
            ),
        ]
        result = _build_allowed_types_by_file(configs)
        assert result == {"a.json": {"type1"}}

    def test_original_normalized_excluded_from_mapping(self):
        configs = [
            FileLoadConfig(
                path="a.json",
                perturb_types=[
                    PerturbationType.ORIGINAL_NORMALIZED,
                    PerturbationType.PERTURB_TYPE2,
                ],
            ),
        ]
        result = _build_allowed_types_by_file(configs)
        assert result == {"a.json": {"type2"}}

    def test_multiple_types_combined(self):
        configs = [
            FileLoadConfig(
                path="a.json",
                perturb_types=[
                    PerturbationType.PERTURB_TYPE1,
                    PerturbationType.PERTURB_TYPE3_RANDOM,
                    PerturbationType.PERTURB_TYPE3_MASKED,
                ],
            ),
        ]
        result = _build_allowed_types_by_file(configs)
        assert result == {"a.json": {"type1", "type3_random", "type3_masked"}}

    def test_untyped_configs_omitted_from_mapping(self):
        configs = [
            FileLoadConfig(
                path="typed.json",
                perturb_types=[PerturbationType.PERTURB_TYPE1],
            ),
            FileLoadConfig(path="untyped.json"),
        ]
        result = _build_allowed_types_by_file(configs)
        assert "typed.json" in result
        assert "untyped.json" not in result
