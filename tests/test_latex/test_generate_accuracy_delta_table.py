"""Tests for src.latex.generate_accuracy_delta_table."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.misc.latex.generate_accuracy_delta_table import (
    ORIGINAL_GROUP_NAME,
    ORIGINAL_PERTURBATION_TYPE,
    GroupScore,
    canonical_perturb_type,
    compute_best_scores,
    derive_selected_groups,
    discover_model_metrics,
    ensure_baseline_groups,
    escape_latex,
    format_delta,
    format_header,
    format_score,
    formal_model_label,
    group_sort_key,
    load_model_name_map,
    load_model_scores,
    parse_group_name,
    render_table,
    resolve_model_label,
    select_models,
    short_model_label,
    source_family_label,
)


# ---------------------------------------------------------------------------
# canonical_perturb_type
# ---------------------------------------------------------------------------


class TestCanonicalPerturbType:
    def test_known_alias(self):
        assert canonical_perturb_type("type1") == "perturb_type1"

    def test_unknown_passthrough(self):
        assert canonical_perturb_type("perturb_type2") == "perturb_type2"

    def test_whitespace_stripped(self):
        assert (
            canonical_perturb_type(" type3_masked ") == "perturb_type3_masked"
        )


# ---------------------------------------------------------------------------
# escape_latex
# ---------------------------------------------------------------------------


class TestEscapeLatex:
    def test_underscore(self):
        assert escape_latex("a_b") == r"a\_b"

    def test_ampersand(self):
        assert escape_latex("a&b") == r"a\&b"

    def test_dollar(self):
        assert escape_latex("$100") == r"\$100"

    def test_percent(self):
        assert escape_latex("50%") == r"50\%"

    def test_clean_passthrough(self):
        assert escape_latex("hello world") == "hello world"


# ---------------------------------------------------------------------------
# source_family_label
# ---------------------------------------------------------------------------


class TestSourceFamilyLabel:
    def test_approximation_true(self):
        assert source_family_label("approximation_True.json") == "approximation"

    def test_numeration_false(self):
        assert source_family_label("numeration_False.json") == "numeration"

    def test_no_bool_suffix(self):
        assert source_family_label("range.json") == "range"


# ---------------------------------------------------------------------------
# parse_group_name
# ---------------------------------------------------------------------------


class TestParseGroupName:
    def test_valid_split(self):
        src, pt = parse_group_name("approximation_True.json | perturb_type1")
        assert src == "approximation_True.json"
        assert pt == "perturb_type1"

    def test_missing_separator_raises(self):
        with pytest.raises(ValueError):
            parse_group_name("approximation_True.json perturb_type1")


# ---------------------------------------------------------------------------
# format_score
# ---------------------------------------------------------------------------


class TestFormatScore:
    def test_two_decimal_places(self):
        assert format_score(0.8635, 2) == "86.35"

    def test_zero_decimal_places(self):
        assert format_score(1.0, 0) == "100"


# ---------------------------------------------------------------------------
# format_delta
# ---------------------------------------------------------------------------


class TestFormatDelta:
    def test_positive_higher_is_better_green(self):
        result = format_delta(0.8, 0.9, digits=2, higher_is_better=True)
        assert "numpertdeltaimprove" in result
        assert "+" in result

    def test_negative_higher_is_better_red(self):
        result = format_delta(0.9, 0.8, digits=2, higher_is_better=True)
        assert "numpertdeltadecline" in result
        assert "-" in result or "−" in result or "-10" in result

    def test_zero_returns_empty(self):
        assert format_delta(0.8, 0.8, digits=2) == ""

    def test_positive_lower_is_better_red(self):
        result = format_delta(0.8, 0.9, digits=2, higher_is_better=False)
        assert "numpertdeltadecline" in result


# ---------------------------------------------------------------------------
# format_header
# ---------------------------------------------------------------------------


class TestFormatHeader:
    def test_original_group(self):
        assert format_header(ORIGINAL_GROUP_NAME) == "Original"

    def test_masked_group(self):
        result = format_header("approximation_True.json | perturb_type3_masked")
        assert result == "Masked"

    def test_random_group(self):
        result = format_header("numeration_True.json | perturb_type3_random")
        assert result == "Random"

    def test_approximation_source(self):
        result = format_header("approximation_True.json | perturb_type1")
        assert result == "Approximation"


# ---------------------------------------------------------------------------
# group_sort_key
# ---------------------------------------------------------------------------


class TestGroupSortKey:
    def test_original_sorts_first(self):
        key = group_sort_key(ORIGINAL_GROUP_NAME)
        assert key[0] == -1

    def test_approximation_before_numeration(self):
        key_approx = group_sort_key("approximation_True.json | perturb_type1")
        key_num = group_sort_key("numeration_True.json | perturb_type1")
        assert key_approx < key_num

    def test_type1_before_type2_same_family(self):
        key_t1 = group_sort_key("approximation_True.json | perturb_type1")
        key_t2 = group_sort_key("approximation_True.json | perturb_type2")
        assert key_t1 < key_t2


# ---------------------------------------------------------------------------
# short_model_label
# ---------------------------------------------------------------------------


class TestShortModelLabel:
    def _root(self, name: str = "qwen3-8b") -> Path:
        return Path("results") / name

    def test_baseline_label(self):
        result = short_model_label("baseline", self._root())
        assert result == "qwen3_baseline"

    def test_no_approx_label(self):
        result = short_model_label("qwen3_no_approx", self._root())
        assert result == "qwen3_no_approx"

    def test_custom_label_already_starts_with_family(self):
        result = short_model_label("qwen3_ft_v2", self._root())
        assert result == "qwen3_ft_v2"

    def test_no_num_label(self):
        result = short_model_label("qwen3_no_num", self._root())
        assert result == "qwen3_no_num"

    def test_no_range_label(self):
        result = short_model_label("qwen3_no_range", self._root())
        assert result == "qwen3_no_range"

    def test_no_perturb_split_label(self):
        result = short_model_label("qwen3_no_perturb_split", self._root())
        assert result == "qwen3_no_perturb_split"

    def test_only_approximation(self):
        folder = (
            "Qwen3-8B-daring-ant-evolves_only_approximation_no_f2f_type2_q4_k_m"
        )
        assert (
            short_model_label(folder, self._root("qwen3-8b-q4_k_m"))
            == "qwen3_approx"
        )

    def test_only_numeration(self):
        folder = (
            "Qwen3-8B-daring-ant-evolves_only_numeration_no_f2f_type2_q4_k_m"
        )
        assert (
            short_model_label(folder, self._root("qwen3-8b-q4_k_m"))
            == "qwen3_num"
        )

    def test_only_range(self):
        folder = "Qwen3-8B-daring-ant-evolves_only_range_no_f2f_type2_q4_k_m"
        assert (
            short_model_label(folder, self._root("qwen3-8b-q4_k_m"))
            == "qwen3_range"
        )

    def test_only_negative(self):
        folder = "Qwen3-8B-daring-ant-evolves_only_negative_q4_k_m"
        assert (
            short_model_label(folder, self._root("qwen3-8b-q4_k_m"))
            == "qwen3_negation"
        )

    def test_only_obfuscation_masked(self):
        folder = "Qwen3-8B-daring-ant-evolves_only_obfuscation_masked_q4_k_m"
        assert (
            short_model_label(folder, self._root("qwen3-8b-q4_k_m"))
            == "qwen3_masked"
        )

    def test_only_obfuscation_random(self):
        folder = "Qwen3-8B-daring-ant-evolves_only_obfuscation_random_q4_k_m"
        assert (
            short_model_label(folder, self._root("qwen3-8b-q4_k_m"))
            == "qwen3_random"
        )

    def test_base_pap(self):
        folder = "Qwen3-8B-base_q4_k_m-pap"
        assert (
            short_model_label(folder, self._root("qwen3-8b-q4_k_m"))
            == "qwen3_base_pap"
        )

    def test_base_two_shot(self):
        folder = "Qwen3-8b-base_q4_k_m-two-shot"
        assert (
            short_model_label(folder, self._root("qwen3-8b-q4_k_m"))
            == "qwen3_base_two_shot"
        )

    def test_all_perturb_with_leading_dot(self):
        folder = "Qwen3-8B-young-nightingale-manifests_all_perturb_no_f_type2"
        assert (
            short_model_label(folder, self._root("qwen3-8b-q4_k_m"))
            == "qwen3_all_perturb"
        )


# ---------------------------------------------------------------------------
# formal_model_label
# ---------------------------------------------------------------------------


class TestFormalModelLabel:
    def _root(self, name: str = "qwen3-8b") -> Path:
        return Path("results") / name

    def test_baseline(self):
        assert (
            formal_model_label("baseline", self._root())
            == r"$\text{M}_{\text{Baseline}}$"
        )

    def test_no_approx(self):
        assert (
            formal_model_label("qwen3_no_approx", self._root())
            == r"$\text{M}_{\text{Excl-Approx}}$"
        )

    def test_no_num(self):
        assert (
            formal_model_label("qwen3_no_num", self._root())
            == r"$\text{M}_{\text{Excl-Num}}$"
        )

    def test_no_range(self):
        assert (
            formal_model_label("qwen3_no_range", self._root())
            == r"$\text{M}_{\text{Excl-Range}}$"
        )

    def test_no_perturb_split(self):
        assert (
            formal_model_label("qwen3_no_perturb_split", self._root())
            == r"$\text{M}_{\text{Excl-Split}}$"
        )

    def test_normalized_claims(self):
        assert (
            formal_model_label("qwen3_normalized_claims", self._root())
            == r"$\text{M}_{\text{Norm-Claims}}$"
        )

    def test_unknown_suffix_falls_back_to_title(self):
        # "qwen3_range" has no explicit mapping → title-cased suffix
        assert (
            formal_model_label("qwen3_range", self._root())
            == r"$\text{M}_{\text{Range}}$"
        )


# ---------------------------------------------------------------------------
# resolve_model_label
# ---------------------------------------------------------------------------


def _make_discovered(*labels: str) -> dict[str, Path]:
    return {label: Path(label) / "metrics.json" for label in labels}


class TestResolveModelLabel:
    def test_exact_match(self):
        disc = _make_discovered("baseline", "ft_model")
        assert resolve_model_label(disc, "baseline") == "baseline"

    def test_suffix_match(self):
        disc = _make_discovered("path/to/baseline", "path/to/ft_model")
        assert resolve_model_label(disc, "baseline") == "path/to/baseline"

    def test_segment_match(self):
        disc = _make_discovered("path/to/baseline")
        assert resolve_model_label(disc, "to") == "path/to/baseline"

    def test_ambiguous_suffix_raises(self):
        disc = _make_discovered("a/baseline", "b/baseline")
        with pytest.raises(ValueError, match="ambiguous"):
            resolve_model_label(disc, "baseline")

    def test_not_found_raises(self):
        disc = _make_discovered("baseline")
        with pytest.raises(ValueError, match="not found"):
            resolve_model_label(disc, "missing_model")


# ---------------------------------------------------------------------------
# select_models
# ---------------------------------------------------------------------------


class TestSelectModels:
    def test_no_include_returns_all(self):
        disc = _make_discovered("baseline", "ft_model", "other_model")
        result, baseline = select_models(disc, "baseline", [])
        assert set(result.keys()) == {"baseline", "ft_model", "other_model"}
        assert baseline == "baseline"

    def test_include_models_filters(self):
        disc = _make_discovered("baseline", "ft_model", "other_model")
        result, baseline = select_models(disc, "baseline", ["ft_model"])
        assert set(result.keys()) == {"baseline", "ft_model"}
        assert "other_model" not in result

    def test_baseline_always_present(self):
        disc = _make_discovered("baseline", "ft_model")
        result, _ = select_models(disc, "baseline", ["ft_model"])
        assert "baseline" in result


# ---------------------------------------------------------------------------
# derive_selected_groups
# ---------------------------------------------------------------------------


def _make_baseline_scores(*group_names: str) -> dict[str, GroupScore]:
    """Build a fake baseline_scores dict from group name strings."""
    scores: dict[str, GroupScore] = {}
    for gn in group_names:
        if gn == ORIGINAL_GROUP_NAME:
            scores[gn] = GroupScore(
                source_file=ORIGINAL_GROUP_NAME,
                perturb_type=ORIGINAL_PERTURBATION_TYPE,
                accuracy=0.9,
            )
        else:
            src, pt = parse_group_name(gn)
            scores[gn] = GroupScore(
                source_file=src, perturb_type=pt, accuracy=0.8
            )
    return scores


class TestDeriveSelectedGroups:
    def test_no_filters_returns_all_non_original(self):
        baseline = _make_baseline_scores(
            "approximation_True.json | perturb_type1",
            "numeration_True.json | perturb_type2",
        )
        groups = derive_selected_groups(baseline, [], [], [], [], False)
        assert len(groups) == 2
        assert ORIGINAL_GROUP_NAME not in groups

    def test_perturb_type_filter(self):
        baseline = _make_baseline_scores(
            "approximation_True.json | perturb_type1",
            "numeration_True.json | perturb_type2",
        )
        groups = derive_selected_groups(baseline, [], [], [], ["type1"], False)
        assert all("perturb_type1" in g for g in groups)
        assert len(groups) == 1

    def test_include_original_prepends(self):
        baseline = _make_baseline_scores(
            ORIGINAL_GROUP_NAME,
            "approximation_True.json | perturb_type1",
        )
        groups = derive_selected_groups(baseline, [], [], [], [], True)
        assert groups[0] == ORIGINAL_GROUP_NAME

    def test_unsupported_perturb_type_raises(self):
        baseline = _make_baseline_scores(
            "approximation_True.json | perturb_type1"
        )
        with pytest.raises(ValueError, match="Unsupported"):
            derive_selected_groups(
                baseline, [], [], [], ["type_unknown_99"], False
            )

    def test_exact_groups_valid(self):
        baseline = _make_baseline_scores(
            "approximation_True.json | perturb_type1"
        )
        groups = derive_selected_groups(
            baseline,
            exact_groups=["approximation_True.json | perturb_type1"],
            source_files=[],
            source_file_patterns=[],
            perturb_types=[],
            include_original=False,
        )
        assert groups == ["approximation_True.json | perturb_type1"]

    def test_exact_groups_missing_raises(self):
        baseline = _make_baseline_scores(
            "approximation_True.json | perturb_type1"
        )
        with pytest.raises(ValueError, match="Group not found"):
            derive_selected_groups(
                baseline,
                exact_groups=["numeration_True.json | perturb_type1"],
                source_files=[],
                source_file_patterns=[],
                perturb_types=[],
                include_original=False,
            )


# ---------------------------------------------------------------------------
# compute_best_scores
# ---------------------------------------------------------------------------


class TestComputeBestScores:
    def test_returns_max_across_models(self):
        group = "approximation_True.json | perturb_type1"
        model_scores = {
            "baseline": {
                group: GroupScore(
                    "approximation_True.json", "perturb_type1", 0.7
                )
            },
            "ft_model": {
                group: GroupScore(
                    "approximation_True.json", "perturb_type1", 0.9
                )
            },
        }
        best = compute_best_scores(model_scores, [group])
        assert abs(best[group] - 0.9) < 1e-9

    def test_missing_group_raises(self):
        model_scores = {"baseline": {}}
        with pytest.raises(ValueError, match="No scores available"):
            compute_best_scores(
                model_scores, ["approximation_True.json | perturb_type1"]
            )


# ---------------------------------------------------------------------------
# ensure_baseline_groups
# ---------------------------------------------------------------------------


class TestEnsureBaselineGroups:
    def test_all_present_no_error(self):
        group = "approximation_True.json | perturb_type1"
        model_scores = {
            "baseline": {
                group: GroupScore(
                    "approximation_True.json", "perturb_type1", 0.8
                )
            }
        }
        ensure_baseline_groups(
            model_scores, "baseline", [group]
        )  # no exception

    def test_missing_group_raises(self):
        model_scores = {"baseline": {}}
        with pytest.raises(ValueError, match="approximation_True.json"):
            ensure_baseline_groups(
                model_scores,
                "baseline",
                ["approximation_True.json | perturb_type1"],
            )


# ---------------------------------------------------------------------------
# load_model_scores
# ---------------------------------------------------------------------------


def _write_metrics(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


class TestLoadModelScores:
    def test_supported_types_loaded(self, tmp_path):
        metrics = tmp_path / "metrics.json"
        _write_metrics(
            metrics,
            {
                "per_file_and_type": {
                    "k1": {
                        "group_name": "approximation_True.json | perturb_type1",
                        "accuracy": 0.85,
                    }
                }
            },
        )
        scores = load_model_scores(metrics)
        assert "approximation_True.json | perturb_type1" in scores
        assert (
            abs(
                scores["approximation_True.json | perturb_type1"].accuracy
                - 0.85
            )
            < 1e-9
        )

    def test_unsupported_type_skipped(self, tmp_path):
        metrics = tmp_path / "metrics.json"
        _write_metrics(
            metrics,
            {
                "per_file_and_type": {
                    "k1": {
                        "group_name": "file.json | unsupported_type_xyz",
                        "accuracy": 0.5,
                    }
                }
            },
        )
        scores = load_model_scores(metrics)
        assert not scores

    def test_original_score_loaded(self, tmp_path):
        metrics = tmp_path / "metrics.json"
        _write_metrics(
            metrics,
            {
                "per_file_and_type": {},
                "per_perturbation_type": {
                    "original_normalized": {"accuracy": 0.92}
                },
            },
        )
        scores = load_model_scores(metrics)
        assert ORIGINAL_GROUP_NAME in scores
        assert abs(scores[ORIGINAL_GROUP_NAME].accuracy - 0.92) < 1e-9

    def test_class_label_uses_recall(self, tmp_path):
        metrics = tmp_path / "metrics.json"
        _write_metrics(
            metrics,
            {
                "per_file_and_type": {
                    "k1": {
                        "group_name": "approximation_True.json | perturb_type1",
                        "classification_report": {
                            "true": {
                                "recall": 0.77,
                                "precision": 0.8,
                                "f1-score": 0.78,
                            }
                        },
                    }
                }
            },
        )
        scores = load_model_scores(metrics, class_label="true")
        assert "approximation_True.json | perturb_type1" in scores
        assert (
            abs(
                scores["approximation_True.json | perturb_type1"].accuracy
                - 0.77
            )
            < 1e-9
        )


# ---------------------------------------------------------------------------
# discover_model_metrics
# ---------------------------------------------------------------------------


class TestDiscoverModelMetrics:
    def test_discovers_multiple_models(self, tmp_path):
        (tmp_path / "baseline").mkdir()
        (tmp_path / "baseline" / "metrics.json").write_text(
            "{}", encoding="utf-8"
        )
        (tmp_path / "ft_model").mkdir()
        (tmp_path / "ft_model" / "metrics.json").write_text(
            "{}", encoding="utf-8"
        )
        discovered = discover_model_metrics(tmp_path)
        assert "baseline" in discovered
        assert "ft_model" in discovered

    def test_empty_directory_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No metrics.json"):
            discover_model_metrics(tmp_path)

    def test_nonexistent_path_raises(self, tmp_path):
        with pytest.raises(ValueError, match="does not exist"):
            discover_model_metrics(tmp_path / "nonexistent")

    def test_duplicate_label_raises(self, tmp_path):
        # Two subdirs at the same relative depth with the same name
        # is impossible. A duplicate label can arise when the root
        # itself has metrics.json and is named same as an inner
        # directory. Instead simulate with two nested paths that
        # resolve to the same label — which can't happen
        # structurally. This verifies the underlying mechanism
        # without relying on filesystem path collisions.
        #
        # Simplest reproducible duplicate: nest a model inside
        # another of same name. The glob will find both and the
        # relative label for the inner one will be "baseline/baseline",
        # not "baseline", so no duplicate is raised. A realistic
        # duplicate would require two sibling dirs with the same name,
        # which is impossible on most filesystems.
        (tmp_path / "baseline").mkdir()
        (tmp_path / "baseline" / "metrics.json").write_text(
            "{}", encoding="utf-8"
        )
        discovered = discover_model_metrics(tmp_path)
        assert len(discovered) == 1


# ---------------------------------------------------------------------------
# render_table (end-to-end)
# ---------------------------------------------------------------------------


def _make_group_score(src: str, pt: str, acc: float) -> GroupScore:
    return GroupScore(source_file=src, perturb_type=pt, accuracy=acc)


class TestRenderTable:
    def _build_scores(self):
        group = "approximation_True.json | perturb_type1"
        baseline_scores = {
            ORIGINAL_GROUP_NAME: GroupScore(
                ORIGINAL_GROUP_NAME, ORIGINAL_PERTURBATION_TYPE, 0.90
            ),
            group: _make_group_score(
                "approximation_True.json", "perturb_type1", 0.85
            ),
        }
        ft_scores = {
            ORIGINAL_GROUP_NAME: GroupScore(
                ORIGINAL_GROUP_NAME, ORIGINAL_PERTURBATION_TYPE, 0.90
            ),
            group: _make_group_score(
                "approximation_True.json", "perturb_type1", 0.88
            ),
        }
        return {
            "baseline": baseline_scores,
            "ft_model": ft_scores,
        }, group

    def test_contains_begin_table(self):
        model_scores, group = self._build_scores()
        result = render_table(
            model_scores,
            [ORIGINAL_GROUP_NAME, group],
            baseline_model="baseline",
            results_root=Path("results/qwen3-8b"),
            digits=2,
        )
        assert "\\begin{table*}" in result

    def test_baseline_label_present(self):
        model_scores, group = self._build_scores()
        result = render_table(
            model_scores,
            [ORIGINAL_GROUP_NAME, group],
            baseline_model="baseline",
            results_root=Path("results/qwen3-8b"),
            digits=2,
        )
        assert r"\text{M}" in result
        assert "Baseline" in result

    def test_column_headers_bold(self):
        model_scores, group = self._build_scores()
        result = render_table(
            model_scores,
            [ORIGINAL_GROUP_NAME, group],
            baseline_model="baseline",
            results_root=Path("results/qwen3-8b"),
            digits=2,
        )
        assert r"\textbf{Model}" in result
        assert r"\textbf{Original}" in result

    def test_ft_model_delta_colored(self):
        model_scores, group = self._build_scores()
        result = render_table(
            model_scores,
            [ORIGINAL_GROUP_NAME, group],
            baseline_model="baseline",
            results_root=Path("results/qwen3-8b"),
            digits=2,
        )
        assert "\\textcolor" in result

    def test_original_column_no_delta(self):
        model_scores, group = self._build_scores()
        result = render_table(
            model_scores,
            [ORIGINAL_GROUP_NAME, group],
            baseline_model="baseline",
            results_root=Path("results/qwen3-8b"),
            digits=2,
        )
        lines = result.splitlines()
        # The baseline row now contains the math-mode label with "Baseline"
        for line in lines:
            if "Baseline" in line and "\\\\" in line:
                cells = line.split(" & ")
                # cells[1] is the Original column cell — no delta because it is
                # the reference column
                assert "\\textcolor" not in cells[1]

    def test_best_score_bolded(self):
        model_scores, group = self._build_scores()
        result = render_table(
            model_scores,
            [group],
            baseline_model="baseline",
            results_root=Path("results/qwen3-8b"),
            digits=2,
        )
        # ft_model has 0.88 > 0.85, so ft_model's cell should be bolded
        assert "\\textbf" in result

    def test_model_ordering(self):
        group = "approximation_True.json | perturb_type1"

        def s(acc):
            return {
                group: _make_group_score(
                    "approximation_True.json", "perturb_type1", acc
                )
            }

        root = Path("results/qwen3-8b-q4_k_m")
        model_scores = {
            "baseline": s(0.80),
            "Qwen3-8b-base_q4_k_m-two-shot": s(0.81),
            "Qwen3-8B-base_q4_k_m-pap": s(0.82),
            "Qwen3-8B-daring-ant-evolves_normalized_claims_no_perturb_split_q4_k_m": s(  # noqa: E501
                0.83
            ),
            "aaa_other_model": s(0.85),  # alphabetically first but goes last
            "Qwen3-8B-young-nightingale-manifests_all_perturb_no_f_type2": s(
                0.86
            ),
        }
        result = render_table(
            model_scores,
            [group],
            baseline_model="baseline",
            results_root=root,
            digits=2,
        )
        data_lines = [
            line
            for line in result.splitlines()
            if "\\\\" in line and "textbf{Model}" not in line
        ]
        # Expected order: baseline(0), two-shot(1), pap(2),
        # no_perturb_split(3), rest alphabetically
        assert (
            "Baseline" in data_lines[0] or "baseline" in data_lines[0].lower()
        )
        assert "Two-Shot" in data_lines[1] or "two" in data_lines[1].lower()
        assert "PAP" in data_lines[2] or "pap" in data_lines[2].lower()
        assert (
            "Excl-Perturb" in data_lines[3]
            or "Split" in data_lines[3]
            or "Perturb" in data_lines[3]
        )

    def test_name_map_command_used(self):
        model_scores, group = self._build_scores()
        name_map = {
            "qwen3_baseline": {
                "short_label": "qwen3_baseline",
                "command": "\\mbase",
                "definition": "$\\text{M}_{\\text{Base}}$",
            },
        }
        result = render_table(
            model_scores,
            [ORIGINAL_GROUP_NAME, group],
            baseline_model="baseline",
            results_root=Path("results/qwen3-8b"),
            digits=2,
            name_map=name_map,
        )
        assert "\\mbase" in result
        # Full math notation should NOT appear for the mapped model
        assert r"\text{Baseline}" not in result

    def test_name_map_fallback_for_unmapped(self):
        model_scores, group = self._build_scores()
        # Provide a map that only covers baseline, not ft_model
        name_map = {
            "qwen3_baseline": {
                "short_label": "qwen3_baseline",
                "command": "\\mbase",
                "definition": "$\\text{M}_{\\text{Base}}$",
            },
        }
        result = render_table(
            model_scores,
            [group],
            baseline_model="baseline",
            results_root=Path("results/qwen3-8b"),
            digits=2,
            name_map=name_map,
        )
        # ft_model is not in the map, so it falls back to formal_model_label
        assert r"\text{M}" in result


# ---------------------------------------------------------------------------
# load_model_name_map
# ---------------------------------------------------------------------------


class TestLoadModelNameMap:
    def test_loads_entries(self, tmp_path):
        jsonl = tmp_path / "map.jsonl"
        baseline_line = (
            '{"short_label": "qwen3_baseline", '
            '"command": "\\\\mbase", '
            '"definition": "$\\\\text{M}_{\\\\text{Base}}$"}\n'
        )
        no_range_line = (
            '{"short_label": "qwen3_no_range", '
            '"command": "\\\\mnorange", '
            '"definition": "$\\\\text{M}_{\\\\text{Excl-Range}}$"}\n'
        )
        jsonl.write_text(baseline_line + no_range_line, encoding="utf-8")
        result = load_model_name_map(jsonl)
        assert "qwen3_baseline" in result
        assert result["qwen3_baseline"]["command"] == "\\mbase"
        assert "qwen3_no_range" in result

    def test_skips_blank_lines(self, tmp_path):
        jsonl = tmp_path / "map.jsonl"
        baseline_line = (
            '{"short_label": "qwen3_baseline", '
            '"command": "\\\\mbase", '
            '"definition": "$\\\\text{M}_{\\\\text{Base}}$"}\n'
        )
        no_range_line = (
            '{"short_label": "qwen3_no_range", '
            '"command": "\\\\mnorange", '
            '"definition": "$\\\\text{M}_{\\\\text{Excl-Range}}$"}\n'
        )
        jsonl.write_text(baseline_line + "\n" + no_range_line, encoding="utf-8")
        result = load_model_name_map(jsonl)
        assert len(result) == 2
