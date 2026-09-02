"""Compares baseline and adversarially fine-tuned models on VitaminC.

The VitaminC evaluation differs from the paper's other cross-dataset
experiments in one important way: its contrastive pairs are naturally
occurring.  Each claim is paired with two snapshots of the same Wikipedia
sentence, so the *evidence* rather than the claim carries the numerical
edit, and the gold label flips between the two snapshots.

Alongside the usual accuracy and macro-F1 comparison, this module
therefore reports a *pair accuracy*: the share of contrastive pairs for
which a model labels both snapshots correctly.  A model that ignores the
quantity and keys off surface overlap scores near zero on that metric
even when its per-instance accuracy looks reasonable, which makes it the
closest natural analogue of the paper's label-flipping evaluation.

Significance is assessed with a paired percentile bootstrap over
instances (or pairs) and an exact two-sided McNemar test, matching
``scripts/analysis/rebuttal_significance.py``.

Usage:
    python scripts/analysis/vitaminc_comparison.py \
        --baseline results/inference/vitaminc/qwen3-8b-q4_k_m/baseline \
        --finetuned \
            results/inference/vitaminc/qwen3-8b-q4_k_m/young_nightingale \
        --dataset data/vitaminc/vitaminc_test_numerical.json \
        --output results/vitaminc_stats
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 3407

NUMBER_PATTERN = re.compile(r"\d[\d,]*\.?\d*")

MULTIPLIERS = {
    "thousand": 1_000,
    "million": 1_000_000,
    "billion": 1_000_000_000,
    "trillion": 1_000_000_000_000,
}

SCALE_PATTERN = re.compile(
    r"(?P<number>\d[\d,]*\.?\d*)\s*"
    r"(?P<scale>thousand|million|billion|trillion)\b",
    re.IGNORECASE,
)

EDIT_SIZE_BINS = (
    (0.10, "(0, 10%]"),
    (0.25, "(10%, 25%]"),
    (0.50, "(25%, 50%]"),
    (1.00, "(50%, 100%]"),
    (float("inf"), "> 100%"),
)


@dataclass
class Instance:
    """One model's outcome on a single evaluation instance.

    Attributes:
        gold: Gold label as a boolean.
        predicted: Predicted label, or ``None`` on a parse failure.
    """

    gold: bool
    predicted: Optional[bool]

    @property
    def parsed(self) -> bool:
        """Whether the model produced a parseable verdict."""
        return self.predicted is not None

    @property
    def correct(self) -> bool:
        """Whether the parsed verdict matches the gold label."""
        return self.predicted is not None and self.predicted == self.gold


def parse_bool_json(value: str) -> Optional[bool]:
    """Parses a ``'{"label": true}'`` cell into a boolean.

    Args:
        value: Raw CSV cell contents.

    Returns:
        The boolean verdict, or ``None`` when the cell is absent or does
        not carry a usable boolean label.
    """
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(parsed, dict):
        label = parsed.get("label")
        if isinstance(label, bool):
            return label
    return None


def resolve_predictions_csv(run_dir: Path) -> Path:
    """Locates ``predictions.csv`` for one inference run.

    Args:
        run_dir: Directory passed on the command line.

    Returns:
        Path to the predictions file.

    Raises:
        FileNotFoundError: If no predictions file exists beneath
            ``run_dir``.
    """
    top = run_dir / "predictions.csv"
    if top.exists():
        return top
    nested = sorted(run_dir.glob("*/predictions.csv"))
    nested += sorted(run_dir.glob(".models*/predictions.csv"))
    if nested:
        return nested[0]
    raise FileNotFoundError(f"No predictions.csv under {run_dir}")


def load_run(run_dir: Path) -> Dict[str, Instance]:
    """Loads one run's per-instance outcomes keyed by claim identifier.

    Args:
        run_dir: Directory holding the run's ``predictions.csv``.

    Returns:
        Mapping from claim identifier to :class:`Instance`.

    Raises:
        ValueError: If a gold label is missing or a claim identifier is
            duplicated.
    """
    path = resolve_predictions_csv(run_dir)
    instances: Dict[str, Instance] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            claim_id = row["claim_id"]
            gold = parse_bool_json(row["label"])
            if gold is None:
                raise ValueError(f"Missing gold label for {claim_id}")
            if claim_id in instances:
                raise ValueError(f"Duplicate claim_id {claim_id} in {path}")
            instances[claim_id] = Instance(
                gold=gold, predicted=parse_bool_json(row["prediction"])
            )
    logger.info("Loaded %d instances from %s", len(instances), path)
    return instances


def load_pair_index(dataset: Path) -> Dict[str, Tuple[str, str]]:
    """Maps each claim identifier to its contrastive-pair group key.

    A VitaminC case may contain more than one distinct claim, so the case
    identifier alone does not identify a contrastive pair: grouping on it
    would allow one claim's supporting snapshot to be matched against a
    different claim's refuting snapshot.  The group key is therefore the
    (case, claim text) combination, which isolates the two snapshots of
    one revision judged against one claim.

    Args:
        dataset: The evaluation subset written by
            ``scripts/data/prepare_vitaminc.py``.

    Returns:
        Mapping from claim identifier to its (case, claim) group key.
    """
    with dataset.open(encoding="utf-8") as handle:
        records = json.load(handle)
    return {r["claim_id"]: (r["vitaminc_case_id"], r["claim"]) for r in records}


def load_evidence(dataset: Path) -> Dict[str, str]:
    """Maps each claim identifier to its evidence text.

    Args:
        dataset: The evaluation subset written by
            ``scripts/data/prepare_vitaminc.py``.

    Returns:
        Mapping from claim identifier to evidence document.
    """
    with dataset.open(encoding="utf-8") as handle:
        records = json.load(handle)
    return {r["claim_id"]: r["doc"] for r in records}


def mcnemar_exact(discordant_a: int, discordant_b: int) -> float:
    """Computes the exact two-sided McNemar p-value.

    Args:
        discordant_a: Instances the first model gets right and the
            second wrong.
        discordant_b: Instances the second model gets right and the
            first wrong.

    The test is the exact binomial test on the discordant pairs.  It is
    delegated to SciPy rather than summed directly, because ``2 ** n``
    overflows a float once the discordant count reaches a few thousand,
    which it does on this dataset.

    Returns:
        The two-sided p-value; ``1.0`` when there are no discordant
        instances.
    """
    total = discordant_a + discordant_b
    if total == 0:
        return 1.0
    smaller = min(discordant_a, discordant_b)
    return float(
        stats.binomtest(smaller, total, 0.5, alternative="two-sided").pvalue
    )


def macro_f1(gold: np.ndarray, correct: np.ndarray) -> float:
    """Computes binary macro-F1 from gold labels and correctness flags.

    Args:
        gold: Boolean gold labels.
        correct: Boolean flags marking correctly predicted instances.
            Instances that are incorrect are assumed to carry the
            opposite label, which holds for a binary task.

    Returns:
        The unweighted mean of the two per-class F1 scores.
    """
    scores: List[float] = []
    for positive in (True, False):
        is_gold = gold == positive
        # Predicted positive: correct where gold is positive, plus
        # incorrect where gold is negative.
        predicted = (is_gold & correct) | (~is_gold & ~correct)
        true_positives = float(np.sum(is_gold & predicted))
        precision_denom = float(np.sum(predicted))
        recall_denom = float(np.sum(is_gold))
        if precision_denom == 0 or recall_denom == 0:
            scores.append(0.0)
            continue
        precision = true_positives / precision_denom
        recall = true_positives / recall_denom
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append(2 * precision * recall / (precision + recall))
    return float(np.mean(scores))


def bootstrap_ci(
    values: np.ndarray,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> Tuple[float, float]:
    """Computes a percentile bootstrap confidence interval for a mean.

    Args:
        values: Per-unit paired differences.
        resamples: Number of bootstrap resamples.
        seed: Seed for the random generator.

    Returns:
        The 2.5th and 97.5th percentiles of the resampled means.
    """
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(resamples, values.size))
    means = values[indices].mean(axis=1)
    return (
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def describe_run(instances: Dict[str, Instance]) -> Dict[str, float]:
    """Summarises one run over all of its instances.

    Args:
        instances: Per-instance outcomes for one model.

    Returns:
        Accuracy over all instances (parse failures counted as errors),
        accuracy over parseable instances, macro-F1 over parseable
        instances, and the parse-failure rate, all as percentages.
    """
    total = len(instances)
    parsed = [i for i in instances.values() if i.parsed]
    correct_all = sum(1 for i in instances.values() if i.correct)
    gold = np.array([i.gold for i in parsed], dtype=bool)
    correct = np.array([i.correct for i in parsed], dtype=bool)
    return {
        "n": total,
        "accuracy_all": 100.0 * correct_all / total if total else 0.0,
        "n_parsed": len(parsed),
        "accuracy_parsed": (100.0 * float(np.mean(correct)) if parsed else 0.0),
        "macro_f1_parsed": (100.0 * macro_f1(gold, correct) if parsed else 0.0),
        "parse_failure_rate": (
            100.0 * (total - len(parsed)) / total if total else 0.0
        ),
    }


def compare_instances(
    baseline: Dict[str, Instance],
    finetuned: Dict[str, Instance],
    claim_ids: Sequence[str],
) -> Dict[str, object]:
    """Compares two runs over a shared set of instances.

    Parse failures are counted as errors so that the comparison covers
    every instance in ``claim_ids``; the parseable-only view is reported
    separately by :func:`describe_run`.

    Args:
        baseline: Baseline run outcomes.
        finetuned: Fine-tuned run outcomes.
        claim_ids: Instances to compare.

    Returns:
        Accuracy delta in percentage points with a bootstrap confidence
        interval, discordant counts, and the exact McNemar p-value.
    """
    base_correct = np.array(
        [baseline[c].correct for c in claim_ids], dtype=bool
    )
    fine_correct = np.array(
        [finetuned[c].correct for c in claim_ids], dtype=bool
    )
    differences = fine_correct.astype(float) - base_correct.astype(float)
    low, high = bootstrap_ci(differences)
    only_base = int(np.sum(base_correct & ~fine_correct))
    only_fine = int(np.sum(fine_correct & ~base_correct))
    return {
        "n": len(claim_ids),
        "baseline_accuracy": 100.0 * float(np.mean(base_correct)),
        "finetuned_accuracy": 100.0 * float(np.mean(fine_correct)),
        "delta_pp": 100.0 * float(np.mean(differences)),
        "ci95_pp": [100.0 * low, 100.0 * high],
        "only_baseline_correct": only_base,
        "only_finetuned_correct": only_fine,
        "mcnemar_p": mcnemar_exact(only_base, only_fine),
    }


def build_pairs(
    claim_ids: Sequence[str],
    pair_index: Dict[str, Tuple[str, str]],
    reference: Dict[str, Instance],
) -> List[Tuple[str, str]]:
    """Finds contrastive pairs among the evaluated instances.

    A contrastive pair consists of the two snapshots of one revision
    judged against one claim, whose gold labels disagree because the
    numerical edit flipped the verdict.  Instances are grouped on the
    (case, claim) key rather than the case alone, since a case may carry
    several distinct claims.

    Args:
        claim_ids: Instances available for both models.
        pair_index: Mapping from claim identifier to (case, claim) key.
        reference: Any run's outcomes, used to read gold labels.

    Returns:
        A list of (supporting instance, refuting instance) tuples.
    """
    by_group: Dict[Tuple[str, str], List[str]] = {}
    for claim_id in claim_ids:
        group = pair_index.get(claim_id)
        if group is None:
            continue
        by_group.setdefault(group, []).append(claim_id)

    pairs: List[Tuple[str, str]] = []
    for members in by_group.values():
        supporting = [c for c in members if reference[c].gold]
        refuting = [c for c in members if not reference[c].gold]
        for true_id, false_id in zip(sorted(supporting), sorted(refuting)):
            pairs.append((true_id, false_id))
    return pairs


def compare_pairs(
    baseline: Dict[str, Instance],
    finetuned: Dict[str, Instance],
    pairs: Sequence[Tuple[str, str]],
) -> Dict[str, object]:
    """Compares two runs on contrastive-pair accuracy.

    Args:
        baseline: Baseline run outcomes.
        finetuned: Fine-tuned run outcomes.
        pairs: Contrastive pairs from :func:`build_pairs`.

    Returns:
        Pair accuracies, their delta with a bootstrap confidence
        interval, and the exact McNemar p-value over pairs.
    """
    base = np.array(
        [baseline[a].correct and baseline[b].correct for a, b in pairs],
        dtype=bool,
    )
    fine = np.array(
        [finetuned[a].correct and finetuned[b].correct for a, b in pairs],
        dtype=bool,
    )
    differences = fine.astype(float) - base.astype(float)
    low, high = bootstrap_ci(differences)
    only_base = int(np.sum(base & ~fine))
    only_fine = int(np.sum(fine & ~base))
    return {
        "n_pairs": len(pairs),
        "baseline_pair_accuracy": 100.0 * float(np.mean(base)),
        "finetuned_pair_accuracy": 100.0 * float(np.mean(fine)),
        "delta_pp": 100.0 * float(np.mean(differences)),
        "ci95_pp": [100.0 * low, 100.0 * high],
        "only_baseline_correct": only_base,
        "only_finetuned_correct": only_fine,
        "mcnemar_p": mcnemar_exact(only_base, only_fine),
    }


def normalize_scale_words(text: str) -> str:
    """Rewrites ``"10 million"`` style quantities into plain digits.

    This mirrors ``replace_number_phrases`` in
    ``src/data_preprocessing/raw_dataset_preprocessing/
    process_tag_claims_spacy.py``.  That module is not imported here
    because importing it configures file logging as a side effect, which
    would suppress this script's own log output.  Only the multiplier
    handling needed for evidence comparison is reproduced.

    Args:
        text: Evidence text.

    Returns:
        The text with scale words folded into their numeric values.
    """

    def expand(match: re.Match) -> str:
        """Multiplies the matched number by its scale word."""
        try:
            value = float(match.group("number").replace(",", ""))
        except ValueError:
            return match.group(0)
        factor = MULTIPLIERS[match.group("scale").lower()]
        return f"{value * factor:,.0f}"

    return SCALE_PATTERN.sub(expand, text)


def extract_numbers(text: str) -> List[float]:
    """Extracts numeric values from text after scale normalisation.

    Args:
        text: Evidence text.

    Returns:
        Every parseable number in the text.
    """
    values: List[float] = []
    for token in NUMBER_PATTERN.findall(normalize_scale_words(text)):
        try:
            values.append(float(token.replace(",", "")))
        except ValueError:
            continue
    return values


def edit_size(doc_a: str, doc_b: str) -> Optional[float]:
    """Estimates the relative size of the numerical edit between snapshots.

    The two snapshots of a revision usually differ in exactly one
    quantity.  When several differ, the smallest relative change is used,
    which is a heuristic: it can select a changed date or an incidental
    figure rather than the quantity the claim is about.  Results binned
    on it should be read as indicative, not exact.

    Args:
        doc_a: Evidence text of the supporting snapshot.
        doc_b: Evidence text of the refuting snapshot.

    Returns:
        The smallest relative change, or ``None`` when no changed number
        can be aligned between the snapshots.
    """
    counts_a = collections.Counter(extract_numbers(doc_a))
    counts_b = collections.Counter(extract_numbers(doc_b))
    only_a = list((counts_a - counts_b).elements())
    only_b = list((counts_b - counts_a).elements())
    ratios = [
        abs(after - before) / abs(before)
        for before in only_a
        for after in only_b
        if before != 0
    ]
    return min(ratios) if ratios else None


def bin_for_edit_size(size: float) -> str:
    """Returns the bin label for a relative edit size."""
    for upper, label in EDIT_SIZE_BINS:
        if size <= upper:
            return label
    return EDIT_SIZE_BINS[-1][1]


def compare_by_edit_size(
    baseline: Dict[str, Instance],
    finetuned: Dict[str, Instance],
    pairs: Sequence[Tuple[str, str]],
    evidence: Dict[str, str],
) -> Dict[str, Dict[str, object]]:
    """Compares pair accuracy grouped by the size of the evidence edit.

    Each bin also reports which side of the pair each model gets wrong:
    refuting a supporting snapshot versus confirming a refuting one.
    Those two error modes distinguish a model that compares quantities
    from one that merely checks whether they are close.

    Args:
        baseline: Baseline run outcomes.
        finetuned: Fine-tuned run outcomes.
        pairs: Contrastive pairs from :func:`build_pairs`.
        evidence: Mapping from claim identifier to evidence text.

    Returns:
        A mapping from bin label to the per-bin comparison, including the
        bootstrap confidence interval of the pair-accuracy delta.
    """
    grouped: Dict[str, List[Tuple[str, str]]] = {}
    unaligned = 0
    for support_id, refute_id in pairs:
        size = edit_size(evidence[support_id], evidence[refute_id])
        if size is None:
            unaligned += 1
            continue
        grouped.setdefault(bin_for_edit_size(size), []).append(
            (support_id, refute_id)
        )

    results: Dict[str, Dict[str, object]] = {}
    for _, label in EDIT_SIZE_BINS:
        members = grouped.get(label)
        if not members:
            continue
        cells = compare_pairs(baseline, finetuned, members)
        for name, run in (("baseline", baseline), ("finetuned", finetuned)):
            total = len(members)
            wrongly_refutes = sum(1 for a, _ in members if not run[a].correct)
            wrongly_confirms = sum(1 for _, b in members if not run[b].correct)
            cells[f"{name}_wrongly_refutes_pct"] = (
                100.0 * wrongly_refutes / total
            )
            cells[f"{name}_wrongly_confirms_pct"] = (
                100.0 * wrongly_confirms / total
            )
        results[label] = cells
    if unaligned:
        logger.info("Pairs with no alignable changed number: %d", unaligned)
    results["_unaligned_pairs"] = {"n": unaligned}
    return results


def by_gold_label(
    baseline: Dict[str, Instance],
    finetuned: Dict[str, Instance],
    claim_ids: Sequence[str],
) -> Dict[str, Dict[str, object]]:
    """Compares the two runs separately for each gold label.

    Args:
        baseline: Baseline run outcomes.
        finetuned: Fine-tuned run outcomes.
        claim_ids: Instances available for both models.

    Returns:
        A mapping from ``"supports"``/``"refutes"`` to the comparison
        produced by :func:`compare_instances`.
    """
    groups = {
        "supports": [c for c in claim_ids if baseline[c].gold],
        "refutes": [c for c in claim_ids if not baseline[c].gold],
    }
    return {
        name: compare_instances(baseline, finetuned, ids)
        for name, ids in groups.items()
        if ids
    }


def format_report(results: Dict[str, object]) -> str:
    """Renders the analysis as a Markdown report.

    Args:
        results: The assembled results dictionary.

    Returns:
        The report text.
    """
    runs = results["runs"]
    overall = results["overall"]
    pairs = results["pairs"]
    lines = [
        "# VitaminC numerical subset: baseline vs adversarial fine-tuning",
        "",
        f"Instances evaluated by both models: {overall['n']}",
        "",
        "## Per-model summary",
        "",
        "| Model | n | Acc (all) | Acc (parsed) | Macro-F1 | Parse fail |",
        "|---|---|---|---|---|---|",
    ]
    for name in ("baseline", "finetuned"):
        row = runs[name]
        lines.append(
            f"| {name} | {row['n']} | {row['accuracy_all']:.2f} | "
            f"{row['accuracy_parsed']:.2f} | "
            f"{row['macro_f1_parsed']:.2f} | "
            f"{row['parse_failure_rate']:.2f} |"
        )

    lines += [
        "",
        "## Paired comparison (parse failures counted as errors)",
        "",
        "| Scope | n | Baseline | Fine-tuned | Delta (pp) | 95% CI | "
        "McNemar p |",
        "|---|---|---|---|---|---|---|",
    ]

    def row(name: str, cells: Dict[str, object]) -> str:
        """Formats one comparison row."""
        size = cells.get("n") or cells.get("n_pairs")
        base_key = (
            "baseline_accuracy"
            if "baseline_accuracy" in cells
            else "baseline_pair_accuracy"
        )
        fine_key = (
            "finetuned_accuracy"
            if "finetuned_accuracy" in cells
            else "finetuned_pair_accuracy"
        )
        ci = cells["ci95_pp"]
        return (
            f"| {name} | {size} | {cells[base_key]:.2f} | "
            f"{cells[fine_key]:.2f} | {cells['delta_pp']:+.2f} | "
            f"[{ci[0]:+.2f}, {ci[1]:+.2f}] | "
            f"{cells['mcnemar_p']:.3g} |"
        )

    lines.append(row("all instances", overall))
    for name, cells in results["by_gold_label"].items():
        lines.append(row(f"gold = {name}", cells))
    lines.append(row("contrastive pairs (both correct)", pairs))
    lines += [
        "",
        "Pair accuracy requires the model to label BOTH snapshots of a "
        "revision correctly, so it is the natural analogue of the "
        "paper's label-flipping evaluation.",
        "",
        "## Pair accuracy by size of the numerical edit in the evidence",
        "",
        "`refutes` = wrongly refuted the supporting snapshot; `confirms` "
        "= wrongly confirmed the refuting snapshot. Bin assignment uses "
        "the smallest relative change between the two snapshots and is "
        "indicative only (see `edit_size`).",
        "",
        "| Edit size | Pairs | Base pair acc | FT pair acc | Delta (pp) "
        "| 95% CI | Base refutes/confirms | FT refutes/confirms |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label, cells in results["by_edit_size"].items():
        if label.startswith("_"):
            continue
        ci = cells["ci95_pp"]
        lines.append(
            f"| {label} | {cells['n_pairs']} | "
            f"{cells['baseline_pair_accuracy']:.2f} | "
            f"{cells['finetuned_pair_accuracy']:.2f} | "
            f"{cells['delta_pp']:+.2f} | "
            f"[{ci[0]:+.2f}, {ci[1]:+.2f}] | "
            f"{cells['baseline_wrongly_refutes_pct']:.1f} / "
            f"{cells['baseline_wrongly_confirms_pct']:.1f} | "
            f"{cells['finetuned_wrongly_refutes_pct']:.1f} / "
            f"{cells['finetuned_wrongly_confirms_pct']:.1f} |"
        )
    unaligned = results["by_edit_size"].get("_unaligned_pairs", {})
    lines += [
        "",
        f"Pairs excluded because no changed number could be aligned: "
        f"{unaligned.get('n', 0)}.",
        "",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Builds the command-line parser for this script."""
    parser = argparse.ArgumentParser(
        description=(
            "Compares baseline and fine-tuned model predictions on the "
            "VitaminC numerical subset."
        )
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Run directory of the zero-shot baseline model.",
    )
    parser.add_argument(
        "--finetuned",
        type=Path,
        required=True,
        help="Run directory of the adversarially fine-tuned model.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/vitaminc/vitaminc_test_numerical.json"),
        help="The VitaminC evaluation subset used for both runs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/vitaminc_stats"),
        help="Directory for the JSON results and Markdown report.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """Runs the VitaminC comparison end to end."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)

    baseline = load_run(args.baseline)
    finetuned = load_run(args.finetuned)
    pair_index = load_pair_index(args.dataset)
    evidence = load_evidence(args.dataset)

    shared = sorted(set(baseline) & set(finetuned))
    if not shared:
        logger.error("The two runs share no instances; nothing to compare.")
        sys.exit(1)
    if len(shared) != len(baseline) or len(shared) != len(finetuned):
        logger.warning(
            "Runs differ in coverage: baseline=%d finetuned=%d shared=%d",
            len(baseline),
            len(finetuned),
            len(shared),
        )
    for claim_id in shared:
        if baseline[claim_id].gold != finetuned[claim_id].gold:
            raise ValueError(f"Gold label mismatch for {claim_id}")

    pairs = build_pairs(shared, pair_index, baseline)
    logger.info("Contrastive pairs available: %d", len(pairs))

    results: Dict[str, object] = {
        "runs": {
            "baseline": describe_run(baseline),
            "finetuned": describe_run(finetuned),
        },
        "overall": compare_instances(baseline, finetuned, shared),
        "by_gold_label": by_gold_label(baseline, finetuned, shared),
        "pairs": compare_pairs(baseline, finetuned, pairs),
        "by_edit_size": compare_by_edit_size(
            baseline, finetuned, pairs, evidence
        ),
        "config": {
            "baseline_dir": str(args.baseline),
            "finetuned_dir": str(args.finetuned),
            "dataset": str(args.dataset),
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
    }

    args.output.mkdir(parents=True, exist_ok=True)
    with open(args.output / "vitaminc_stats.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    report = format_report(results)
    with open(args.output / "report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
