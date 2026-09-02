"""Builds a numerical evaluation subset from the VitaminC dataset.

VitaminC (Schuster et al., NAACL 2021) pairs each claim with two
snapshots of the same Wikipedia sentence, taken before and after a
factual revision.  Roughly half of its real revisions alter a numeric
value, which makes the dataset a naturally occurring counterpart to the
synthetic claim-side perturbations used elsewhere in this project: here
the *evidence* changes while the claim is held fixed.

This module isolates the subset that is genuinely numerical and
compatible with the binary True/False protocol used throughout the
paper, then writes it in the record schema consumed by
``src.llm_finetune.utils.data_loading.pipeline.load_split``.

Selection criteria (applied in order):
    1. ``revision_type == "real"`` — drops the third of the corpus that
       is annotator-written synthetic data.
    2. ``label`` in ``{"SUPPORTS", "REFUTES"}`` — drops NOT ENOUGH INFO,
       which has no counterpart in the binary protocol.
    3. The revision itself is numerical: the multiset of numeric tokens
       differs between the two evidence snapshots of the same case.
    4. The claim carries a quantity: spaCy tags at least one entity of a
       genuinely quantitative type (CARDINAL, MONEY, PERCENT, QUANTITY),
       mirroring the numerical tagging used to build QuanTemp splits.

Every emitted instance receives a unique ``claim_id`` because the same
claim text appears twice with different evidence; downstream loading
must therefore run with text-deduplication disabled.

Usage:
    python scripts/data/prepare_vitaminc.py \
        --input-dir /path/to/vitaminc \
        --split test \
        --output data/vitaminc/vitaminc_test_numerical.json
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import re
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import spacy

logger = logging.getLogger(__name__)

NUMERIC_TOKEN_PATTERN = re.compile(r"\d[\d,.]*")

QUANTITATIVE_ENTITIES = frozenset({"CARDINAL", "MONEY", "PERCENT", "QUANTITY"})

LABEL_MAP = {"SUPPORTS": "True", "REFUTES": "False"}

SPACY_MODEL = "en_core_web_sm"


def load_vitaminc_split(input_dir: Path, split: str) -> List[Dict[str, Any]]:
    """Loads one VitaminC split from its JSON-lines file.

    Args:
        input_dir: Directory holding ``train.jsonl``, ``dev.jsonl`` and
            ``test.jsonl`` as released on the HuggingFace Hub.
        split: Split name to load.

    Returns:
        The split's records in file order.
    """
    path = input_dir / f"{split}.jsonl"
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def numeric_tokens(text: str) -> collections.Counter:
    """Returns the multiset of numeric tokens occurring in ``text``."""
    return collections.Counter(NUMERIC_TOKEN_PATTERN.findall(text))


def find_numerical_cases(records: Iterable[Dict[str, Any]]) -> Set[str]:
    """Identifies cases whose revision changes at least one number.

    A VitaminC case bundles the two evidence snapshots that straddle one
    Wikipedia revision.  The revision counts as numerical when the two
    snapshots disagree on their numeric tokens.

    Args:
        records: VitaminC records, each carrying ``case_id`` and
            ``evidence``.

    Returns:
        The identifiers of the numerical cases.
    """
    evidence_by_case: Dict[str, Set[str]] = collections.defaultdict(set)
    for record in records:
        evidence_by_case[record["case_id"]].add(record["evidence"])

    numerical: Set[str] = set()
    for case_id, snapshots in evidence_by_case.items():
        if len(snapshots) < 2:
            continue
        counters = [numeric_tokens(snapshot) for snapshot in snapshots]
        if any(counter != counters[0] for counter in counters[1:]):
            numerical.add(case_id)
    return numerical


def load_spacy_model(model_name: str = SPACY_MODEL) -> spacy.language.Language:
    """Loads the spaCy pipeline used for numerical entity tagging."""
    return spacy.load(model_name, disable=["lemmatizer", "textcat"])


def find_quantitative_claims(
    claims: List[str],
    nlp: spacy.language.Language,
    batch_size: int = 256,
) -> Set[str]:
    """Selects claims containing at least one quantitative entity.

    Args:
        claims: Distinct claim strings to tag.
        nlp: Loaded spaCy pipeline.
        batch_size: Number of claims per spaCy batch.

    Returns:
        The subset of ``claims`` carrying a CARDINAL, MONEY, PERCENT or
        QUANTITY entity.
    """
    quantitative: Set[str] = set()
    docs = nlp.pipe(claims, batch_size=batch_size)
    for claim, doc in zip(claims, docs):
        if any(ent.label_ in QUANTITATIVE_ENTITIES for ent in doc.ents):
            quantitative.add(claim)
    return quantitative


def build_records(
    records: List[Dict[str, Any]],
    numerical_cases: Set[str],
    quantitative_claims: Set[str],
    split: str,
) -> List[Dict[str, Any]]:
    """Converts selected VitaminC records into the project schema.

    Args:
        records: All records of the source split.
        numerical_cases: Case identifiers whose revision is numerical.
        quantitative_claims: Claim strings carrying a quantity.
        split: Split name, used to build claim identifiers.

    Returns:
        Schema-compatible records, one per (claim, evidence) instance.
    """
    output: List[Dict[str, Any]] = []
    for record in records:
        if record.get("revision_type") != "real":
            continue
        if record["label"] not in LABEL_MAP:
            continue
        if record["case_id"] not in numerical_cases:
            continue
        claim = record["claim"].strip()
        if claim not in quantitative_claims:
            continue

        claim_id = f"vitaminc_{split}_{record['unique_id']}"
        output.append(
            {
                "claim_id": claim_id,
                "label": LABEL_MAP[record["label"]],
                "claim": claim,
                "normalized_claim": claim,
                "doc": record["evidence"].strip(),
                "lang": "en",
                "taxonomy_label": "statistical",
                "label_original": record["label"],
                "country_of_origin": "unknown",
                "crawled_date": "",
                "url": "",
                "vitaminc_case_id": record["case_id"],
                "vitaminc_unique_id": record["unique_id"],
                "vitaminc_page": record.get("page", ""),
                "vitaminc_revision_type": record["revision_type"],
                "vitaminc_wiki_revision_id": record.get("wiki_revision_id", ""),
            }
        )
    return output


def summarize(
    source: List[Dict[str, Any]],
    selected: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Computes funnel and distribution statistics for the built subset.

    Args:
        source: All records of the source split.
        selected: The records retained by :func:`build_records`.

    Returns:
        A dictionary of counts and distributions suitable for logging or
        reporting in the paper's appendix.
    """
    evidence_words = [len(r["doc"].split()) for r in selected] or [0]
    claim_words = [len(r["claim"].split()) for r in selected] or [0]
    return {
        "source_instances": len(source),
        "source_by_revision_type": dict(
            collections.Counter(
                r.get("revision_type", "unknown") for r in source
            )
        ),
        "source_by_label": dict(
            collections.Counter(r["label"] for r in source)
        ),
        "selected_instances": len(selected),
        "selected_by_label": dict(
            collections.Counter(r["label"] for r in selected)
        ),
        "unique_claims": len(set(r["claim"] for r in selected)),
        "unique_cases": len(set(r["vitaminc_case_id"] for r in selected)),
        "evidence_words_median": statistics.median(evidence_words),
        "evidence_words_mean": round(statistics.fmean(evidence_words), 1),
        "claim_words_median": statistics.median(claim_words),
    }


def write_outputs(
    records: List[Dict[str, Any]],
    stats: Dict[str, Any],
    output: Path,
) -> None:
    """Writes the subset and its statistics sidecar to disk.

    Args:
        records: Schema-compatible records to serialise.
        stats: Statistics as produced by :func:`summarize`.
        output: Destination path for the subset JSON file.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)

    stats_path = output.with_name(f"{output.stem}_stats.json")
    with open(stats_path, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)


def build_parser() -> argparse.ArgumentParser:
    """Builds the command-line parser for this script."""
    parser = argparse.ArgumentParser(
        description=(
            "Isolates the numerical, real-revision, binary-label subset "
            "of VitaminC for cross-dataset evaluation."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing VitaminC {train,dev,test}.jsonl.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "dev", "test"],
        help="VitaminC split to process (default: test).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSON path (default: "
            "data/vitaminc/vitaminc_<split>_numerical.json)."
        ),
    )
    parser.add_argument(
        "--spacy-model",
        default=SPACY_MODEL,
        help=f"spaCy model for entity tagging (default: {SPACY_MODEL}).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """Runs the VitaminC subset extraction end to end."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)

    output = args.output or Path(
        f"data/vitaminc/vitaminc_{args.split}_numerical.json"
    )

    logger.info("Loading VitaminC %s split...", args.split)
    source = load_vitaminc_split(args.input_dir, args.split)
    logger.info("Loaded %d instances", len(source))

    logger.info("Identifying numerical revisions...")
    numerical_cases = find_numerical_cases(source)
    logger.info("Numerical cases: %d", len(numerical_cases))

    candidates = sorted(
        {
            r["claim"].strip()
            for r in source
            if r.get("revision_type") == "real"
            and r["label"] in LABEL_MAP
            and r["case_id"] in numerical_cases
        }
    )
    logger.info("Tagging %d distinct claims with spaCy...", len(candidates))
    nlp = load_spacy_model(args.spacy_model)
    quantitative = find_quantitative_claims(candidates, nlp)
    logger.info("Claims with a quantitative entity: %d", len(quantitative))

    selected = build_records(source, numerical_cases, quantitative, args.split)
    stats = summarize(source, selected)
    write_outputs(selected, stats, output)

    logger.info("Wrote %d instances to %s", len(selected), output)
    for key, value in stats.items():
        logger.info("  %s: %s", key, value)


if __name__ == "__main__":
    main()
