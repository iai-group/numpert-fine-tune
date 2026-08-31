"""Deterministic claim-level aggregation of per-number NLI results.

A separate pass from LLM scoring so relevance thresholds can be re-tuned
without re-querying the model: ``aggregate_record`` rolls each record's
per-number ``nli`` results up to a pair-level relation plus the
``perturbable`` gate Stage 3 consumes (which pairs can be flipped, in which
direction, using which evidence numbers). The CLI re-aggregates an
already-scored file in place (or to ``--output``).

Semantics (spec 2026-07-02 §4): only spans with ``relevance >= min_relevance``
count as *relevant*; ``counts.supports/refutes/neutral`` count relevant spans
by relation, while ``max_relevance`` ranges over all scored spans. A span with
``nli`` missing/None is counted under ``counts.errored`` and never as a
relation. The ``perturbable`` gate keys on the record's (normalized) label:
True-labeled pairs can flip ``to_false`` by breaking a relevant supporting
number; False-labeled pairs can flip ``to_true`` by fixing the relevant
refuting number(s). Conflicting or unrecognized labels are never perturbable.
"""

import argparse
import json
from typing import Any, Dict, List

DEFAULT_MIN_RELEVANCE = 4


def aggregate_record(
    record: Dict[str, Any], min_relevance: int = DEFAULT_MIN_RELEVANCE
) -> Dict[str, Any]:
    """Attach ``nli_aggregate`` to ``record`` (mutates + returns it)."""
    counts = {"supports": 0, "refutes": 0, "neutral": 0, "errored": 0}
    supporting: List[int] = []
    refuting: List[int] = []
    max_relevance = 0
    for e in record.get("evidence_numbers", []):
        nli = e.get("nli")
        if not nli:
            counts["errored"] += 1
            continue
        max_relevance = max(max_relevance, nli["relevance"])
        if nli["relevance"] < min_relevance:
            continue
        relation = nli["relation"]
        counts[relation] += 1
        if relation == "supports":
            supporting.append(e["id"])
        elif relation == "refutes":
            refuting.append(e["id"])

    if supporting and refuting:
        pair_relation = "conflicting"
    elif supporting:
        pair_relation = "supports"
    elif refuting:
        pair_relation = "refutes"
    else:
        pair_relation = "neutral"

    label = str(record.get("label", "")).strip().lower()
    record["nli_aggregate"] = {
        "pair_relation": pair_relation,
        "counts": counts,
        "max_relevance": max_relevance,
        "relevant_supporting_ids": supporting,
        "relevant_refuting_ids": refuting,
        "perturbable": {
            "to_false": label == "true" and len(supporting) >= 1,
            "to_true": label == "false" and len(refuting) >= 1,
        },
    }
    return record


def aggregate_records(
    records: List[Dict[str, Any]],
    min_relevance: int = DEFAULT_MIN_RELEVANCE,
) -> List[Dict[str, Any]]:
    """Aggregate every record in place; returns the same list."""
    for record in records:
        aggregate_record(record, min_relevance=min_relevance)
    return records


def main() -> None:
    """Re-aggregate an already-scored NLI output file."""
    parser = argparse.ArgumentParser(
        description="Roll per-number NLI up to claim-level aggregates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output", default=None,
        help="Output path (default: rewrite --input in place).",
    )
    parser.add_argument(
        "--min-relevance", type=int, default=DEFAULT_MIN_RELEVANCE,
        dest="min_relevance",
        help="Spans below this relevance never count as supporting/refuting.",
    )
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as fh:
        records = json.load(fh)
    aggregate_records(records, min_relevance=args.min_relevance)
    output = args.output or args.input
    with open(output, "w", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=2)
    dist: Dict[str, int] = {}
    for r in records:
        pr = r["nli_aggregate"]["pair_relation"]
        dist[pr] = dist.get(pr, 0) + 1
    print(f"Aggregated {len(records)} records -> {output} ({dist})")


if __name__ == "__main__":
    main()
