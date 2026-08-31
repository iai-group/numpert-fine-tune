"""Orchestrate claim↔evidence number NLI scoring over a dataset.

Each record's evidence numbers are grouped into one call per sentence window;
calls are scored concurrently via a thread pool (the vLLM server batches them).
Results are attached to each evidence-number dict under an ``nli`` key.
"""

import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from src.number_annotation.core.schema import span_from_dict
from src.number_annotation.nli.aggregate import (
    DEFAULT_MIN_RELEVANCE,
    aggregate_records,
)
from src.number_annotation.nli.client import (
    DEFAULT_NLI_BASE_URL,
    DEFAULT_NLI_MAX_TOKENS,
    DEFAULT_NLI_MODEL,
    NliClient,
)
from src.number_annotation.nli.grouping import (
    DEFAULT_MAX_MARKERS_PER_CALL,
    WINDOWS,
    SentenceGroup,
    group_by_sentence,
)
from src.number_annotation.nli.marker import render_marked_sentence

INPUT = "data/spacy_number_annotated/test_full_spacy.json"
OUTPUT = "data/evidence_nli/test_nli.json"


def sample_records(
    records: List[Dict[str, Any]], n: int, seed: int
) -> List[Dict[str, Any]]:
    """Return a deterministic sample of ``n`` records (all if n <= 0)."""
    if n <= 0 or n >= len(records):
        return records
    return random.Random(seed).sample(records, n)


def _nli_dict(item) -> Dict[str, Any]:
    """Convert an NliItem to the per-span ``nli`` dict (marker dropped)."""
    return {
        "relevance": item.relevance,
        "relation": item.relation,
        "claim_num": item.claim_num,
        "rationale": item.rationale,
    }


def _score_one_record(
    record: Dict[str, Any],
    client,
    window: str = "sentence",
    max_markers: int = DEFAULT_MAX_MARKERS_PER_CALL,
) -> Dict[str, Any]:
    """Score a single record's evidence numbers and attach ``nli`` to each.

    Each sentence-group call is isolated: if a call fails (e.g. the model
    truncates output), only that group's spans are marked with ``nli=None``
    and an ``nli_error`` breadcrumb, and scoring continues for the rest.
    """
    spans = [span_from_dict(d) for d in record.get("evidence_numbers", [])]
    claim_numbers = [span_from_dict(d) for d in record.get("claim_numbers", [])]
    groups: List[SentenceGroup] = group_by_sentence(
        spans, window=window, max_markers_per_call=max_markers
    )

    by_span_id: Dict[int, Optional[Dict[str, Any]]] = {}
    err_by_span_id: Dict[int, str] = {}
    for group in groups:
        marked = render_marked_sentence(
            group.sentence_text, group.sentence_start, group.markers
        )
        try:
            items = client.score(
                record.get("claim", ""), claim_numbers, marked
            )
        except Exception as exc:  # noqa: BLE001 — isolate one bad group.
            for _label, span in group.markers:
                by_span_id[span.id] = None
                err_by_span_id[span.id] = str(exc)
            continue
        item_by_marker = {it.marker: it for it in items}
        for label, span in group.markers:
            item = item_by_marker.get(label)
            by_span_id[span.id] = _nli_dict(item) if item else None

    for d in record.get("evidence_numbers", []):
        d["nli"] = by_span_id.get(d["id"])
        if d["id"] in err_by_span_id:
            d["nli_error"] = err_by_span_id[d["id"]]
    return record


def score_records(
    records: List[Dict[str, Any]],
    client,
    max_workers: int = 16,
    window: str = "sentence",
    max_markers: int = DEFAULT_MAX_MARKERS_PER_CALL,
    min_relevance: int = DEFAULT_MIN_RELEVANCE,
    aggregate: bool = True,
) -> List[Dict[str, Any]]:
    """Score all records concurrently; returns the same records, mutated.

    A record that fails entirely (e.g. grouping error) is isolated too: its
    evidence numbers get ``nli=None`` + ``nli_error`` and the run continues.
    Unless ``aggregate`` is False, each record then gets its deterministic
    claim-level ``nli_aggregate`` rollup (see ``aggregate.py``).
    """
    def _safe(record: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return _score_one_record(
                record, client, window=window, max_markers=max_markers
            )
        except Exception as exc:  # noqa: BLE001 — isolate one bad record.
            for d in record.get("evidence_numbers", []):
                d["nli"] = None
                d["nli_error"] = str(exc)
            return record

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        list(pool.map(_safe, records))
    if aggregate:
        aggregate_records(records, min_relevance=min_relevance)
    return records


def write_meta(
    output_path: str,
    *,
    model: str,
    window: str,
    max_markers: int,
    min_relevance: int,
    n_records: int,
    n_spans: int,
    n_attached: int,
    n_errored: int,
    wall_seconds: float,
    input_path: str,
) -> str:
    """Write the ``<output>.meta.json`` compute-accounting sidecar.

    Ties a concrete run (input, knobs, span counts, wall time) to its SLURM
    job id so ``sacct`` GPU-hours can be attributed for the paper.
    """
    meta = {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "model": model,
        "window": window,
        "max_markers": max_markers,
        "min_relevance": min_relevance,
        "n_records": n_records,
        "n_spans": n_spans,
        "n_attached": n_attached,
        "n_errored": n_errored,
        "wall_seconds": round(wall_seconds, 1),
        "input": input_path,
        "output": output_path,
    }
    meta_path = output_path + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return meta_path


def main() -> None:
    """CLI entry point: load, sample, score, write JSON (+ optional HTML)."""
    parser = argparse.ArgumentParser(
        description="Score claim-evidence number relevance/NLI (vLLM).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", default=INPUT)
    parser.add_argument("--output", default=OUTPUT)
    parser.add_argument("--html", default=None, help="HTML report path.")
    parser.add_argument("--n", type=int, default=50, help="Sample (0=all).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default=DEFAULT_NLI_MODEL)
    parser.add_argument(
        "--base-url", default=DEFAULT_NLI_BASE_URL, dest="base_url"
    )
    parser.add_argument(
        "--max-workers", type=int, default=16, dest="max_workers"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=DEFAULT_NLI_MAX_TOKENS,
        dest="max_tokens",
        help="Generation budget per sentence call (truncation -> nli_error).",
    )
    parser.add_argument(
        "--system-role",
        action="store_true",
        help="Send a standalone system role (default: folded into user).",
    )
    parser.add_argument(
        "--window",
        default="sentence",
        choices=list(WINDOWS),
        help="Context window shown to the model (markers stay on the anchor "
        "sentence's numbers).",
    )
    parser.add_argument(
        "--max-markers", type=int, default=DEFAULT_MAX_MARKERS_PER_CALL,
        dest="max_markers",
        help="Max numbers marked per call; denser sentences are chunked "
        "into several calls (bounds output size below max_tokens).",
    )
    parser.add_argument(
        "--min-relevance", type=int, default=DEFAULT_MIN_RELEVANCE,
        dest="min_relevance",
        help="Aggregation threshold: spans below this relevance never count "
        "as supporting/refuting.",
    )
    parser.add_argument(
        "--no-aggregate", action="store_true",
        help="Skip the claim-level nli_aggregate rollup.",
    )
    parser.add_argument(
        "--html-max-records", type=int, default=0, dest="html_max_records",
        help="Render at most this many record blocks in the HTML report "
        "(0 = all; the summary always covers all records).",
    )
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as fh:
        records = json.load(fh)
    records = sample_records(records, args.n, args.seed)

    client = NliClient(
        model=args.model,
        base_url=args.base_url,
        max_tokens=args.max_tokens,
        system_in_user=not args.system_role,
    )
    started = time.monotonic()
    scored = score_records(
        records,
        client,
        max_workers=args.max_workers,
        window=args.window,
        max_markers=args.max_markers,
        min_relevance=args.min_relevance,
        aggregate=not args.no_aggregate,
    )
    wall_seconds = time.monotonic() - started

    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(scored, fh, ensure_ascii=False, indent=2)

    spans = [e for r in scored for e in r.get("evidence_numbers", [])]
    n_attached = sum(1 for e in spans if e.get("nli"))
    n_errored = sum(1 for e in spans if e.get("nli_error"))
    print(
        f"Scored {len(scored)} records (window={args.window}) -> {args.output} "
        f"({n_attached}/{len(spans)} spans attached, {n_errored} errored)"
    )
    if n_errored:
        print(f"WARNING: {n_errored} spans failed scoring (see nli_error).")

    meta_path = write_meta(
        args.output,
        model=args.model,
        window=args.window,
        max_markers=args.max_markers,
        min_relevance=args.min_relevance,
        n_records=len(scored),
        n_spans=len(spans),
        n_attached=n_attached,
        n_errored=n_errored,
        wall_seconds=wall_seconds,
        input_path=args.input,
    )
    print(f"Meta -> {meta_path}")

    if args.html:
        from src.number_annotation.nli.report import write_nli_report

        write_nli_report(scored, args.html,
                         max_records=args.html_max_records, seed=args.seed)
        print(f"Report -> {args.html}")


if __name__ == "__main__":
    main()
