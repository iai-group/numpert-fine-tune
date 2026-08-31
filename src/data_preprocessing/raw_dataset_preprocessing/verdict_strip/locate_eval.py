"""Feasibility spike harness for LLM-assisted verdict locating.

Builds eval cohorts, then per record: initial audit (partition recall vs
clean-control) -> locate -> apply both removal strategies -> re-audit ->
aggregate metrics into a markdown report. The LLM never edits; code removes
only verbatim substrings (see llm_locate.apply_located_removals).
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from src.data_preprocessing.perturbations_engine.utils.load_save import load_data
from src.data_preprocessing.raw_dataset_preprocessing.verdict_strip.audit_llm import (  # noqa: E501
    AuditClient,
    build_audit_records,
)
from src.data_preprocessing.raw_dataset_preprocessing.verdict_strip.llm_locate import (  # noqa: E501
    LocateClient,
    apply_located_removals,
)

STRATEGIES = ("phrase", "sentence")


def build_eval_set(
    cleaned_records: List[Dict[str, Any]],
    difflog: List[Dict[str, Any]],
    per_cohort: int = 40,
) -> List[Dict[str, Any]]:
    """Label records as `changed` (a rule fired) or `control_norule`."""
    joined = build_audit_records(cleaned_records, difflog)
    changed: List[Dict[str, Any]] = []
    norule: List[Dict[str, Any]] = []
    for r in joined:
        rec = {
            "id": r["id"],
            "datasplit": r.get("datasplit"),
            "domain": r["domain"],
            "cleaned_doc": r["cleaned_doc"],
            "removed": r["removed"],
        }
        if r["changed"]:
            rec["cohort"] = "changed"
            changed.append(rec)
        else:
            rec["cohort"] = "control_norule"
            norule.append(rec)
    return changed[:per_cohort] + norule[:per_cohort]


def evaluate(
    eval_records: List[Dict[str, Any]],
    locate_client: Any,
    audit_client: Any,
) -> Dict[str, Any]:
    """Per record: initial audit -> locate -> remove (both) -> re-audit."""
    records = [
        evaluate_one(rec, locate_client, audit_client)
        for rec in eval_records
    ]
    return _aggregate(records)


def evaluate_one(
    rec: Dict[str, Any],
    locate_client: Any,
    audit_client: Any,
) -> Dict[str, Any]:
    """Evaluate a single record (initial audit -> locate -> remove -> re-audit).

    Failures are caught and reported in `out["error"]`, so this is safe to run
    concurrently across records.
    """
    out: Dict[str, Any] = {
        "id": rec["id"],
        "domain": rec["domain"],
        "cohort": rec["cohort"],
        "label": rec.get("label"),
        "error": None,
    }
    try:
        doc = rec["cleaned_doc"]
        initial = audit_client.check(
            {"id": rec["id"], "cleaned_doc": doc, "removed": []}
        )
        still = initial["still_has_verdict"]
        if rec["cohort"] == "control_norule":
            out["cohort_resolved"] = "control_norule"
        else:
            out["cohort_resolved"] = "recall" if still else "clean_control"
        out["initial_still_has_verdict"] = still

        located = locate_client.locate(
            {"id": rec["id"], "cleaned_doc": doc}
        )
        out["has_verdict"] = located["has_verdict"]
        quotes = located["verdict_quotes"]

        variants: Dict[str, Any] = {}
        for strat in STRATEGIES:
            rem = apply_located_removals(doc, quotes, strat)
            post = audit_client.check(
                {"id": rec["id"], "cleaned_doc": rem.cleaned,
                 "removed": [a[2] for a in rem.applied]}
            )
            variants[strat] = {
                "located_count": len(rem.applied),
                "unlocatable_count": len(rem.unlocatable),
                "char_delta": len(doc) - len(rem.cleaned),
                "post_still_has_verdict": post["still_has_verdict"],
                "post_removed_real_evidence": post[
                    "removed_real_evidence"
                ],
            }
        out["variants"] = variants
        out["locatable_any"] = any(
            v["located_count"] > 0 for v in variants.values()
        )
    except Exception as exc:  # isolate one bad record
        out["error"] = str(exc)
    return out


def _aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r for r in records if not r["error"]]
    recall = [r for r in ok if r.get("cohort_resolved") == "recall"]
    clean = [r for r in ok if r.get("cohort_resolved") == "clean_control"]
    with_verdict = [r for r in ok if r.get("has_verdict")]
    cohorts = Counter(r.get("cohort_resolved") for r in ok)

    def cleared(strat: str) -> float:
        if not recall:
            return 0.0
        n = sum(
            1 for r in recall
            if not r["variants"][strat]["post_still_has_verdict"]
        )
        return n / len(recall)

    fp = (
        sum(1 for r in clean if r["locatable_any"]) / len(clean)
        if clean else 0.0
    )
    return {
        "n_eval": len(records),
        "n_errors": sum(1 for r in records if r["error"]),
        "cohorts": dict(cohorts),
        "verbatim_hit_rate": (
            sum(1 for r in with_verdict if r["locatable_any"])
            / len(with_verdict) if with_verdict else 0.0
        ),
        "audit_cleared": {s: cleared(s) for s in STRATEGIES},
        "false_positive_rate": fp,
        "records": records,
    }


def format_locate_eval_report(report: Dict[str, Any]) -> str:
    """Render the eval report as reviewable markdown."""
    ac = report["audit_cleared"]
    lines = [
        "# LLM verdict-locate feasibility spike",
        "",
        f"- eval records: {report['n_eval']} (errors: {report['n_errors']})",
        "- cohorts: "
        + ", ".join(f"{k}: {v}" for k, v in report["cohorts"].items()),
        f"- verbatim-hit rate: {report['verbatim_hit_rate']:.2f}",
        f"- audit-cleared (phrase): {ac['phrase']:.2f}",
        f"- audit-cleared (sentence): {ac['sentence']:.2f}",
        f"- false-positive rate (clean controls): "
        f"{report['false_positive_rate']:.2f}",
        "",
        "## Records",
        "",
    ]
    for r in report["records"]:
        if r["error"]:
            lines.append(f"- id={r['id']} {r['domain']} ERROR: {r['error']}")
            continue
        v = r.get("variants", {})
        ph = v.get("phrase", {})
        se = v.get("sentence", {})
        lines.append(
            f"- id={r['id']} {r['domain']} "
            f"[{r.get('cohort_resolved')}] "
            f"has_verdict={r.get('has_verdict')} "
            f"located={ph.get('located_count')} "
            f"phrase_cleared={not ph.get('post_still_has_verdict')} "
            f"sentence_cleared={not se.get('post_still_has_verdict')}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM verdict-locate feasibility spike."
    )
    parser.add_argument("--cleaned", required=True, help="Cleaned split JSON.")
    parser.add_argument("--difflog", required=True, help="diff_log.jsonl.")
    parser.add_argument(
        "--report",
        default="data/raw_data/verdict_stripped/rulebased/llm_locate_eval.md",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--per-cohort", type=int, default=40)
    args = parser.parse_args()

    cleaned = load_data(args.cleaned)
    with open(args.difflog, encoding="utf-8") as f:
        difflog = [json.loads(line) for line in f if line.strip()]
    eval_records = build_eval_set(cleaned, difflog, args.per_cohort)
    locate_client = LocateClient(model=args.model, base_url=args.base_url)
    audit_client = AuditClient(model=args.model, base_url=args.base_url)
    report = evaluate(eval_records, locate_client, audit_client)

    out_md = Path(args.report)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(
        format_locate_eval_report(report), encoding="utf-8"
    )
    out_json = out_md.with_suffix(".json")
    out_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"eval {report['n_eval']} records "
        f"(errors {report['n_errors']}) -> {args.report}"
    )


if __name__ == "__main__":
    main()
