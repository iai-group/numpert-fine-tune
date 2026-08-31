"""Hybrid (rules -> LLM long-tail) evaluation over a label-balanced sample.

Samples N records per veracity label (True/False/Conflicting) from the
rule-cleaned split, then for each record: audit the rule-cleaned doc (residual
after rules), LLM-locate + phrase-remove, re-audit (residual after rules+LLM).
Aggregates per label so we can see how much the LLM stage adds on top of the
(broadened) regex rules, and at what over-removal cost. Concurrent across
records so 500 x ~4 calls finishes in reasonable wall-time.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

from src.data_preprocessing.perturbations_engine.utils.load_save import load_data
from src.data_preprocessing.raw_dataset_preprocessing.verdict_strip.audit_llm import (  # noqa: E501
    AuditClient,
)
from src.data_preprocessing.raw_dataset_preprocessing.verdict_strip.llm_locate import (  # noqa: E501
    LocateClient,
)
from src.data_preprocessing.raw_dataset_preprocessing.verdict_strip.locate_eval import (  # noqa: E501
    evaluate_one,
)
from src.data_preprocessing.raw_dataset_preprocessing.verdict_strip.rules import (  # noqa: E501
    domain_of,
)

LABELS = ("True", "False", "Conflicting")


def build_label_balanced_set(
    cleaned_records: List[Dict[str, Any]],
    difflog: List[Dict[str, Any]],
    per_label: int = 167,
    labels=LABELS,
) -> List[Dict[str, Any]]:
    """Stride-sample `per_label` records for each veracity label.

    Records changed by a rule are tagged cohort `changed` (with the removed
    spans); untouched ones `control_norule`. A stride spreads the sample across
    each label's records (deterministic, no RNG).
    """
    by_key = {(e.get("datasplit"), e.get("id")): e for e in difflog}
    out: List[Dict[str, Any]] = []
    for lab in labels:
        pool = [r for r in cleaned_records if r.get("label") == lab]
        step = max(1, len(pool) // per_label) if per_label else 1
        for rec in pool[::step][:per_label]:
            entry = by_key.get((rec.get("datasplit"), rec.get("id")))
            if entry is not None:
                domain = entry.get("domain", "")
                removed = [
                    r.get("removed_text", "")
                    for r in entry.get("removals", [])
                ]
                cohort = "changed"
            else:
                domain = domain_of(rec.get("url", "") or "")
                removed = []
                cohort = "control_norule"
            out.append({
                "id": rec.get("id"),
                "datasplit": rec.get("datasplit"),
                "label": lab,
                "domain": domain,
                "cohort": cohort,
                "cleaned_doc": rec.get("doc", ""),
                "removed": removed,
            })
    return out


def _phrase(r: Dict[str, Any]) -> Dict[str, Any]:
    return r["variants"]["phrase"]


def _label_metrics(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r for r in recs if not r["error"]]
    n = len(ok)

    def rate(pred) -> float:
        return round(sum(1 for r in ok if pred(r)) / n, 3) if n else 0.0

    residual_rules = rate(lambda r: r.get("initial_still_has_verdict"))
    residual_hybrid = rate(lambda r: _phrase(r)["post_still_has_verdict"])
    resid = [r for r in ok if r.get("initial_still_has_verdict")]
    locate_recall = (
        round(
            sum(1 for r in resid if _phrase(r)["located_count"] > 0)
            / len(resid),
            3,
        )
        if resid else 0.0
    )
    removed = [r for r in ok if _phrase(r)["located_count"] > 0]
    over = (
        round(
            sum(1 for r in removed if _phrase(r)["post_removed_real_evidence"])
            / len(removed),
            3,
        )
        if removed else 0.0
    )
    return {
        "n": n,
        "n_errors": sum(1 for r in recs if r["error"]),
        "residual_after_rules": residual_rules,
        "residual_after_hybrid": residual_hybrid,
        "improvement": round(residual_rules - residual_hybrid, 3),
        "locate_recall_on_residual": locate_recall,
        "llm_removed_records": len(removed),
        "over_removal_rate": over,
    }


def aggregate_by_label(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_label = {}
    for lab in LABELS:
        recs = [r for r in records if r.get("label") == lab]
        if recs:
            by_label[lab] = _label_metrics(recs)
    return {
        "overall": _label_metrics(records),
        "by_label": by_label,
        "records": records,
    }


def evaluate_hybrid(
    eval_records: List[Dict[str, Any]],
    locate_client: Any,
    audit_client: Any,
    max_workers: int = 16,
) -> Dict[str, Any]:
    """Run `evaluate_one` concurrently across records; aggregate by label."""
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        records = list(
            ex.map(
                lambda r: evaluate_one(r, locate_client, audit_client),
                eval_records,
            )
        )
    return aggregate_by_label(records)


def format_hybrid_report(report: Dict[str, Any]) -> str:
    def row(name: str, m: Dict[str, Any]) -> str:
        return (
            f"| {name} | {m['n']} | {m['residual_after_rules']} | "
            f"{m['residual_after_hybrid']} | {m['improvement']} | "
            f"{m['locate_recall_on_residual']} | {m['over_removal_rate']} "
            f"({m['llm_removed_records']}) |"
        )

    o = report["overall"]
    lines = [
        "# Hybrid (rules -> LLM long-tail) eval",
        "",
        f"- records: {o['n']} (errors: {o['n_errors']})",
        "",
        "| cohort | n | residual_rules | residual_hybrid | improvement | "
        "locate_recall | over_removal (n) |",
        "|---|---|---|---|---|---|---|",
        row("overall", o),
    ]
    for lab, m in report["by_label"].items():
        lines.append(row(lab, m))
    lines.append("")
    lines.append(
        "residual_* = fraction the audit still flags as containing a verdict "
        "after rules / after rules+LLM. improvement = rules - hybrid. "
        "over_removal = fraction of LLM-removed records the audit flags as "
        "having lost real evidence."
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hybrid (rules -> LLM) label-balanced eval."
    )
    parser.add_argument("--cleaned", required=True)
    parser.add_argument("--difflog", required=True)
    parser.add_argument(
        "--report",
        default="data/raw_data/verdict_stripped/rulebased/hybrid_eval.md",
    )
    # Single served model for both roles (back-compat) ...
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    # ... or split roles for a fixed-judge comparison (locate vs audit served
    # by different models/endpoints).
    parser.add_argument("--locate-model")
    parser.add_argument("--locate-base-url")
    parser.add_argument("--audit-model")
    parser.add_argument("--audit-base-url")
    parser.add_argument("--per-label", type=int, default=167)
    parser.add_argument("--max-workers", type=int, default=16)
    args = parser.parse_args()

    locate_model = args.locate_model or args.model
    locate_url = args.locate_base_url or args.base_url
    audit_model = args.audit_model or args.model
    audit_url = args.audit_base_url or args.base_url
    if not all([locate_model, locate_url, audit_model, audit_url]):
        parser.error(
            "provide --model/--base-url, or the per-role "
            "--locate-model/--locate-base-url/--audit-model/--audit-base-url"
        )

    cleaned = load_data(args.cleaned)
    with open(args.difflog, encoding="utf-8") as f:
        difflog = [json.loads(line) for line in f if line.strip()]
    eval_records = build_label_balanced_set(cleaned, difflog, args.per_label)
    locate_client = LocateClient(model=locate_model, base_url=locate_url)
    audit_client = AuditClient(model=audit_model, base_url=audit_url)
    report = evaluate_hybrid(
        eval_records, locate_client, audit_client, args.max_workers
    )

    out_md = Path(args.report)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(format_hybrid_report(report), encoding="utf-8")
    out_md.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"hybrid eval {report['overall']['n']} records "
        f"(errors {report['overall']['n_errors']}) -> {args.report}"
    )


if __name__ == "__main__":
    main()
