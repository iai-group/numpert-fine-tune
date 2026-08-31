"""A/B re-audit of gate-rejected LLM cuts under alternative gate prompts.

Replays every `rejected` span from a hybrid run's `llm_diff_log.jsonl`
against the corresponding hybrid doc (which still contains the span,
because rejection means it was never removed): excise the span exactly the
way the gate did (whole-sentence via `apply_located_removals`) and ask the
audit model twice - once with the baseline prompt, once with a candidate
prompt (default: the greedy variant). The output quantifies how many
rejections flip to acceptance and on what kind of span.

Experiment log: docs/greedy_gate_experiment.md.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

from src.data_preprocessing.perturbations_engine.utils.load_save import load_data
from src.data_preprocessing.raw_dataset_preprocessing.verdict_strip.audit_llm import (  # noqa: E501
    PROMPTS,
    AuditClient,
)
from src.data_preprocessing.raw_dataset_preprocessing.verdict_strip.llm_locate import (  # noqa: E501
    apply_located_removals,
    locate_span,
)


def load_rejected_cuts(difflog_path: str) -> List[Dict[str, Any]]:
    """One row per rejected span: (datasplit, id, domain, quote)."""
    cuts: List[Dict[str, Any]] = []
    with open(difflog_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            for q in entry.get("rejected", []):
                cuts.append(
                    {
                        "id": entry.get("id"),
                        "datasplit": entry.get("datasplit"),
                        "domain": entry.get("domain"),
                        "quote": q,
                    }
                )
    return cuts


def load_docs(hybrid_dir: str) -> Dict[Any, str]:
    """Map (datasplit, id) -> doc for all three hybrid splits."""
    docs: Dict[Any, str] = {}
    for split in ("train", "val", "test"):
        path = Path(hybrid_dir) / f"{split}_claims_quantemp.json"
        if not path.exists():
            continue
        for rec in load_data(str(path)):
            docs[(rec.get("datasplit"), rec.get("id"))] = rec.get("doc", "")
    return docs


def reaudit_cut(
    cut: Dict[str, Any],
    doc: str,
    baseline_client: Any,
    candidate_client: Any,
) -> Dict[str, Any]:
    """Excise one rejected span and audit it under both prompts."""
    out = dict(cut)
    if locate_span(doc, cut["quote"]) is None:
        out["status"] = "unlocatable"
        return out
    cand = apply_located_removals(doc, [cut["quote"]], "sentence")
    rec = {
        "id": cut["id"],
        "cleaned_doc": cand.cleaned,
        "removed": [cut["quote"]],
    }
    try:
        out["baseline"] = baseline_client.check(rec)
        out["candidate"] = candidate_client.check(rec)
        out["status"] = "ok"
    except Exception as exc:  # isolate one bad cut from the rest
        out["status"] = "error"
        out["error"] = str(exc)
    return out


def reaudit_all(
    cuts: List[Dict[str, Any]],
    docs: Dict[Any, str],
    baseline_client: Any,
    candidate_client: Any,
    max_workers: int = 16,
) -> List[Dict[str, Any]]:
    def one(cut: Dict[str, Any]) -> Dict[str, Any]:
        doc = docs.get((cut["datasplit"], cut["id"]))
        if doc is None:
            return {**cut, "status": "missing_doc"}
        return reaudit_cut(cut, doc, baseline_client, candidate_client)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(one, cuts))


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate flip counts overall and per domain."""
    ok = [r for r in results if r.get("status") == "ok"]
    summary: Dict[str, Any] = {
        "total": len(results),
        "ok": len(ok),
        "unlocatable": sum(
            1 for r in results if r.get("status") == "unlocatable"
        ),
        "missing_doc": sum(
            1 for r in results if r.get("status") == "missing_doc"
        ),
        "errored": sum(1 for r in results if r.get("status") == "error"),
        # baseline reproducibility: does the old prompt still reject?
        "baseline_rejects_again": sum(
            1 for r in ok if r["baseline"]["removed_real_evidence"]
        ),
        "candidate_rejects": sum(
            1 for r in ok if r["candidate"]["removed_real_evidence"]
        ),
        # the interesting flips: baseline rejects, candidate accepts
        "flipped_to_accept": sum(
            1
            for r in ok
            if r["baseline"]["removed_real_evidence"]
            and not r["candidate"]["removed_real_evidence"]
        ),
        "by_domain": {},
    }
    for r in ok:
        d = summary["by_domain"].setdefault(
            r.get("domain") or "?",
            {"ok": 0, "baseline_rejects": 0, "candidate_rejects": 0},
        )
        d["ok"] += 1
        d["baseline_rejects"] += int(r["baseline"]["removed_real_evidence"])
        d["candidate_rejects"] += int(r["candidate"]["removed_real_evidence"])
    return summary


def format_report(
    results: List[Dict[str, Any]], summary: Dict[str, Any]
) -> str:
    ok = [r for r in results if r.get("status") == "ok"]
    lines = [
        "# Greedy-gate re-audit of rejected cuts",
        "",
        f"- cuts replayed: {summary['total']} (ok {summary['ok']}, "
        f"unlocatable {summary['unlocatable']}, missing_doc "
        f"{summary['missing_doc']}, errored {summary['errored']})",
        f"- baseline (old prompt) rejects again: "
        f"{summary['baseline_rejects_again']}/{summary['ok']}",
        f"- candidate (greedy prompt) rejects: "
        f"{summary['candidate_rejects']}/{summary['ok']}",
        f"- flipped to accept (baseline reject -> candidate accept): "
        f"{summary['flipped_to_accept']}",
        "",
        "| domain | ok | baseline rejects | candidate rejects |",
        "|---|---:|---:|---:|",
    ]
    for dom, d in sorted(
        summary["by_domain"].items(), key=lambda kv: -kv[1]["ok"]
    ):
        lines.append(
            f"| {dom} | {d['ok']} | {d['baseline_rejects']} "
            f"| {d['candidate_rejects']} |"
        )
    lines += ["", "## Disagreements (baseline reject, candidate accept)", ""]
    for r in ok:
        if r["baseline"]["removed_real_evidence"] and not (
            r["candidate"]["removed_real_evidence"]
        ):
            lines.append(
                f"- [{r['datasplit']} id={r['id']} {r['domain']}] "
                f"{r['quote'][:220]!r}"
            )
    lines += ["", "## Candidate still rejects", ""]
    for r in ok:
        if r["candidate"]["removed_real_evidence"]:
            lines.append(
                f"- [{r['datasplit']} id={r['id']} {r['domain']}] "
                f"{r['quote'][:220]!r}"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-audit gate-rejected cuts under an alternate prompt."
    )
    parser.add_argument(
        "--difflog", required=True, help="llm_diff_log.jsonl of a hybrid run."
    )
    parser.add_argument(
        "--hybrid-dir",
        required=True,
        help="Directory with that run's {split}_claims_quantemp.json.",
    )
    parser.add_argument(
        "--out-prefix",
        required=True,
        help="Writes <prefix>.jsonl (rows) and <prefix>.md (summary).",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument(
        "--baseline-prompt",
        choices=tuple(PROMPTS),
        default="baseline",
        help="Reproducibility control prompt (default: baseline).",
    )
    parser.add_argument(
        "--candidate-prompt",
        choices=tuple(PROMPTS),
        default="greedy",
        help="Prompt under test (default: greedy). Use greedy_v2 for the "
        "item-8 greedy-v1-vs-v2 A/B (--baseline-prompt greedy).",
    )
    args = parser.parse_args()

    cuts = load_rejected_cuts(args.difflog)
    docs = load_docs(args.hybrid_dir)
    print(
        f"replaying {len(cuts)} rejected cuts over {len(docs)} docs "
        f"(baseline={args.baseline_prompt}, candidate={args.candidate_prompt})"
    )
    baseline = AuditClient(
        model=args.model,
        base_url=args.base_url,
        system_prompt=PROMPTS[args.baseline_prompt],
    )
    candidate = AuditClient(
        model=args.model,
        base_url=args.base_url,
        system_prompt=PROMPTS[args.candidate_prompt],
    )
    results = reaudit_all(cuts, docs, baseline, candidate, args.max_workers)
    summary = summarize(results)

    out_jsonl = Path(args.out_prefix + ".jsonl")
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    Path(args.out_prefix + ".md").write_text(
        format_report(results, summary), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "by_domain"}))
    print(f"-> {out_jsonl} / {args.out_prefix}.md")


if __name__ == "__main__":
    main()
