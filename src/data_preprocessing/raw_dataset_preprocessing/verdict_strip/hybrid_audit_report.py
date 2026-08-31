"""Visual HTML audit of the HYBRID (LLM locate+gate) verdict stripping.

Samples records that the hybrid pass changed (``kept`` non-empty), runs the
audit LLM on each, and renders an HTML page showing, per sample:
  - the claim,
  - the FULL evidence document the hybrid pass saw (the rule-cleaned
    ``verdict_stripped`` doc), with the removed (``kept``) spans highlighted,
  - the two audit booleans (still_has_verdict / removed_real_evidence).

The audit LLM never edits text; it only judges. Data sources:
  - input docs + claim : verdict_stripped/rulebased/{split}.json
  - hybrid cleaned docs: verdict_stripped/hybrid/{split}.json
  - removed spans (kept): verdict_stripped/hybrid/llm_diff_log.jsonl
  (all under data/raw_data/)
"""

import argparse
import html as html_lib
import json
from pathlib import Path
from typing import Any, Dict, List

from src.data_preprocessing.perturbations_engine.utils.load_save import load_data
from src.data_preprocessing.raw_dataset_preprocessing.verdict_strip.audit_llm import (  # noqa: E501
    AuditClient,
)
from src.data_preprocessing.raw_dataset_preprocessing.verdict_strip.llm_locate import (  # noqa: E501
    apply_located_removals,
    locate_span,
)


def build_hybrid_audit_records(
    pristine_records: List[Dict[str, Any]],
    hybrid_records: List[Dict[str, Any]],
    hybrid_difflog: List[Dict[str, Any]],
    rule_difflog: List[Dict[str, Any]],
    only_keys: Any = None,
    rulebased_records: Any = None,
) -> List[Dict[str, Any]]:
    """Join pristine docs, hybrid outputs, and both diff logs.

    Keyed on ``(datasplit, id)``. Only entries the hybrid pass actually
    changed (``kept`` non-empty) are returned. The displayed ``input_doc`` is
    the **pristine** doc so that BOTH the rule-based removals (``rule_removed``,
    from the rule diff log) and the hybrid removals (``removed`` = kept) can be
    highlighted in distinct colours. ``cleaned_doc`` (the hybrid output) is what
    the audit model judges.

    ``only_keys`` (a set of ``(datasplit, id)``) switches to targeted mode:
    exactly those diff-log entries are returned, INCLUDING ones with no kept
    cuts (all-rejected / errored records) — used to audit problem samples.

    ``rulebased_records`` (the rule-cleaned splits the LLM pass consumed)
    enables ``removed_extents``: each record's kept quotes are replayed
    through ``apply_located_removals(..., "sentence")`` — the exact pipeline
    application — so the report can highlight the TRUE removed extent
    (whole sentence, glue-bounded), not just the located quote fragment.
    """
    pr_by = {(r.get("datasplit"), r.get("id")): r for r in pristine_records}
    hy_by = {(r.get("datasplit"), r.get("id")): r for r in hybrid_records}
    rb_by = {
        (r.get("datasplit"), r.get("id")): r
        for r in (rulebased_records or [])
    }
    rule_by: Dict[Any, List[str]] = {}
    footer_by: Dict[Any, List[str]] = {}
    for e in rule_difflog:
        key = (e.get("datasplit"), e.get("id"))
        for rm in e.get("removals", []):
            text = rm.get("removed_text", "")
            if not text:
                continue
            bucket = footer_by if rm.get("category") == "footer" else rule_by
            bucket.setdefault(key, []).append(text)
    out: List[Dict[str, Any]] = []
    for e in hybrid_difflog:
        kept = e.get("kept") or []
        key = (e.get("datasplit"), e.get("id"))
        if only_keys is not None:
            if key not in only_keys:
                continue
        elif not kept:
            continue
        src = pr_by.get(key, {})
        hy = hy_by.get(key, {})
        rb_doc = rb_by.get(key, {}).get("doc", "")
        extents: List[str] = []
        if kept and rb_doc:
            applied = apply_located_removals(rb_doc, kept, "sentence").applied
            # strip the pipeline's whitespace repair margins; locate_span
            # is whitespace-tolerant when re-finding these for display
            extents = [t.strip() for _s, _e, t in applied if t.strip()]
        out.append({
            "id": e.get("id"),
            "datasplit": e.get("datasplit"),
            "domain": e.get("domain", ""),
            "claim": src.get("claim", ""),
            "input_doc": src.get("doc", ""),
            "cleaned_doc": hy.get("doc", ""),
            "removed": kept,
            "removed_extents": extents,
            "rule_removed": rule_by.get(key, []),
            "footer_removed": footer_by.get(key, []),
            "rejected": e.get("rejected") or [],
            "unlocatable": e.get("unlocatable") or [],
        })
    return out


def diverse_sample(
    records: List[Dict[str, Any]], total: int = 100
) -> List[Dict[str, Any]]:
    """Round-robin across domains so rare fact-checkers are represented."""
    by_domain: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        by_domain.setdefault(r.get("domain", ""), []).append(r)
    queues = [list(v) for v in by_domain.values()]
    sample: List[Dict[str, Any]] = []
    while queues and len(sample) < total:
        next_queues = []
        for q in queues:
            if len(sample) >= total:
                break
            sample.append(q.pop(0))
            if q:
                next_queues.append(q)
        queues = next_queues
    return sample


def audit_all(
    records: List[Dict[str, Any]], client: Any
) -> List[Dict[str, Any]]:
    """Run the audit on every record; attach its verdict (errors isolated)."""
    out: List[Dict[str, Any]] = []
    for rec in records:
        try:
            verdict = client.check(rec)
        except Exception as exc:  # isolate one bad record from the rest
            verdict = {
                "still_has_verdict": None,
                "removed_real_evidence": None,
                "error": str(exc),
            }
        out.append({**rec, "audit": verdict})
    return out


def _locate_spans(doc: str, phrases: List[str], cls: str):
    """(start, end, cls) spans for `phrases`, tolerating case/whitespace."""
    spans = []
    for p in phrases:
        if not p:
            continue
        found = locate_span(doc, p)
        if found is not None:
            spans.append((found[0], found[1], cls))
    return spans


_SPAN_PRIORITY = {"removed": 0, "rule": 1, "footer": 2, "rejected": 3}


def highlight_html(
    doc: str, removed: List[str], rejected=(), rule_removed=(),
    footer_removed=(),
) -> str:
    """HTML-escape `doc`, wrapping spans by source:
      - removed (red)    : the LLM/hybrid cut that was applied,
      - rule (blue)      : verdict rule-based removal,
      - footer (purple)  : footer / sign-off rule removal,
      - rejected (amber) : locator proposed, gate kept as evidence.
    On overlap, higher-priority (removed > rule > footer > rejected) wins.
    """
    spans = (
        _locate_spans(doc, list(removed), "removed")
        + _locate_spans(doc, list(rule_removed), "rule")
        + _locate_spans(doc, list(footer_removed), "footer")
        + _locate_spans(doc, list(rejected), "rejected")
    )
    spans.sort(key=lambda s: (s[0], _SPAN_PRIORITY.get(s[2], 9)))
    parts: List[str] = []
    cursor = 0
    for start, end, cls in spans:
        if start < cursor:  # overlaps an already-emitted span; skip
            continue
        parts.append(html_lib.escape(doc[cursor:start]))
        parts.append(
            f'<span class="{cls}">'
            + html_lib.escape(doc[start:end])
            + "</span>"
        )
        cursor = end
    parts.append(html_lib.escape(doc[cursor:]))
    return "".join(parts)


_CSS = """
body{font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;
background:#f6f7f9;color:#1a1a1a}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;
padding:12px 20px;box-shadow:0 1px 3px rgba(0,0,0,.06)}
h1{font-size:18px;margin:0}
.sub{color:#666;font-size:13px;margin-top:4px}
.card{background:#fff;margin:16px 20px;border:1px solid #e3e3e3;
border-radius:8px;padding:16px 18px}
.meta{font-size:12px;color:#888;font-family:ui-monospace,monospace}
.claim{margin:8px 0 12px;padding:8px 12px;background:#eef3fb;border-left:3px
solid #4a7fd0;border-radius:4px}
.doc{white-space:pre-wrap;word-wrap:break-word;background:#fafafa;border:1px
solid #eee;border-radius:6px;padding:12px;max-height:420px;overflow:auto}
.removed{background:#ffd6d6;color:#8a1c1c;text-decoration:line-through;
border-radius:3px;padding:0 2px}
.rule{background:#d7e6ff;color:#1a3a73;text-decoration:line-through;
border-radius:3px;padding:0 2px}
.footer{background:#e8dcff;color:#4b2a86;text-decoration:line-through;
border-radius:3px;padding:0 2px}
.rejected{background:#fff1c2;color:#7a5800;border-radius:3px;padding:0 2px}
.unloc{margin-top:10px;font-size:13px;color:#7a5800}
.unloc ul{margin:4px 0 0;padding-left:18px}
.legend{font-size:12px;color:#555;margin-top:6px}
.legend .removed,.legend .rule,.legend .footer,.legend .rejected{
padding:1px 6px}
.badges{margin:10px 0}
.badge{display:inline-block;font-size:12px;font-weight:600;border-radius:999px;
padding:3px 10px;margin-right:8px}
.t{background:#fde2e1;color:#a11}.f{background:#e3f3e5;color:#176b2c}
.n{background:#eee;color:#777}
"""


def _badge(label: str, value: Any) -> str:
    cls = {True: "t", False: "f"}.get(value, "n")
    return f'<span class="badge {cls}">{label}={value}</span>'


def _unlocatable_html(unlocatable: List[str]) -> str:
    """Render the locator's quotes that were not found verbatim in the doc."""
    if not unlocatable:
        return ""
    items = "".join(
        f"<li>{html_lib.escape(u)}</li>" for u in unlocatable if u
    )
    return (
        '<div class="unloc"><b>Unlocatable</b> (locator quoted, not found '
        f"verbatim — ignored):<ul>{items}</ul></div>"
    )


def render_html(audited: List[Dict[str, Any]]) -> str:
    """Render the audited sample as a self-contained HTML page."""
    n = len(audited)
    leaks = sum(1 for r in audited if r["audit"].get("still_has_verdict"))
    over = sum(1 for r in audited if r["audit"].get("removed_real_evidence"))
    by_dom: Dict[str, int] = {}
    for r in audited:
        by_dom[r["domain"]] = by_dom.get(r["domain"], 0) + 1
    dom_summary = ", ".join(f"{d}:{c}" for d, c in sorted(by_dom.items()))
    cards: List[str] = []
    for r in audited:
        a = r["audit"]
        cards.append(
            '<div class="card">'
            f'<div class="meta">id={r["id"]} · {r["datasplit"]} · '
            f'{html_lib.escape(r["domain"])}</div>'
            '<div class="badges">'
            + _badge("still_has_verdict", a.get("still_has_verdict"))
            + _badge("removed_real_evidence", a.get("removed_real_evidence"))
            + ('<span class="badge n">error</span>' if a.get("error") else "")
            + "</div>"
            '<div class="claim"><b>Claim:</b> '
            + html_lib.escape(r["claim"])
            + "</div>"
            '<div class="doc">'
            # Extents first: the full sentence-snapped cut is wrapped where
            # locatable in the pristine doc; the bare quote (contained in
            # its extent) is skipped by the overlap logic and only serves
            # as a fallback where an extent is not locatable (e.g. a rule
            # cut broke its contiguity in the pristine text).
            + highlight_html(
                r["input_doc"],
                (r.get("removed_extents") or []) + r["removed"],
                r.get("rejected", []),
                r.get("rule_removed", []), r.get("footer_removed", []),
            )
            + "</div>"
            + _unlocatable_html(r.get("unlocatable", []))
            + "</div>"
        )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<style>{_CSS}</style><title>Hybrid verdict-strip audit</title></head>"
        "<body><header><h1>Hybrid verdict-strip audit</h1>"
        f'<div class="sub">{n} samples · still_has_verdict (missed leak): '
        f"{leaks} · removed_real_evidence (over-removal): {over} · "
        f"domains — {html_lib.escape(dom_summary)}</div>"
        '<div class="legend">'
        '<span class="removed">LLM removed</span> = hybrid cut '
        "(full sentence-snapped extent) &nbsp; "
        '<span class="rule">rule removed</span> = rule-based cut &nbsp; '
        '<span class="footer">footer</span> = footer/sign-off rule cut '
        "&nbsp; "
        '<span class="rejected">rejected</span> = locator proposed, '
        "gate kept as evidence &nbsp; · unlocatable listed per card"
        "</div></header>"
        + "".join(cards)
        + "</body></html>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HTML audit of hybrid verdict stripping (sampled)."
    )
    parser.add_argument(
        "--pristine-dir", default="data/raw_data/QuanTemp-main-data",
        help="Pristine splits (full evidence doc + claim) for display.",
    )
    parser.add_argument(
        "--input-dir", default="data/raw_data/verdict_stripped/rulebased",
        help="Rule-cleaned splits + diff_log.jsonl (rule removals).",
    )
    parser.add_argument(
        "--hybrid-dir", default="data/raw_data/verdict_stripped/hybrid",
        help="Hybrid-cleaned splits + llm_diff_log.jsonl.",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val", "test"],
    )
    parser.add_argument("--total", type=int, default=100)
    parser.add_argument(
        "--ids",
        help="Targeted mode: comma-separated datasplit:id keys, or "
        "@<file> with one key per line. Renders exactly these records "
        "(including all-rejected/errored ones) instead of sampling.",
    )
    parser.add_argument(
        "--report",
        default="data/raw_data/verdict_stripped/hybrid/audit_report.html",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    args = parser.parse_args()

    pristine_records: List[Dict[str, Any]] = []
    hybrid_records: List[Dict[str, Any]] = []
    rulebased_records: List[Dict[str, Any]] = []
    for s in args.splits:
        fname = f"{s}_claims_quantemp.json"
        pristine_records += load_data(str(Path(args.pristine_dir) / fname))
        hybrid_records += load_data(str(Path(args.hybrid_dir) / fname))
        rulebased_records += load_data(str(Path(args.input_dir) / fname))
    with open(Path(args.hybrid_dir) / "llm_diff_log.jsonl", encoding="utf-8") as f:  # noqa: E501
        hybrid_difflog = [json.loads(line) for line in f if line.strip()]
    with open(Path(args.input_dir) / "diff_log.jsonl", encoding="utf-8") as f:
        rule_difflog = [json.loads(line) for line in f if line.strip()]

    only_keys = None
    if args.ids:
        if args.ids.startswith("@"):
            specs = [
                ln.strip()
                for ln in Path(args.ids[1:]).read_text().splitlines()
                if ln.strip()
            ]
        else:
            specs = [s for s in args.ids.split(",") if s.strip()]
        only_keys = set()
        for spec in specs:
            split, _, rid = spec.partition(":")
            only_keys.add((split, int(rid)))

    records = build_hybrid_audit_records(
        pristine_records, hybrid_records, hybrid_difflog, rule_difflog,
        only_keys=only_keys, rulebased_records=rulebased_records,
    )
    sample = (
        records if only_keys is not None
        else diverse_sample(records, args.total)
    )
    client = AuditClient(model=args.model, base_url=args.base_url)
    audited = audit_all(sample, client)
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(audited), encoding="utf-8")
    print(f"audited {len(audited)} samples -> {args.report}")


if __name__ == "__main__":
    main()
