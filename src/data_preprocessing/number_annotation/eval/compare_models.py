"""Compare smoke-run annotations across models.

Loads `smoke_<model>.json` files and reports, per model, the located /
failure counts and validity, plus span agreement against a reference
model (default `opus`) via `evaluate.score_spans`. Use it to pick the
cheapest model that holds annotation quality before scaling to the seed.

All models must have been run on the SAME sample (same `--n`/`--seed`),
so offsets are comparable across files.
"""

import argparse
import html as html_lib
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.data_preprocessing.perturbations_engine.utils.load_save import load_data
from src.number_annotation.eval.evaluate import score_spans
from src.number_annotation.core.schema import NumberSpan, span_from_dict

SMOKE = "data/spacy_number_annotated/smoke_{model}_four_samples.json"
_AGREEMENT_FIELDS = [
    "overlap_f1",
    "exact_f1",
    "value_accuracy",
    "type_accuracy",
]

# Pastel colours assigned to each model in order
_MODEL_COLOURS = [
    "#a8d8ea",  # blue
    "#ffcba4",  # orange
    "#b5ead7",  # green
    "#ffd6e0",  # pink
    "#e2cfea",  # purple
    "#ffffb5",  # yellow
]


def _spans(record: Dict[str, Any], key: str) -> List[NumberSpan]:
    """Deserialize a record's span list under `key`."""
    return [span_from_dict(s) for s in record.get(key, [])]


def _index(records: List[Dict[str, Any]]) -> Dict[Any, Dict[str, Any]]:
    """Index records by `claim_id`."""
    return {r.get("claim_id"): r for r in records}


def summarize(records: List[Dict[str, Any]]) -> Dict[str, int]:
    """Return aggregate located/failure/validity counts for one model."""
    def ann(r: Dict[str, Any], field: str) -> int:
        return r.get("_annotation", {}).get(field, 0)

    return {
        "records": len(records),
        "claim_numbers": sum(
            len(r.get("claim_numbers", [])) for r in records
        ),
        "evidence_numbers": sum(
            len(r.get("evidence_numbers", [])) for r in records
        ),
        "claim_failures": sum(ann(r, "claim_failures") for r in records),
        "evidence_failures": sum(
            ann(r, "evidence_failures") for r in records
        ),
        "evidence_spans_valid_records": sum(
            1
            for r in records
            if r.get("_annotation", {}).get("evidence_spans_valid")
        ),
    }


def compare(
    ref_records: List[Dict[str, Any]],
    cand_records: List[Dict[str, Any]],
    key: str = "evidence_numbers",
) -> Dict[str, float]:
    """Score candidate spans against reference spans, averaged.

    Treats the reference model's spans as 'gold' and averages
    `score_spans` over records present in both files.
    """
    ref = _index(ref_records)
    cand = _index(cand_records)
    shared = [cid for cid in cand if cid in ref]
    if not shared:
        return {"shared": 0}

    totals = {f: 0.0 for f in _AGREEMENT_FIELDS}
    for cid in shared:
        scores = score_spans(_spans(cand[cid], key), _spans(ref[cid], key))
        for f in _AGREEMENT_FIELDS:
            totals[f] += getattr(scores, f)

    result: Dict[str, float] = {"shared": len(shared)}
    result.update({f: totals[f] / len(shared) for f in _AGREEMENT_FIELDS})
    return result


def _highlight_text(text: str, spans: List[NumberSpan], colour: str) -> str:
    """Return HTML with each span wrapped in a coloured <mark> tag."""
    if not spans:
        return html_lib.escape(text)

    # Sort spans by start offset; skip overlapping ones naively
    ordered = sorted(spans, key=lambda s: s.char_start)
    parts: List[str] = []
    cursor = 0
    for sp in ordered:
        if sp.char_start < cursor:
            continue
        parts.append(html_lib.escape(text[cursor:sp.char_start]))
        title = (
            f"{sp.surface} | {sp.value_type} | {sp.modifier} | val={sp.value}"
        )
        parts.append(
            f'<mark style="background:{colour};border-radius:3px;padding:0 2px" title="{html_lib.escape(title)}">'
            f"{html_lib.escape(text[sp.char_start:sp.char_end])}</mark>"
        )
        cursor = sp.char_end
    parts.append(html_lib.escape(text[cursor:]))
    return "".join(parts)


_REASON_LABELS = {
    "surface_not_found": (
        "surface_not_found",
        "Surface absent from text (hallucinated or normalized)",
    ),
    "all_occurrences_claimed": (
        "all_occurrences_claimed",
        "Surface occurs but every slot already taken (likely duplicate)",
    ),
}


def _reason_cell(reason: str, duplicate_of: Any) -> str:
    """Return an HTML cell describing the failure reason + duplicate flag."""
    label, tip = _REASON_LABELS.get(reason, (reason, ""))
    cell = (
        f"<span title=\"{html_lib.escape(tip)}\">{html_lib.escape(label)}</span>"
        if tip
        else html_lib.escape(str(reason))
    )
    if duplicate_of is not None:
        cell += (
            f" <span class='dup-flag' title='Exact duplicate of "
            f"already-located span #{duplicate_of}'>dup of #{duplicate_of}</span>"
        )
    return cell


def _failure_section(failures: List[Dict[str, Any]]) -> str:
    """Return an HTML block listing LocationFailure details."""
    if not failures:
        return ""
    rows = "\n".join(
        f"<tr>"
        f"<td><code>{html_lib.escape(f.get('surface', ''))}</code></td>"
        f"<td>{html_lib.escape(f.get('context', ''))}</td>"
        f"<td>{_reason_cell(f.get('reason', ''), f.get('duplicate_of'))}</td>"
        f"</tr>"
        for f in failures
    )
    return (
        "<details open><summary class='fail-summary'>"
        f"{len(failures)} location failure(s)</summary>"
        "<table class='fail-table'><thead><tr>"
        "<th>surface (LLM output)</th><th>context snippet</th><th>reason</th>"
        "</tr></thead><tbody>"
        + rows
        + "</tbody></table></details>"
    )


def _span_table(spans: List[NumberSpan], ref_spans: Optional[List[NumberSpan]] = None) -> str:
    """Return an HTML table of spans.

    Rows whose character range does not overlap any span in *ref_spans* receive
    the ``not-in-ref`` class so they can be styled distinctively.  When
    *ref_spans* is ``None`` (e.g. for the reference model itself) no rows are
    highlighted.
    """
    if not spans:
        return "<p><em>none</em></p>"

    def _overlaps_ref(s: NumberSpan) -> bool:
        if ref_spans is None:
            return True
        return any(
            s.char_start < r.char_end and s.char_end > r.char_start
            for r in ref_spans
        )

    rows = "\n".join(
        (
            f"<tr class='not-in-ref'><td>{s.id}</td><td>{html_lib.escape(s.surface)}</td>"
            if not _overlaps_ref(s)
            else f"<tr><td>{s.id}</td><td>{html_lib.escape(s.surface)}</td>"
        )
        + f"<td>{s.char_start}–{s.char_end}</td>"
        f"<td>{s.value_type}</td><td>{s.modifier}</td>"
        f"<td>{s.value if s.value is not None else ''}</td>"
        f"<td>{html_lib.escape(s.unit or '')}</td></tr>"
        for s in sorted(spans, key=lambda s: s.char_start)
    )
    return (
        "<table><thead><tr>"
        "<th>#</th><th>surface</th><th>offsets</th>"
        "<th>type</th><th>modifier</th><th>value</th><th>unit</th>"
        "</tr></thead><tbody>"
        + rows
        + "</tbody></table>"
    )


def _missing_span_table(missing: List[NumberSpan]) -> str:
    """Return an HTML table for reference spans absent from the model."""
    if not missing:
        return "<p><em>all reference spans covered</em></p>"
    return _span_table(sorted(missing, key=lambda s: s.char_start))


def generate_html(
    models: List[str],
    loaded: Dict[str, List[Dict[str, Any]]],
    summaries: Dict[str, Dict[str, int]],
    agreements: Dict[str, Dict[str, float]],
    ref: str,
    key: str,
) -> str:
    """Build a self-contained HTML report string."""
    colour_map = {
        m: _MODEL_COLOURS[i % len(_MODEL_COLOURS)] for i, m in enumerate(models)
    }

    # ── summary tables ────────────────────────────────────────────────────────
    count_headers = [
        "model", "records", "claim_numbers", "evidence_numbers",
        "claim_failures", "evidence_failures", "evidence_spans_valid_records",
    ]
    count_rows = "".join(
        "<tr>"
        + "".join(
            f"<td style='background:{colour_map[m]}'>{m}</td>"
            if h == "model"
            else (
                f"<td class='fail'>{summaries[m][h]}</td>"
                if h in ("claim_failures", "evidence_failures") and summaries[m][h] > 0
                else f"<td>{summaries[m][h]}</td>"
            )
            for h in count_headers
        )
        + "</tr>"
        for m in models
    )
    count_table = (
        "<table><thead><tr>"
        + "".join(f"<th>{h}</th>" for h in count_headers)
        + "</tr></thead><tbody>"
        + count_rows
        + "</tbody></table>"
    )

    agree_headers = ["model", "shared"] + _AGREEMENT_FIELDS
    agree_rows = "".join(
        "<tr>"
        + f"<td style='background:{colour_map[m]}'>{m}</td>"
        + "".join(
            f"<td>{agreements[m].get(h, '—')}</td>"
            if h == "shared"
            else f"<td>{agreements[m].get(h, '—'):.3f}</td>"
            if h in agreements[m]
            else "<td>—</td>"
            for h in agree_headers[1:]
        )
        + "</tr>"
        for m in models
        if m != ref and m in agreements
    )
    agree_section = (
        f"<h2>Agreement vs <em>{ref}</em> on <code>{key}</code></h2>"
        "<table><thead><tr>"
        + "".join(f"<th>{h}</th>" for h in agree_headers)
        + "</tr></thead><tbody>"
        + agree_rows
        + "</tbody></table>"
    ) if agree_rows else ""

    # ── per-record detail ─────────────────────────────────────────────────────
    # Build a union of all claim_ids across models
    all_ids: List[Any] = []
    seen_ids: set = set()
    for m in models:
        for r in loaded[m]:
            cid = r.get("claim_id")
            if cid not in seen_ids:
                all_ids.append(cid)
                seen_ids.add(cid)

    indexed = {m: _index(loaded[m]) for m in models}

    record_html_parts: List[str] = []
    for rec_idx, cid in enumerate(all_ids):
        # Use first available record for the document text / claim text
        first_rec = next(
            (indexed[m][cid] for m in models if cid in indexed[m]), None
        )
        if first_rec is None:
            continue

        claim_text = first_rec.get("claim", "")
        doc_text = first_rec.get("doc", "")
        label = first_rec.get("label", "")
        label_original = first_rec.get("label_original", "")

        # Build highlighted doc for each model
        panels: List[str] = []
        for m in models:
            rec = indexed[m].get(cid)
            if rec is None:
                panels.append(
                    f"<div class='model-panel'>"
                    f"<h4 style='background:{colour_map[m]}'>{m} — <em>not present</em></h4>"
                    f"</div>"
                )
                continue

            ann = rec.get("_annotation", {})
            spans = _spans(rec, key)
            fail_count = ann.get(
                "evidence_failures" if key == "evidence_numbers" else "claim_failures", 0
            )
            fail_details_key = (
                "evidence_failure_details" if key == "evidence_numbers"
                else "claim_failure_details"
            )
            fail_details: List[Dict[str, Any]] = ann.get(fail_details_key, [])
            span_valid = ann.get(
                "evidence_valid" if key == "evidence_numbers" else "claim_valid", True
            )
            fail_badge = (
                f"<span class='badge fail'>{fail_count} failure(s)</span>" if fail_count else ""
            )
            invalid_badge = (
                "<span class='badge warn'>spans_invalid</span>" if not span_valid else ""
            )
            highlighted = _highlight_text(doc_text, spans, colour_map[m])
            ref_rec = indexed.get(ref, {}).get(cid) if m != ref else None
            ref_spans_for_table = _spans(ref_rec, key) if ref_rec is not None else None
            not_in_ref_count = (
                sum(
                    1 for s in spans
                    if not any(
                        s.char_start < r.char_end and s.char_end > r.char_start
                        for r in ref_spans_for_table
                    )
                )
                if ref_spans_for_table is not None else 0
            )
            missing_spans = (
                [
                    r for r in ref_spans_for_table
                    if not any(
                        r.char_start < s.char_end and r.char_end > s.char_start
                        for s in spans
                    )
                ]
                if ref_spans_for_table is not None else []
            )
            missing_count = len(missing_spans)
            extra_badge = (
                f"<span class='badge extra'>{not_in_ref_count} not in {ref}</span>"
                if not_in_ref_count else ""
            )
            missing_badge = (
                f"<span class='badge missing'>{missing_count} missing</span>"
                if missing_count else ""
            )
            if ref_spans_for_table is not None:
                tab_id = f"tg_{rec_idx}_{m}"
                span_details = (
                    f"<div class='tab-group' id='{tab_id}'>"
                    f"<div class='tab-bar'>"
                    f"<button class='tab-btn active' data-tab='found'"
                    f" onclick=\"showTab('{tab_id}','found')\">Found ({len(spans)})</button>"
                    f"<button class='tab-btn' data-tab='missing'"
                    f" onclick=\"showTab('{tab_id}','missing')\">Missing ({missing_count})</button>"
                    f"</div>"
                    f"<div class='tab-pane' data-tab='found'>{_span_table(spans, ref_spans_for_table)}</div>"
                    f"<div class='tab-pane' data-tab='missing' style='display:none'>"
                    f"{_missing_span_table(missing_spans)}</div>"
                    f"</div>"
                )
            else:
                span_details = (
                    f"<details><summary>{len(spans)} spans</summary>"
                    f"{_span_table(spans)}</details>"
                )
            panels.append(
                f"<div class='model-panel'>"
                f"<h4 style='background:{colour_map[m]}'>{m} &nbsp;"
                f"{fail_badge}{invalid_badge}{extra_badge}{missing_badge}</h4>"
                f"{_failure_section(fail_details)}"
                f"{span_details}"
                f"<div class='doc-text'>{highlighted}</div>"
                f"</div>"
            )

        record_html_parts.append(
            f"<section class='record'>"
            f"<h3>{html_lib.escape(str(cid))} "
            f"<span class='label'>{html_lib.escape(label)}</span>"
            f"<span class='label-orig'>{html_lib.escape(label_original)}</span></h3>"
            f"<p class='claim'><strong>Claim:</strong> {html_lib.escape(claim_text)}</p>"
            f"<div class='panels'>{''.join(panels)}</div>"
            f"</section>"
        )

    records_html = "\n".join(record_html_parts)

    # ── legend ────────────────────────────────────────────────────────────────
    legend = " ".join(
        f"<span style='background:{colour_map[m]};padding:2px 8px;border-radius:4px'>{m}</span>"
        for m in models
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Model annotation comparison — {key}</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 0; padding: 1rem 2rem; color: #222; }}
  h1 {{ border-bottom: 2px solid #ddd; padding-bottom: .4rem; }}
  h2 {{ margin-top: 2rem; }}
  table {{ border-collapse: collapse; font-size: .85rem; margin: .5rem 0 1.5rem; }}
  th, td {{ border: 1px solid #ccc; padding: 4px 10px; text-align: left; }}
  th {{ background: #f0f0f0; }}
  td.fail {{ background: #ffe0e0; font-weight: bold; }}
  .badge {{ font-size: .75rem; border-radius: 4px; padding: 1px 6px; margin-left: 4px; }}
  .badge.fail {{ background: #f99; }}
  .badge.warn {{ background: #fd9; }}
  .record {{ border: 1px solid #ddd; border-radius: 6px; padding: 1rem; margin-bottom: 1.5rem; }}
  .record h3 {{ margin: 0 0 .4rem; font-size: 1rem; }}
  .label {{ font-size: .8rem; background: #d4edda; padding: 2px 7px; border-radius: 4px; margin-left: 6px; }}
  .label-orig {{ font-size: .8rem; background: #e9ecef; padding: 2px 7px; border-radius: 4px; margin-left: 4px; }}
  .claim {{ font-style: italic; color: #555; margin: .3rem 0 .7rem; }}
  .panels {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 1rem; }}
  .model-panel {{ border: 1px solid #ddd; border-radius: 4px; overflow: hidden; }}
  .model-panel h4 {{ margin: 0; padding: 4px 10px; font-size: .9rem; }}
  .doc-text {{ font-size: .82rem; line-height: 1.6; padding: .5rem; white-space: pre-wrap; word-break: break-word; max-height: 300px; overflow-y: auto; }}
  details summary {{ cursor: pointer; font-size: .82rem; padding: 4px 10px; background: #f8f8f8; }}
  details table {{ margin: 0; width: 100%; }}
  .fail-summary {{ background: #ffe0e0 !important; color: #900; font-weight: bold; }}
  .fail-table td, .fail-table th {{ font-size: .8rem; }}
  .fail-table td:first-child {{ font-family: monospace; background: #fff5f5; color: #c00; white-space: nowrap; }}
  .dup-flag {{ background: #fde68a; color: #92400e; border-radius: 3px; padding: 0 5px; font-size: .72rem; font-weight: bold; cursor: help; }}
  mark {{ cursor: help; }}
  tr.not-in-ref td {{ background: #fde8c8; }}
  .badge.extra {{ background: #f6c26b; color: #7a3e00; }}
  .badge.missing {{ background: #f0b8b8; color: #7a0000; }}
  .tab-group {{ border-top: 1px solid #ddd; }}
  .tab-bar {{ display: flex; padding: 4px 4px 0; background: #f8f8f8; border-bottom: 1px solid #ddd; }}
  .tab-btn {{ border: 1px solid #ccc; border-bottom: none; background: #eee; padding: 3px 10px; cursor: pointer; font-size: .82rem; margin-right: 2px; border-radius: 3px 3px 0 0; }}
  .tab-btn.active {{ background: #fff; border-color: #ddd; font-weight: 600; position: relative; bottom: -1px; border-bottom-color: #fff; }}
  .tab-pane table {{ margin: 0; width: 100%; }}
</style>
<script>
function showTab(id, name) {{
  var g = document.getElementById(id);
  g.querySelectorAll('.tab-pane').forEach(function(p) {{ p.style.display = 'none'; }});
  g.querySelectorAll('.tab-btn').forEach(function(b) {{ b.classList.remove('active'); }});
  g.querySelector('.tab-pane[data-tab="' + name + '"]').style.display = '';
  g.querySelector('.tab-btn[data-tab="' + name + '"]').classList.add('active');
}}
</script>
</head>
<body>
<h1>Model annotation comparison &mdash; <code>{key}</code></h1>
<p><strong>Legend:</strong> {legend}</p>

<h2>Per-model counts</h2>
{count_table}

{agree_section}

<h2>Per-record detail</h2>
{records_html}
</body>
</html>"""


def main() -> None:
    """Load smoke files for several models and print the comparison."""
    parser = argparse.ArgumentParser(
        description="Compare smoke annotations across models."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model labels to compare (e.g. opus haiku sonnet).",
    )
    parser.add_argument(
        "--ref",
        default="opus",
        help="Reference model for agreement scoring.",
    )
    parser.add_argument(
        "--key",
        default="evidence_numbers",
        choices=["evidence_numbers", "claim_numbers"],
    )
    parser.add_argument(
        "--html",
        metavar="PATH",
        default=None,
        help="Write an HTML report to this path (e.g. compare.html).",
    )
    args = parser.parse_args()

    loaded = {m: load_data(SMOKE.format(model=m)) for m in args.models}
    summaries = {m: summarize(loaded[m]) for m in args.models}

    print("Per-model counts:")
    for model in args.models:
        print(f"  {model}: {summaries[model]}")

    agreements: Dict[str, Dict[str, float]] = {}
    if args.ref in loaded:
        print(f"\nAgreement vs '{args.ref}' on {args.key}:")
        for model in args.models:
            if model == args.ref:
                continue
            agree = compare(loaded[args.ref], loaded[model], args.key)
            agreements[model] = agree
            print(f"  {model}: {agree}")

    if args.html:
        html_path = Path(args.html)
        html_path.write_text(
            generate_html(
                models=args.models,
                loaded=loaded,
                summaries=summaries,
                agreements=agreements,
                ref=args.ref,
                key=args.key,
            ),
            encoding="utf-8",
        )
        print(f"\nHTML report written to: {html_path}")


if __name__ == "__main__":
    main()
