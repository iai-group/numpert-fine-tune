"""Self-contained HTML report for NLI scoring results.

Each record's evidence document is rendered with scored numbers highlighted;
background colour scales with relevance (1=pale → 5=strong) and the hover title
carries the relation + rationale. Unscored numbers (``nli is None``) are left as
plain text.
"""

import html as html_lib
import random
from typing import Any, Dict, List

# Pale -> strong, indexed by relevance 1..5.
_RELEVANCE_COLOURS = {
    1: "#f0f0f0",
    2: "#d9ecff",
    3: "#a9d4ff",
    4: "#ffd699",
    5: "#ff9e80",
}


def _highlight_doc(doc: str, evidence_numbers: List[Dict[str, Any]]) -> str:
    """Wrap each scored evidence number in a coloured <mark> tag."""
    scored = [
        e for e in evidence_numbers
        if e.get("nli") and e.get("char_start") is not None
    ]
    if not scored:
        return html_lib.escape(doc)
    ordered = sorted(scored, key=lambda e: e["char_start"])
    parts: List[str] = []
    cursor = 0
    for e in ordered:
        start, end = e["char_start"], e["char_end"]
        if start < cursor:
            continue
        nli = e["nli"]
        colour = _RELEVANCE_COLOURS.get(nli["relevance"], "#f0f0f0")
        title = (
            f"rel={nli['relevance']} | {nli['relation']} | "
            f"claim_num={nli['claim_num']} | {nli['rationale']}"
        )
        parts.append(html_lib.escape(doc[cursor:start]))
        parts.append(
            f'<mark style="background:{colour};border-radius:3px;'
            f'padding:0 2px" title="{html_lib.escape(title)}">'
            f"{html_lib.escape(doc[start:end])}</mark>"
        )
        cursor = end
    parts.append(html_lib.escape(doc[cursor:]))
    return "".join(parts)


def _dist(d: Dict[Any, int]) -> str:
    """Render a {key: count} distribution as 'k: v, k: v' (sorted by key)."""
    return ", ".join(f"{k}: {v}" for k, v in sorted(d.items(), key=str))


def _summary_html(records: List[Dict[str, Any]]) -> str:
    """Run-level rollup: span counts, distributions, perturbable gate."""
    spans = [e for r in records for e in r.get("evidence_numbers", [])]
    attached = [e for e in spans if e.get("nli")]
    relations: Dict[str, int] = {}
    relevance: Dict[int, int] = {}
    for e in attached:
        nli = e["nli"]
        relations[nli["relation"]] = relations.get(nli["relation"], 0) + 1
        relevance[nli["relevance"]] = relevance.get(nli["relevance"], 0) + 1
    pair: Dict[str, int] = {}
    to_false = to_true = 0
    for r in records:
        agg = r.get("nli_aggregate")
        if not agg:
            continue
        pair[agg["pair_relation"]] = pair.get(agg["pair_relation"], 0) + 1
        to_false += bool(agg["perturbable"]["to_false"])
        to_true += bool(agg["perturbable"]["to_true"])
    return (
        "<section style='padding:1em;border:2px solid #444;"
        "background:#fafafa'>"
        "<h2>Summary</h2>"
        f"<p>Spans attached: {len(attached)}/{len(spans)} "
        f"(errored: {len(spans) - len(attached)})</p>"
        f"<p>Relations: {_dist(relations)}</p>"
        f"<p>Relevance histogram: {_dist(relevance)}</p>"
        f"<p>Pair relations ({len(records)} records): {_dist(pair)}</p>"
        f"<p>Perturbable — to_false: {to_false}, to_true: {to_true}</p>"
        "</section>"
    )


def _record_html(record: Dict[str, Any]) -> str:
    """Render one record block (claim + highlighted evidence)."""
    claim = html_lib.escape(record.get("claim", ""))
    claim_id = html_lib.escape(str(record.get("claim_id", "")))
    body = _highlight_doc(
        record.get("doc", ""), record.get("evidence_numbers", [])
    )
    return (
        f"<section style='margin:1.5em 0;padding:1em;border:1px solid #ccc'>"
        f"<h3>{claim_id}</h3>"
        f"<p><strong>Claim:</strong> {claim}</p>"
        f"<p style='line-height:1.8'>{body}</p>"
        f"</section>"
    )


def generate_nli_html(
    records: List[Dict[str, Any]], max_records: int = 0, seed: int = 0
) -> str:
    """Build a self-contained HTML report string.

    ``max_records`` > 0 renders only a deterministic sample of that many
    record blocks (a full split would produce a browser-choking file); the
    summary block always covers *all* records.
    """
    legend = (
        "<p>Relevance legend: "
        + " ".join(
            f"<mark style='background:{c};padding:0 4px'>{r}</mark>"
            for r, c in sorted(_RELEVANCE_COLOURS.items())
        )
        + " — hover a number for relation + rationale.</p>"
    )
    summary = _summary_html(records)
    shown = records
    note = ""
    if max_records and len(records) > max_records:
        shown = random.Random(seed).sample(records, max_records)
        note = (
            f"<p><em>Showing {max_records} of {len(records)} records "
            f"(sample seed {seed}).</em></p>"
        )
    blocks = note + "".join(_record_html(r) for r in shown)
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>Claim↔Evidence Number NLI</title></head><body>"
        "<h1>Claim↔Evidence Number Relevance / NLI</h1>"
        f"{summary}{legend}{blocks}</body></html>"
    )


def write_nli_report(
    records: List[Dict[str, Any]],
    path: str,
    max_records: int = 0,
    seed: int = 0,
) -> None:
    """Write the HTML report to ``path``."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(generate_nli_html(records, max_records=max_records,
                                   seed=seed))
