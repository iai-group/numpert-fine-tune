"""Local HTML audit of stripper removals for manual eyeballing.

Renders ONE box per document (not per removal): the FULL pristine doc with all
its removed spans highlighted in place — footer/sign-off cuts in purple,
verdict cuts in blue — grouped by domain. Lets a human confirm that only
boilerplate/verdict was cut and the surrounding evidence reads intact.
Self-contained HTML written locally (no external upload).

Run:  python3 scripts/footer_audit_view.py                 # footer cuts only
      python3 scripts/footer_audit_view.py --category all   # verdict + footer
      python3 scripts/footer_audit_view.py --category verdict
Output: data/raw_data/verdict_stripped/rulebased/footer_audit.html
"""

import argparse
import html as html_lib
import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PRISTINE = REPO / "data/raw_data/QuanTemp-main-data"
RULEBASED = REPO / "data/raw_data/verdict_stripped/rulebased"
SPLITS = ["train", "val", "test"]


def load_pristine():
    by_key = {}
    for s in SPLITS:
        for r in json.load(open(PRISTINE / f"{s}_claims_quantemp.json")):
            by_key[(r.get("datasplit"), r.get("id"))] = r
    return by_key


def _compact(s):
    """Non-whitespace chars of `s` and their indices in `s`.

    Comparing removals on their non-whitespace sequence makes matching immune to
    every whitespace repair the stripper applies between handlers (collapsing
    `[ \\t]{2,}`, dropping spaces before punctuation, end-stripping) — none of
    which is reflected in the recorded `removed_text`.
    """
    chars, idx = [], []
    for i, ch in enumerate(s):
        if not ch.isspace():
            chars.append(ch)
            idx.append(i)
    return "".join(chars), idx


def locate_spans(doc, removals):
    """Map each removal to a (start, end, category) span in the pristine `doc`.

    The stripper records each removal's `removed_text` against the doc state at
    its OWN invocation — footer rules run before verdict rules, and each handler
    also collapses whitespace at the excision seam. So a later `section`
    removal's text can have both a content gap (where an earlier footer cut sat)
    and collapsed whitespace, making it NOT a verbatim substring of the pristine
    doc. Rather than search pristine, we REPLAY the removals cumulatively
    against a shrinking working copy — the same sequence of states the stripper
    saw — locating each on its non-whitespace sequence and mapping the match
    back to pristine indices via a surviving-index list. Returns
    (start, end, category) spans over pristine plus any removals whose content
    genuinely could not be found at replay time (flagged in the UI).
    """
    work = doc
    orig = list(range(len(doc)))  # pristine index of each surviving char
    colored = {}                  # pristine index -> category (first wins)
    misses = []
    for rm in removals:
        text = rm.get("removed_text", "")
        cat = rm.get("category", "verdict")
        tnw, _ = _compact(text)
        if not tnw:
            continue
        wnw, wpos = _compact(work)
        # Rightmost surviving occurrence: handlers excise right-to-left and
        # record removals in that order, so when a phrase repeats (and one copy
        # is nested in a larger block another removal cuts), matching from the
        # right consumes the copies in the same order the stripper did.
        i = wnw.rfind(tnw)
        if i < 0:
            misses.append(rm)
            continue
        start = wpos[i]                       # first matched non-ws char
        end = wpos[i + len(tnw) - 1] + 1      # just past the last one
        for p in orig[start:end]:
            colored.setdefault(p, cat)
        work = work[:start] + work[end:]
        orig = orig[:start] + orig[end:]
    # coalesce coloured pristine indices into contiguous same-category spans
    spans = []
    for p in sorted(colored):
        cat = colored[p]
        if spans and spans[-1][1] == p and spans[-1][2] == cat:
            spans[-1] = (spans[-1][0], p + 1, cat)
        else:
            spans.append((p, p + 1, cat))
    return spans, misses


def render_doc(doc, spans):
    """HTML-escape `doc`, wrapping each span; drop spans overlapping an
    already-emitted one (higher-priority footer wins ties by sorting)."""
    order = {"footer": 0, "verdict": 1}
    spans = sorted(spans, key=lambda s: (s[0], order.get(s[2], 9)))
    parts, cursor = [], 0
    for start, end, cat in spans:
        if start < cursor:
            continue
        parts.append(html_lib.escape(doc[cursor:start]))
        cls = "cut-footer" if cat == "footer" else "cut-verdict"
        parts.append(f'<span class="{cls}">'
                     + html_lib.escape(doc[start:end]) + "</span>")
        cursor = end
    parts.append(html_lib.escape(doc[cursor:]))
    return "".join(parts)


_CSS = """
body{font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;
background:#f6f7f9;color:#1a1a1a}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;
padding:12px 20px;z-index:5}
h1{font-size:18px;margin:0}
.sub{color:#666;font-size:13px;margin-top:4px}
.legend{margin:6px 20px 0;font-size:13px}
.legend span{padding:1px 8px;border-radius:3px;margin-right:6px}
.nav{margin:6px 20px;font-size:13px}
.nav a{margin-right:12px;white-space:nowrap}
h2{font-size:16px;margin:22px 20px 6px;padding-top:8px;
border-top:2px solid #ccc}
.card{background:#fff;margin:10px 20px;border:1px solid #e3e3e3;
border-radius:8px;padding:10px 14px}
.meta{font-size:12px;color:#888;font-family:ui-monospace,monospace;
margin-bottom:6px}
.claim{font-size:13px;color:#333;background:#eef3fb;border-left:3px solid
#4a7fd0;border-radius:4px;padding:6px 10px;margin-bottom:8px}
.doc{white-space:pre-wrap;word-wrap:break-word;background:#fafafa;
border:1px solid #eee;border-radius:6px;padding:12px;max-height:460px;
overflow:auto}
.cut-footer{background:#e8dcff;color:#4b2a86;text-decoration:line-through;
border-radius:3px;padding:0 2px}
.cut-verdict{background:#d7e6ff;color:#1a3a73;text-decoration:line-through;
border-radius:3px;padding:0 2px}
.miss{color:#a11;font-size:12px;margin-top:6px}
.miss ul{margin:2px 0 0;padding-left:18px}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="footer",
                    choices=["footer", "verdict", "all"])
    ap.add_argument("--out", default=str(RULEBASED / "footer_audit.html"))
    args = ap.parse_args()

    pristine = load_pristine()
    difflog = [
        json.loads(line)
        for line in open(RULEBASED / "diff_log.jsonl", encoding="utf-8")
    ]

    # domain -> list of (entry, [removals in scope])
    by_domain = defaultdict(list)
    n_docs = n_rem = 0
    for e in difflog:
        rems = [
            rm for rm in e.get("removals", [])
            if args.category == "all" or rm.get("category") == args.category
        ]
        if not rems:
            continue
        by_domain[e.get("domain", "")].append((e, rems))
        n_docs += 1
        n_rem += len(rems)

    domains = sorted(by_domain, key=lambda d: -len(by_domain[d]))
    nav = " ".join(
        f'<a href="#{html_lib.escape(d)}">{html_lib.escape(d)} '
        f"({len(by_domain[d])})</a>" for d in domains
    )
    parts = [
        f"<style>{_CSS}</style>",
        f"<header><h1>Removal audit ({html_lib.escape(args.category)})</h1>"
        f'<div class="sub">{n_rem} removals in {n_docs} documents across '
        f"{len(domains)} domains — full document shown, removed spans struck "
        "through in place.</div>"
        '<div class="legend"><span class="cut-footer">footer</span>'
        '<span class="cut-verdict">verdict</span></div></header>',
        f'<div class="nav">{nav}</div>',
    ]
    for d in domains:
        parts.append(f'<h2 id="{html_lib.escape(d)}">{html_lib.escape(d)} '
                     f"— {len(by_domain[d])} docs</h2>")
        for e, rems in by_domain[d]:
            key = (e.get("datasplit"), e.get("id"))
            rec = pristine.get(key) or {}
            doc = rec.get("doc", "")
            spans, misses = locate_spans(doc, rems)
            claim = rec.get("claim", "")
            rule_names = ",".join(sorted({r.get("rule", "") for r in rems}))
            miss_html = ""
            if misses:
                items = "".join(
                    f"<li>[{html_lib.escape(m.get('category', ''))}/"
                    f"{html_lib.escape(m.get('rule', ''))}] "
                    f"{html_lib.escape((m.get('removed_text') or '')[:160])}"
                    "</li>" for m in misses
                )
                miss_html = ('<div class="miss">⚠ not located verbatim: '
                             f"<ul>{items}</ul></div>")
            parts.append(
                '<div class="card"><div class="meta">'
                f"id={e.get('id')} split={e.get('datasplit')} "
                f"rules={html_lib.escape(rule_names)}"
                "</div>"
                + (f'<div class="claim">{html_lib.escape(claim)}</div>'
                   if claim else "")
                + f'<div class="doc">{render_doc(doc, spans)}</div>'
                + miss_html
                + "</div>"
            )
    out = Path(args.out)
    out.write_text("".join(parts), encoding="utf-8")
    print(f"{n_rem} {args.category} removals in {n_docs} docs across "
          f"{len(domains)} domains")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
