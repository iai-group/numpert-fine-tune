"""Empirical discovery for footer / sign-off stripping.

Two scans over the pristine QuanTemp corpus (all 3 splits):
  (A) targeted signature scan  -> validate the 6 known footer signatures
  (B) trailing-block discovery -> per-domain trailing sentences repeated
      across many docs (templated footers the handoff didn't enumerate).

Run:  python3 footer_discovery.py  (from repo root or anywhere; paths absolute)
Prints a markdown report to stdout.
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

REPO = Path("/mnt/beegfs/home/u2929246/claim-evidence-perturbs")
SPLITS = [
    REPO / "data/raw_data/QuanTemp-main-data/train_claims_quantemp.json",
    REPO / "data/raw_data/QuanTemp-main-data/val_claims_quantemp.json",
    REPO / "data/raw_data/QuanTemp-main-data/test_claims_quantemp.json",
]


def domain_of(url: str) -> str:
    if not url:
        return ""
    netloc = urlparse(url).netloc.lower()
    return netloc[4:] if netloc.startswith("www.") else netloc


# ---- known signatures from the handoff (label -> regex) --------------------
SIGNATURES = {
    "Ferret Fact Service (FFS) is": r"Ferret Fact Service \(FFS\) is",
    "VERA reader-request / truth bot": r"Have you seen any dubious claims|VERA, the truth bot",
    "IFCN FactChat … WhatsApp": r"This fact check is available at IFCN",
    "FactCheckNI is Northern Ireland": r"FactCheckNI is Northern Ireland",
    "IFCN / Code of Principles": r"International Fact-?Checking Network|Code of Principles",
    "methodology": r"fact-?checking methodology|our methodology",
}

# sentence splitter: keep it dumb but stable (mirror engine's [.!?] style)
_SENT = re.compile(r"[^.!?]*[.!?]+|\S[^.!?]*$")


def sentences(text):
    return [m.group(0).strip() for m in _SENT.finditer(text) if m.group(0).strip()]


def norm(s):
    s = s.lower()
    s = re.sub(r"https?://\S+", "<url>", s)
    s = re.sub(r"\S+@\S+", "<email>", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load():
    rows = []
    for p in SPLITS:
        rows.extend(json.load(open(p)))
    return rows


def scan_signatures(rows):
    print("## (A) Targeted signature scan\n")
    print("| Signature | docs | top domains |")
    print("|---|---:|---|")
    for label, pat in SIGNATURES.items():
        rx = re.compile(pat, re.IGNORECASE)
        dom = Counter()
        n = 0
        for r in rows:
            if rx.search(r.get("doc") or ""):
                n += 1
                dom[domain_of(r.get("url") or "")] += 1
        top = ", ".join(f"{d} ({c})" for d, c in dom.most_common(5))
        print(f"| {label} | {n} | {top} |")
    print()


def scan_trailing(rows, last_k=3, min_docs=5, min_frac=0.15):
    """Per-domain: count normalized trailing sentences across docs."""
    by_dom_docs = defaultdict(int)
    # trailing sentence -> set of doc indices (per domain)
    tail_counts = defaultdict(Counter)
    tail_example = {}
    for r in rows:
        d = domain_of(r.get("url") or "")
        doc = r.get("doc") or ""
        by_dom_docs[d] += 1
        sents = sentences(doc)
        seen = set()
        for s in sents[-last_k:]:
            ns = norm(s)
            if len(ns) < 25:  # skip trivially short tails
                continue
            if ns in seen:
                continue
            seen.add(ns)
            tail_counts[d][ns] += 1
            tail_example.setdefault((d, ns), s.strip())

    print("## (B) Trailing-block discovery "
          f"(last {last_k} sentences, repeated in >= {min_docs} docs "
          f"and >= {int(min_frac*100)}% of domain)\n")
    print(f"Scanned {len(rows)} docs across {len(by_dom_docs)} domains.\n")
    candidates = []
    for d, ctr in tail_counts.items():
        total = by_dom_docs[d]
        for ns, c in ctr.items():
            if c >= min_docs and c / total >= min_frac:
                candidates.append((c, c / total, d, total, tail_example[(d, ns)]))
    candidates.sort(reverse=True)
    print(f"Found {len(candidates)} repeated trailing-sentence templates.\n")
    print("| docs | frac | domain (domain docs) | example trailing sentence |")
    print("|---:|---:|---|---|")
    for c, frac, d, total, ex in candidates[:60]:
        ex = ex.replace("|", "\\|")
        if len(ex) > 160:
            ex = ex[:157] + "…"
        print(f"| {c} | {frac:.0%} | {d} ({total}) | {ex} |")
    print()


def main():
    rows = load()
    print(f"# Footer-stripping empirical discovery\n")
    print(f"Corpus: {len(rows)} pristine docs (train+val+test).\n")
    scan_signatures(rows)
    scan_trailing(rows)


if __name__ == "__main__":
    main()
