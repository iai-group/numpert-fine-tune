"""Guarded LLM second pass on top of the rule-cleaned splits.

For each record the 12B locator quotes verdict spans; a per-removal keep-good
gate tests each verbatim span in isolation (excise it, re-audit; keep only if
the audit does NOT flag removed_real_evidence). Kept spans are applied via the
existing phrase removal. The LLM never edits directly and never deletes a span
the audit calls evidence; every decision is logged.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.data_preprocessing.perturbations_engine.utils.load_save import (
    load_data,
    save_data,
)
from src.data_preprocessing.raw_dataset_preprocessing.verdict_strip.audit_llm import (  # noqa: E501
    PROMPTS,
    AuditClient,
)
from src.data_preprocessing.raw_dataset_preprocessing.verdict_strip.llm_locate import (  # noqa: E501
    LocateClient,
    apply_located_removals,
    locate_span,
)
from src.data_preprocessing.raw_dataset_preprocessing.verdict_strip.rules import (  # noqa: E501
    domain_of,
)
from src.number_annotation.core.segmentation import segment

# Docs longer than this are located chunk-by-chunk: dw.com's ~19k-char
# multi-fact-check compilations carry dozens of verdict spans, and a
# single locate response listing them all truncates at any sane
# max_tokens (observed at 512/2048/3072/6144 across jobs 463632-466486).
LOCATE_CHUNK_CHARS = 6000


def chunk_doc(doc: str, chunk_chars: int = LOCATE_CHUNK_CHARS) -> List[str]:
    """Split `doc` into contiguous sentence-aligned slices <= chunk_chars.

    Each chunk is a verbatim substring of `doc`, so quotes located inside a
    chunk are verbatim in the full doc too (the gate needs that).
    """
    if len(doc) <= chunk_chars:
        return [doc]
    sentences = segment(doc).sentences
    if not sentences:
        return [doc]
    chunks: List[str] = []
    start = 0
    end = 0
    for s in sentences:
        if end > start and s.end - start > chunk_chars:
            chunks.append(doc[start:end])
            start = end
        end = s.end
    if end > start:
        chunks.append(doc[start:end])
    return chunks or [doc]


def locate_quotes(
    doc: str, locate_client: Any, rec_id: Any
) -> Tuple[bool, List[str]]:
    """Locate verdict quotes, chunking long docs. Returns (has, quotes)."""
    has_verdict = False
    quotes: List[str] = []
    for chunk in chunk_doc(doc):
        located = locate_client.locate(
            {"id": rec_id, "cleaned_doc": chunk}
        )
        has_verdict = has_verdict or bool(located.get("has_verdict"))
        for q in located.get("verdict_quotes", []):
            if q and q not in quotes:
                quotes.append(q)
    return has_verdict, quotes


@dataclass
class GateResult:
    cleaned: str
    kept: List[str] = field(default_factory=list)
    rejected: List[str] = field(default_factory=list)
    unlocatable: List[str] = field(default_factory=list)
    applied: List[Tuple[int, int, str]] = field(default_factory=list)


def gate_removals(
    doc: str,
    quotes: List[str],
    audit_client: Any,
    rec_id: Any,
) -> GateResult:
    """Keep only located cuts the audit does not flag as removing evidence.

    Removal is whole-SENTENCE: a located verdict quote excises the PySBD
    sentence(s) it overlaps, not just the quoted fragment, so no dangling
    sentence remnants are left behind.
    """
    kept: List[str] = []
    rejected: List[str] = []
    unlocatable: List[str] = []
    for q in quotes:
        if not q:
            continue
        if locate_span(doc, q) is None:
            unlocatable.append(q)
            continue
        cand = apply_located_removals(doc, [q], "sentence")
        verdict = audit_client.check(
            {"id": rec_id, "cleaned_doc": cand.cleaned, "removed": [q]}
        )
        if verdict["removed_real_evidence"]:
            rejected.append(q)
        else:
            kept.append(q)
    final = apply_located_removals(doc, kept, "sentence")
    return GateResult(
        cleaned=final.cleaned,
        kept=kept,
        rejected=rejected,
        unlocatable=unlocatable,
        applied=final.applied,
    )


def hybrid_strip_record(
    rec: Dict[str, Any],
    locate_client: Any,
    audit_client: Any,
) -> Dict[str, Any]:
    """Locate verdicts in a rule-cleaned record and apply the gated removals."""
    doc = rec.get("doc", "")
    out: Dict[str, Any] = {
        "id": rec.get("id"),
        "datasplit": rec.get("datasplit"),
        "domain": domain_of(rec.get("url", "") or ""),
        "doc": doc,
        "had_locate": False,
        "kept": [],
        "rejected": [],
        "unlocatable": [],
        "error": None,
    }
    try:
        has_verdict, quotes = locate_quotes(
            doc, locate_client, rec.get("id")
        )
        out["had_locate"] = has_verdict
        gate = gate_removals(doc, quotes, audit_client, rec.get("id"))
        out["doc"] = gate.cleaned
        out["kept"] = gate.kept
        out["rejected"] = gate.rejected
        out["unlocatable"] = gate.unlocatable
    except Exception as exc:  # isolate one bad record; keep rule-cleaned doc
        out["doc"] = doc
        out["error"] = str(exc)
    return out


def hybrid_strip_split(
    records: List[Dict[str, Any]],
    locate_client: Any,
    audit_client: Any,
    max_workers: int = 16,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Process a split concurrently. Returns (cleaned_records, llm_difflog).

    A record's `doc` is replaced only when the gate kept >=1 cut. The diff log
    has one entry per record where the LLM kept or rejected something, or that
    errored.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(
            ex.map(
                lambda r: hybrid_strip_record(r, locate_client, audit_client),
                records,
            )
        )
    cleaned: List[Dict[str, Any]] = []
    difflog: List[Dict[str, Any]] = []
    for rec, res in zip(records, results):
        out = dict(rec)
        if res["kept"]:
            out["doc"] = res["doc"]
        cleaned.append(out)
        if res["kept"] or res["rejected"] or res["error"]:
            difflog.append({
                "id": res["id"],
                "datasplit": res["datasplit"],
                "domain": res["domain"],
                "kept": res["kept"],
                "rejected": res["rejected"],
                "unlocatable": res["unlocatable"],
                "error": res["error"],
            })
    return cleaned, difflog


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Guarded LLM second pass over rule-cleaned splits."
    )
    parser.add_argument(
        "files", nargs="+", help="Rule-cleaned split JSON files."
    )
    parser.add_argument(
        "--out-dir",
        default="data/raw_data/verdict_stripped/hybrid",
        help="Directory for hybrid-cleaned splits + llm_diff_log.jsonl.",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=LocateClient.max_tokens,
        help="Locate-call completion cap (dw.com compilations need >512).",
    )
    parser.add_argument(
        "--gate-prompt",
        choices=tuple(PROMPTS),
        default="baseline",
        help="Audit prompt for the keep-good gate. 'greedy' accepts cuts "
        "whose span is verdict language even when entangled with evidence; "
        "'greedy_v2' also accepts fact-bearing spans whose framing resolves "
        "the claim (Conclusion/Fact/rating recaps). "
        "See docs/greedy_gate_experiment.md.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the diff log only; skip writing cleaned splits.",
    )
    args = parser.parse_args()

    locate_client = LocateClient(
        model=args.model, base_url=args.base_url, max_tokens=args.max_tokens
    )
    audit_client = AuditClient(
        model=args.model,
        base_url=args.base_url,
        system_prompt=PROMPTS[args.gate_prompt],
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    difflog_path = out_dir / "llm_diff_log.jsonl"
    total_changed = 0

    with difflog_path.open("w", encoding="utf-8") as logf:
        for f in args.files:
            src = Path(f)
            records = load_data(str(src))
            cleaned, difflog = hybrid_strip_split(
                records, locate_client, audit_client, args.max_workers
            )
            changed = sum(1 for d in difflog if d["kept"])
            total_changed += changed
            for line in difflog:
                logf.write(json.dumps(line, ensure_ascii=False) + "\n")
            if not args.dry_run:
                save_data(cleaned, str(out_dir / src.name))
            print(
                f"{src.name}: {changed} changed, "
                f"{len(difflog)} logged / {len(records)} records"
            )
    print(f"total changed: {total_changed}; diff log -> {difflog_path}")


if __name__ == "__main__":
    main()
