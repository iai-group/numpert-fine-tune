"""Sample audit of stripped evidence via the vLLM-served model.

The LLM NEVER edits text. For each sampled record it answers two yes/no
checks:
  - still_has_verdict   : does the cleaned doc still state a verdict?
                          (missed leak)
  - removed_real_evidence: did the removed span include genuine evidence?
                          (over-removal)
A record flagged on either check is surfaced for human review.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List

import requests

from src.data_preprocessing.perturbations_engine.utils.load_save import load_data
from src.data_preprocessing.raw_dataset_preprocessing.verdict_strip.rules import (  # noqa: E501
    domain_of,
)

AUDIT_SYSTEM_PROMPT = (
    "You are auditing automated removal of fact-checker verdict language "
    "from evidence text. Answer ONLY with the requested JSON.\n"
    "still_has_verdict = true ONLY if the cleaned text still contains an "
    "EXPLICIT verdict/rating STATEMENT by the fact-checker that directly "
    "resolves the claim - e.g. 'We rate this False', 'Verdict: incorrect', "
    "'Three Pinocchios', 'This claim is misleading', or a standalone rating "
    "label.\n"
    "still_has_verdict = false for EVERYTHING else, including investigative "
    "findings or analysis that merely imply a conclusion - e.g. 'we found no "
    "credible sources', 'there is no evidence that this is true', 'the UK has "
    "not banned X', attributions to other organisations ('debunked by Full "
    "Fact'), or meta-notes like 'the rating is not changed'. Those are "
    "EVIDENCE, not a verdict.\n"
    "removed_real_evidence = true if the removed spans include genuine "
    "evidence/analysis rather than only verdict language."
)

# Greedy-gate variant (2026-07-05 experiment, see
# docs/greedy_gate_experiment.md): the baseline prompt rejects a cut whenever
# the span "includes genuine evidence", which keeps hedged/claim-restating
# verdict sentences in the corpus (561 rejected cuts in the 463632 run, 30%
# of them hedged rulings). This variant flips the bias: a span whose PRIMARY
# function is rendering the verdict is removable even when evidence is
# entangled; reject only when the span carries substantive facts the
# remaining text does not already state. still_has_verdict is unchanged.
GREEDY_AUDIT_SYSTEM_PROMPT = (
    "You are auditing automated removal of fact-checker verdict language "
    "from evidence text. Answer ONLY with the requested JSON.\n"
    "still_has_verdict = true ONLY if the cleaned text still contains an "
    "EXPLICIT verdict/rating STATEMENT by the fact-checker that directly "
    "resolves the claim - e.g. 'We rate this False', 'Verdict: incorrect', "
    "'Three Pinocchios', 'This claim is misleading', or a standalone rating "
    "label.\n"
    "still_has_verdict = false for EVERYTHING else, including investigative "
    "findings or analysis that merely imply a conclusion - e.g. 'we found no "
    "credible sources', 'there is no evidence that this is true', 'the UK has "
    "not banned X', attributions to other organisations ('debunked by Full "
    "Fact'), or meta-notes like 'the rating is not changed'. Those are "
    "EVIDENCE, not a verdict.\n"
    "removed_real_evidence = true ONLY if the removed spans contain "
    "SUBSTANTIVE factual content that the CLEANED text does not state "
    "anywhere else - specific figures, dates, named sources, or "
    "investigative findings that would be lost. A sentence whose main "
    "purpose is to state, restate, hedge, or justify the fact-checker's "
    "conclusion is VERDICT language, NOT evidence, even when it (a) "
    "restates the claim being checked ('his claim that X is true'), (b) "
    "qualifies the rating ('accurate but needs clarification'), or (c) "
    "summarises the reasoning behind the rating ('We rated it False "
    "because ...') while the supporting facts remain in the cleaned text.\n"
    "When uncertain, prefer removed_real_evidence = false: a leaked verdict "
    "is worse than losing a redundant concluding sentence."
)

# Greedy-v2 variant (2026-07-09, item 8, see docs/greedy_gate_experiment.md):
# greedy-v1's few surviving rejections all read as fact-checker CONCLUSIONS
# that carry some unique facts (indiatoday "Conclusion ...", factly "Fact:
# ...", politifact rating recaps, snopes refutations) whose removal improves
# the doc per user direction ("lean further toward acceptance"). v2 tightens
# removed_real_evidence further: a fact-bearing span is STILL removable when
# its framing RESOLVES the checked claim; reject only standalone background
# reporting that does not itself answer the claim. still_has_verdict is
# unchanged from baseline/v1.
GREEDY_V2_AUDIT_SYSTEM_PROMPT = (
    "You are auditing automated removal of fact-checker verdict language "
    "from evidence text. Answer ONLY with the requested JSON.\n"
    "still_has_verdict = true ONLY if the cleaned text still contains an "
    "EXPLICIT verdict/rating STATEMENT by the fact-checker that directly "
    "resolves the claim - e.g. 'We rate this False', 'Verdict: incorrect', "
    "'Three Pinocchios', 'This claim is misleading', or a standalone rating "
    "label.\n"
    "still_has_verdict = false for EVERYTHING else, including investigative "
    "findings or analysis that merely imply a conclusion - e.g. 'we found no "
    "credible sources', 'there is no evidence that this is true', 'the UK has "
    "not banned X', attributions to other organisations ('debunked by Full "
    "Fact'), or meta-notes like 'the rating is not changed'. Those are "
    "EVIDENCE, not a verdict.\n"
    "removed_real_evidence = true ONLY if the removed span is ESSENTIALLY "
    "ONLY standalone factual reporting with NO verdict or conclusion "
    "function - it neither states, restates, hedges, nor justifies the "
    "fact-checker's conclusion, AND does not itself RESOLVE the checked "
    "claim. A span that carries specific figures, dates, or names is STILL "
    "verdict language (removable) when its framing resolves the claim - for "
    "example a 'Conclusion', 'Fact:', 'Our rating', or 'Our ruling' section, "
    "or a sentence that refutes or corrects the checked claim ('X did not "
    "happen', 'the photo actually shows Y', 'the clip is from an earlier "
    "event') - because its primary function is delivering the fact-checker's "
    "answer, and downstream evidence must not contain that answer. Reject "
    "the cut (removed_real_evidence = true) only for spans of background "
    "reporting that do NOT themselves resolve the claim and whose facts the "
    "cleaned text would otherwise lose entirely.\n"
    "When uncertain, prefer removed_real_evidence = false: a leaked verdict "
    "or conclusion is worse than losing a concluding sentence, and "
    "downstream evidence should never contain the fact-checker's own "
    "resolution of the claim."
)

# Greedy-v3 (2026-07-09, item 8 second iteration): the full run with v2
# (job 466703) REGRESSED on the broad distribution -- 138 rejections vs
# v1's 21, and 123 of the 138 quotes were KEPT by v1 in the archived
# 466490 run (pure hedged-rating/true-confirmation boilerplate like
# "Half True", "The statement is partially accurate but leaves out
# important details"). Root cause: v2 REPLACED v1's explicit carve-outs
# (a)/(b)/(c) with the claim-resolution framing; the 466689 A/B only
# replayed v1's rejections, so the regression on v1's ACCEPTANCES was
# invisible. v3 = v1's removed_real_evidence text verbatim + the
# claim-resolution carve-out APPENDED as (d). still_has_verdict unchanged.
GREEDY_V3_AUDIT_SYSTEM_PROMPT = (
    "You are auditing automated removal of fact-checker verdict language "
    "from evidence text. Answer ONLY with the requested JSON.\n"
    "still_has_verdict = true ONLY if the cleaned text still contains an "
    "EXPLICIT verdict/rating STATEMENT by the fact-checker that directly "
    "resolves the claim - e.g. 'We rate this False', 'Verdict: incorrect', "
    "'Three Pinocchios', 'This claim is misleading', or a standalone rating "
    "label.\n"
    "still_has_verdict = false for EVERYTHING else, including investigative "
    "findings or analysis that merely imply a conclusion - e.g. 'we found no "
    "credible sources', 'there is no evidence that this is true', 'the UK has "
    "not banned X', attributions to other organisations ('debunked by Full "
    "Fact'), or meta-notes like 'the rating is not changed'. Those are "
    "EVIDENCE, not a verdict.\n"
    "removed_real_evidence = true ONLY if the removed spans contain "
    "SUBSTANTIVE factual content that the CLEANED text does not state "
    "anywhere else - specific figures, dates, named sources, or "
    "investigative findings that would be lost. A sentence whose main "
    "purpose is to state, restate, hedge, or justify the fact-checker's "
    "conclusion is VERDICT language, NOT evidence, even when it (a) "
    "restates the claim being checked ('his claim that X is true'), (b) "
    "qualifies the rating ('accurate but needs clarification'), (c) "
    "summarises the reasoning behind the rating ('We rated it False "
    "because ...') while the supporting facts remain in the cleaned text, "
    "or (d) delivers the fact-checker's RESOLUTION of the claim - a "
    "'Conclusion' or 'Fact:' section answer, or a sentence that refutes, "
    "confirms, or corrects the checked claim ('X did not happen', 'the "
    "photo actually shows Y', 'the numbers check out') - even when that "
    "resolving sentence carries specific figures, dates, or names, because "
    "its primary function is delivering the fact-checker's answer and "
    "downstream evidence must not contain that answer.\n"
    "When uncertain, prefer removed_real_evidence = false: a leaked verdict "
    "or conclusion is worse than losing a redundant concluding sentence."
)

# Name -> gate/audit system prompt. Shared by hybrid_strip (--gate-prompt)
# and reaudit_rejected (--baseline-prompt / --candidate-prompt) so the
# name->prompt mapping is defined once.
PROMPTS = {
    "baseline": AUDIT_SYSTEM_PROMPT,
    "greedy": GREEDY_AUDIT_SYSTEM_PROMPT,
    "greedy_v2": GREEDY_V2_AUDIT_SYSTEM_PROMPT,
    "greedy_v3": GREEDY_V3_AUDIT_SYSTEM_PROMPT,
}

AUDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "still_has_verdict": {"type": "boolean"},
        "removed_real_evidence": {"type": "boolean"},
    },
    "required": ["still_has_verdict", "removed_real_evidence"],
    "additionalProperties": False,
}


def _user_prompt(rec: Dict[str, Any]) -> str:
    removed = "\n".join(f"- {t}" for t in rec.get("removed", []))
    return (
        f"CLEANED EVIDENCE:\n{rec.get('cleaned_doc', '')}\n\n"
        f"REMOVED SPANS:\n{removed}\n\n"
        'Return JSON: {"still_has_verdict": bool, '
        '"removed_real_evidence": bool}'
    )


@dataclass
class AuditClient:
    model: str
    base_url: str
    # 300 s read-timed-out on huge docs under load (2026-07-07 run).
    timeout: int = 600
    system_prompt: str = AUDIT_SYSTEM_PROMPT
    post: Callable[..., Any] = requests.post

    def check(self, rec: Dict[str, Any]) -> Dict[str, bool]:
        content = f"{self.system_prompt}\n\n{_user_prompt(rec)}"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 200,
            "temperature": 0.0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "audit",
                    "strict": True,
                    "schema": AUDIT_SCHEMA,
                },
            },
        }
        url = self.base_url.rstrip("/") + "/chat/completions"
        resp = self.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        choice = resp.json()["choices"][0]
        if choice.get("finish_reason") == "length":
            raise RuntimeError(
                "vLLM audit response truncated at max_tokens=200 "
                "(finish_reason=length). Increase max_tokens."
            )
        content = choice["message"]["content"]
        return json.loads(content)


@dataclass
class FakeAuditClient:
    """Test double: canned verdicts keyed by record id."""

    verdicts: Dict[Any, Dict[str, bool]]

    def check(self, rec: Dict[str, Any]) -> Dict[str, bool]:
        return self.verdicts.get(
            rec["id"],
            {"still_has_verdict": False, "removed_real_evidence": False},
        )


def build_audit_records(
    cleaned_records: List[Dict[str, Any]],
    difflog: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Join a cleaned split with its diff log into audit-ready records.

    Bridges the shapes that `clean_splits` emits (a cleaned split keyed by
    `id`/`url`/`doc`, and a `diff_log.jsonl` keyed by `id`/`domain`/`removals`)
    to the shape `stratified_sample`/`audit_records` consume: each record gets
    `id`, `domain`, `changed`, `cleaned_doc`, and `removed` (the list of removed
    text spans; empty for untouched records). A record is `changed` iff it has
    a diff-log entry. The join keys on `(datasplit, id)` because the combined
    diff log spans all splits while each split's ids restart at 0.
    """
    by_key = {(e.get("datasplit"), e.get("id")): e for e in difflog}
    out: List[Dict[str, Any]] = []
    for rec in cleaned_records:
        entry = by_key.get((rec.get("datasplit"), rec.get("id")))
        if entry is not None:
            domain = entry.get("domain", "")
            removed = [
                r.get("removed_text", "")
                for r in entry.get("removals", [])
            ]
            changed = True
        else:
            domain = domain_of(rec.get("url", "") or "")
            removed = []
            changed = False
        out.append(
            {
                "id": rec.get("id"),
                "domain": domain,
                "changed": changed,
                "cleaned_doc": rec.get("doc", ""),
                "removed": removed,
            }
        )
    return out


def stratified_sample(
    records: List[Dict[str, Any]],
    per_domain: int = 5,
    untouched: int = 50,
) -> List[Dict[str, Any]]:
    """Sample changed records per domain plus some untouched ones.

    Up to ``per_domain`` changed records per domain plus ``untouched``
    clean ones.
    """
    by_domain: Dict[str, List[Dict[str, Any]]] = {}
    untouched_pool: List[Dict[str, Any]] = []
    for r in records:
        if r.get("changed"):
            by_domain.setdefault(r["domain"], []).append(r)
        else:
            untouched_pool.append(r)
    sample: List[Dict[str, Any]] = []
    for domain, recs in by_domain.items():
        sample.extend(recs[:per_domain])
    sample.extend(untouched_pool[:untouched])
    return sample


def audit_records(
    records: List[Dict[str, Any]], client: Any
) -> Dict[str, Any]:
    """Run the audit over ``records``; return a report with flagged items.

    Per-record failures are isolated: a record whose `client.check` raises
    (e.g. a transient HTTP error, or a payload vLLM rejects) is recorded under
    `errored` and the run continues, rather than aborting the whole audit.
    """
    flagged: List[Dict[str, Any]] = []
    errored: List[Dict[str, Any]] = []
    for rec in records:
        try:
            verdict = client.check(rec)
        except Exception as exc:  # isolate one bad record from the rest
            errored.append({**rec, "audit_error": str(exc)})
            continue
        if verdict["still_has_verdict"] or verdict["removed_real_evidence"]:
            flagged.append({**rec, "audit": verdict})
    return {
        "n_audited": len(records),
        "flagged": flagged,
        "errored": errored,
    }


def format_audit_report(report: Dict[str, Any]) -> str:
    """Render an `audit_records` report as a human-reviewable markdown doc."""
    flagged = report["flagged"]
    errored = report.get("errored", [])
    leaks = [r for r in flagged if r["audit"]["still_has_verdict"]]
    over = [r for r in flagged if r["audit"]["removed_real_evidence"]]
    lines = [
        "# Verdict-strip sample audit",
        "",
        f"- audited: {report['n_audited']}",
        f"- flagged: {len(flagged)}",
        f"- still_has_verdict (missed leak): {len(leaks)}",
        f"- removed_real_evidence (over-removal): {len(over)}",
        f"- errored (not audited): {len(errored)}",
        "",
    ]
    for r in flagged:
        a = r["audit"]
        lines.append(
            f"## id={r.get('id')} domain={r.get('domain')}"
        )
        lines.append(
            f"- still_has_verdict={a['still_has_verdict']} "
            f"removed_real_evidence={a['removed_real_evidence']}"
        )
        lines.append(f"- cleaned_doc: {r.get('cleaned_doc', '')[:500]}")
        removed = r.get("removed", [])
        if removed:
            lines.append("- removed:")
            for t in removed:
                lines.append(f"  - {t[:300]}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample-audit stripped evidence via a vLLM model."
    )
    parser.add_argument("--cleaned", required=True, help="Cleaned split JSON.")
    parser.add_argument("--difflog", required=True, help="diff_log.jsonl.")
    parser.add_argument(
        "--report",
        default="data/raw_data/verdict_stripped/rulebased/audit_report.md",
        help="Markdown report output path.",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--per-domain", type=int, default=5)
    parser.add_argument("--untouched", type=int, default=50)
    args = parser.parse_args()

    cleaned = load_data(args.cleaned)
    with open(args.difflog, encoding="utf-8") as f:
        difflog = [json.loads(line) for line in f if line.strip()]
    records = build_audit_records(cleaned, difflog)
    sample = stratified_sample(records, args.per_domain, args.untouched)
    client = AuditClient(model=args.model, base_url=args.base_url)
    report = audit_records(sample, client)
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(format_audit_report(report), encoding="utf-8")
    print(
        f"audited {report['n_audited']}, "
        f"flagged {len(report['flagged'])} -> {args.report}"
    )


if __name__ == "__main__":
    main()
